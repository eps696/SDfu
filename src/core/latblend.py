# edited from https://github.com/lunarring/latentblending
# Copyright 2022 Lunar Ring, originally written by Johannes Stelzer, email stelzer@lunar-ring.ai twitter @j_stelzer
# Licensed under the Apache License, Version 2.0 (the "License")

import os, sys
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import yaml
from contextlib import nullcontext

sys.path.append(os.path.join(os.path.dirname(__file__), '../xtra'))
import lpips

import torch
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)

from .utils import slerp, slerp2, lerp, blend, cvshow, progbar, isset
try: # colab
    get_ipython().__class__.__name__
    iscolab = True
except: iscolab = False

try:
    import xformers; isxf = True
except: isxf = False

class LatentBlending():
    def __init__(self, sd, steps, cfg_scale=7, time_ids=None, strength=1., verbose=True):
        """ Initializes the latent blending class.
            cfg_scale: float
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `cfg_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `cfg_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            scale_mid_damper: float = 0.5
                Reduces the guidance scale towards the middle of the transition.
                A value of 0.5 would decrease the cfg_scale towards the middle linearly by 0.5.
            mid_compression_scaler: float = 2.0
                Increases the sampling density in the middle (where most changes happen). Higher value
                imply more values in the middle. However the inflection point can occur outside the middle,
                thus high values can give rough transitions. Values around 2 should be fine.
        """
        self.sd = sd
        self.device = self.sd.device
        self.set_steps(steps, strength)

        self.seed1 = int((time.time()%1)*69696)
        self.seed2 = int((time.time()%1)*69696)

        # Initialize vars
        self.tree_lats = [None, None]
        self.tree_fracts = None
        self.idx_injection = []
        self.tree_imgs = []

        self.isxl = time_ids is not None
        self.time_ids = time_ids
        
        self.lat_init1 = None
        self.lat_init2 = None
        self.text_emb1 = None
        self.text_emb2 = None
        self.pool_emb1 = None # sdxl
        self.pool_emb2 = None # sdxl

        # Mixing parameters
        self.branch1_cross_power = 0.1
        self.branch1_cross_range = 0.6
        self.branch1_cross_decay = 0.8
        self.parent_cross_power = 0.1
        self.parent_cross_range = 0.8
        self.parent_cross_power_decay = 0.8

        self.cfg_scale = cfg_scale
        self.verbose = verbose

        self.dt_per_diff = 0
        self.lpips = lpips.LPIPS(net='alex', verbose=False).to(self.device)
        self.slerp = slerp2 if self.sd.use_kdiff or self.isxl else slerp

    def set_steps(self, steps, strength):
        self.steps = min(int(steps * strength), steps)
        self.sd.set_steps(steps, strength)

    def set_conds(self, cond1, cond2, uc, pool1=None, pool2=None, pool_uc=None, cws=None, cnimg=None, c_img=None):
        self.text_emb1 = cond1
        self.text_emb2 = cond2
        self.pool_emb1 = pool1 # sdxl
        self.pool_emb2 = pool2 # sdxl
        self.cws       = cws
        self.uc        = uc
        self.pool_uc   = pool_uc # sdxl
        self.cnimg     = cnimg
        if c_img is not None: # pair of lists of [2,1,...], already with uc
            self.img_emb1 = [c_.chunk(2)[1] if self.cfg_scale in [0,1] else c_ for c_ in c_img[0]]
            self.img_emb2 = [c_.chunk(2)[1] if self.cfg_scale in [0,1] else c_ for c_ in c_img[1]]
        else:
            self.img_emb1 = self.img_emb2 = None

    def init_lats(self, lat1, lat2, **kwargs):
        self.branch1_cross_power = 0. # to keep keypoint latents intact
        self.lat_init1 = lat1
        self.lat_init2 = lat2

    def run_transition(self, w, h, depth_strength, max_branches, reuse=False, seeds=None):
        """ Function for computing transitions. Returns a list of transition images using spherical latent blending.
            depth_strength:
                Determines how deep the first injection will happen.
                Deeper injections (low values) may cause (unwanted) formation of new structures, shallow (high) values will go into alpha-blendy land.
            max_branches: int
                The maximum number of branches to be computed. Higher values give better
                results. Use this if you want to have controllable results independent of your computer.
            reuse: Optional[bool]:
                Don't recompute the latents (purely prompt1). Saves compute.
            seeds: Optional[List[int)]:
                You can supply two seeds that are used for the first and second keyframe (prompt1 and prompt2).
                Otherwise random seeds will be taken.
        """
        if self.cfg_scale > 0:
            assert self.text_emb1 is not None and self.text_emb2 is not None, 'Set text embeddings with .set_prompts(...) first'

        self.shape = [1, 4, h // self.sd.vae_scale, w // self.sd.vae_scale]

        if seeds is not None:
            if 'rand' in seeds:
                seeds = list(np.random.randint(0, 1000000, 2).astype(np.int32))
            else:
                assert len(seeds) == 2, "Supply two seeds"
            self.seed1, self.seed2 = seeds

        lats1 = self.compute_latents1() if not reuse or len(self.tree_lats[-1]) != self.steps else self.tree_lats[-1]
        lats2 = self.compute_latents2()

        # Reset the tree, injecting the edge latents1/2 we just generated/recycled
        self.tree_lats = [lats1, lats2]
        self.tree_fracts = [0.0, 1.0]
        self.tree_imgs = [self.lat2img((self.tree_lats[0][-1])), self.lat2img((self.tree_lats[-1][-1]))]
        self.tree_idx_inject = [0, 0]

        # Set up branching scheme (dependent on provided compute time)
        idxs_injection, list_num_stems = self.get_time_based_branching(depth_strength, max_branches)
        log = str(idxs_injection) + ' ' + str(list_num_stems)

        # Run iteratively, starting with the longest trajectory.
        # Always inserting new branches where they are needed most according to image similarity
        if self.verbose and not iscolab: pbar = progbar(sum(list_num_stems))
        for s_idx in range(len(idxs_injection)):
            num_stems = list_num_stems[s_idx]
            idx_injection = idxs_injection[s_idx]
            for i in range(num_stems):
                fract_mixing, b_parent1, b_parent2 = self.get_mixing_parameters(idx_injection)
                lats = self.compute_latents_mix(fract_mixing, b_parent1, b_parent2, idx_injection)
                self.insert_into_tree(fract_mixing, idx_injection, lats)
                # print(f"fract_mixing: {fract_mixing} idx_injection {idx_injection}")
                if self.verbose and not iscolab: pbar.upd(log)

    def compute_latents1(self):
        # diffusion trajectory 1
        cond = self.text_emb1
        pool_c = self.pool_emb1 # sdxl
        im_cond = self.img_emb1
        t0 = time.time()
        lat_start = self.get_noise(self.seed1) if self.lat_init1 is None else self.lat_init1
        lats1 = self.run_diffusion(cond, pool_c, im_cond, lat_start)
        t1 = time.time()
        self.dt_per_diff = (t1 - t0) / self.steps
        self.tree_lats[0] = lats1
        return lats1

    def compute_latents2(self):
        # diffusion trajectory 2, may be affected by trajectory 1
        cond = self.text_emb2
        pool_c = self.pool_emb2 # sdxl
        im_cond = self.img_emb2
        lat_start = self.get_noise(self.seed2) if self.lat_init2 is None else self.lat_init2
        if self.branch1_cross_power > 0.0: # influenced by branch1 ?
            idx_mixing_stop = int(round(self.steps * self.branch1_cross_range))
            mix_coeffs = list(np.linspace(self.branch1_cross_power, self.branch1_cross_power * self.branch1_cross_decay, idx_mixing_stop))
            mix_coeffs.extend((self.steps - idx_mixing_stop) * [0])
            lats_mixing = self.tree_lats[0]
            lats2 = self.run_diffusion(cond, pool_c, im_cond, lat_start, 0, lats_mixing, mix_coeffs)
        else:
            lats2 = self.run_diffusion(cond, pool_c, im_cond, lat_start)
        self.tree_lats[-1] = lats2
        return lats2

    def compute_latents_mix(self, fract_mixing, b_parent1, b_parent2, idx_injection):
        """ Runs a diffusion trajectory, using the latents from the respective parents
            fract_mixing: float
                the fraction along the transition axis [0, 1]
            b_parent1: int
                index of parent1 to be used
            b_parent2: int
                index of parent2 to be used
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
        """
        if self.cfg_scale == 0 and not self.isxl: # no guidance
            cond = None
            pool_c = None
        else:
            if isset(self.sd.a, 'lguide') and self.sd.a.lguide is True and not self.isxl: # multi guidance
                cond = [self.text_emb1, self.text_emb2, fract_mixing]
            else: # cond lerp
                cond = lerp(self.text_emb1, self.text_emb2, fract_mixing)
                pool_c = lerp(self.pool_emb1, self.pool_emb2, fract_mixing) if self.isxl else None
        if self.img_emb1 is not None:
            im_cond = [lerp(self.img_emb1[i], self.img_emb2[i], fract_mixing) for i in range(len(self.img_emb1))]
        else:
            im_cond = None

        fract_mixing_parental = (fract_mixing - self.tree_fracts[b_parent1]) / (self.tree_fracts[b_parent2] - self.tree_fracts[b_parent1])

        lats_parent_mix = []
        for i in range(self.steps):
            latents_p1 = self.tree_lats[b_parent1][i]
            latents_p2 = self.tree_lats[b_parent2][i]
            if latents_p1 is None or latents_p2 is None:
                latents_parental = None
            else:
                latents_parental = self.slerp(latents_p1, latents_p2, fract_mixing_parental)
            lats_parent_mix.append(latents_parental)

        idx_mixing_stop = int(round(self.steps * self.parent_cross_range))
        mix_coeffs = idx_injection * [self.parent_cross_power]
        nmb_mixing = idx_mixing_stop - idx_injection
        if nmb_mixing > 0:
            mix_coeffs.extend(list(np.linspace(self.parent_cross_power, self.parent_cross_power * self.parent_cross_power_decay, nmb_mixing)))
        mix_coeffs.extend((self.steps - len(mix_coeffs)) * [0])
        lat_start = lats_parent_mix[idx_injection - 1]
        lats = self.run_diffusion(cond, pool_c, im_cond, lat_start, idx_injection, lats_parent_mix, mix_coeffs)
        return lats

    def get_time_based_branching(self, depth_strength, max_branches):
        r"""
        Sets up the branching scheme dependent on the time that is granted for compute.
        The scheme uses an estimation derived from the first image's computation speed.
            depth_strength:
                Determines how deep the first injection will happen.
                Deeper injections will cause (unwanted) formation of new structures,
                more shallow values will go into alpha-blendy land.
            max_branches: int
                The maximum number of branches to be computed. Higher values give better
                results. Use this if you want to have controllable results independent
                of your computer.
        """
        idx_injection_base = int(round(self.steps * depth_strength))
        idxs_injection = np.arange(idx_injection_base, self.steps - 1, 3)
        list_num_stems = np.ones(len(idxs_injection), dtype=np.int32)
        max_branches -= 2  # Discounting the outer frames
        stop_criterion_reached = False
        is_first_iteration = True

        while not stop_criterion_reached:
            increase_done = False
            for s_idx in range(len(list_num_stems) - 1):
                if list_num_stems[s_idx + 1] / list_num_stems[s_idx] >= 2:
                    list_num_stems[s_idx] += 1
                    increase_done = True
                    break
            if not increase_done:
                list_num_stems[-1] += 1

            if np.sum(list_num_stems) >= max_branches:
                stop_criterion_reached = True
                if is_first_iteration:
                    # Need to undersample.
                    idxs_injection = np.linspace(idxs_injection[0], idxs_injection[-1], max_branches).astype(np.int32)
                    list_num_stems = np.ones(len(idxs_injection), dtype=np.int32)
            else:
                is_first_iteration = False

        return idxs_injection, list_num_stems

    def get_mixing_parameters(self, idx_injection):
        """ Computes which parental latents should be mixed together to achieve a smooth blend.
        As metric, we are using lpips image similarity. The insertion takes place where the metric is maximal.
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
        """
        similarities = []
        for i in range(len(self.tree_imgs) - 1):
            similarities.append(self.get_lpips_similarity(self.tree_imgs[i], self.tree_imgs[i + 1]))
        b_closest1 = np.argmax(similarities)
        b_closest2 = b_closest1 + 1
        fract_closest1 = self.tree_fracts[b_closest1]
        fract_closest2 = self.tree_fracts[b_closest2]

        # Ensure that the parents are indeed older!
        b_parent1 = b_closest1
        while True:
            if self.tree_idx_inject[b_parent1] < idx_injection:
                break
            else:
                b_parent1 -= 1
        b_parent2 = b_closest2
        while True:
            if self.tree_idx_inject[b_parent2] < idx_injection:
                break
            else:
                b_parent2 += 1
        fract_mixing = (fract_closest1 + fract_closest2) / 2
        return fract_mixing, b_parent1, b_parent2

    def insert_into_tree(self, fract_mixing, idx_injection, lats):
        """ Inserts all necessary parameters into the trajectory tree.
            fract_mixing: float
                the fraction along the transition axis [0, 1]
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
            lats: list
                list of the latents to be inserted
        """
        b_parent1, b_parent2 = self.get_closest_idx(fract_mixing)
        self.tree_lats.insert(b_parent1 + 1, lats)
        self.tree_imgs.insert(b_parent1 + 1, self.lat2img(lats[-1]))
        self.tree_fracts.insert(b_parent1 + 1, fract_mixing)
        self.tree_idx_inject.insert(b_parent1 + 1, idx_injection)

    def get_closest_idx(self, fract_mixing: float):
        """ Helper function to retrieve the parents for any given mixing.
        Example: fract_mixing = 0.4 and self.tree_fracts = [0, 0.3, 0.6, 1.0]
        Will return the two closest values here, i.e. [1, 2]
        """
        pdist = fract_mixing - np.asarray(self.tree_fracts)
        pdist_pos = pdist.copy()
        pdist_pos[pdist_pos < 0] = np.inf
        b_parent1 = np.argmin(pdist_pos)
        pdist_neg = -pdist.copy()
        pdist_neg[pdist_neg <= 0] = np.inf
        b_parent2 = np.argmin(pdist_neg)

        if b_parent1 > b_parent2:
            tmp = b_parent2
            b_parent2 = b_parent1
            b_parent1 = tmp

        return b_parent1, b_parent2

    @torch.no_grad()
    def run_diffusion(self, cond, pool_c, im_cond, lat_start, idx_start=0, lats_mixing=None, mix_coeffs=0.):
        """
            cond: torch.FloatTensor
                Prompt conditioning (text embedding)
            lat_start: torch.FloatTensor
                Latent for injection
            idx_start: int
                Index of the diffusion process start and where the latents are injected
            lats_mixing: torch.FloatTensor
                List of latents (latent trajectories) that are used for mixing
            mix_coeffs: float or list
                Coefficients, how strong each element of lats_mixing will be mixed in.
        """
        if type(mix_coeffs) == float:
            list_mixing_coeffs = self.steps * [mix_coeffs]
        elif type(mix_coeffs) == list:
            assert len(mix_coeffs) == self.steps
            list_mixing_coeffs = mix_coeffs
        else:
            raise ValueError("mix_coeffs should be float or list with len = steps")

        if np.sum(list_mixing_coeffs) > 0:
            assert len(lats_mixing) == self.steps

        if self.sd.use_kdiff or self.isxl or self.sd.use_lcm or self.sd.a.sampler.lower()=='tcd': # tcd/lcm/k schedulers require reset on every generation
            self.set_steps(self.sd.a.steps, self.sd.a.strength)

        self.run_scope = nullcontext # torch.autocast
        with self.run_scope("cuda"):
            # Collect latents
            lat = lat_start.clone()
            lats_out = []
            
            for i in range(self.steps):
                if i < idx_start:
                    lats_out.append(None)
                    continue
                elif i == idx_start:
                    lat = lat_start.clone() # set the right starting latents
                if i > 0 and list_mixing_coeffs[i] > 0:
                    lat_mixtarget = lats_mixing[i-1].clone()
                    lat = self.slerp(lat, lat_mixtarget, list_mixing_coeffs[i]) # mix latents

                if self.isxl:
                    lat = self.xl_step(lat, cond, pool_c, im_cond, self.sd.timesteps[i])
                else:
                    lat = self.sd_step(lat, cond, im_cond, self.sd.timesteps[i])

                lats_out.append(lat.clone())

            return lats_out

    def sd_step(self, lat, cond, im_cond, t, verbose=True):
        with self.run_scope("cuda"):
            ukwargs = {}
            if im_cond is not None: # encoded img for ip adapter
                ukwargs['added_cond_kwargs'] = {"image_embeds": im_cond}
            lat_in = self.sd.scheduler.scale_model_input(lat.to(self.sd.unet.device, dtype=self.sd.unet.dtype), t) # scales only k-samplers
            if self.cfg_scale > 0:
                if isinstance(cond, list) and len(cond) == 3: # multi guided lerp
                    cond, cond2, mix = cond
                    bs = len(cond)*2 + 1
                    cond_in = torch.cat([self.uc, cond, cond2])
                else:
                    mix = 0.
                    bs = len(cond) + 1
                    cond_in = torch.cat([self.uc, cond])
                lat_in = torch.cat([lat_in] * bs, dim=0)
                if self.sd.use_cnet and self.cnimg is not None: # controlnet
                    ctl_downs, ctl_mid = self.sd.cnet(lat_in, t, cond_in, self.cnimg, self.sd.cnet_ws, return_dict=False)
                    ukwargs = {'down_block_additional_residuals': ctl_downs, 'mid_block_additional_residual': ctl_mid, **ukwargs}
                noises = self.sd.unet(lat_in, t, cond_in, **ukwargs).sample.chunk(bs) # pred noise residual at step t
                noise_pred = noises[0] # uncond
                for n in range(len(cond)): # multi guidance
                    noise_guide = noises[n+1] * (1.-mix) + noises[n+1+len(cond)] * mix - noises[0] if mix > 0 else noises[n+1] - noises[0]
                    noise_pred = noise_pred + noise_guide * self.cfg_scale * self.cws[n % len(self.cws)]
            else: # no guidance
                noise_pred = self.sd.unet(lat_in, t, self.uc, **ukwargs).sample
            lat = self.sd.scheduler.step(noise_pred, t, lat, **self.sd.sched_kwargs).prev_sample
        return lat

    def xl_step(self, lat, cond, pool_c, im_cond, t, verbose=True):
        with self.run_scope("cuda"):
            lat_in = self.sd.scheduler.scale_model_input(lat.to(self.device), t) # scaling only k-samplers
            if not self.cfg_scale in [0,1]:
                cond = torch.cat([self.uc, cond])
                pool_c = torch.cat([self.pool_uc, pool_c])
                lat_in = torch.cat([lat_in] * 2, dim=0)
            if len(lat_in) > len(self.time_ids): self.time_ids = torch.cat([self.time_ids] * 2, dim=0)

            ukwargs = {'added_cond_kwargs': {"text_embeds": pool_c, "time_ids": self.time_ids}}
            if self.sd.use_cnet and self.cnimg is not None: # controlnet
                ctl_downs, ctl_mid = self.sd.cnet(lat_in, t, cond, self.cnimg, 1, **ukwargs, return_dict=False)
                ctl_downs = [ctl_down * self.sd.a.control_scale for ctl_down in ctl_downs]
                ctl_mid *= self.sd.a.control_scale
                ukwargs = {**ukwargs, 'down_block_additional_residuals': ctl_downs, 'mid_block_additional_residual': ctl_mid}
            if im_cond is not None: # encoded img for ip adapter
                ukwargs['added_cond_kwargs']['image_embeds'] = im_cond

            if self.cfg_scale in [0,1]:
                noise_pred = self.sd.unet(lat_in, t, cond, **ukwargs).sample
            else:
                noise_un, noise_c = self.sd.unet(lat_in, t, cond, **ukwargs).sample.chunk(2)
                noise_pred = noise_un + self.cfg_scale * (noise_c - noise_un)
            lat = self.sd.scheduler.step(noise_pred, t, lat).prev_sample.to(self.device, dtype=self.sd.dtype)

        return lat

    def get_noise(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        return torch.randn(self.shape, generator=generator, device=self.device) * self.sd.scheduler.init_noise_sigma

    def lat2img(self, lat):
        if self.isxl:
            self.sd.vae.to(dtype=torch.float32)
            self.sd.vae.post_quant_conv.to(lat.dtype)
            self.sd.vae.decoder.conv_in.to(lat.dtype)
            self.sd.vae.decoder.mid_block.to(lat.dtype)
            lat = lat.to(next(iter(self.sd.vae.post_quant_conv.parameters())).dtype)
        else:
            lat = lat.to(dtype=self.sd.vae.dtype)
        return self.sd.vae.decode(lat / self.sd.vae.config.scaling_factor).sample
    
    def get_lpips_similarity(self, imgA, imgB):
        return float(self.lpips(imgA, imgB)[0][0][0][0]) # high value = low similarity

    def save_imgs(self, save_dir, start_num=0, skiplast=False):
        os.makedirs(save_dir, exist_ok=True)
        if skiplast: self.tree_imgs = self.tree_imgs[:-1]
        for i, img in enumerate(self.tree_imgs):
            img = torch.clamp((img + 1.) / 2., min=0., max=1.) * 255
            img = img[0].permute([1,2,0]).cpu().numpy().astype(np.uint8)
            Image.fromarray(img).save(os.path.join(save_dir, f"{str(i+start_num).zfill(5)}.jpg"))
        return len(self.tree_imgs)

