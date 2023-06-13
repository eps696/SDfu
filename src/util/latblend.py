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

sys.path.append(os.path.join(os.path.dirname(__file__), '../xtra'))
import lpips

import torch
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)

from .utils import slerp, slerp2, lerp, blend, cvshow, progbar
try: # colab
    get_ipython().__class__.__name__
    iscolab = True
except: iscolab = False

class LatentBlending():
    def __init__(self, sd, steps, cfg_scale=7, scale_mid_damper=0.5):
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
        self.device = torch.device('cuda')
        self.steps = steps
        self.sd.set_steps(self.steps)

        assert scale_mid_damper > 0 and scale_mid_damper <= 1.0, f"scale_mid_damper neees to be in interval (0,1], you provided {scale_mid_damper}"
        self.scale_mid_damper = scale_mid_damper
        self.seed1 = int((time.time()%1)*69696)
        self.seed2 = int((time.time()%1)*69696)

        # Initialize vars
        self.tree_lats = [None, None]
        self.tree_fracts = None
        self.idx_injection = []
        self.tree_imgs = []

        self.text_emb1 = None
        self.text_emb2 = None
        self.lat_init1 = None
        self.lat_init2 = None

        # Mixing parameters
        self.branch1_cross_power = 0.1
        self.branch1_cross_range = 0.6
        self.branch1_cross_decay = 0.8
        self.parent_cross_power = 0.1
        self.parent_cross_range = 0.8
        self.parent_cross_power_decay = 0.8

        self.scale_base = cfg_scale
        self.cfg_scale = cfg_scale

        self.dt_per_diff = 0
        self.lpips = lpips.LPIPS(net='alex', verbose=False).cuda()
        self.slerp = slerp2 if self.sd.a.sampler == 'euler' else slerp

    def set_scale_mid_damp(self, fract_mixing):
        # Tunes the guidance scale down as a linear function of fract_mixing, towards 0.5 the minimum will be reached.
        mid_factor = 1 - np.abs(fract_mixing - 0.5) / 0.5
        max_guidance_reduction = self.scale_base * (1 - self.scale_mid_damper) - 1
        guidance_scale_effective = self.scale_base - max_guidance_reduction * mid_factor
        self.cfg_scale = guidance_scale_effective

    def set_conds(self, cond1, cond2, cws, uc):
        self.text_emb1 = cond1
        self.text_emb2 = cond2
        self.cws = cws
        self.uc = uc

    def init_lats(self, lat1, lat2):
        self.branch1_cross_power = 0. # to keep initial latents intact
        self.lat_init1 = lat1
        self.lat_init2 = lat2

    def run_transition(self, w, h, depth_strength=0.4, t_compute_max=None, max_branches=None, reuse=False, seeds=None):
        """ Function for computing transitions. Returns a list of transition images using spherical latent blending.
            reuse_img1: Optional[bool]:
                Don't recompute the latents for the first keyframe (purely prompt1). Saves compute.
            reuse_img2: Optional[bool]:
                Don't recompute the latents for the second keyframe (purely prompt2). Saves compute.
            num_inference_steps:
                Number of diffusion steps. Higher values will take more compute time.
            depth_strength:
                Determines how deep the first injection will happen.
                Deeper injections (low values) may cause (unwanted) formation of new structures, shallow (high) values will go into alpha-blendy land.
            t_compute_max:
                Either provide t_compute_max or max_branches.
                The maximum time allowed for computation. Higher values give better results but take longer.
            max_branches: int
                Either provide t_compute_max or max_branches. The maximum number of branches to be computed. Higher values give better
                results. Use this if you want to have controllable results independent of your computer.
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
        idxs_injection, list_num_stems = self.get_time_based_branching(depth_strength, t_compute_max, max_branches)

        # Run iteratively, starting with the longest trajectory.
        # Always inserting new branches where they are needed most according to image similarity
        if not iscolab: pbar = progbar(sum(list_num_stems))
        for s_idx in range(len(idxs_injection)):
            num_stems = list_num_stems[s_idx]
            idx_injection = idxs_injection[s_idx]
            for i in range(num_stems):
                fract_mixing, b_parent1, b_parent2 = self.get_mixing_parameters(idx_injection)
                # self.set_scale_mid_damp(fract_mixing) # !!! makes too messy
                lats = self.compute_latents_mix(fract_mixing, b_parent1, b_parent2, idx_injection)
                self.insert_into_tree(fract_mixing, idx_injection, lats)
                # print(f"fract_mixing: {fract_mixing} idx_injection {idx_injection}")
                if not iscolab: pbar.upd()

    def compute_latents1(self):
        # diffusion trajectory 1
        cond = self.text_emb1
        t0 = time.time()
        lat_start = self.get_noise(self.seed1) if self.lat_init1 is None else self.lat_init1
        lats1 = self.run_diffusion(cond, lat_start)
        t1 = time.time()
        self.dt_per_diff = (t1 - t0) / self.steps
        self.tree_lats[0] = lats1
        return lats1

    def compute_latents2(self):
        # diffusion trajectory 2, may be affected by trajectory 1
        cond = self.text_emb2
        lat_start = self.get_noise(self.seed2) if self.lat_init2 is None else self.lat_init2
        if self.branch1_cross_power > 0.0: # influenced by branch1 ?
            idx_mixing_stop = int(round(self.steps * self.branch1_cross_range))
            mix_coeffs = list(np.linspace(self.branch1_cross_power, self.branch1_cross_power * self.branch1_cross_decay, idx_mixing_stop))
            mix_coeffs.extend((self.steps - idx_mixing_stop) * [0])
            lats_mixing = self.tree_lats[0]
            lats2 = self.run_diffusion(cond, lat_start, 0, lats_mixing, mix_coeffs)
        else:
            lats2 = self.run_diffusion(cond, lat_start)
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
        if self.cfg_scale == 0: # no guidance
            cond = None
        elif self.sd.a.lguide: # multi guidance
            cond = [self.text_emb1, self.text_emb2, fract_mixing]
        else: # cond lerp
            cond = lerp(self.text_emb1, self.text_emb2, fract_mixing) if self.cfg_scale > 0 else None
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
        lats = self.run_diffusion(cond, lat_start, idx_injection, lats_parent_mix, mix_coeffs)
        return lats

    def get_time_based_branching(self, depth_strength, t_compute_max=None, max_branches=None):
        r"""
        Sets up the branching scheme dependent on the time that is granted for compute.
        The scheme uses an estimation derived from the first image's computation speed.
        Either provide t_compute_max or max_branches
        Args:
            depth_strength:
                Determines how deep the first injection will happen.
                Deeper injections will cause (unwanted) formation of new structures,
                more shallow values will go into alpha-blendy land.
            t_compute_max: float
                The maximum time allowed for computation. Higher values give better results
                but take longer. Use this if you want to fix your waiting time for the results.
            max_branches: int
                The maximum number of branches to be computed. Higher values give better
                results. Use this if you want to have controllable results independent
                of your computer.
        """
        idx_injection_base = int(round(self.steps * depth_strength))
        idxs_injection = np.arange(idx_injection_base, self.steps - 1, 3)
        list_num_stems = np.ones(len(idxs_injection), dtype=np.int32)
        t_compute = 0

        if max_branches is None:
            assert t_compute_max is not None, "Either specify t_compute_max or max_branches"
            stop_criterion = "t_compute_max"
        elif t_compute_max is None:
            assert max_branches is not None, "Either specify t_compute_max or max_branches"
            stop_criterion = "max_branches"
            max_branches -= 2  # Discounting the outer frames
        else:
            raise ValueError("Either specify t_compute_max or max_branches")
        stop_criterion_reached = False
        is_first_iteration = True
        while not stop_criterion_reached:
            list_compute_steps = self.steps - idxs_injection
            list_compute_steps *= list_num_stems
            t_compute = np.sum(list_compute_steps) * self.dt_per_diff + 0.15 * np.sum(list_num_stems)
            increase_done = False
            for s_idx in range(len(list_num_stems) - 1):
                if list_num_stems[s_idx + 1] / list_num_stems[s_idx] >= 2:
                    list_num_stems[s_idx] += 1
                    increase_done = True
                    break
            if not increase_done:
                list_num_stems[-1] += 1

            if stop_criterion == "t_compute_max" and t_compute > t_compute_max:
                stop_criterion_reached = True
            elif stop_criterion == "max_branches" and np.sum(list_num_stems) >= max_branches:
                stop_criterion_reached = True
                if is_first_iteration:
                    # Need to undersample.
                    idxs_injection = np.linspace(idxs_injection[0], idxs_injection[-1], max_branches).astype(np.int32)
                    list_num_stems = np.ones(len(idxs_injection), dtype=np.int32)
            else:
                is_first_iteration = False

            # print(f"t_compute {t_compute} list_num_stems {list_num_stems}")
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
    def run_diffusion(self, cond, lat_start, idx_start=0, lats_mixing=None, mix_coeffs=0.):
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
            raise ValueError("mix_coeffs should be float or list with len=num_inference_steps")

        if np.sum(list_mixing_coeffs) > 0:
            assert len(lats_mixing) == self.steps

        self.precision_scope = torch.autocast
        with self.precision_scope("cuda"):
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

                if self.sd.a.sampler == 'euler':
                    lat = self.euler_step(lat, cond, self.sd.sigmas, i)
                else: # ddim
                    lat = self.ddim_step(lat, cond, self.sd.timesteps[i])

                lats_out.append(lat.clone())

            return lats_out

    def euler_step(self, lat, cond, sigmas, i, verbose=True):
        with self.precision_scope("cuda"):
            sigma = sigmas[i]
            t = sigma * lat.new_ones([lat.shape[0]])
            if self.cfg_scale > 0:
                if isinstance(cond, list) and len(cond) == 3: # multi guidance lerp
                    c1, c2, mix = cond
                    bs = len(c1)*2 + 1
                    noises = self.sd.kdiff_model(torch.cat([lat] * bs), t, cond=torch.cat([self.uc, c1, c2])).chunk(bs)
                    denoised = noises[0] # uncond
                    for n in range(len(c1)):
                        denoised = denoised + (noises[n+1] * (1.-mix) + noises[n+1+len(c1)] * mix - noises[0]) * self.cfg_scale * self.cws[n % len(self.cws)]
                else: # multi guidance
                    bs = len(cond) + 1
                    noises = self.sd.kdiff_model(torch.cat([lat] * bs), t, cond=torch.cat([self.uc, cond])).chunk(bs)
                    denoised = noises[0] # uncond
                    for n in range(len(cond)):
                        denoised = denoised + (noises[n+1] - noises[0]) * self.cfg_scale * self.cws[n % len(self.cws)] # guidance
            else:
                denoised = self.sd.kdiff_model(lat, t, cond=self.uc)
            d = (lat - denoised) / sigma
            dt = sigmas[i + 1] - sigma
            lat = lat + d * dt # Euler method
        return lat

    def ddim_step(self, lat, cond, t, verbose=True):
        with self.precision_scope("cuda"):
            lat = self.sd.scheduler.scale_model_input(lat.cuda(), t) # scales only k-samplers!?
            if self.cfg_scale > 0:
                if isinstance(cond, list) and len(cond) == 3: # multi guidance
                    c1, c2, mix = cond
                    bs = len(c1)*2 + 1
                    noises = self.sd.unet(torch.cat([lat] * bs), t, torch.cat([self.uc, c1, c2])).sample.chunk(bs)
                    noise_pred = noises[0] # uncond
                    for n in range(len(c1)):
                        noise_pred = noise_pred + (noises[n+1] * (1.-mix) + noises[n+1+len(c1)] * mix - noises[0]) * self.cfg_scale * self.cws[n % len(self.cws)]
                else: # usual guidance
                    bs = len(cond) + 1
                    noises = self.sd.unet(torch.cat([lat] * bs), t, torch.cat([self.uc, cond])).sample.chunk(bs) # pred noise residual at step t
                    noise_pred = noises[0] # uncond
                    for n in range(len(cond)):
                        noise_pred = noise_pred + (noises[n+1] - noises[0]) * self.cfg_scale * self.cws[n % len(self.cws)] # guidance
            else:
                noise_pred = self.sd.unet(lat, t, self.uc).sample
            lat = self.sd.scheduler.step(noise_pred.cpu(), t, lat.cpu(), **self.sd.sched_kwargs).prev_sample.cuda().half() # why can't make it on cuda??
        return lat

    def get_noise(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        return torch.randn(self.shape, generator=generator, device=self.device) * self.sd.scheduler.init_noise_sigma

    def lat2img(self, lat):
        return self.sd.vae.decode(lat.to(dtype=self.sd.vae.dtype) / self.sd.vae.config.scaling_factor).sample
    
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

