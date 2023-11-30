
import os, sys
import time
import numpy as np
from contextlib import nullcontext
from einops import rearrange

import torch
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from diffusers.utils import is_accelerate_available, is_accelerate_version

sys.path.append(os.path.join(os.path.dirname(__file__), '../xtra'))

from .text import multiprompt
from .utils import load_img, makemask, isok, isset, progbar, file_list
from .args import models

import logging
logging.getLogger('diffusers').setLevel(logging.ERROR)
try:
    import xformers; isxf = True
except: isxf = False
try: # colab
    get_ipython().__class__.__name__
    iscolab = True
except: iscolab = False

device = torch.device('cuda')

class SDpipe(StableDiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, image_encoder=None, \
                 safety_checker=None, feature_extractor=None, requires_safety_checker=False):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, image_encoder, None, None, requires_safety_checker=False)
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
        if image_encoder is not None:
            self.register_modules(image_encoder=image_encoder)

class ModelWrapper: # for k-sampling
    def __init__(self, model, alphas_cumprod):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
    def apply_model(self, *args, **kwargs):
        if len(args) == 3:
            conds = args[-1]
            args = args[:2]
        if kwargs.get("cond", None) is not None:
            conds = kwargs.pop("cond")
        return self.model(*args, encoder_hidden_states=conds, **kwargs).sample


class SDfu:
    def __init__(self, a, vae=None, text_encoder=None, tokenizer=None, unet=None, scheduler=None):
        # settings
        self.a = a
        self.device = device
        self.run_scope = nullcontext # torch.autocast
        if not isset(a, 'maindir'): a.maindir = './models' # for external scripts
        self.setseed(a.seed if isset(a, 'seed') else None)

        self.use_lcm = False
        if a.model == 'lcm':
            self.use_lcm = True
            a.model = os.path.join(a.maindir, 'lcm')
            self.a.sampler = 'orig'
            if self.a.cfg_scale > 3: self.a.cfg_scale = 2
            if self.a.steps > 12: self.a.steps = 4 # if steps are still set for ldm schedulers

        if a.model in models:
            self.load_model_custom(self.a, vae, text_encoder, tokenizer, unet, scheduler)
        else: # downloaded or url
            self.load_model_external(a.model)
        self.use_kdiff = hasattr(self.scheduler, 'sigmas') # k-diffusion sampling
        assert not (isset(a, 'control_mod') and self.use_kdiff), "Controlnet not supported for k-samplers here"
        if not self.a.lowmem: self.pipe.to(device)

        # load finetuned stuff
        mod_tokens = None
        if isset(a, 'load_lora') and os.path.isfile(a.load_lora): # lora
            self.pipe.load_lora_weights(a.load_lora, low_cpu_mem_usage=True)
            self.pipe.fuse_lora()
            if a.verbose: print(' loaded LoRA', a.load_lora)
        elif isset(a, 'load_custom') and os.path.isfile(a.load_custom): # custom diffusion
            from .finetune import load_delta, custom_diff
            self.pipe.unet = custom_diff(self.pipe.unet, train=False)
            mod_tokens = load_delta(torch.load(a.load_custom), self.pipe.unet, self.pipe.text_encoder, self.pipe.tokenizer)
        elif isset(a, 'load_token') and os.path.exists(a.load_token): # text inversion
            from .finetune import load_embeds
            emb_files = [a.load_token] if os.path.isfile(a.load_token) else file_list(a.load_token, 'pt')
            mod_tokens = []
            for emb_file in emb_files:
                mod_tokens += load_embeds(torch.load(emb_file), self.pipe.text_encoder, self.pipe.tokenizer)
        if mod_tokens is not None: print(' loaded tokens:', mod_tokens[0] if len(mod_tokens)==1 else mod_tokens)

        # load controlnet
        if isset(a, 'control_mod'):
            if not os.path.exists(a.control_mod): a.control_mod = os.path.join(a.maindir, 'control', a.control_mod)
            assert os.path.exists(a.control_mod), "Not found ControlNet model %s" % a.control_mod
            from diffusers import ControlNetModel
            self.cnet = ControlNetModel.from_pretrained(a.control_mod, torch_dtype=torch.float16)
            if not self.a.lowmem: self.cnet.to(device)
            self.pipe.register_modules(controlnet=self.cnet)
            if a.verbose: print(' loaded ControlNet', a.control_mod)
        self.use_cnet = hasattr(self, 'cnet')

        # load animatediff = before ip adapter, after custom diffusion !
        if isset(a, 'animdiff'):
            if not os.path.exists(a.animdiff): a.animdiff = os.path.join(a.maindir, 'anima')
            assert os.path.exists(a.animdiff), "Not found AnimateDiff model %s" % a.animdiff
            from diffusers.models import UNetMotionModel
            from diffusers.models.unet_motion_model import MotionAdapter
            motion_adapter = MotionAdapter.from_pretrained(a.animdiff)
            self.unet = UNetMotionModel.from_unet2d(self.unet, motion_adapter)
            self.scheduler = self.set_scheduler(a) # k-samplers must be loaded after unet
            if not self.a.lowmem: self.unet.to(device)
            self.pipe.register_modules(unet = self.unet, motion_adapter = motion_adapter)
            if a.verbose: print(' loaded AnimateDiff', a.animdiff)

        # load ip adapter = after animatediff !
        if isset(a, 'img_ref'):
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection as CLIPimg
            self.image_preproc = CLIPImageProcessor.from_pretrained(os.path.join(a.maindir, 'image/preproc_config.json')) # openai/clip-vit-base-patch32
            self.image_encoder = CLIPimg.from_pretrained(os.path.join(a.maindir, 'image'), torch_dtype=torch.float16).to(device)
            self.unet._load_ip_adapter_weights(torch.load(os.path.join(a.maindir, 'image/ip-adapter_sd15.bin'), map_location="cpu"))
            self.pipe.register_modules(image_encoder = self.image_encoder)
            self.pipe.set_ip_adapter_scale(a.imgref_weight)
            if a.verbose: print(' loaded IP adapter for image encodings')

        self.final_setup(a)

    def load_model_external(self, model_path):
        SDload = StableDiffusionPipeline.from_single_file if os.path.isfile(model_path) else StableDiffusionPipeline.from_pretrained
        self.pipe = SDload(model_path, torch_dtype=torch.float16, safety_checker=None)
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer    = self.pipe.tokenizer
        self.unet         = self.pipe.unet
        self.vae          = self.pipe.vae
        self.scheduler    = self.pipe.scheduler if self.a.sampler=='orig' else self.set_scheduler(self.a)
        self.sched_kwargs = {}

    def load_model_custom(self, a, vae=None, text_encoder=None, tokenizer=None, unet=None, scheduler=None):
        # paths
        self.clipseg_path = os.path.join(a.maindir, 'clipseg/rd64-uni.pth')
        vtype  = a.model[-1] == 'v'
        vidtype = a.model[0] == 'v'
        self.subdir = 'v2v' if vtype else 'v2' if vidtype or a.model[0]=='2' else 'v1'

        if vtype and not isxf: # scheduler.prediction_type == "v_prediction":
            print(" V-models require xformers! install it or use another model"); exit()

        # text input
        txtenc_path = os.path.join(a.maindir, self.subdir, 'text-' + a.model[2:] if a.model[2:] in ['drm'] else 'text')
        if text_encoder is None:
            text_encoder = CLIPTextModel.from_pretrained(txtenc_path, torch_dtype=torch.float16, local_files_only=True)
        if tokenizer is None:
            tokenizer    = CLIPTokenizer.from_pretrained(txtenc_path, torch_dtype=torch.float16, local_files_only=True)
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        if unet is None:
            unet_path = os.path.join(a.maindir, self.subdir, 'unet' + a.model)
            if vidtype:
                from diffusers.models import UNet3DConditionModel as UNet
            else:
                from diffusers.models import UNet2DConditionModel as UNet
            unet = UNet.from_pretrained(unet_path, torch_dtype=torch.float16, local_files_only=True)
        if not isxf and isinstance(unet.config.attention_head_dim, int): unet.set_attention_slice(unet.config.attention_head_dim // 2) # 8
        self.unet = unet

        if vae is None:
            vae_path = 'vae'
            if a.model[0]=='1' and a.vae != 'orig':
                vae_path = 'vae-ft-mse' if a.vae=='mse' else 'vae-ft-ema'
            vae_path = os.path.join(a.maindir, self.subdir, vae_path)
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        if a.lowmem: vae.enable_tiling() # higher res, more free ram
        elif vidtype or isset(a, 'animdiff') or not isxf: vae.enable_slicing()
        self.vae = vae

        if scheduler is None:
            scheduler = self.set_scheduler(a, self.subdir, vtype)
        self.scheduler = scheduler

        self.pipe = SDpipe(vae, text_encoder, tokenizer, unet, scheduler)

    def set_scheduler(self, a, subdir='', vtype=False):
        if isset(a, 'animdiff'):
            sched_path = os.path.join(a.maindir, 'scheduler_config-linear.json')
        else:
            sched_path = os.path.join(a.maindir, subdir, 'scheduler_config-%s.json' % a.model)
        if not os.path.exists(sched_path):
            sched_path = os.path.join(a.maindir, subdir, 'scheduler_config.json')
        self.sched_kwargs = {}
        if a.sampler == 'lcm':
            from diffusers.schedulers import LCMScheduler
            scheduler = LCMScheduler.from_pretrained(os.path.join(a.maindir, 'lcm/scheduler/scheduler_config.json'))
        elif a.sampler == 'pndm':
            from diffusers.schedulers import PNDMScheduler
            scheduler = PNDMScheduler.from_pretrained(sched_path)
        elif a.sampler == 'dpm':
            from diffusers.schedulers import DPMSolverMultistepScheduler
            scheduler = DPMSolverMultistepScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="scaled_linear", solver_order=2, sample_max_value=1.)
        elif a.sampler == 'uni':
            from diffusers.schedulers import UniPCMultistepScheduler
            scheduler = UniPCMultistepScheduler.from_pretrained(sched_path)
        elif a.sampler == 'ddim':
            from diffusers.schedulers import DDIMScheduler
            scheduler = DDIMScheduler.from_pretrained(sched_path)
            self.sched_kwargs = {"eta": a.ddim_eta}
        else:
            from diffusers.schedulers import LMSDiscreteScheduler
            import k_diffusion as K
            scheduler = LMSDiscreteScheduler.from_pretrained(sched_path)
            model = ModelWrapper(self.unet, scheduler.alphas_cumprod)
            self.kdiff_model = K.external.CompVisVDenoiser(model) if vtype else K.external.CompVisDenoiser(model)
            self.kdiff_model.sigmas     = self.kdiff_model.sigmas.to(device)
            self.kdiff_model.log_sigmas = self.kdiff_model.log_sigmas.to(device)
            if   a.sampler == 'klms':    self.sampling_fn = K.sampling.sample_lms
            elif a.sampler == 'euler':   self.sampling_fn = K.sampling.sample_euler
            elif a.sampler == 'euler_a': self.sampling_fn = K.sampling.sample_euler_ancestral
            elif a.sampler == 'dpm2_a':  self.sampling_fn = K.sampling.sample_dpm_2_ancestral # slow but rich!
            else: print(' Unknown sampler', a.sampler); exit()
        return scheduler

    def final_setup(self, a):
        if isxf and not isset(a, 'img_ref'): self.pipe.enable_xformers_memory_efficient_attention() # !!! breaks ip adapter
        if isset(a, 'freeu'): self.pipe.unet.enable_freeu(s1=1.5, s2=1.6, b1=0.9, b2=0.2) # 2.1 1.4, 1.6, 0.9, 0.2, sdxl 1.3, 1.4, 0.9, 0.2

        self.vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1) # 8
        self.res = self.unet.config.sample_size * self.vae_scale # original model resolution (not for video models!)
        try:
            uchannels = self.unet.config.in_channels
        except: 
            uchannels = self.unet.in_channels
        self.inpaintmod = uchannels==9
        assert not (self.inpaintmod and not isset(a, 'mask')), '!! Inpainting model requires mask !!' 
        self.depthmod = uchannels==5
        if self.depthmod:
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            depth_path = os.path.join(a.maindir, self.subdir, 'depth')
            self.depth_estimator = DPTForDepthEstimation.from_pretrained(depth_path, torch_dtype=torch.float16).to(device)
            self.feat_extractor  = DPTImageProcessor.from_pretrained(depth_path, torch_dtype=torch.float16, device=device)
        if isset(a, 'img_scale') and a.img_scale > 0: # instruct pix2pix
            assert not (self.use_cnet or a.cfg_scale in [0,1]), "Use either Instruct-pix2pix or Controlnet guidance"
            a.strength = 1.

        self.set_steps(a.steps, a.strength) # sampling

        # enable_model_cpu_offload
        if self.a.lowmem:
            assert is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"), " Install accelerate > 0.17 for model offloading"
            from accelerate import cpu_offload_with_hook
            hook = None
            for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
                _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)
            if self.use_cnet:
                _, hook = cpu_offload_with_hook(self.cnet, device, prev_module_hook=hook)
                cpu_offload_with_hook(self.cnet, device)
            self.final_offload_hook = hook


    def setseed(self, seed=None):
        self.seed = seed or int((time.time()%1)*69696)
        self.g_ = torch.Generator("cuda").manual_seed(self.seed)
    
    def set_steps(self, steps, strength=1., warmup=1, device=device):
        if self.use_lcm:
            self.scheduler.set_timesteps(steps, device, strength=strength)
            self.timesteps = self.scheduler.timesteps
            self.lat_timestep = self.timesteps[:1].repeat(self.a.batch)
        else:
            self.scheduler.set_timesteps(steps, device=device)
            steps = min(int(steps * strength), steps) # t_enc .. 50
            if self.use_kdiff:
                self.sigmas = self.scheduler.sigmas[-steps - warmup :]
            else:
                self.timesteps = self.scheduler.timesteps[-steps - warmup :] # 1 warmup step
                self.lat_timestep = self.timesteps[:1].repeat(self.a.batch)

    def next_step_ddim(self, noise, t, sample):
        t, next_t = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), t
        alpha_prod_t      = self.scheduler.alphas_cumprod[t] if t >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_t]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * noise) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def img_c(self, images):
        with self.run_scope('cuda'):
            images = self.image_preproc(images, return_tensors="pt").pixel_values.to(device) # [N,3,224,224]
            return self.image_encoder(images).image_embeds.mean(0, keepdim=True) # [1,1024]

    def img_lat(self, image, deterministic=False):
        with self.run_scope('cuda'):
            postr = self.vae.encode(image.half()).latent_dist
            lats = postr.mean if deterministic else postr.sample()
            lats *= self.vae.config.scaling_factor
        return torch.cat([lats] * self.a.batch)

    def ddim_inv(self, lat, cond, cnimg=None): # ddim inversion, slower, ~exact
        ukwargs = {}
        with self.run_scope('cuda'):
            for t in reversed(self.scheduler.timesteps):
                with torch.no_grad():
                    if self.use_cnet and cnimg is not None: # controlnet
                        ctl_downs, ctl_mid = self.cnet(lat, t, cond, cnimg, 1, return_dict=False)
                        ctl_downs = [ctl_down * self.a.control_scale for ctl_down in ctl_downs]
                        ctl_mid *= self.a.control_scale
                        ukwargs = {'down_block_additional_residuals': ctl_downs, 'mid_block_additional_residual': ctl_mid}
                    noise_pred = self.unet(lat, t, cond, **ukwargs).sample
                lat = self.next_step_ddim(noise_pred, t, lat)
        return lat

    def lat_z(self, lat):
        with self.run_scope('cuda'):
            if self.use_kdiff: # k-samplers, fast, not exact
                return lat + torch.randn_like(lat) * self.sigmas[0]
            else: # ddim stochastic encode, fast, not exact
                return self.scheduler.add_noise(lat, torch.randn(lat.shape, generator=self.g_, device=device, dtype=lat.dtype), self.lat_timestep)

    def img_z(self, image):
        return self.lat_z(self.img_lat(image))

    def rnd_z(self, H, W, frames=None):
        if frames is None: # image
            shape_ = (self.a.batch, 4,         H // self.vae_scale, W // self.vae_scale)
        else: # video
            shape_ = (self.a.batch, 4, frames, H // self.vae_scale, W // self.vae_scale)
        lat = torch.randn(shape_, generator=self.g_, device=device, dtype=torch.float16)
        return self.scheduler.init_noise_sigma * lat

    def prep_mask(self, mask_str, img_path, init_image=None):
        image_pil, _ = load_img(img_path, tensor=False)
        if init_image is None:
            init_image, (W,H) = load_img(img_path)
        with self.run_scope('cuda'):
            mask = makemask(mask_str, image_pil, self.a.invert_mask, model_path=self.clipseg_path)
            masked_lat = self.img_lat(init_image * mask)
            mask = F.interpolate(mask, size = masked_lat.shape[-2:], mode="bicubic", align_corners=False)
        return {'masked_lat': masked_lat, 'mask': mask} # [1,3,64,64], [1,1,64,64]

    def prep_depth(self, init_image):
        [H, W] = init_image.shape[-2:]
        with torch.no_grad(), self.run_scope('cuda'):
            preps = self.feat_extractor(images=[(init_image.squeeze(0)+1)/2], return_tensors="pt").pixel_values
            preps = F.interpolate(preps, size=[384,384], mode="bicubic", align_corners=False).to(device)
            dd = self.depth_estimator(preps).predicted_depth.unsqueeze(0) # [1,1,384,384]
        dd = F.interpolate(dd, size=[H//self.vae_scale, W//self.vae_scale], mode="bicubic", align_corners=False)
        depth_min, depth_max = torch.amin(dd, dim=[1,2,3], keepdim=True), torch.amax(dd, dim=[1,2,3], keepdim=True)
        dd = 2. * (dd - depth_min) / (depth_max - depth_min) - 1.
        return {'depth': dd} # [1,1,64,64]

    def generate(self, lat, cs, uc, c_img=None, cfg_scale=None, cws=None, mask=None, masked_lat=None, depth=None, cnimg=None, ilat=None, offset=0, verbose=True):
        if cfg_scale is None: cfg_scale = self.a.cfg_scale
        with torch.no_grad(), self.run_scope('cuda'):
            if cws is None or not len(cws) == len(cs): cws = [1 / len(cs)] * len(cs)
            conds = uc if cfg_scale==0 else cs[:self.a.batch] if cfg_scale==1 else torch.cat([uc, cs]) if ilat is None else torch.cat([uc, uc, cs]) 
            if self.a.batch > 1: conds = conds.repeat_interleave(self.a.batch, 0)
            bs = len(conds) // self.a.batch
            if cnimg is not None and len(lat.shape)==4: cnimg = cnimg.repeat_interleave(len(conds) // len(cnimg), 0)
            if ilat is not None: ilat = ilat.repeat_interleave(len(conds) // len(ilat), 0)
            if c_img is not None: 
                uc_img = torch.zeros_like(c_img) # [1,1024]
                img_conds = uc_img if cfg_scale==0 else c_img[:self.a.batch] if cfg_scale==1 else torch.cat([uc_img, c_img])
                if self.a.batch > 1: img_conds = img_conds.repeat_interleave(self.a.batch, 0)
            if self.use_lcm: # lcm scheduler requires reset on every generation
                self.set_steps(self.a.steps, self.a.strength)

            if self.use_kdiff:
                def model_fn(x, t, tnum=0):
                    x = x.half()
                    ukwargs = {} # kwargs placeholder for unet
                    if isok(mask, masked_lat) and self.inpaintmod: # inpaint with rml model
                        x = torch.cat([x, 1.-mask, masked_lat], dim=1)
                    elif isok(depth) and self.depthmod: # depth model
                        x = torch.cat([x, depth], dim=1)

                    x, t = torch.cat([x] * bs), torch.cat([t] * bs) # expand latents for guidance

                    if c_img is not None: # encoded img for ip adapter
                        ukwargs['added_cond_kwargs'] = {"image_embeds": img_conds}

                    def calc_pred(x, t, conds, ukwargs={}):
                        if cfg_scale in [0, 1]:
                            pred = self.kdiff_model(x, t, cond=conds, **ukwargs)
                        elif ilat is not None and isset(self.a, 'img_scale'): # instruct pix2pix
                            preds = self.kdiff_model(torch.cat([x, ilat], dim=1), t, cond=conds).chunk(bs)
                            pred = preds[0] + self.a.img_scale * (preds[1] - preds[0]) # uncond + image guidance
                            for n in range(len(preds)-2):
                                pred = pred + (preds[n+2] - preds[1]) * cfg_scale * cws[n % len(cws)] # prompt guidance
                        else:
                            preds = self.kdiff_model(x, t, cond=conds, **ukwargs).chunk(bs)
                            pred = preds[0] # uncond
                            for n in range(len(preds)-1):
                                pred = pred + (preds[n+1] - preds[0]) * cfg_scale * cws[n % len(cws)] # prompt guidance
                        return pred

                    if len(x.shape)==4: # unet2D [b,c,h,w]
                        pred = calc_pred(x, t, conds, ukwargs)
                    else: # unet3D [b,c,f,h,w]
                        frames = x.shape[2]
                        if frames > self.a.ctx_frames: # sliding sampling for long videos
                            pred = torch.zeros_like(x[:1])
                            slide_count = torch.zeros((1, 1, x.shape[2], 1, 1), device=x.device)
                            for slids in uniform_slide(tnum, frames, ctx_size=self.a.ctx_frames, loop=self.a.loop):
                                pred_sub = calc_pred(x[:,:,slids], t, conds, ukwargs)
                                pred[:,:,slids] += pred_sub
                                slide_count[:,:,slids] += 1  # increment which indices were used
                            pred /= slide_count
                        else: # single context video
                            pred = calc_pred(x, t, conds, ukwargs)

                    return pred

                lat = self.sampling_fn(model_fn, lat, self.sigmas[offset:], disable = not verbose)
                if verbose and not iscolab: print() # compensate pbar printout

            else:
                if verbose and not iscolab: pbar = progbar(len(self.timesteps) - offset)
                for tnum, t in enumerate(self.timesteps[offset:]):
                    ukwargs = {} # kwargs placeholder for unet
                    lat_in = self.scheduler.scale_model_input(lat, t) # ~ 1/std(z) ?? https://github.com/huggingface/diffusers/issues/437

                    if isok(mask, masked_lat) and self.inpaintmod: # inpaint with rml model
                        lat_in = torch.cat([lat_in, 1.-mask, masked_lat], dim=1)
                    elif isok(depth) and self.depthmod: # depth model
                        lat_in = torch.cat([lat_in, depth], dim=1)

                    lat_in = torch.cat([lat_in] * bs) # expand latents for guidance

                    if c_img is not None: # encoded img for ip adapter
                        ukwargs['added_cond_kwargs'] = {"image_embeds": img_conds}

                    def calc_noise(x, t, conds, ukwargs={}):
                        if cfg_scale in [0, 1]:
                            noise_pred = self.unet(x, t, conds, **ukwargs).sample
                        elif ilat is not None and isset(self.a, 'img_scale'): # instruct pix2pix
                            noises = self.unet(torch.cat([x, ilat], dim=1), t, conds).sample.chunk(bs)
                            noise_pred = noises[0] + self.a.img_scale * (noises[1] - noises[0]) # uncond + image guidance
                            for n in range(len(noises)-2):
                                noise_pred = noise_pred + (noises[n+2] - noises[1]) * cfg_scale * cws[n % len(cws)] # prompt guidance
                        else:
                            noises = self.unet(x, t, conds, **ukwargs).sample.chunk(bs) # pred noise residual at step t
                            noise_pred = noises[0] # uncond
                            for n in range(len(noises)-1):
                                noise_pred = noise_pred + (noises[n+1] - noises[0]) * cfg_scale * cws[n % len(cws)] # prompt guidance
                        return noise_pred

                    def calc_cnet_batch(x, t, conds, cnimg):
                        bcimg = cnimg.repeat(bs, 1, 1, 1) # bs repeat, frames interleave
                        bconds = conds.repeat_interleave(x.shape[2], 0)
                        bx = rearrange(x, 'b c f h w -> (b f) c h w' )
                        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                            self.unet.to("cpu"); torch.cuda.empty_cache()
                        ctl_downs, ctl_mid = self.cnet(bx, t, bconds, bcimg, self.a.control_scale, return_dict=False)
                        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                            self.cnet.to("cpu"); torch.cuda.empty_cache()
                        return ctl_downs, ctl_mid

                    if len(lat_in.shape)==4: # unet2D [b,c,h,w]
                        if self.use_cnet and cnimg is not None: # controlnet = max 960x640 or 768 on 16gb
                            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                                self.unet.to("cpu"); torch.cuda.empty_cache()
                            ctl_downs, ctl_mid = self.cnet(lat_in, t, conds, cnimg, self.a.control_scale, return_dict=False)
                            ukwargs = {**ukwargs, 'down_block_additional_residuals': ctl_downs, 'mid_block_additional_residual': ctl_mid}
                            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                                self.cnet.to("cpu"); torch.cuda.empty_cache()
                        noise_pred = calc_noise(lat_in, t, conds, ukwargs)

                    else: # unet3D [b,c,f,h,w]
                        frames = lat_in.shape[2]
                        if frames > self.a.ctx_frames: # sliding sampling for long videos
                            noise_pred = torch.zeros_like(lat)
                            slide_count = torch.zeros((1, 1, lat_in.shape[2], 1, 1), device=lat_in.device)
                            for slids in uniform_slide(tnum, frames, ctx_size=self.a.ctx_frames, loop=self.a.loop):
                                if self.use_cnet and cnimg is not None: # controlnet
                                    ctl_downs, ctl_mid = calc_cnet_batch(lat_in[:,:,slids], t, conds, cnimg[slids])
                                    ukwargs = {**ukwargs, 'down_block_additional_residuals': ctl_downs, 'mid_block_additional_residual': ctl_mid}
                                noise_pred_sub = calc_noise(lat_in[:,:,slids], t, conds, ukwargs)
                                noise_pred[:,:,slids] += noise_pred_sub
                                slide_count[:,:,slids] += 1  # increment which indices were used
                            noise_pred /= slide_count

                        else: # single context video
                            if self.use_cnet and cnimg is not None: # controlnet
                                ctl_downs, ctl_mid = calc_cnet_batch(lat_in, t, conds, cnimg)
                                ukwargs = {**ukwargs, 'down_block_additional_residuals': ctl_downs, 'mid_block_additional_residual': ctl_mid}
                            noise_pred = calc_noise(lat_in, t, conds, ukwargs)

                    if self.use_lcm:
                        lat, latend = self.scheduler.step(noise_pred, t, lat, **self.sched_kwargs, return_dict=False)
                    else:
                        lat = self.scheduler.step(noise_pred, t, lat, **self.sched_kwargs).prev_sample # compute previous noisy sample x_t -> x_t-1

                    if verbose and not iscolab: pbar.upd()

            if self.use_lcm:
                lat = latend

            if isok(mask, masked_lat) and not self.inpaintmod: # inpaint with standard models
                lat = masked_lat * mask + lat * (1.-mask)

            # decode latents
            lat /= self.vae.config.scaling_factor

            if len(lat.shape)==5: # video
                lat = lat.permute(0,2,1,3,4).squeeze(0).half() # [f,c,h,w]
                output = torch.cat([self.vae.decode(lat[b : b + self.a.vae_batch]).sample.float().cpu() for b in range(0, len(lat), self.a.vae_batch)])
                output = output[None,:].permute(0,2,1,3,4) # [1,c,f,h,w]
            else: # image
                output = self.vae.decode(lat).sample

            # Offload last model
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.final_offload_hook.offload()

            return output


# sliding sampling for long videos
# from https://github.com/ArtVentureX/comfyui-animatediff/blob/main/animatediff/sliding_schedule.py
def ordered_halving(val, verbose=False): # Returns fraction that has denominator that is a power of 2
    bin_str = f"{val:064b}" # get binary value, padded with 0s for 64 bits
    bin_flip = bin_str[::-1] # flip binary value, padding included
    as_int = int(bin_flip, 2) # convert binary to int
    final = as_int / (1 << 64) # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616, or 1 with 64 zero's
    if verbose: print(f"$$$$ final: {final}")
    return final
# generate lists of latent indices to process
def uniform_slide(step, num_frames, ctx_size=16, ctx_stride=1, ctx_overlap=4, loop=True, verbose=False):
    if num_frames <= ctx_size:
        yield list(range(num_frames))
        return
    ctx_stride = min(ctx_stride, int(np.ceil(np.log2(num_frames / ctx_size))) + 1)
    pad = int(round(num_frames * ordered_halving(step, verbose)))
    fstop = num_frames + pad + (0 if loop else -ctx_overlap)
    for ctx_step in 1 << np.arange(ctx_stride):
        fstart = int(ordered_halving(step) * ctx_step) + pad
        fstep = ctx_size * ctx_step - ctx_overlap
        for j in range(fstart, fstop, fstep):
            yield [e % num_frames for e in range(j, j + ctx_size * ctx_step, ctx_step)]

