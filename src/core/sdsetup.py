
import logging
logging.getLogger('xformers').setLevel(logging.ERROR)
logging.getLogger('diffusers.models.modeling_utils').setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*", message='1Torch*')
warnings.filterwarnings('ignore', category=FutureWarning, module="xformers.*")

import os, sys
import time
import numpy as np
from contextlib import nullcontext
from einops import rearrange
import inspect

import torch
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, ImageProjection
from diffusers.models.embeddings import IPAdapterPlusImageProjection, IPAdapterFullImageProjection
from diffusers.models.embeddings import IPAdapterFaceIDImageProjection, IPAdapterFaceIDPlusImageProjection
from diffusers import StableDiffusionPipeline
from diffusers.utils import is_accelerate_available, is_accelerate_version
logging.getLogger('diffusers').setLevel(logging.ERROR)

sys.path.append(os.path.join(os.path.dirname(__file__), '../xtra'))

from .text import multiprompt
from .utils import file_list, img_list, load_img, makemask, isok, isset, progbar, gpu_ram, clean_vram
from .args import models, unprompt

try:
    import xformers; isxf = True
except: isxf = False
try: # colab
    get_ipython().__class__.__name__
    iscolab = True
except: iscolab = False

is_mac = torch.backends.mps.is_available() and torch.backends.mps.is_built() # M1/M2 chip?
is_cuda = torch.cuda.is_available()
device = 'mps' if is_mac else 'cuda' if is_cuda else 'cpu'
dtype = torch.float16 if is_cuda or is_mac else torch.float32

class SDpipe(StableDiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, image_encoder=None, \
                 safety_checker=None, feature_extractor=None, requires_safety_checker=False):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, image_encoder, None, None, requires_safety_checker=False)
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
        if image_encoder is not None:
            self.register_modules(image_encoder=image_encoder)

class SDfu:
    def __init__(self, a, vae=None, text_encoder=None, tokenizer=None, unet=None, scheduler=None):
        # settings
        self.a = a
        self.device = device
        self.dtype = dtype
        if is_mac: self.a.lowmem = False # no model offloading on mac
        self.run_scope = nullcontext if is_mac else torch.autocast # Don't use autocast for MPS
        if not isset(a, 'maindir'): a.maindir = './models' # for external scripts
        self.setseed(a.seed if isset(a, 'seed') else None)
        a.unprompt = unprompt(a)

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
        if not self.a.lowmem: self.pipe.to(device)
        if is_mac: self.pipe.enable_attention_slicing()
        self.use_kdiff = hasattr(self.scheduler, 'sigmas') # k-diffusion sampling

        # load finetuned stuff
        mod_tokens = None
        if isset(a, 'load_custom') and os.path.isfile(a.load_custom): # custom diffusion
            from .finetune import load_delta, custom_diff
            self.pipe.unet = custom_diff(self.pipe.unet, train=False)
            mod_tokens = load_delta(torch.load(a.load_custom, weights_only=True), self.pipe.unet, self.pipe.text_encoder, self.pipe.tokenizer)
        elif isset(a, 'load_token') and os.path.exists(a.load_token): # text inversion
            from .finetune import load_embeds
            emb_files = [a.load_token] if os.path.isfile(a.load_token) else file_list(a.load_token, 'pt')
            mod_tokens = []
            for emb_file in emb_files:
                mod_tokens += load_embeds(torch.load(emb_file, weights_only=True), self.pipe.text_encoder, self.pipe.tokenizer)
        if mod_tokens is not None: print(' loaded tokens:', mod_tokens[0] if len(mod_tokens)==1 else mod_tokens)

        # load animatediff = before ip adapter, after custom diffusion !
        if isset(a, 'animdiff'):
            if not os.path.exists(a.animdiff): a.animdiff = os.path.join(a.maindir, a.animdiff)
            assert os.path.exists(a.animdiff), "Not found AnimateDiff model %s" % a.animdiff
            if a.verbose: print(' loading AnimateDiff', a.animdiff)
            from diffusers.models import UNetMotionModel, MotionAdapter
            motion_adapter = MotionAdapter.from_pretrained(a.animdiff)
            self.unet = UNetMotionModel.from_unet2d(self.unet, motion_adapter)
            self.scheduler = self.set_scheduler(a) # k-samplers must be loaded after unet
            if not self.a.lowmem: self.unet.to(device)
            self.pipe.register_modules(unet = self.unet)

        if isset(a, 'load_lora'): # lora = after animdiff !
            self.pipe.load_lora_weights(a.load_lora, low_cpu_mem_usage=True)
            self.pipe.fuse_lora() # (lora_scale=0.7)
            if a.verbose: print(' loaded LoRA', a.load_lora)

        if isset(a, 'animdiff'): # after lora !
            self.pipe.register_modules(motion_adapter = motion_adapter)

        # load ip adapters = after animatediff
        if isset(a, 'img_ref'):
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection as CLIPimg
            self.image_preproc = CLIPImageProcessor.from_pretrained(os.path.join(a.maindir, 'image/preproc_config.json')) # openai/clip-vit-base-patch32
            self.image_encoder = CLIPimg.from_pretrained(os.path.join(a.maindir, 'image'), torch_dtype=dtype).to(device)
            self.pipe.register_modules(image_encoder = self.image_encoder)
            ip_dicts, ip_ws, self.ips = [], [], []
            srcs, ipas, iptypes, ws = a.img_ref.split('+'), a.ipa.split('+'), a.ip_type.split('+'), a.imgref_weight.split('+')
            refcount = max([len(srcs), len(ipas), len(iptypes), len(ws)])
            for i in range(refcount):
                ipa_name = ipas[i % len(ipas)]
                ipa_path = ipa_name if os.path.exists(ipa_name) else os.path.join(a.maindir, 'image', ipa_name)
                if not os.path.isfile(ipa_path): ipa_path = os.path.join(a.maindir, 'image/%s.bin' % ('ipa-' + ipa_name if len(ipa_name) > 0 else 'ipa'))
                iptype = iptypes[i % len(iptypes)]
                ipw = float(ws[i % len(ws)])
                if 'face' in iptype:
                    import insightface
                    self.face_align = insightface.utils.face_align
                    self.faceidr = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider' if is_cuda else 'CPUExecutionProvider'])
                    self.faceidr.prepare(ctx_id=0)
                ip_dicts += [torch.load(ipa_path, weights_only=True, map_location="cpu")]
                ip_ws += [{'down': {'block_2':[ipw]*2}} if 'scen' in iptype else {'up': {'block_1': [ipw]*3, 'block_2':[ipw]*3}} if 'styl' in iptype else ipw]
                self.ips += [iptype]
                if a.verbose: print(' loading IP adapter %s for images' % os.path.basename(ipa_path), srcs[i % len(srcs)])
            self.unet._load_ip_adapter_weights(ip_dicts)
            self.pipe.set_ip_adapter_scale(ip_ws)

        # load controlnet = after lora
        if isset(a, 'control_mod'):
            from diffusers import ControlNetModel
            from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
            cns, cis, cnws = a.control_mod.split('+'), a.control_img.split('+'), [float(s) for s in a.control_scale.split('+')]
            cn_count = max([len(cns), len(cis), len(cnws)])
            self.cns = [cns[i % len(cns)] for i in range(cn_count)]
            self.cnet_ws = [cnws[i % len(cnws)] for i in range(cn_count)]
            cnets = []
            for cn in self.cns:
                if not os.path.exists(cn): cn = os.path.join(self.a.maindir, 'control', cn)
                assert os.path.exists(cn), "Not found ControlNet model %s" % cn
                cnets += [ControlNetModel.from_pretrained(cn, torch_dtype=dtype)]
            self.cnet = MultiControlNetModel(cnets)
            if a.verbose: print(' loading ControlNets', self.cns)
            if not self.a.lowmem: self.cnet.to(device)
            self.pipe.register_modules(controlnet=self.cnet)
        self.use_cnet = hasattr(self, 'cnet')

        self.final_setup(a)

    def load_model_external(self, model_path):
        SDload = StableDiffusionPipeline.from_single_file if os.path.isfile(model_path) else StableDiffusionPipeline.from_pretrained
        self.pipe = SDload(model_path, torch_dtype=dtype, safety_checker=None)
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer    = self.pipe.tokenizer
        self.unet         = self.pipe.unet
        self.vae          = self.pipe.vae
        self.scheduler    = self.pipe.scheduler if self.a.sampler=='orig' else self.set_scheduler(self.a, path=os.path.join(model_path, 'scheduler'))

    def load_model_custom(self, a, vae=None, text_encoder=None, tokenizer=None, unet=None, scheduler=None):
        # paths
        self.clipseg_path = os.path.join(a.maindir, 'xtra/clipseg/rd64-uni.pth')
        vidtype = a.model[0] == 'v'
        self.subdir = 'v2' if vidtype or a.model[0]=='2' else 'v1'

        # text input
        txtenc_path = os.path.join(a.maindir, self.subdir, 'text-' + a.model[2:] if a.model[2:] in ['drm'] else 'text')
        if text_encoder is None:
            text_encoder = CLIPTextModel.from_pretrained(txtenc_path, torch_dtype=dtype, local_files_only=True)
        if tokenizer is None:
            tokenizer    = CLIPTokenizer.from_pretrained(txtenc_path, torch_dtype=dtype, local_files_only=True)
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        if unet is None:
            unet_path = os.path.join(a.maindir, self.subdir, 'unet' + a.model)
            if vidtype:
                from diffusers.models import UNet3DConditionModel as UNet
            else:
                from diffusers.models import UNet2DConditionModel as UNet
            unet = UNet.from_pretrained(unet_path, torch_dtype=dtype, local_files_only=True)
        self.unet = unet

        if vae is None:
            vae_path = 'vae'
            if a.model[0]=='1' and a.vae != 'orig':
                vae_path = 'vae-ft-mse' if a.vae=='mse' else 'vae-ft-ema'
            vae_path = os.path.join(a.maindir, self.subdir, vae_path)
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
        if a.lowmem: vae.enable_tiling() # higher res, more free ram
        elif vidtype or isset(a, 'animdiff') or not isxf: vae.enable_slicing()
        self.vae = vae

        if scheduler is None:
            scheduler = self.set_scheduler(a, self.subdir)
        self.scheduler = scheduler

        self.pipe = SDpipe(vae, text_encoder, tokenizer, unet, scheduler)

    def set_scheduler(self, a, subdir='', path=None):
        sched_args = {}
        if isset(a, 'animdiff'): sched_args = {'beta_schedule': "linear", 'timestep_spacing': "linspace", 'clip_sample': False, 'steps_offset':1, **sched_args} 
        if not isset(self, 'sched_path'):
            self.sched_path = os.path.join(a.maindir, subdir, 'scheduler_config-%s.json' % a.model)
            if not os.path.exists(self.sched_path):
                self.sched_path = os.path.join(a.maindir, subdir, 'scheduler_config.json')
            if not os.path.exists(self.sched_path) and path is not None:
                self.sched_path = os.path.join(path, 'scheduler_config.json')
        if a.sampler == 'lcm':
            from diffusers.schedulers import LCMScheduler
            scheduler = LCMScheduler.from_pretrained(os.path.join(a.maindir, 'lcm/scheduler/scheduler_config.json'))
        elif a.sampler == 'dpm':
            from diffusers.schedulers import DPMSolverMultistepScheduler
            scheduler = DPMSolverMultistepScheduler(beta_schedule="scaled_linear", **sched_args)
        else:
            if a.sampler == 'ddim':
                from diffusers.schedulers import DDIMScheduler as Sched
            elif a.sampler == 'pndm':
                from diffusers.schedulers import PNDMScheduler as Sched
            elif a.sampler == 'euler':
                from diffusers.schedulers import EulerDiscreteScheduler as Sched
            elif a.sampler == 'euler_a':
                from diffusers.schedulers import EulerAncestralDiscreteScheduler as Sched
            elif a.sampler == 'ddpm':
                from diffusers.schedulers import DDPMScheduler as Sched
            elif a.sampler == 'uni':
                from diffusers.schedulers import UniPCMultistepScheduler as Sched
            elif a.sampler == 'lms':
                from diffusers.schedulers import LMSDiscreteScheduler as Sched
            elif a.sampler == 'tcd':
                from diffusers.schedulers import TCDScheduler as Sched
            else:
                print(' Unknown sampler', a.sampler); exit()
            scheduler = Sched.from_pretrained(self.sched_path, **sched_args)
        return scheduler

    def final_setup(self, a):
        if isxf and not (isset(a, 'img_ref') or isset(a, 'animdiff')): # !!! breaks ip adapter or animdiff
            self.pipe.enable_xformers_memory_efficient_attention()
        if isset(a, 'freeu'): self.pipe.unet.enable_freeu(s1=1.5, s2=1.6, b1=0.9, b2=0.2) # 2.1 1.4, 1.6, 0.9, 0.2, sdxl 1.3, 1.4, 0.9, 0.2

        self.sched_kwargs = {'eta': a.eta} if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()) else {}
        if "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            self.sched_kwargs['generator'] = self.g_

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
            self.depth_estimator = DPTForDepthEstimation.from_pretrained(depth_path, torch_dtype=dtype).to(device)
            self.feat_extractor  = DPTImageProcessor.from_pretrained(depth_path, torch_dtype=dtype, device=device)
        if isset(a, 'img_scale') and a.img_scale > 0: # instruct pix2pix
            assert not (self.use_cnet or a.cfg_scale in [0,1]), "Use either Instruct-pix2pix or Controlnet guidance"
            a.strength = 1.

        self.set_steps(a.steps, a.strength) # sampling

        # enable_model_cpu_offload
        if self.a.lowmem and is_cuda:
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
        self.g_ = torch.Generator(device).manual_seed(self.seed)
    
    def set_steps(self, steps, strength=1., warmup=1, device=device):
        if self.use_lcm:
            self.scheduler.set_timesteps(steps, device, strength=strength)
            self.timesteps = self.scheduler.timesteps
        else:
            self.scheduler.set_timesteps(steps, device=device)
            steps = min(int(steps * strength), steps) # t_enc .. 50
            self.timesteps = self.scheduler.timesteps[-steps - warmup :] # 1 warmup step
        self.lat_timestep = self.timesteps[:1].expand(self.a.batch, -1) # MPS-friendly expand

    def next_step_ddim(self, noise, t, sample):
        t, next_t = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), t
        alpha_prod_t      = self.scheduler.alphas_cumprod[t] if t >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_t]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * noise) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def ddim_inv(self, lat, cond, cnimg=None, c_img=None): # ddim inversion, slower, ~exact
        ukwargs = {}
        if c_img is not None: # list of [2,1,...] encodings for ip adapter, already with uc_img
            img_conds = [c_.chunk(2)[1] if self.a.cfg_scale in [0,1] else c_ for c_ in c_img]
            ukwargs['added_cond_kwargs'] = {"image_embeds": img_conds}
        with self.run_scope(device):
            for t in reversed(self.scheduler.timesteps):
                with torch.no_grad():
                    if self.use_cnet and cnimg is not None: # controlnet
                        ctl_downs, ctl_mid = self.cnet(lat, t, cond, cnimg, self.cnet_ws, return_dict=False)
                        ukwargs = {'down_block_additional_residuals': ctl_downs, 'mid_block_additional_residual': ctl_mid}
                    noise_pred = self.unet(lat, t, cond, **ukwargs).sample
                lat = self.next_step_ddim(noise_pred, t, lat)
        return lat

    def img_cus(self, img_path, ipnum, allref=False): # ucs and cs together
        assert os.path.exists(img_path), "!! Image ref %s not found !!" % img_path
        if allref is True: # all files at once
            img_conds = [self.img_cu([load_img(im, tensor=False)[0] for im in img_list(img_path, subdir=True)], ipnum)]
        else:
            if os.path.isfile(img_path): # single image
                img_conds = [self.img_cu(load_img(img_path, tensor=False)[0], ipnum)] # list 1 of [2,1,..]
            else: # every image/subfolder separately
                subs = sorted(img_list(img_path) + [f.path for f in os.scandir(img_path) if f.is_dir()])
                img_conds = [self.img_cu([load_img(im, tensor=False)[0] for im in img_list(sub)] if os.path.isdir(sub) \
                                     else load_img(sub,tensor=False)[0], ipnum) for sub in subs] # list N of [2,1,..]
        return img_conds # list of [2,1,1024] base or [2,1,257,1280] plus or [2,1,512] faceid

    def img_cu(self, images, ipnum): # uc and c together
        with self.run_scope(device):
            if not isinstance(images, list): images = [images]
            if 'face' in self.ips[ipnum]:
                faces = [self.faceidr.get(np.array(img)) for img in images]
                faces = [face[0].normed_embedding if len(face)>0 else np.zeros([512]) for face in faces] # list of [512], first face on every image
            images_t = self.image_preproc(images, return_tensors="pt").pixel_values.to(device) # [N,3,224,224]

            ip_layer = self.unet.encoder_hid_proj.image_projection_layers[ipnum]
            if isinstance(ip_layer, ImageProjection): # base ip adapter
                cs = self.image_encoder(images_t).image_embeds.mean(0, keepdim=True) # [1,1024]
                ucs = torch.zeros_like(cs)
            elif isinstance(ip_layer, IPAdapterFaceIDImageProjection): # face id adapter
                cs = torch.stack([torch.tensor(face).to(device, dtype=dtype) for face in faces]).mean(0, keepdim=True) # [1,512]
                ucs = torch.zeros_like(cs)
            elif isinstance(ip_layer, IPAdapterFaceIDPlusImageProjection): # face id plus adapter, fixed for all frames
                cs = torch.stack([torch.tensor(face).to(device, dtype=dtype) for face in faces]).mean(0, keepdim=True) # [1,512]
                ucs = torch.zeros_like(cs)
                clip_cs = self.image_encoder(images_t, output_hidden_states=True).hidden_states[-2].mean(0, keepdim=True) # [1,257,1280]
                clip_ucs = self.image_encoder(torch.zeros_like(images_t), output_hidden_states=True).hidden_states[-2].mean(0, keepdim=True)
                ip_layer.clip_embeds = torch.stack([clip_ucs, clip_cs]).to(dtype=dtype)
                if isset(self.a, 'animdiff'): # fix for framesets
                    clip_embeds = ip_layer.clip_embeds.detach().clone()
                    ip_layer.clip_embeds = clip_embeds.repeat_interleave(self.a.ctx_frames, dim=0) # MPS-friendly
                    del clip_embeds
                ip_layer.shortcut = 'plus2' in self.a.ipa.split('+')[ipnum % len(self.a.ipa.split('+'))] # True if plus v2
            elif isinstance(ip_layer, (IPAdapterPlusImageProjection, IPAdapterFullImageProjection)): # plus-family adapters
                cs = self.image_encoder(images_t, output_hidden_states=True).hidden_states[-2].mean(0, keepdim=True) # [1,257,1280]
                ucs = self.image_encoder(torch.zeros_like(images_t), output_hidden_states=True).hidden_states[-2].mean(0, keepdim=True)
            else:
                print(' unknown IP adapter \n', ip_layer); exit()
            return torch.stack([ucs, cs]) # [2,1,1024] base or [2,1,257,1280] plus or [2,1,512] faceid

    def img_lat(self, image, deterministic=False):
        with self.run_scope(device):
            if image.device.type != device: image = image.to(device)
            if image.dtype != dtype: image = image.to(dtype=dtype)
            postr = self.vae.encode(image).latent_dist
            lats = postr.mean if deterministic else postr.sample(self.g_)
            lats *= self.vae.config.scaling_factor
        return torch.cat([lats] * self.a.batch)

    def lat_z(self, lat): # ddim stochastic encode, fast, not exact
        with self.run_scope(device):
            timestep = self.lat_timestep[:1].expand(lat.shape[0], -1) # MPS-friendly expand 
            return self.scheduler.add_noise(lat, torch.randn(lat.shape, generator=self.g_, device=device, dtype=lat.dtype), timestep)

    def img_z(self, image):
        return self.lat_z(self.img_lat(image))

    def rnd_z(self, H, W, frames=None):
        shape_ = [self.a.batch, 4, H // self.vae_scale, W // self.vae_scale] # image b,4,h,w
        if frames is not None: shape_.insert(2, frames) # video b,4,f,h,w
        lat = torch.randn(shape_, generator=self.g_, device=device, dtype=dtype)
        return self.scheduler.init_noise_sigma * lat

    def prep_mask(self, mask_str, img_path, init_image=None):
        image_pil, _ = load_img(img_path, tensor=False)
        if init_image is None:
            init_image, (W,H) = load_img(img_path)
        with self.run_scope(device):
            mask = makemask(mask_str, image_pil, self.a.invert_mask, model_path=self.clipseg_path)
            masked_lat = self.img_lat(init_image * mask)
            mask = F.interpolate(mask, size = masked_lat.shape[-2:], mode="bicubic", align_corners=False)
        return {'masked_lat': masked_lat, 'mask': mask} # [1,3,64,64], [1,1,64,64]

    def prep_depth(self, init_image):
        [H, W] = init_image.shape[-2:]
        with torch.no_grad(), self.run_scope(device):
            preps = self.feat_extractor(images=[(init_image.squeeze(0)+1)/2], return_tensors="pt").pixel_values
            preps = F.interpolate(preps, size=[384,384], mode="bicubic", align_corners=False).to(device)
            dd = self.depth_estimator(preps).predicted_depth.unsqueeze(0) # [1,1,384,384]
        dd = F.interpolate(dd, size=[H//self.vae_scale, W//self.vae_scale], mode="bicubic", align_corners=False)
        depth_min, depth_max = torch.amin(dd, dim=[1,2,3], keepdim=True), torch.amax(dd, dim=[1,2,3], keepdim=True)
        dd = 2. * (dd - depth_min) / (depth_max - depth_min) - 1.
        return {'depth': dd} # [1,1,64,64]

    def sag_masking(self, orig_lats, attn_map, map_size, t, eps):
        bh, hw1, hw2 = attn_map.shape # [8*f,h,w]
        h = self.unet.config.attention_head_dim if isset(self.unet.config, 'attention_head_dim') else self.unet.config.num_attention_heads
        if isinstance(h, list): h = h[-1]
        if len(orig_lats.shape)==4:
            orig_lats = orig_lats.unsqueeze(2) # make compatible with unet3D
        b, ch, seq, lath, latw = orig_lats.shape # [b,4,f,h,w]
        attn_map = attn_map.reshape(b * seq, h, hw1, hw2) # [b*f,8,144,144]
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0 # [b*f,144]
        attn_mask = attn_mask.reshape(b, seq, map_size[0], map_size[1]).unsqueeze(1)
        attn_mask = attn_mask.repeat(1, ch, 1, 1, 1).type(attn_map.dtype) # [b,ch,f,12,12] # orig
        attn_mask = torch.stack([F.interpolate(attn_mask[:,:,i], (lath, latw)) for i in range(seq)]).permute(1,2,0,3,4)
        degraded_lats = torch.stack([gaussian_blur_2d(orig_lats[:,:,i], kernel_size=9, sigma=1.0) for i in range(seq)]).permute(1,2,0,3,4)
        degraded_lats = degraded_lats * attn_mask + orig_lats * (1 - attn_mask)
        if len(eps.shape) < len(degraded_lats.shape):
            degraded_lats = degraded_lats.squeeze(2) # get back to unet2D
        t_expanded = torch.cat([t[None]] * len(degraded_lats), dim=0) # MPS-friendly expand
        degraded_lats = self.scheduler.add_noise(degraded_lats, noise=eps, timesteps=t_expanded) # Noise it again to match the noise level
        return degraded_lats

    def pred_x0(self, sample, model_output, timestep): # from DDIMScheduler.step
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep.cpu().to(dtype=torch.int32)].to(sample.device)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample


    def generate(self, lat, cs, uc, c_img=None, cfg_scale=None, cws=None, mask=None, masked_lat=None, depth=None, cnimg=None, ilat=None, offset=0, verbose=True):
        if cfg_scale is None: cfg_scale = self.a.cfg_scale
        if self.a.sag_scale > 0:
            assert self.scheduler.config.prediction_type == "epsilon", "Only prediction_type 'epsilon' is implemented here"
            store_processor = CrossAttnStoreProcessor()
            self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = store_processor
            map_size = None
            def get_map_size(module, input, output):
                nonlocal map_size
                map_size = output[0].shape[-2:]
        loop_context = self.unet.mid_block.attentions[0].register_forward_hook(get_map_size) if self.a.sag_scale > 0 else nullcontext() # SAG
        with torch.no_grad(), self.run_scope(device), loop_context:
            if cws is None or not len(cws) == len(cs): cws = [len(uc) / len(cs)] * len(cs)
            conds = uc if cfg_scale==0 else cs if cfg_scale==1 else torch.cat([uc,cs], dim=0) if ilat is None else torch.cat([uc,uc,cs], dim=0)
            if self.a.batch > 1: conds = conds.repeat_interleave(self.a.batch, 0)
            bs = len(conds) // (len(uc) * self.a.batch)
            if cnimg is not None and len(lat.shape)==4: cnimg = [ci.repeat_interleave(len(conds) // len(ci), 0) for ci in cnimg]
            if ilat is not None: ilat = ilat.repeat_interleave(len(conds) // len(ilat), 0)
            if c_img is not None:  # already with uc_img
                c_imgs = c_img if isinstance(c_img, list) else [c_img]
                img_conds, img_uncond = [], []
                for c_img in c_imgs:
                    if len(cs) > 1 and len(c_img)==2: c_img = c_img.repeat_interleave(len(cs), 0)
                    img_cond = c_img.chunk(2)[1] if cfg_scale in [0,1] else c_img # only c if no scale, to keep correct shape
                    img_uncond += [c_img.chunk(2)[0]]
                    if self.a.batch > 1: img_cond = img_cond.repeat_interleave(self.a.batch, 0)
                    img_conds += [img_cond] # [2*f, 1, ...]
            if self.use_kdiff or self.use_lcm or self.a.sampler=='tcd': # trailing (lcm) or k- schedulers require reset on every generation
                self.set_steps(self.a.steps, self.a.strength)
            sagkwargs = {}

            if verbose: pbar = progbar(len(self.timesteps) - offset)
            for tnum, t in enumerate(self.timesteps[offset:]):
                ukwargs = {} # kwargs placeholder for unet
                lat_in = self.scheduler.scale_model_input(lat, t) # ~ 1/std(z) ?? https://github.com/huggingface/diffusers/issues/437

                if isok(mask, masked_lat) and self.inpaintmod: # inpaint with rml model
                    lat_in = torch.cat([lat_in, 1.-mask, masked_lat], dim=1)
                elif isok(depth) and self.depthmod: # depth model
                    lat_in = torch.cat([lat_in, depth], dim=1)

                lat_in = torch.cat([lat_in] * bs, dim=0) # expand latents for guidance

                if c_img is not None: # encoded img for ip adapter
                    ukwargs['added_cond_kwargs'] = {"image_embeds": img_conds}
                    if self.a.sag_scale > 0:
                        sagkwargs['added_cond_kwargs'] = {"image_embeds": img_conds} if cfg_scale in [0,1] else {"image_embeds": img_uncond}

                def calc_noise(x, t, conds, ukwargs={}):
                    self.unet.to(device)
                    if len(x) == 1 and len(conds) == 1: # no guidance
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
                    if self.a.sag_scale > 0:
                        if cfg_scale in [0,1]:
                            pred_x0 = self.pred_x0(x[:1], noise_pred, t) # DDIM-like prediction of x0
                            cond_attn = store_processor.attention_probs # get the stored attention maps
                            degraded_lats = self.sag_masking(pred_x0, cond_attn, map_size, t, noise_pred)
                            degraded_pred = self.unet(degraded_lats, t, conds, **sagkwargs).sample
                            noise_pred += self.a.sag_scale * (noise_pred - degraded_pred)
                        else:
                            pred_x0 = self.pred_x0(x[:1], noises[0], t) # DDIM-like prediction of x0
                            uncond_attn, cond_attn = store_processor.attention_probs.chunk(2) # get the stored attention maps
                            degraded_lats = self.sag_masking(pred_x0, uncond_attn, map_size, t, noise_pred)
                            degraded_pred = self.unet(degraded_lats, t, conds.chunk(2)[0], **sagkwargs).sample
                            noise_pred += self.a.sag_scale * (noises[0] - degraded_pred)
                    return noise_pred

                def calc_cnet_batch(x, t, conds, cnimg):
                    bx = rearrange(x, 'b c f h w -> (b f) c h w' ) # bs repeat, frames interleave
                    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                        self.unet.to("cpu"); clean_vram()
                    ctl_downs, ctl_mid = self.cnet(bx, t, conds, [ci.repeat(bs,1,1,1) for ci in cnimg], self.cnet_ws, return_dict=False)
                    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                        self.cnet.to("cpu"); clean_vram()
                    return ctl_downs, ctl_mid

                if len(lat_in.shape)==4: # unet2D [b,c,h,w]
                    if self.use_cnet and cnimg is not None: # controlnet = max 960x640 or 768 on 16gb
                        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                            self.unet.to("cpu"); clean_vram()
                        ctl_downs, ctl_mid = self.cnet(lat_in, t, conds, cnimg, self.cnet_ws, return_dict=False)
                        ukwargs = {**ukwargs, 'down_block_additional_residuals': ctl_downs, 'mid_block_additional_residual': ctl_mid}
                        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                            self.cnet.to("cpu"); clean_vram()
                    noise_pred = calc_noise(lat_in, t, conds, ukwargs)

                else: # unet3D [b,c,f,h,w]
                    frames = lat_in.shape[2]
                    if frames > self.a.ctx_frames: # sliding sampling for long videos
                        noise_pred = torch.zeros_like(lat)
                        slide_count = torch.zeros((1, 1, lat_in.shape[2], 1, 1), device=lat_in.device)
                        for slids in uniform_slide(tnum, frames, ctx_size=self.a.ctx_frames, loop=self.a.loop):
                            conds_ = conds if 'vzs' in self.a.model else conds[slids] if cfg_scale in [0,1] else torch.cat([cc[slids] for cc in conds.chunk(bs)], dim=0)
                            if c_img is not None: # ip adapter
                                ukwargs['added_cond_kwargs']['image_embeds'] = [torch.cat([cc[slids] for cc in imcond.chunk(bs)], dim=0) for imcond in img_conds]
                                if self.a.sag_scale > 0:
                                    imcc = img_conds if cfg_scale in [0,1] else img_uncond
                                    sagkwargs['added_cond_kwargs']['image_embeds'] = [imcond[slids] for imcond in imcc]
                            if self.use_cnet and cnimg is not None: # controlnet
                                ctl_downs, ctl_mid = calc_cnet_batch(lat_in[:,:,slids], t, conds_, [ci[slids] for ci in cnimg])
                                ukwargs = {**ukwargs, 'down_block_additional_residuals': ctl_downs, 'mid_block_additional_residual': ctl_mid}
                            noise_pred_sub = calc_noise(lat_in[:,:,slids], t, conds_, ukwargs)
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

                if verbose: pbar.upd()

            if self.use_lcm:
                lat = latend

            if isok(mask, masked_lat) and not self.inpaintmod: # inpaint with standard models
                lat = masked_lat * mask + lat * (1.-mask)

            # decode latents
            lat /= self.vae.config.scaling_factor

            if len(lat.shape)==5: # video
                lat = lat.permute(0,2,1,3,4).squeeze(0).to(dtype=dtype) # [f,c,h,w]
                output = torch.cat([self.vae.decode(lat[b : b + self.a.vae_batch]).sample.float().cpu() for b in range(0, len(lat), self.a.vae_batch)])
                output = output.unsqueeze(0).permute(0,2,1,3,4) # [1,c,f,h,w]
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

class CrossAttnStoreProcessor: # processes and stores attention probabilities
    def __init__(self):
        self.attention_probs = None
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states) # linear proj
        hidden_states = attn.to_out[1](hidden_states) # dropout
        return hidden_states

def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])
    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])
    return img
