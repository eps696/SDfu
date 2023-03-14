
import os, sys
import time
from easydict import EasyDict
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

sys.path.append(os.path.join(os.path.dirname(__file__), '../xtra'))
import k_diffusion as K

from util.finetune import load_embeds, load_delta
from util.utils import load_img, makemask, isok, isset, progbar

import logging
logging.getLogger('diffusers').setLevel(logging.ERROR)
try:
    import xformers; isxf = True
except: isxf = False

samplers = ['klms', 'pndm', 'dpm', 'euler_a', 'dpm2_a',   'ddim', 'euler']
models = ['15', '15i', '2i', '2d', '21', '21v'] # !! only 15 is uncensored !!
device = torch.device('cuda')

class SDpipe(DiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler):
        super().__init__()
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

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

def set_sampler(scheduler_type: str):
    library = importlib.import_module("k_diffusion")
    sampling = getattr(library, "sampling")
    sampler = getattr(sampling, scheduler_type)
    return sampler


def sd_setup(a, vae=None, text_encoder=None, tokenizer=None, unet=None, scheduler=None):
    # settings
    use_half = isset(a, 'precision') and a.precision not in ['full', 'float32', 'fp32']
    precision_scope = torch.autocast if use_half else nullcontext
    if isset(a, 'ddim_inv') and a.ddim_inv is True: a.sampler = 'ddim'
    if not isset(a, 'seed'): a.seed = int((time.time()%1)*69696)
    seed_everything(a.seed)

    # paths
    if not isset(a, 'maindir'): a.maindir = './models' # for external scripts
    clipseg_path = os.path.join(a.maindir, 'clipseg/rd64-uni.pth')
    vtype = a.model[-1] == 'v'
    subdir = 'v2v' if vtype else 'v2' if a.model[0]=='2' else 'v1'

    if vtype and not isxf: # scheduler.prediction_type == "v_prediction":
        print(" V-models require xformers! install it or use another model"); exit()

    txtenc_path = os.path.join(a.maindir, subdir, 'text')
    if text_encoder is None:
        text_encoder = CLIPTextModel.from_pretrained(txtenc_path, torch_dtype=torch.float16).to(device)
    if tokenizer is None:
        tokenizer    = CLIPTokenizer.from_pretrained(txtenc_path, torch_dtype=torch.float16)

    if unet is None:
        unet_path = os.path.join(a.maindir, subdir, 'unet' + a.model)
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16).to(device)
    if not isxf: unet.set_attention_slice(unet.config.attention_head_dim // 2)

    # load finetuned stuff
    if  isset(a, 'delta_ckpt') and os.path.isfile(a.delta_ckpt): # custom diffusion
        mod_tokens = load_delta(torch.load(a.delta_ckpt), text_encoder, tokenizer, unet)
        print(' loaded tokens:', mod_tokens)
    elif isset(a, 'token_emb') and os.path.exists(a.token_emb): # text inversion
        emb_files = [a.token_emb] if os.path.isfile(a.token_emb) else file_list(a.token_emb, 'pt')
        for emb_file in emb_files:
            mod_tokens = load_embeds(torch.load(emb_file), text_encoder, tokenizer)
            print(' loaded token:', mod_tokens)

    if vae is None:
        vae_path = 'vae'
        if a.model[0]=='1' and a.vae != 'orig':
            vae_path = 'vae-ft-mse' if a.vae=='mse' else 'vae-ft-ema'
        vae_path = os.path.join(a.maindir, subdir, vae_path)
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16).to(device)
    if not isxf: vae.enable_slicing()

    if scheduler is None:
        sched_path = os.path.join(a.maindir, subdir, 'scheduler_config.json')
        sched_kwargs = {}
        if a.sampler == 'pndm':
            scheduler = PNDMScheduler.from_pretrained(sched_path)
        elif a.sampler == 'dpm':
            scheduler = DPMSolverMultistepScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="scaled_linear", solver_order=2, sample_max_value=1.)
        elif a.sampler == 'ddim':
            scheduler = DDIMScheduler.from_pretrained(sched_path)
            sched_kwargs = {"eta": a.ddim_eta}
        else:
            scheduler = LMSDiscreteScheduler.from_pretrained(sched_path)
            model = ModelWrapper(unet, scheduler.alphas_cumprod)
            kdiff_model = K.external.CompVisVDenoiser(model) if vtype else K.external.CompVisDenoiser(model)
            kdiff_model.sigmas, kdiff_model.log_sigmas = kdiff_model.sigmas.to(device), kdiff_model.log_sigmas.to(device)

    # sampling
    scheduler.set_timesteps(a.steps, device=device)
    if not isset(a, 'in_img'): a.strength = 1.
    a.steps = min(int(a.steps * a.strength), a.steps)
    a.use_kdiff = a.sampler not in ['pndm', 'ddim', 'dpm']
    if a.use_kdiff:
        sigmas = scheduler.sigmas[-a.steps-1 :]
        if   a.sampler == 'klms':    sampling_fn = K.sampling.sample_lms
        elif a.sampler == 'euler':   sampling_fn = K.sampling.sample_euler
        elif a.sampler == 'euler_a': sampling_fn = K.sampling.sample_euler_ancestral
        elif a.sampler == 'dpm2_a':  sampling_fn = K.sampling.sample_dpm_2_ancestral # slow but rich!
        else: print(' Unknown sampler', a.sampler); exit()
    else:
        timesteps = scheduler.timesteps[-a.steps :]
        lat_timestep = timesteps[:1].repeat(a.batch)

    pipe = SDpipe(vae, text_encoder, tokenizer, unet, scheduler).to(device).to("cuda")
    if isxf: pipe.enable_xformers_memory_efficient_attention()

    vae_scale = 2 ** (len(vae.config.block_out_channels) - 1) # 8
    a.res = unet.config.sample_size * vae_scale # original model resolution
    a.depth = unet.in_channels==5
    a.inpaint = unet.in_channels==9
    assert not (a.inpaint and not isset(a, 'mask')), '!! Inpainting model requires mask !!' 
    assert not (unet.in_channels != 4 and a.use_kdiff), "!! K-samplers don't work with depth/inpaint models !!"

# # # main functions

    def next_step_ddim(noise, t, sample):
        t, next_t = min(t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999), t
        alpha_prod_t      = scheduler.alphas_cumprod[t] if t >= 0 else scheduler.final_alpha_cumprod
        alpha_prod_t_next = scheduler.alphas_cumprod[next_t]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * noise) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    @precision_scope('cuda')
    def img_lat(image):
        lats = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
        return torch.cat([lats])
    @precision_scope('cuda')
    def lat_z(lat, cond=None):
        if a.ddim_inv: # ddim inversion, slower, ~exact
            for t in reversed(scheduler.timesteps):
                with torch.no_grad():
                    noise_pred = unet(lat, t, cond).sample
                lat = next_step_ddim(noise_pred, t, lat)
            return lat
        elif a.use_kdiff: # k-samplers, fast, not exact
            return lat + torch.randn_like(lat) * sigmas[0]
        else: # ddim stochastic encode, fast, not exact
            return scheduler.add_noise(lat, torch.randn(lat.shape, device=device, dtype=lat.dtype), lat_timestep)
    def img_z(image, cond=None): # cond - for ddim inversion only
        return lat_z(img_lat(image), cond)
    def rnd_z(H, W):
        shape_ = (a.batch, 4, H // vae_scale, W // vae_scale)
        lat = torch.randn(shape_, device=device)
        return scheduler.init_noise_sigma * lat

    func = EasyDict(img_lat=img_lat, img_z=img_z, rnd_z=rnd_z)

    if isset(a, 'mask'):
        def prep_mask(mask_str, img_path, init_image=None):
            image_pil = load_img(img_path, tensor=False)
            if init_image is None:
                init_image, (W,H) = load_img(img_path)
            mask = makemask(mask_str, image_pil, a.invert_mask, model_path=clipseg_path)
            masked_lat = img_lat(init_image * mask)
            mask = F.interpolate(mask, size = masked_lat.shape[-2:], mode="bicubic", align_corners=False)
            return {'masked_lat': masked_lat, 'mask': mask} # [1,3,64,64], [1,1,64,64]
        func.prep_mask = prep_mask

    if a.depth:
        from transformers import DPTForDepthEstimation, DPTFeatureExtractor
        depth_path = os.path.join(a.maindir, subdir, 'depth')
        depth_estimator = DPTForDepthEstimation.from_pretrained(depth_path, torch_dtype=torch.float16).to(device)
        feature_extractor = DPTFeatureExtractor.from_pretrained(depth_path, torch_dtype=torch.float16, device=device)
        def prep_depth(init_image):
            [H, W] = init_image.shape[-2:]
            with torch.no_grad(), torch.autocast("cuda"):
                preps = feature_extractor(images=[init_image.squeeze(0)], return_tensors="pt").pixel_values
                preps = F.interpolate(preps, size=[384,384], mode="bicubic", align_corners=False).to(device)
                dd = depth_estimator(preps).predicted_depth.unsqueeze(0) # [1,1,384,384]
            dd = F.interpolate(dd, size=[H//vae_scale, W//vae_scale], mode="bicubic", align_corners=False)
            depth_min, depth_max = torch.amin(dd, dim=[1,2,3], keepdim=True), torch.amax(dd, dim=[1,2,3], keepdim=True)
            dd = 2. * (dd - depth_min) / (depth_max - depth_min) - 1.
            return {'depth': dd} # [1,1,64,64]
        func.prep_depth = prep_depth

    def generate(lat, c_, uc, mask=None, masked_lat=None, depth=None, a=a, verbose=True):
        with torch.no_grad(), precision_scope('cuda'):
            if a.batch > 1:
                uc, c_ = uc.repeat(a.batch, 1, 1), c_.repeat(a.batch, 1, 1)
            conds = uc if a.cfg_scale==0 else c_ if a.cfg_scale==1 else torch.cat([uc, c_])

            if a.use_kdiff:
                def model_fn(x, t):
                    if a.cfg_scale in [0, 1]:
                        noise_pred = kdiff_model(x, t, cond=conds)
                    else:
                        lat_in, t = torch.cat([x]*2), torch.cat([t]*2)
                        nz_uncond, nz_cond = kdiff_model(lat_in, t, cond=conds).chunk(2)
                        noise_pred = nz_uncond + a.cfg_scale * (nz_cond - nz_uncond)
                    return noise_pred
                lat = sampling_fn(model_fn, lat, sigmas)
                if verbose: print() # compensate pbar printout

            else:
                log = 'gen sched %d, ts %d' % (len(scheduler.timesteps), len(timesteps))
                if verbose: pbar = progbar(len(timesteps))
                for i, t in enumerate(timesteps):
                    lat_in = scheduler.scale_model_input(lat, t) # scales only k-samplers ~ 1/std(z), not needed here

                    if isok(mask, masked_lat) and a.inpaint: # inpaint with rml model
                        lat_in = torch.cat([lat_in, 1.-mask, masked_lat], dim=1)
                    elif isok(depth) and a.depth: # depth model
                        lat_in = torch.cat([lat_in, depth], dim=1)

                    if a.cfg_scale in [0, 1]:
                        noise_pred = unet(lat_in, t, conds).sample # pred noise residual at step t # ..cross_attn_kwargs
                    else:
                        lat_in = torch.cat([lat_in] * 2) # expand latents for classifier free guidance
                        nz_uncond, nz_cond = unet(lat_in, t, conds).sample.chunk(2) # pred noise residual at step t # ..cross_attn_kwargs
                        noise_pred = nz_uncond + a.cfg_scale * (nz_cond - nz_uncond) # guidance here

                    lat = scheduler.step(noise_pred, t, lat, **sched_kwargs).prev_sample # compute previous noisy sample x_t -> x_t-1
                    if verbose: pbar.upd(log)

            if isok(mask, masked_lat) and not a.inpaint: # inpaint with standard models
                lat = masked_lat * mask + lat * (1.-mask)

            # decode latents
            lat /= vae.config.scaling_factor
            return vae.decode(lat).sample

    return [a, func, pipe, generate]

