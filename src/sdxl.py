
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import time
import numpy as np
import imageio
from contextlib import nullcontext
from huggingface_hub import hf_hub_download

import torch
from safetensors.torch import load_file

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler

from core.args import main_args, unprompt
from core.text import read_txt, txt_clean
from core.utils import load_img, save_img, lerp, slerp, makemask, blend, cvshow, calc_size, isok, isset

try:
    import xformers; isxf = True
except: isxf = False
try: # colab
    get_ipython().__class__.__name__
    iscolab = True
except: iscolab = False

from eps import img_list, basename, progbar, save_cfg

device = torch.device('cuda')

def get_args(parser):
    parser.add_argument('-mdir', '--models_dir', default='models/xl')
    # parser.add_argument('-m',  '--model',   default='base', help='base or refiner')
    parser.add_argument('-lg',  '--lightning', action='store_true', help='Use SDXL-Lightning Unet')
    parser.add_argument('-fs', '--fstep',   default=1, type=int, help="number of frames for each interpolation step (1 = no interpolation)")
    parser.add_argument('-lb', '--latblend', default=0, type=float, help="Strength of latent blending, if > 0: 0.1 ~ alpha-blend, 0.9 ~ full rebuild")
    parser.add_argument('-cts', '--control_scale', default=0.5, type=float, help="ControlNet effect scale")
    parser.add_argument(       '--curve',   default='linear', help="Interpolating curve: bezier, parametric, custom or linear")
    # UNUSED
    parser.add_argument('-sm', '--sampler', default=None)
    parser.add_argument(       '--vae',     default=None)
    parser.add_argument('-cg', '--cguide',  action=None)
    parser.add_argument('-lo', '--lowmem',  action=None)
    parser.add_argument('-rt', '--load_token', default=None)
    parser.add_argument('-rd', '--load_custom', default=None)
    parser.add_argument('-rl', '--load_lora', default=None)
    return parser.parse_args()


def read_multitext(in_txt, prefix=None, postfix=None):
    if in_txt is None or len(in_txt)==0: return []
    lines = [tt.strip() for tt in read_txt(in_txt) if tt.strip()[0] != '#']
    if prefix is not None: lines = [prefix + tt for tt in lines]
    if postfix is not None: lines = [tt + postfix for tt in lines]
    prompts = [tt.split('|')[:2] for tt in lines] # 2 text encoders = 2 subprompts
    texts   = [txt_clean(tt) for tt in lines]
    return prompts, texts

class SDfu:
    def __init__(self, a):
        self.a = a
        self.device = device
        self.run_scope = nullcontext # torch.autocast
        if not isset(a, 'maindir'): a.maindir = './models' # for external scripts
        self.setseed(a.seed if isset(a, 'seed') else None)
        if not isset(a, 'in_img'): a.strength = 1.
        a.unprompt = unprompt(a)
        
        def get_model(name, url):
            return os.path.join(a.models_dir, name) if os.path.exists(os.path.join(a.models_dir, name)) else url

        base_url = 'stabilityai/stable-diffusion-xl-base-1.0'
        self.pipe = DiffusionPipeline.from_pretrained(get_model('base', base_url), torch_dtype=torch.float16, variant="fp16")
        if a.lightning is True and a.steps in [2,4,8]: # lightning
            a.cfg_scale = 0
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(get_model('base', base_url), subfolder='scheduler', timestep_spacing="trailing")
            unet_path = os.path.join(a.models_dir, 'sdxl_lightning_%dstep_unet.safetensors' % a.steps)
            if not os.path.exists(unet_path): unet_path = hf_hub_download('ByteDance/SDXL-Lightning', 'sdxl_lightning_%dstep_unet.safetensors' % a.steps)
            self.pipe.unet.load_state_dict(load_file(unet_path))
        self.pipe.to(device)
        self.unet           = self.pipe.unet
        self.vae            = self.pipe.vae
        self.scheduler      = self.pipe.scheduler
        self.text_encoder   = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        self.tokenizer      = self.pipe.tokenizer
        self.tokenizer_2    = self.pipe.tokenizer_2

        # load controlnet
        if isset(a, 'control_mod'):
            if a.verbose: print(' loading ControlNet', a.control_mod)
            from diffusers import ControlNetModel
            self.cnet = ControlNetModel.from_pretrained(get_model(a.control_mod, 'diffusers/controlnet-canny-sdxl-1.0-mid'), torch_dtype=torch.float16)
            if not self.a.lowmem: self.cnet.to(device)
            self.pipe.register_modules(controlnet=self.cnet)
        self.use_cnet = hasattr(self, 'cnet')

        # load ip adapter = after animatediff
        if isset(a, 'img_ref'):
            if a.verbose: print(' loading IP adapter for images', a.img_ref)
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection as CLIPimg
            if os.path.exists(os.path.join(a.models_dir, 'image')):
                self.image_preproc = CLIPImageProcessor.from_pretrained(os.path.join(a.models_dir, 'image/preproc_config.json'))
                self.image_encoder = CLIPimg.from_pretrained(os.path.join(a.models_dir, 'image'), torch_dtype=torch.float16).to(device)
                self.unet._load_ip_adapter_weights(torch.load(os.path.join(a.models_dir, 'image/ip-adapter_sdxl.bin'), map_location="cpu"))
            else:
                self.image_preproc = CLIPImageProcessor.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='feature_extractor')
                self.image_encoder = CLIPimg.from_pretrained('h94/IP-Adapter', subfolder='sdxl_models/image_encoder', torch_dtype=torch.float16).to(device)
                self.unet._load_ip_adapter_weights(torch.load(hf_hub_download('h94/IP-Adapter', 'sdxl_models/ip-adapter_sdxl.bin'), map_location="cpu"))
                
            self.pipe.register_modules(image_encoder = self.image_encoder)
            self.pipe.set_ip_adapter_scale(a.imgref_weight)

        self.vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1) # 8
        self.res = self.unet.config.sample_size * self.vae_scale # original model resolution
        self.set_steps(a.steps, a.strength)

    def setseed(self, seed=None):
        self.seed = seed or int((time.time()%1)*69696)
        self.g_ = torch.Generator("cuda").manual_seed(self.seed)
    
    def set_steps(self, steps, strength=1., device=device):
        self.scheduler.set_timesteps(steps, device=device)
        steps = min(int(steps * strength), steps)
        self.timesteps = self.scheduler.timesteps[-steps:]
        self.lat_timestep = self.timesteps[:1].repeat(self.a.batch)

    def encode_prompt(self, prompt, prompt2=None, unprompt=None, unprompt2=None, do_cfg=True, num=1):
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        if isinstance(prompt,  str): prompt  = [prompt]
        if isinstance(prompt2, str): prompt2 = [prompt2]
        prompt2 = prompt2 or prompt
        prompts = [prompt, prompt2]
        cs = []
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            tokens = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            c_ = text_encoder(tokens.input_ids.to(device), output_hidden_states=True)
            pool_c = c_[0]
            cs.append(c_.hidden_states[-2])
        cs = torch.concat(cs, dim=-1)
        cs = cs.repeat(num,1,1).to(device, dtype = self.unet.dtype if self.text_encoder_2 is None else self.text_encoder_2.dtype)
        pool_c = pool_c.repeat(num,1)
        uc      = torch.zeros_like(cs)     if do_cfg else None
        pool_uc = torch.zeros_like(pool_c) if do_cfg else None
        return cs, uc, pool_c, pool_uc

    def img_cus(self, img_path, allref=False): # ucs and cs together
        assert os.path.exists(img_path), "!! Image ref %s not found !!" % img_path
        if allref is True: # all files at once
            img_conds = [self.img_cu([load_img(im, 224, tensor=False)[0] for im in [os.path.join(dp, f) for dp,dn,fn in os.walk(img_path) for f in fn]])]
        else:
            if os.path.isfile(img_path): # single image
                img_conds = [self.img_cu(load_img(img_path, 224, tensor=False)[0])] # list 1 of [2,1,1024]
            else:
                subdirs = sorted([f.path for f in os.scandir(img_path) if f.is_dir()])
                if len(subdirs) > 0: # every subfolder at once
                    img_conds = [self.img_cu([load_img(im, 224, tensor=False)[0] for im in img_list(sub)]) for sub in subdirs] # list N of [2,1,1024]
                else: # every image separately
                    img_conds = [self.img_cu(load_img(im, 224, tensor=False)[0]) for im in img_list(img_path)] # list N of [2,1,1024]
        return img_conds # list of [2,1,1024]

    def img_cu(self, images): # uc and c together
        with self.run_scope('cuda'):
            images = self.image_preproc(images, return_tensors="pt").pixel_values.to(device) # [N,3,224,224]
            cs = self.image_encoder(images).image_embeds.mean(0, keepdim=True) # [1,1024]
            ucs = torch.zeros_like(cs)
            return torch.stack([ucs, cs]) # [2,1,1024]

    def img_lat(self, image):
        with self.run_scope('cuda'):
            self.vae.to(dtype=torch.float32)
            lats = self.vae.encode(image.float()).latent_dist.sample(self.g_).to(dtype=torch.float16)
            self.vae.to(torch.float16)
            lats *= self.vae.config.scaling_factor
        return torch.cat([lats] * self.a.batch)

    def lat_z(self, lat):
        with self.run_scope('cuda'):
            return self.scheduler.add_noise(lat, torch.randn(lat.shape, generator=self.g_, device=device, dtype=lat.dtype), self.lat_timestep)

    def img_z(self, image):
        return self.lat_z(self.img_lat(image))

    def rnd_z(self, H, W, frames=None):
        shape_ = [self.a.batch, 4, H // self.vae_scale, W // self.vae_scale] # image b,4,h,w
        if frames is not None: shape_.insert(2, frames) # video b,4,f,h,w
        lat = torch.randn(shape_, generator=self.g_, device=device, dtype=torch.float16)
        return self.scheduler.init_noise_sigma * lat

    def generate(self, lat, cs, pool_c, time_ids, cfg_scale, c_img=None, cnimg=None, verbose=True):
        if cfg_scale is None: cfg_scale = self.a.cfg_scale
        with torch.no_grad(), self.run_scope('cuda'):
            self.set_steps(self.a.steps, self.a.strength) # trailing (lcm) schedulers require reset on every generation
            bs = 1 if cfg_scale in [0,1] else 2
            conds = cs.repeat_interleave(len(lat), 0)
            pool_c = pool_c.repeat_interleave(len(lat), 0)
            time_ids = time_ids.repeat_interleave(len(lat), 0)
            ukwargs = {'added_cond_kwargs': {"text_embeds": pool_c, "time_ids": time_ids}}
            if cnimg is not None: cnimg = cnimg.repeat_interleave(len(conds) // len(cnimg), 0)
            if c_img is not None: # already with uc
                c_imgs = c_img if isinstance(c_img, list) else [c_img]
                img_conds = []
                for c_img in c_imgs:
                    img_cond = c_img[len(c_img)//2:] if cfg_scale in [0,1] else c_img # only c if no scale
                    img_conds += [img_cond.repeat_interleave(len(lat), 0)]
                ukwargs['added_cond_kwargs']['image_embeds'] = img_conds

            def calc_noise(x, t, conds, **ukwargs):
                if cfg_scale in [0,1]:
                    return self.unet(x, t, conds, **ukwargs).sample
                else:
                    noise_un, noise_c = self.unet(x, t, conds, **ukwargs).sample.chunk(2)
                    return noise_un + cfg_scale * (noise_c - noise_un)

            if verbose and not iscolab: pbar = progbar(len(self.timesteps))
            for t in self.timesteps:
                lat_in = self.scheduler.scale_model_input(lat, t)
                lat_in = torch.cat([lat_in] * bs)

                cnkwargs = {}
                if self.use_cnet and cnimg is not None:
                    ctl_downs, ctl_mid = self.cnet(lat_in, t, conds, cnimg, self.a.control_scale, **ukwargs, return_dict=False)
                    cnkwargs = {'down_block_additional_residuals': ctl_downs, 'mid_block_additional_residual': ctl_mid}
                noise_pred = calc_noise(lat_in, t, conds, **ukwargs, **cnkwargs)

                lat = self.scheduler.step(noise_pred, t, lat).prev_sample

                if verbose and not iscolab: pbar.upd()

            self.vae.to(dtype=torch.float32)
            self.vae.post_quant_conv.to(lat.dtype)
            self.vae.decoder.conv_in.to(lat.dtype)
            self.vae.decoder.mid_block.to(lat.dtype)
            lat = lat.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            lat /= self.vae.config.scaling_factor
            if len(lat.shape)==5: # video
                lat = lat.permute(0,2,1,3,4).squeeze(0) # [f,c,h,w]
                output = torch.cat([self.vae.decode(lat[b : b + self.a.vae_batch]).sample.float().cpu() for b in range(0, len(lat), self.a.vae_batch)])
                output = output[None,:].permute(0,2,1,3,4) # [1,c,f,h,w]
            else: # image
                output = torch.cat([self.vae.decode(l).sample.float().cpu() for l in lat.chunk(len(lat))]) # OOM if batch

            return output


@torch.no_grad()
def main():
    a = get_args(main_args())
    if a.models_dir is None: a.models_dir = os.path.join(a.maindir, 'xl')
    if a.in_img is not None and a.fstep > 1: print('Interpolation with images not supported with SDXL'); exit()
    sd = SDfu(a)
    a = sd.a
    a.seed = sd.seed
    size = None if not isset(a, 'size') else calc_size(a.size)
    do_cfg = a.cfg_scale not in [0,1]
    os.makedirs(a.out_dir, exist_ok=True)
    if a.verbose: save_cfg(a, a.out_dir)
    if a.verbose: print(' sd xl ..', a.steps, '..', a.cfg_scale, '..', a.strength, '..', a.seed)
    gendict = {}

    prompts, texts = read_multitext(a.in_txt, a.pretxt, a.postxt)
    cs, pool_cs = [], []
    for prompt_ in prompts:
        p1, p2 = prompt_ if len(prompt_)==2 else prompt_ * 2
        c_, uc, pool_c, pool_uc = sd.encode_prompt(p1, p2, a.unprompt, a.unprompt, do_cfg, a.num)
        cs.append(c_)
        pool_cs.append(pool_c)
    cs = torch.cat(cs).unsqueeze(1) # [N,1,77,2048]
    pool_cs = torch.cat(pool_cs).unsqueeze(1) # [N,1,1280]
    count = len(cs)
    
    img_conds = []
    if isset(a, 'img_ref'):
        img_conds = sd.img_cus(a.img_ref, isset(a, 'allref')) # list of [2,1,1024]
        count = max(count, len(img_conds))

    cn_imgs = []
    if sd.use_cnet and isset(a, 'control_img'):
        assert os.path.exists(a.control_img), "!! ControlNet image(s) %s not found !!" % a.control_img
        cn_imgs = img_list(a.control_img) if os.path.isdir(a.control_img) else [a.control_img]
        count = max(count, len(cn_imgs))

    if isset(a, 'in_img'): # img2img
        assert os.path.exists(a.in_img), "!! Image(s) %s not found !!" % a.in_img
        img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
        zs = torch.stack([sd.img_z(load_img(im, size)[0]) for im in img_paths])
        H, W = [s * sd.vae_scale for s in zs[0].shape[-2:]]
        count = max(count, len(img_paths))
    else: # txt2img
        W, H = [sd.res]*2 if size is None else size
        zs = torch.stack([sd.rnd_z(H, W) for i in range(count)])

    if a.latblend > 0:
        from core.latblend import LatentBlending
        time_ids = torch.tensor([[H, W, 0, 0, H, W]], device=device, dtype=cs.dtype)
        lb = LatentBlending(sd, a.steps, a.cfg_scale, time_ids)
        img_count = 0

    def genmix(i=0, tt=0, cnimg=None, img=None, mask=None):
        z_ = slerp(zs[i % len(zs)], zs[(i+1) % len(zs)], tt) # [1,4,128,128]
        c_ =  lerp(cs[i % len(cs)], cs[(i+1) % len(cs)], tt) # [1,77,2048]
        pool_c = lerp(pool_cs[i % len(pool_cs)], pool_cs[(i+1) % len(pool_cs)], tt) # [1,1280]
        c_img = lerp(img_conds[i % len(img_conds)], img_conds[(i+1) % len(img_conds)], tt) if len(img_conds) > 0 else None # [2,1,1280]
        time_ids = torch.tensor([[H, W, 0, 0, H, W]], device=device, dtype=cs.dtype) # [1,6] orig_size, crops_top_left, tgt_size, ..
        if do_cfg:
            c_ = torch.cat([uc, c_], dim=0) # [2,77,2048]
            pool_c = torch.cat([pool_uc, pool_c], dim=0) # [2,1280]
            time_ids = torch.cat([time_ids]*2, dim=0) # [2,6]
        c_ = c_.to(device)
        pool_c = pool_c.to(device)
        image = sd.generate(z_, c_, pool_c, time_ids, a.cfg_scale, c_img, cnimg)
        return image

    pbar = progbar(count if a.latblend > 0 else count * a.fstep)
    for i in range(count):
        log = texts[i % len(texts)][:44] if len(texts) > 0 else ''

        if len(cn_imgs) > 0:
            gendict['cnimg'] = (load_img(cn_imgs[i % len(cn_imgs)], (W,H))[0] + 1) / 2

        if a.fstep <= 1: # single
            if isset(a, 'in_img'):
                img_path = img_paths[i % len(img_paths)]
                file_out = basename(img_path) if len(img_paths) == count else '%06d' % i
                log += ' .. %s' % os.path.basename(img_path)
            else:
                file_out = '%03d-%s-%d' % (i, log, sd.seed)

            images = genmix(i, **gendict)
            torch.cuda.empty_cache()

            if a.verbose: cvshow(np.array(images.permute(0,2,3,1)[0] *.5 + .5))
            if len(images) > 1:
                for j in range(len(images)):
                    save_img(images[j], j, a.out_dir, filepath=file_out + '-%02d.jpg' % j)
            else:
                save_img(images[0], 0, a.out_dir, filepath=file_out + '.jpg')
            pbar.upd(log, uprows=2)

        else: # interpolate
            if a.latblend > 0:
                if len(img_conds) > 0:
                    gendict['c_img'] = [ img_conds[i % len(img_conds)], img_conds[(i+1) % len(img_conds)] ]
                lb.set_conds(cs[i % len(cs)], cs[(i+1) % len(cs)], uc, pool_cs[i % len(pool_cs)], pool_cs[(i+1) % len(pool_cs)], pool_uc, **gendict)
                lb.init_lats( zs[i % len(zs)],   zs[(i+1) % len(zs)])
                lb.run_transition(W, H, 1.- a.latblend, a.fstep, reuse = i>0)
                img_count += lb.save_imgs(a.out_dir, img_count, skiplast=True)
                pbar.upd(uprows=2)
            else:
                for f in range(a.fstep):
                    tt = blend(f / a.fstep, a.curve)
                    images = genmix(i, tt, **gendict)
                    if a.verbose: cvshow(np.array(images.permute(0,2,3,1)[0] *.5 + .5))
                    save_img(images[0], i * a.fstep + f, a.out_dir)
                    pbar.upd(log, uprows=2)


if __name__ == '__main__':
    main()
