
import os
import time
import argparse

import torch
from pytorch_lightning import seed_everything

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, PNDMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available

from core.text import txt_clean, read_txt
from core.utils import load_img, save_img, calc_size, isok, isset, img_list, basename, progbar

def get_args():
    parser = argparse.ArgumentParser()
    # inputs & paths
    parser.add_argument('-t',  '--in_txt',  default=None, help='Text string or file to process')
    parser.add_argument('-im', '--in_img',  default=None, help='input image or directory with images (overrides width and height)')
    parser.add_argument('-o',  '--out_dir', default="_out", help="Output directory for generated images")
    parser.add_argument('-md', '--maindir', default='./models', help='Main SD models directory')
    # mandatory params
    parser.add_argument('-m',  '--model',   default='15', choices=['14','15','21'], help="model version")
    parser.add_argument('-sm', '--sampler', default='pndm', choices=['pndm', 'ddim'])
    parser.add_argument('-C','--cfg_scale', default=13, type=float, help="prompt guidance scale")
    parser.add_argument('-f', '--strength', default=0.75, type=float, help="strength of image processing. 0 = preserve img, 1 = replace it completely")
    parser.add_argument('-s',  '--steps',   default=50, type=int, help="number of diffusion steps")
    parser.add_argument('-S',  '--seed',    type=int, help="image seed")
    # misc
    parser.add_argument('-sz', '--size',    default=None, help="image size, multiple of 8")
    return parser.parse_args()

device = torch.device('cuda')

class SDpipe(DiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler):
        super().__init__()
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

def sd_setup(a):
    # settings
    if not isset(a, 'seed'): a.seed = int((time.time()%1)*69696)
    seed_everything(a.seed)
    subdir = 'v2' if a.model[0]=='2' else 'v1'

    # text encoder & tokenizer
    txtenc_path = os.path.join(a.maindir, subdir, 'text')
    text_encoder = CLIPTextModel.from_pretrained(txtenc_path, torch_dtype=torch.float16).to(device)
    tokenizer    = CLIPTokenizer.from_pretrained(txtenc_path, torch_dtype=torch.float16)

    # unet
    unet_path = os.path.join(a.maindir, subdir, 'unet' + a.model)
    unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16).to(device)

    # vae
    vae_path = os.path.join(a.maindir, subdir, 'vae')
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16).to(device)

    # scheduler
    sched_path = os.path.join(a.maindir, subdir, 'scheduler_config.json')
    if a.sampler == 'pndm':
        scheduler = PNDMScheduler.from_config(sched_path)
    else: # ddim
        scheduler = DDIMScheduler.from_config(sched_path)

    # sampling
    scheduler.set_timesteps(a.steps, device=device)
    if not isset(a, 'in_img'): a.strength = 1.
    a.steps = min(int(a.steps * a.strength), a.steps)
    timesteps = scheduler.timesteps[-a.steps :]
    lat_timestep = timesteps[:1]

    vae_scale = 8
    a.res = unet.config.sample_size * vae_scale # original model resolution

    # main functions
    def img_lat(image):
        image = image.half()
        lats = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
        return torch.cat([lats])
    def lat_z(lat):
        return scheduler.add_noise(lat, torch.randn(lat.shape, device=device, dtype=lat.dtype), lat_timestep)
    def img_z(image):
        return lat_z(img_lat(image))
    def rnd_z(H, W):
        shape_ = (1, 4, H // vae_scale, W // vae_scale)
        lat = torch.randn(shape_, device=device)
        return scheduler.init_noise_sigma * lat # scale initial noise by std required by the scheduler; not needed for ddim/pndm
    def txt_c(txt):
        prompt_tokens = tokenizer(txt)
        prompt_embeds = text_encoder(prompt_tokens.input_ids.to(device))[0]
        return prompt_embeds.to(dtype=text_encoder.dtype)

    pipe = SDpipe(vae, text_encoder, tokenizer, unet, scheduler).to(device)
    if is_xformers_available: pipe.enable_xformers_memory_efficient_attention()

    uc = txt_c([""])

    def generate(lat, c_):
        with torch.no_grad(), torch.autocast('cuda'):
            conds = torch.cat([uc, c_])

            pbar = progbar(len(timesteps) - 1)
            for i, t in enumerate(timesteps):
                lat_in = torch.cat([lat] * 2) # expand latents for classifier free guidance
                lat_in = scheduler.scale_model_input(lat_in, t) # not needed for ddim/pndm

                noise_pred_uncond, noise_pred_cond = unet(lat_in, t, conds).sample.chunk(2) # pred noise residual at step t
                noise_pred = noise_pred_uncond + a.cfg_scale * (noise_pred_cond - noise_pred_uncond) # guidance here

                lat = scheduler.step(noise_pred, t, lat, **sched_kwargs).prev_sample # compute previous noisy sample x_t -> x_t-1
                pbar.upd()

            # decode latents
            lat /= vae.config.scaling_factor
            decoded_image = vae.decode(lat).sample
            return decoded_image

    return [a, img_z, rnd_z, txt_c, generate]

@torch.no_grad()
def main():
    a = get_args()
    [a, img_z, rnd_z, txt_c, generate] = sd_setup(a)

    os.makedirs(a.out_dir, exist_ok=True)
    print('.. model', a.model, '..', a.sampler, '..', a.cfg_scale, '..', a.strength)

    prompts = read_txt(a.in_txt)
    count = len(prompts)

    if isset(a, 'in_img') and os.path.exists(a.in_img):
        img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
        count = max(count, len(img_paths))

    pbar = progbar(count)
    for i in range(count):
        prompt = prompts[i % len(prompts)]
        c_ = txt_c(prompt)

        if isset(a, 'in_img'): # img2img
            img_path = img_paths[i % len(img_paths)]
            file_out = basename(img_path)
            init_image, (W,H) = load_img(img_path)
            z_ = img_z(init_image)

        else: # txt2img
            file_out = '%s-m%s-%s-%d' % (txt_clean(prompt)[:88], a.model, a.sampler, a.seed)
            W, H = [a.res]*2 if a.size is None else calc_size(a.size)
            z_ = rnd_z(H, W)

        images = generate(z_, c_)

        save_img(images[0], 0, a.out_dir, filepath = file_out + '.jpg')
        pbar.upd(uprows=2)

if __name__ == '__main__':
    main()
