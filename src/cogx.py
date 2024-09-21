
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import imageio
import inspect
import math
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="xformers.*")
warnings.filterwarnings('ignore', category=UserWarning, message='1Torch was not compiled')
import logging
logging.getLogger('xformers').setLevel(logging.ERROR) # shutup triton
logging.getLogger('diffusers.models.modeling_utils').setLevel(logging.CRITICAL)

import torch

from transformers import T5EncoderModel, T5Tokenizer
from diffusers import CogVideoXPipeline
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.video_processor import VideoProcessor
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.utils import load_image, load_video

from core.args import main_args, unprompt
from core.text import read_txt, txt_clean
from core.utils import isset, gpu_ram, basename, progbar, save_cfg

def get_args(parser):
    parser.add_argument('-iv', '--in_vid',  default=None, help='input video or directory with images')
    parser.add_argument('-vf', '--frames',  default=49, type=int, help="Frame count for generated video")
    parser.add_argument('-cf', '--ctx_frames', default=13, type=int, help="latent count to process at once with sliding window sampling")
    parser.add_argument('-dc', '--dyn_cfg', action='store_true', help='Dynamic CFG scale - good for txt2vid')
    parser.add_argument('-fps', '--fps',    default=12, type=int, help="Frame rate")
    parser.add_argument('--loop',           action='store_true')
    # override
    parser.add_argument('-C','--cfg_scale', default=6, type=float, help="prompt guidance scale")
    parser.add_argument('-s',  '--steps',   default=50, type=int, help="number of diffusion steps")
    return parser.parse_args()

device = torch.device("cuda")

class CogXpipe(CogVideoXPipeline):
    def __init__(self, tokenizer, text_encoder, vae, transformer, scheduler):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler)

class CogX:
    def __init__(self, a):
        self.a = a
        self.device = device
        self.seed = a.seed or int((time.time()%1)*69696)
        a.unprompt = unprompt(a)

        mod_path = os.path.join("models/cogx", 'i2v' if isset(a, 'in_img') and os.path.isfile(a.in_img) else 'main')
        if not os.path.isdir(mod_path): mod_path = 'THUDM/CogVideoX-5b-I2V' if isset(a, 'in_img') and os.path.isfile(a.in_img) else 'THUDM/CogVideoX-5b'
        dtype = torch.bfloat16

        self.tokenizer = T5Tokenizer.from_pretrained(mod_path, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(mod_path, subfolder="text_encoder", torch_dtype=dtype)
        self.vae = AutoencoderKLCogVideoX.from_pretrained(mod_path, subfolder="vae", torch_dtype=dtype)
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(mod_path, subfolder="transformer", torch_dtype=dtype)
        self.scheduler = CogVideoXDPMScheduler.from_pretrained(mod_path, subfolder="scheduler") # CogVideoXDDIMScheduler CogVideoXDPMScheduler
        self.pipe = CogXpipe(self.tokenizer, self.text_encoder, self.vae, self.transformer, self.scheduler)

        self.vae_scale_s = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_scale_t = self.vae.config.temporal_compression_ratio
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_s)

        if a.lowmem or gpu_ram() < 20:
            try:
                from optimum.quanto import freeze, quantize, qfloat8 # qfloat8_e4m3fn, qfloat8_e5m2, qint8, qint4, qint2
                quantize(self.text_encoder, weights=qfloat8); freeze(self.text_encoder)
                quantize(self.transformer, weights=qfloat8);  freeze(self.transformer)
                quantize(self.vae, weights=qfloat8);          freeze(self.vae)
                self.pipe.to("cuda")
            except:
                print(' Quantization support requires Optimum library with Python 3.10+ and Torch 2.4')

        self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_sequential_cpu_offload()
        self.transformer.to(memory_format=torch.channels_last)
        self.vae.enable_slicing()
        self.vae.enable_tiling()

        self.g_ = torch.Generator("cuda").manual_seed(self.seed)

        self.sched_kwargs = {'eta': a.eta} if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()) else {}
        if "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            self.sched_kwargs['generator'] = self.g_


    def get_rot_pos_emb(self, height, width, num_frm=13):
        gridH = height // (self.vae_scale_s * self.transformer.config.patch_size)
        gridW = width  // (self.vae_scale_s * self.transformer.config.patch_size)
        baseW =    720 // (self.vae_scale_s * self.transformer.config.patch_size)
        baseH =    480 // (self.vae_scale_s * self.transformer.config.patch_size)
        gridcrop_coords = get_resize_crop_region_for_grid((gridH, gridW), baseW, baseH)
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(self.transformer.config.attention_head_dim, gridcrop_coords, (gridH, gridW), num_frm, use_real=True)
        return freqs_cos.to(self.device), freqs_sin.to(self.device)

    def set_steps(self, steps=None, sigmas=None, strength=1.):
        if sigmas is None:
            self.scheduler.set_timesteps(steps, device=self.device)
        else:
            assert "sigmas" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
            self.scheduler.set_timesteps(sigmas=sigmas, device=self.device)
        self.timesteps = self.scheduler.timesteps
        if sigmas is not None: steps = len(self.timesteps)
        new_steps = min(int(steps * strength), steps)
        self.timesteps = self.timesteps[(steps - new_steps) * self.scheduler.order :]
        self.lat_timestep = self.timesteps[:1].repeat(self.a.batch)
        return new_steps

    def get_t5_emb(self, prompt=None, num=1, maxlen=226):
        tokens = self.tokenizer(prompt, padding="max_length", max_length=maxlen, truncation=True, add_special_tokens=True, return_tensors="pt")
        cs = self.text_encoder(tokens.input_ids.to(self.device))[0]
        return cs.to(self.device, dtype=self.text_encoder.dtype).repeat(num, 1, 1)

    def encode_prompt(self, prompt, unprompt=None, num=1, maxlen=226):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        bs = len(prompt)
        cs = self.get_t5_emb(prompt, num=num, maxlen=maxlen)
        if self.a.cfg_scale not in [0,1]:
            if isinstance(unprompt, str): unprompt = bs * [unprompt]
            uc = self.get_t5_emb(unprompt, num=num, maxlen=maxlen)
            cs = torch.cat([uc, cs])
        return cs

    def get_vaelat(self, enc_out, sample_mode="sample"):
        if hasattr(enc_out, "latent_dist") and sample_mode == "sample":
            return enc_out.latent_dist.sample(self.g_)
        elif hasattr(enc_out, "latent_dist") and sample_mode == "argmax":
            return enc_out.latent_dist.mode()
        elif hasattr(enc_out, "latents"):
            return enc_out.latents
        else:
            raise AttributeError("Could not access latents of provided enc_out")

    def init_lat(self, src, dtype):
        lat = [self.get_vaelat(self.vae.encode(x.unsqueeze(0))) for x in src]
        lat = torch.cat(lat, dim=0).to(dtype).permute(0,2,1,3,4)
        lat = self.vae.config.scaling_factor * lat
        return lat # [b,f,c,h,w]

    def make_lat(self, bs=1, ch=16, num_frm=13, height=60, width=90, image=None, video=None, dtype=None):
        if video is not None: num_frm = video.size(2)
        num_frm = (num_frm - 1) // self.vae_scale_t + 1
        shape = (bs, num_frm, ch, height // self.vae_scale_s, width // self.vae_scale_s)

        img_lat = None
        if image is not None:
            pad_shape = (bs, num_frm - 1, ch, height // self.vae_scale_s, width // self.vae_scale_s)
            lat_pad = torch.zeros(pad_shape, device=self.device, dtype=dtype)
            img_lat = self.init_lat(image.unsqueeze(2), dtype)
            img_lat = torch.cat([img_lat, lat_pad], dim=1)

        lat = torch.randn(shape, generator=self.g_, device=self.device, dtype=dtype)
        if video is not None:
            vid_lat = self.init_lat(video, dtype)
            lat = self.scheduler.add_noise(vid_lat, lat, self.lat_timestep)
        lat *= self.scheduler.init_noise_sigma

        return lat, img_lat

    def lat_vid(self, lat):
        lat = lat.permute(0,2,1,3,4) # [b,c,f,h,w]
        video = self.vae.decode(lat / self.vae.config.scaling_factor).sample
        video = (video / 2 + 0.5).clamp(0,1)
        video = video.permute(0,2,1,3,4) # [b,f,c,h,w]
        return video


    def generate(self, cs, H, W, num_frames, image=None, video=None):
        bs = 1 if self.a.cfg_scale in [0,1] else 2
        steps = self.set_steps(self.a.steps, strength = self.a.strength)

        lat_ch = self.transformer.config.in_channels
        if image is not None: lat_ch = lat_ch // 2
        lat, img_lat = self.make_lat(self.a.num, lat_ch, num_frames, H, W, image, video, dtype=cs.dtype)
        fcount = lat.size(1)
        img_rot_emb = self.get_rot_pos_emb(H, W, self.a.ctx_frames) if self.transformer.config.use_rotary_positional_embeddings else None

        def calc_noise(x, t, conds):
            noise_pred = self.transformer(x, conds, t.expand(x.shape[0]), image_rotary_emb=img_rot_emb, return_dict=False)[0]
            if len(x) > 1 and len(conds) > 1:
                cfg_scale = 1 + self.a.cfg_scale * ((1 - math.cos(math.pi * ((steps - t.item()) / steps) ** 5.)) / 2) if self.a.dyn_cfg else self.a.cfg_scale
                noise_un, noise_c = noise_pred.chunk(bs)
                noise_pred = noise_un + cfg_scale * (noise_c - noise_un)
            return noise_pred

        pbar = progbar(len(self.timesteps))
        old_pred = None # for DPM-solver++
        for i, t in enumerate(self.timesteps):
            lat_in = self.scheduler.scale_model_input(lat, t)
            lat_in = torch.cat([lat_in] * bs) # [2,13,16,60,90]
            if image is not None:
                lat_img_in = torch.cat([img_lat] * bs)
                lat_in = torch.cat([lat_in, lat_img_in], dim=2)

            if fcount > self.a.ctx_frames: # sliding sampling for long videos
                noise_pred = torch.zeros_like(lat)
                slide_count = torch.zeros((1, fcount, 1, 1, 1), device=lat_in.device)
                for slids in uniform_slide(i, fcount, self.a.ctx_frames, loop=self.a.loop):
                    noise_pred_sub = calc_noise(lat_in[:,slids], t, cs)
                    noise_pred[:,slids] += noise_pred_sub
                    slide_count[:,slids] += 1 # increment which indices were used
                noise_pred /= slide_count
            else: # single context video
                noise_pred = calc_noise(lat_in, t, cs)

            if isinstance(self.scheduler, CogVideoXDPMScheduler):
                lat, old_pred = self.scheduler.step(noise_pred, old_pred, t, self.timesteps[i-1] if i > 0 else None, lat, **self.sched_kwargs, return_dict=False)
            else:
                lat = self.scheduler.step(noise_pred, t, lat, **self.sched_kwargs).prev_sample
            lat = lat.to(cs.dtype) # [b,f,c,h,w]
            pbar.upd()

        video = self.lat_vid(lat) # [b,f,c,h,w]
        return video

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
def uniform_slide(step, num_frames, ctx_size=16, ctx_stride=1, ctx_overlap=5, loop=True, verbose=False):
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

def save_video(video, path, fps):
    if torch.is_tensor(video):
        video = (video * 255).round().clamp(0,255).permute(0,2,3,1).to(torch.uint8).cpu().numpy() # [f,h,w,c]
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in video:
            writer.append_data(frame)

def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))
    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


@torch.no_grad()
def main():
    a = get_args(main_args())
    os.makedirs(a.out_dir, exist_ok=True)
    gendict = {}
    if a.verbose: save_cfg(a, a.out_dir)

    cogx = CogX(a)
    a = cogx.a

    if a.size is None: a.size = [720,480] # [x * cogx.vae_scale_s for x in [cogx.transformer.config.sample_width, cogx.transformer.config.sample_height]]

    W, H = [int(x) for x in a.size]
    a.in_txt = ['. '.join([a.pretxt, s, a.postxt]) for s in read_txt(a.in_txt)]

    cs = cogx.encode_prompt(a.in_txt, a.unprompt, a.num) # [2,226,4096]

    if isset(a, 'in_img') and os.path.isfile(a.in_img):
        assert os.path.isfile(a.in_img), "Not found %s" % a.in_img
        in_img = load_image(a.in_img)
        in_img = cogx.video_processor.preprocess(in_img, height=H, width=W).to(device, dtype=cs.dtype)
        gendict['image'] = in_img

    if isset(a, 'in_vid'):
        assert os.path.exists(a.in_vid), "Not found %s" % a.in_vid
        in_vid = load_video(a.in_vid)
        a.frames = len(in_vid)
        video = cogx.video_processor.preprocess_video(in_vid, height=H, width=W).to(device, dtype=cs.dtype)
        size = list(video.shape[-2:])
        gendict['video'] = video

    video = cogx.generate(cs, H, W, a.frames, **gendict) # [b,f,c,h,w]

    outname = []
    if isset(a, 'in_vid'): outname += [basename(a.in_vid)]
    if isset(a, 'in_img'): outname += [basename(a.in_img)]
    if len(a.in_txt) > 0: outname += [txt_clean(basename(a.in_txt[0] if isinstance(a.in_txt, list) else a.in_txt))]
    outname = '-'.join(outname)[:69]
    outpath = os.path.join(a.out_dir, '%s-%d.mp4' % (outname, cogx.seed))
    save_video(video[0], outpath, a.fps)


if __name__ == '__main__':
    main()
