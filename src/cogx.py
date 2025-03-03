
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
from PIL import Image
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
from diffusers.utils import load_image, load_video

from core.args import main_args
from core.text import read_txt, txt_clean
from core.utils import isset, gpu_ram, img_list, basename, progbar, save_cfg

def get_args(parser):
    parser.add_argument('-iv', '--in_vid',  default=None, help='input video or directory with images')
    parser.add_argument('-fs', '--fstep',   default=None, type=int, help="number of frames for each interpolation step (1 = no interpolation)")
    parser.add_argument('-vf', '--frames',  default=None, type=int, help="Frame count for generated video")
    parser.add_argument('-ov', '--overlap', default=None, type=int, help="Cut these frames from the input video to expand further")
    parser.add_argument('-oo', '--ctx_over', default=2, type=int, help="Overlap for sliding window denoising")
    parser.add_argument('-cf', '--ctx_frames', default=13, type=int, help="latent count to process at once with sliding window sampling")
    parser.add_argument('-dc', '--dyn_cfg', action='store_true', help='Dynamic CFG scale - good for txt2vid')
    parser.add_argument('-re', '--rot_emb', action='store_true', help='Extend rotary-position embeddings to sequence length')
    parser.add_argument('-fps', '--fps',    default=12, type=int, help="Frame rate")
    parser.add_argument('--loop',           action='store_true')
    # override
    parser.add_argument('-C','--cfg_scale', default=6, type=float, help="prompt guidance scale")
    parser.add_argument('-s',  '--steps',   default=50, type=int, help="number of diffusion steps")
    parser.add_argument('-post','--postxt', default='. hyperrealistic photography with intricate details, cinematic scene, professional lighting')
    parser.add_argument('-un','--unprompt', default='low quality, oversaturated', help='Negative prompt')
    return parser.parse_args()

is_mac = torch.backends.mps.is_available() and torch.backends.mps.is_built() # M1/M2 chip?
is_cuda = torch.cuda.is_available()
device = 'mps' if is_mac else 'cuda' if is_cuda else 'cpu'
dtype = torch.bfloat16 if is_cuda or is_mac else torch.float32

class CogXpipe(CogVideoXPipeline):
    def __init__(self, tokenizer, text_encoder, vae, transformer, scheduler):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler)

class CogX:
    def __init__(self, a):
        self.a = a
        self.device = device
        self.seed = a.seed or int((time.time()%1)*69696)

        mod_path = os.path.join(a.maindir, 'xtra/cogx', 'i2v' if isset(a, 'img2vid') else 'main')
        if not os.path.isdir(mod_path): mod_path = 'THUDM/CogVideoX-5b-I2V' if isset(a, 'img2vid') else 'THUDM/CogVideoX-5b'

        self.tokenizer = T5Tokenizer.from_pretrained(mod_path, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(mod_path, subfolder="text_encoder", torch_dtype=dtype)
        self.vae = AutoencoderKLCogVideoX.from_pretrained(mod_path, subfolder="vae", torch_dtype=dtype)
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(mod_path, subfolder="transformer", torch_dtype=dtype)
        self.scheduler = CogVideoXDPMScheduler.from_pretrained(mod_path, subfolder="scheduler") # CogVideoXDDIMScheduler CogVideoXDPMScheduler
        self.pipe = CogXpipe(self.tokenizer, self.text_encoder, self.vae, self.transformer, self.scheduler)

        if is_cuda and (a.lowmem or gpu_ram() < 20):
            try:
                from optimum.quanto import freeze, quantize, qfloat8 # qfloat8_e4m3fn, qfloat8_e5m2, qint8, qint4, qint2
                quantize(self.text_encoder, weights=qfloat8); freeze(self.text_encoder)
                quantize(self.transformer, weights=qfloat8);  freeze(self.transformer)
                quantize(self.vae, weights=qfloat8);          freeze(self.vae)
                self.pipe.to(device)
            except:
                print(' Quantization support requires optimum-quanto library with Python 3.10+ and Torch 2.4')

        if is_cuda:
            self.pipe.enable_model_cpu_offload()
            # self.pipe.enable_sequential_cpu_offload()
        if not is_mac:
            self.transformer.to(memory_format=torch.channels_last)
        self.vae.enable_slicing()
        self.vae.enable_tiling()

        self.lat_ch = self.transformer.config.in_channels # 16 or 32
        if isset(self.a, 'img2vid'): self.lat_ch = self.lat_ch // 2
        self.vae_scale_s = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_scale_t = self.vae.config.temporal_compression_ratio
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_s)
        self.g_ = torch.Generator(device).manual_seed(self.seed)

        self.sched_kwargs = {'eta': a.eta} if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()) else {}
        if "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            self.sched_kwargs['generator'] = self.g_


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
        return cs.to(self.device, dtype=self.text_encoder.dtype).expand(num, -1, -1) # mps-friendly

    def encode_prompt(self, prompt, unprompt=None, num=1, maxlen=226):
        if isinstance(prompt, str): prompt = [prompt]
        cs = self.get_t5_emb(prompt, num=num, maxlen=maxlen)
        if self.a.cfg_scale not in [0,1]:
            if isinstance(unprompt, str): unprompt = [unprompt]
            if len(unprompt) == 1: unprompt = unprompt * len(prompt)
            uc = self.get_t5_emb(unprompt, num=num, maxlen=maxlen)
            cs = torch.cat([uc, cs])
        return cs

    def lat_vid(self, lat):
        lat = lat.permute(0,2,1,3,4) # [b,c,f,h,w]
        video = self.vae.decode(lat / self.vae.config.scaling_factor).sample
        video = (video / 2 + 0.5).clamp(0,1)
        video = video.permute(0,2,1,3,4) # [b,f,c,h,w]
        return video

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
        return lat * self.vae.config.scaling_factor # [b,f,c,h,w]

    @staticmethod
    def calc_ids(length, count):
        if count == 1: return [0]
        return [int(i * (length-1) / (count-1)) for i in range(count)]

    def make_lat(self, bs=1, num_frames=49, height=60, width=90, image=None, imgpos=None, video=None, dtype=None):
        num_frm = (num_frames - 1) // self.vae_scale_t + 1
        shape = (bs, num_frm, self.lat_ch, height // self.vae_scale_s, width // self.vae_scale_s) # [1,13,16,60,90]
        img_lat = None
        vid_lat = None
        if image is not None: # [n,c,h,w]
            if imgpos is None: imgpos = self.calc_ids(num_frm, len(image))
            img_lat = torch.zeros(shape, device=self.device, dtype=dtype) # [1,13,16,60,90]
            in_lats = self.init_lat(image.unsqueeze(2), dtype) # [n,1,16,60,90]
            for i, n in enumerate(imgpos):
                img_lat[:,n] = in_lats[min(i, len(in_lats)-1)]
        elif video is not None and num_frames > video.size(2): # video expansion
            vid_lat = self.init_lat(video, dtype) # [b,f,16,60,90]  [b,f,c,h,w]
            pad_shape = (bs, num_frm - vid_lat.shape[1], *shape[2:])
            pad_lat = torch.zeros(pad_shape, device=self.device, dtype=dtype)
            img_lat = torch.cat([vid_lat, pad_lat], dim=1) # [b,F,c,h,w]
        lat = torch.randn(shape, generator=self.g_, device=self.device, dtype=dtype)
        if video is not None: # [b,c,f,h,w]
            if vid_lat is None: vid_lat = self.init_lat(video, dtype) # [b,f,16,60,90]  [b,f,c,h,w]
            vid_frm = vid_lat.shape[1]
            lat[:,:vid_frm] = self.scheduler.add_noise(vid_lat, lat[:,:vid_frm], self.lat_timestep)
        lat *= self.scheduler.init_noise_sigma
        return lat, img_lat

    def get_rot_pos_emb(self, H, W, num_frm=13, time_ids=None):
        gridH, gridW, baseH, baseW = [x // (self.vae_scale_s * self.transformer.config.patch_size) for x in [H, W, 480, 720]]
        gridcrop_coords = get_resize_crop_region_for_grid(gridH, gridW, baseH, baseW)
        if time_ids is None: time_ids = torch.arange(num_frm, dtype=torch.float32)
        freqs_cos, freqs_sin = get_3d_rot_pos_emb(self.transformer.config.attention_head_dim, gridcrop_coords, gridH, gridW, time_ids)
        return freqs_cos.to(self.device), freqs_sin.to(self.device)


    def generate(self, cs, H, W, num_frames, image=None, imgpos=None, video=None):
        bs = 1 if self.a.cfg_scale in [0,1] else 2
        steps = self.set_steps(self.a.steps, strength = self.a.strength)

        lat, img_lat = self.make_lat(self.a.num, num_frames, H, W, image, imgpos, video, dtype=cs.dtype)
        fcount = lat.size(1)
        img_rot_emb = self.get_rot_pos_emb(H, W, self.a.ctx_frames) if self.transformer.config.use_rotary_positional_embeddings else None

        def calc_noise(x, t, conds, img_rot_emb):
            noise_pred = self.transformer(x, conds, t.expand(x.shape[0]), image_rotary_emb=img_rot_emb).sample
            if len(x) > 1 and len(conds) > 1: # cfguidance
                cfg_scale = 1 + self.a.cfg_scale * ((1 - math.cos(math.pi * ((steps - t.item()) / steps) ** 5.)) / 2) if self.a.dyn_cfg else self.a.cfg_scale
                noise_un, noise_c = noise_pred.chunk(bs)
                noise_pred = noise_un + cfg_scale * (noise_c - noise_un)
            return noise_pred

        pbar = progbar(len(self.timesteps))
        old_pred = None # for DPM-solver++
        for i, t in enumerate(self.timesteps):
            lat_in = self.scheduler.scale_model_input(lat, t)
            lat_in = torch.cat([lat_in] * bs) # [2,L,c,h,w]
            if isset(self.a, 'img2vid'):
                lat_img_in = torch.cat([img_lat] * bs)
                lat_in = torch.cat([lat_in, lat_img_in], dim=2)

            if fcount > self.a.ctx_frames: # sliding sampling for long videos
                noise_pred = torch.zeros_like(lat)
                slide_count = torch.zeros((1, fcount, 1, 1, 1), device=lat_in.device)
                for slids in uniform_slide(i, fcount, self.a.ctx_frames, self.a.ctx_over, loop=self.a.loop):
                    if self.a.rot_emb and img_rot_emb is not None: img_rot_emb = self.get_rot_pos_emb(H, W, time_ids = slids)
                    noise_pred_sub = calc_noise(lat_in[:,slids], t, cs, img_rot_emb)
                    noise_pred[:,slids] += noise_pred_sub[:,:len(slids)]
                    slide_count[:,slids] += 1 # increment which indices were used
                noise_pred /= slide_count
            else: # single context video
                noise_pred = calc_noise(lat_in, t, cs, img_rot_emb)

            if isinstance(self.scheduler, CogVideoXDPMScheduler):
                lat, old_pred = self.scheduler.step(noise_pred, old_pred, t, self.timesteps[i-1] if i > 0 else None, lat, **self.sched_kwargs, return_dict=False)
            else:
                lat = self.scheduler.step(noise_pred, t, lat, **self.sched_kwargs).prev_sample
            lat = lat.to(cs.dtype) # [b,f,c,h,w]
            pbar.upd()

        video = self.lat_vid(lat) # [b,f,c,h,w]
        return video

# slightly edited diffusers.models.embeddings.get_3d_rotary_pos_embed
def get_3d_rot_pos_emb(embed_dim, crops_coords, h, w, time_ids, theta=10000, use_real=True, rep_inter_real=True): # RoPE for video tokens with 3D structure.
    # embed_dim = hidden_size_head, theta = Scaling factor for frequency computation
    if not isinstance(time_ids, torch.Tensor): time_ids = torch.tensor(time_ids).float()

    (h0, w0), (h1, w1) = crops_coords # top-left and bottom-right coordinates of the crop
    steps_h, steps_w, steps_t = h1-h0, w1-w0, len(time_ids) # (h1-h0)//h, (w1-w0)//w, ..

    def make_freq(grid, dim):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float()[: (dim//2)] / dim))
        freqs = torch.outer(grid, freqs) # [S, D/2]
        if use_real and rep_inter_real: # flux, hunyuan-dit, cogvideox
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1) # [S,D]
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1) # [S,D]
            return freqs_cos, freqs_sin
        elif use_real: # stable audio
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1) # [S,D]
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1) # [S,D]
            return freqs_cos, freqs_sin
        else: # lumina
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # [S, D/2] not float?
            return freqs_cis

    # BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d tensor
    def combine_thw(freqs_t, freqs_h, freqs_w):
        freqs_t = freqs_t[:, None, None, :].expand(-1, steps_h, steps_w, -1)  # steps_t, steps_h, steps_w, dim_t
        freqs_h = freqs_h[None, :, None, :].expand(steps_t, -1, steps_w, -1)  # steps_t, steps_h, steps_w, dim_h
        freqs_w = freqs_w[None, None, :, :].expand(steps_t, steps_h, -1, -1)  # steps_t, steps_h, steps_w, dim_w
        freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)  # steps_t, steps_h, steps_w, (dim_t + dim_h + dim_w)
        freqs = freqs.view(steps_t * steps_h * steps_w, -1)  # (steps_t * steps_h * steps_w), (dim_t + dim_h + dim_w)
        return freqs

    grid_t = time_ids # [0..12]
    grid_h = torch.arange(h0, h1, dtype=torch.float32) # step = steps_h
    grid_w = torch.arange(w0, w1, dtype=torch.float32) # step = steps_w
    t_cos, t_sin = make_freq(grid_t, embed_dim // 4)     # freqs_t  [steps_t, dim_t] [13,16]
    h_cos, h_sin = make_freq(grid_h, embed_dim // 8 * 3) # freqs_h  [steps_h, dim_h] [30,24]
    w_cos, w_sin = make_freq(grid_w, embed_dim // 8 * 3) # freqs_w  [steps_w, dim_w] [45,24]
    cos = combine_thw(t_cos, h_cos, w_cos)
    sin = combine_thw(t_sin, h_sin, w_sin)
    return cos, sin

def get_resize_crop_region_for_grid(h, w, th, tw):
    if h/w > th/tw:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))
    crop_top = int(round((th - resize_height) / 2.))
    crop_left = int(round((tw - resize_width) / 2.))
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

# sliding sampling for long videos
# from https://github.com/ArtVentureX/comfyui-animatediff/blob/main/animatediff/sliding_schedule.py
def uniform_slide(step, num_frames, size=13, overlap=0, loop=True): # stride = 1
    pad = int(round(num_frames * ordered_halving(step)))
    fstop = num_frames + pad + (0 if loop else -overlap)
    fstart = int(ordered_halving(step)) + pad
    fstep = size - overlap
    for j in range(fstart, fstop, fstep):
        yield [e % num_frames for e in range(j, j + size)]
def ordered_halving(val): # Returns fraction that has denominator that is a power of 2
    bin_str = f"{val:064b}" # get binary value, padded with 0s for 64 bits
    bin_flip = bin_str[::-1] # flip binary value, padding included
    as_int = int(bin_flip, 2) # convert binary to int
    final = as_int / (1 << 64) # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616, or 1 with 64 zero's
    return final

def save_video(video, path, fps=12, ext='jpg'):
    if torch.is_tensor(video):
        video = (video * 255).round().clamp(0,255).permute(0,2,3,1).to(torch.uint8).cpu().numpy() # [f,h,w,c]
    with imageio.get_writer(path + '.mp4', fps=fps) as writer:
        for frame in video:
            writer.append_data(frame)
    os.makedirs(path, exist_ok=True)
    for i, frame in enumerate(video):
        Image.fromarray(frame).save(os.path.join(path, '%05d.%s' % (i, ext)), quality=95, subsampling=0)


@torch.no_grad()
def main():
    a = get_args(main_args())
    os.makedirs(a.out_dir, exist_ok=True)
    gendict = {}

    if isset(a, 'in_img'):
        assert os.path.exists(a.in_img), "Not found %s" % a.in_img
        img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
        in_img = [load_image(f) for f in img_paths]
        if isset(a, 'fstep') and not isset(a, 'frames'): a.frames = a.fstep * len(in_img)
        a.img2vid = True

    elif isset(a, 'in_vid'):
        assert os.path.exists(a.in_vid), "Not found %s" % a.in_vid
        in_vid = load_video(a.in_vid) # list of pil
        if isset(a, 'overlap'):
            start_vid = in_vid[:-a.overlap]
            in_vid = in_vid[-a.overlap:]
            assert isset(a, 'frames') and a.frames > len(in_vid), '!! Set video length higher than overlap %d' % len(in_vid)
        elif isset(a, 'frames') and a.frames < len(in_vid):
            in_vid = in_vid[:a.frames]
        if not isset(a, 'frames'):
            a.frames = len(in_vid)
        if a.frames > len(in_vid): a.img2vid = True

    cogx = CogX(a)
    a = cogx.a

    W, H = [720,480] if a.size is None else [int(s) for s in a.size.split('-')]
    # [x * cogx.vae_scale_s for x in [cogx.transformer.config.sample_width, cogx.transformer.config.sample_height]]
    if not isset(a, 'frames'): a.frames = 49
    if a.verbose: save_cfg(a, a.out_dir)

    a.in_txt = ['. '.join([a.pretxt, s, a.postxt]) for s in read_txt(a.in_txt)]
    cs = cogx.encode_prompt(a.in_txt, a.unprompt, a.num) # [2,226,4096]

    if isset(a, 'in_img'):
        in_img = cogx.video_processor.preprocess(in_img, height=H, width=W).to(device, dtype=cs.dtype) # [n,c,h,w]
        gendict['image'] = in_img

    if isset(a, 'in_vid'):
        video = cogx.video_processor.preprocess_video(in_vid, height=H, width=W).to(device, dtype=cs.dtype) # [b,c,f,h,w]
        if isset(a, 'overlap'):
            start_vid = cogx.video_processor.preprocess_video(start_vid, height=H, width=W).to(device, dtype=cs.dtype) / 2 + 0.5
        gendict['video'] = video

    video = cogx.generate(cs, H, W, a.frames, **gendict)[0] # [f,c,h,w]

    if isset(a, 'in_vid') and isset(a, 'overlap'):
        video = torch.cat([start_vid.permute(0,2,1,3,4)[0], video]) # [f,c,h,w]

    outname = []
    if isset(a, 'in_vid'): outname += [basename(a.in_vid)]
    if isset(a, 'in_img'): outname += [basename(a.in_img)]
    if len(a.in_txt) > 0: outname += [txt_clean(basename(a.in_txt[0] if isinstance(a.in_txt, list) else a.in_txt))]
    outname = '-'.join(outname)[:69]
    outname = os.path.join(a.out_dir, '%s-%d' % (outname, cogx.seed))
    save_video(video, outname)


if __name__ == '__main__':
    main()
