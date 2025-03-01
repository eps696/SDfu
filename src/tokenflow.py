# edited from https://github.com/omerbt/TokenFlow

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import write_video

from diffusers import DDIMScheduler

from core.sdsetup import SDfu
from core.args import main_args
from core.utils import load_img, isset, img_list, file_list, save_cfg, basename, clean_vram, progbar
from core.tokenflow_utils import reg_var, get_var, reg_time, reg_extended_attention_pnp, reg_conv_injection, set_tokenflow, reg_extended_attention

is_mac = torch.backends.mps.is_available() and torch.backends.mps.is_built() # M1/M2 chip?
is_cuda = torch.cuda.is_available()
device = 'mps' if is_mac else 'cuda' if is_cuda else 'cpu'
dtype = torch.float16 if is_cuda or is_mac else torch.float32

def get_args(parser):
    parser.add_argument(       '--edit_type', default='pnp', choices=['pnp', 'sde'])
    parser.add_argument('-st', '--src_txt', default='')
    parser.add_argument('-max','--max_len', default=300, type=int)
    parser.add_argument('-bs', '--batch_size', default=None, type=int, help="Batch size (different from main SDfu!)")
    parser.add_argument('-bp', '--batch_pivot', action='store_true', help='Do pivots in batches? Recommended to fit longer sequences')
    parser.add_argument(       '--cpu', action='store_true', help='Unload registers on CPU? Recommended to fit bigger/longer sequences')
    parser.add_argument(       '--recon',   action='store_true', help='Reconstruct inverted latents to images')
    # pnp
    parser.add_argument('--pnp_attn_t',     default=0.5, type=float, help='injection thresholds [0, 1]')
    parser.add_argument('--pnp_f_t',        default=0.8, type=float, help='injection thresholds [0, 1]')
    # sdedit
    parser.add_argument('--start',          default=0.7, type=float, help='start sampling from t = start * 1000')
    # override standard args
    parser.add_argument('-im', '--in_img',  default='_in', help='input image or directory with images (overrides width and height)')
    parser.add_argument('-sm', '--sampler', default='ddim', choices=['ddim'], help="Using only DDIM for inversion")
    parser.add_argument('-C','--cfg_scale', default=13, type=float, help="prompt guidance scale")
    return parser.parse_args()


def get_timesteps(scheduler, steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(int(steps * strength), steps)
    t_start = max(steps - init_timestep, 0)
    return scheduler.timesteps[t_start:]

def ddim_inversion(sd, x, cond, batch, cnet_conds=None, ts_to_save=None, save_dir=None):
    if save_dir is not None: os.makedirs(save_dir, exist_ok=True)
    timesteps = reversed(sd.scheduler.timesteps)
    if ts_to_save is None: ts_to_save = timesteps
    pbar_t = progbar(len(timesteps))
    for i, t in enumerate(timesteps):
        alpha_prod_t = sd.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (sd.scheduler.alphas_cumprod[timesteps[i-1]] if i > 0 else sd.scheduler.final_alpha_cumprod)
        mu      = alpha_prod_t      ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma      = (1 - alpha_prod_t)      ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        pbar_b = progbar(int(math.ceil(len(x) / batch)))
        for b in range(0, len(x), batch):
            x_batch = x[b : b+batch]
            cond_batch = cond.expand(len(x_batch), -1, -1)
            if sd.use_cnet:
                eps = cnet_pred(sd.unet, sd.cnet, x_batch, t, cond_batch, torch.cat([cnet_conds[b : b+batch]]))
            else:
                eps = sd.unet(x_batch, t, cond_batch).sample
            pred_x0 = (x_batch - sigma_prev * eps) / mu_prev
            x[b : b+batch] = mu * pred_x0 + sigma * eps
            pbar_b.upd()

        if save_dir is not None and t in ts_to_save:
            torch.save(x, os.path.join(save_dir, 'noisy_lats-%04d.pt' % t))
        pbar_t.upd(uprows=2)

    torch.save(x, os.path.join(save_dir, 'noisy_lats-%04d.pt' % t))
    return x

def ddim_sample(sd, x, cond, batch, cnet_conds=None):
    timesteps = sd.scheduler.timesteps
    pbar_t = progbar(len(timesteps))
    for i, t in enumerate(timesteps):
        alpha_prod_t = sd.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (sd.scheduler.alphas_cumprod[timesteps[i+1]] if i < len(timesteps)-1 else sd.scheduler.final_alpha_cumprod)
        mu      = alpha_prod_t      ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma      = (1 - alpha_prod_t)      ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        pbar_b = progbar(int(math.ceil(len(x) / batch)))
        for b in range(0, len(x), batch):
            x_batch = x[b : b+batch]
            cond_batch = cond.expand(len(x_batch), -1, -1)
            if sd.use_cnet:
                eps = cnet_pred(sd.unet, sd.cnet, x_batch, t, cond_batch, torch.cat([cnet_conds[b : b+batch]]))
            else:
                eps = sd.unet(x_batch, t, cond_batch).sample
            pred_x0 = (x_batch - sigma * eps) / mu
            x[b : b+batch] = mu_prev * pred_x0 + sigma_prev * eps
            pbar_b.upd()
        pbar_t.upd(uprows=2)
    return x

def cnet_pred(unet, cnet, lat_in, t, cond, cnet_cond):
    ctl_downs, ctl_mid = cnet(lat_in, t, cond, cnet_cond, conditioning_scale=1, return_dict=False)
    ukwargs = {'down_block_additional_residuals': ctl_downs, 'mid_block_additional_residual': ctl_mid}
    noise_pred = unet(lat_in, t, cond, **ukwargs, return_dict=False)[0]
    return noise_pred
    
def get_conds(sd, prompt, unprompt=None, batch=1):
    tokens = sd.tokenizer(prompt, padding='max_length', max_length=sd.tokenizer.model_max_length, truncation=True, return_tensors='pt')
    conds = sd.text_encoder(tokens.input_ids.to(device))[0]
    if unprompt is None: return torch.cat([conds] * batch)
    untokens = sd.tokenizer(unprompt, padding='max_length', max_length=sd.tokenizer.model_max_length, return_tensors='pt')
    unconds = sd.text_encoder(untokens.input_ids.to(device))[0]
    conds = torch.cat([unconds] * batch + [conds] * batch)
    return conds


class TokenFlow(nn.Module):
    def __init__(self, sd, prompt, src_prompt, unprompt, cnet_conds, lat_dir, cfg_scale, batch_size):
        super().__init__()

        self.bs = batch_size
        self.lat_dir = lat_dir
        self.cfg_scale = cfg_scale
        self.cnet_conds = cnet_conds

        self.vae = sd.vae
        self.unet = sd.unet
        self.scheduler = sd.scheduler
        self.use_cnet = sd.use_cnet
        if sd.use_cnet:
            self.cnet = sd.cnet

        self.conds = get_conds(sd, prompt, unprompt)
        self.pnp_cond = get_conds(sd, src_prompt)
    
    def init_sde(self):
        reg_extended_attention(self)
        set_tokenflow(self.unet)
        
    def init_pnp(self, conv_inject_t, qk_inject_t):
        self.qk_inject_timesteps = self.scheduler.timesteps[:qk_inject_t] if qk_inject_t >= 0 else []
        self.conv_inject_timesteps = self.scheduler.timesteps[:conv_inject_t] if conv_inject_t >= 0 else []
        reg_extended_attention_pnp(self, self.qk_inject_timesteps)
        reg_conv_injection(self, self.conv_inject_timesteps)
        set_tokenflow(self.unet)

    def denoise_step(self, x, t, saved_lats, cnet_conds):
        reg_time(self, t.item()) # register the time step and features in pnp injection modules
        lat_in = torch.cat([saved_lats] + [x] * 2) # [3*f,4,h,w]
        conds = torch.cat([self.pnp_cond.repeat(len(x), 1, 1), torch.repeat_interleave(self.conds, len(x), dim=0)])
        if self.use_cnet:
            noise_pred = cnet_pred(self.unet, self.cnet, lat_in, t, conds, cnet_conds.repeat(3, 1, 1, 1))
        else:
            noise_pred = self.unet(lat_in, t, conds).sample
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_cond - noise_pred_uncond)
        denoised_lat = self.scheduler.step(noise_pred, t, x).prev_sample
        return denoised_lat

    def sample_loop(self, lats, batch_pivot=False, oncpu=False):
        mx = len(lats)
        pnum = int(math.ceil(mx / self.bs)) # count of pivots
        bnum = int(math.ceil(pnum / self.bs)) # count of pivotal batches
        print('frames %d ' % mx + 'batch %d, ' % self.bs + 'pivots %d ' % pnum + ('pivot steps %d' % bnum) if batch_pivot else '')

        pbar_t = progbar(len(self.scheduler.timesteps))
        for t in self.scheduler.timesteps:
            saved_lats = torch.load(os.path.join(self.lat_dir, 'noisy_lats-%04d.pt' % t), weights_only=True) # [f,4,h,w]

            pivot_ids = torch.randint(self.bs, (pnum,)) + torch.arange(0, mx, self.bs)
            dvx = (mx // self.bs) * self.bs
            pivot_ids[-1] = min(pivot_ids[-1], int((pivot_ids[-1] - dvx) * (mx - dvx) / self.bs) + dvx) # for non divisible frame counts

            # make pivots
            reg_var(self, 'pivotal_pass', True)
            reg_var(self, 'batch_pivots', batch_pivot)
            if batch_pivot:
                reg_var(self, 'pivot_hidden_states', [])
                reg_var(self, 'attn_outs', [])
                pbar_p = progbar(bnum)
                pivot_hs_list = []
                attn_list = []
                for i, b in enumerate(range(0, len(pivot_ids), self.bs)):
                    ids = [i for i in list(range(b, b+self.bs)) if i < len(pivot_ids)]
                    # populating pivot_hidden_states & attn_outs for further generations
                    self.denoise_step(lats[pivot_ids[ids]], t, saved_lats[pivot_ids[ids]], self.cnet_conds[pivot_ids[ids]])
                    if oncpu:
                        pivot_hs_list += [x.cpu() for x in get_var(self, 'pivot_hidden_states')]
                        attn_list += [x.cpu() for x in get_var(self, 'attn_outs')]
                    else:
                        pivot_hs_list += get_var(self, 'pivot_hidden_states')
                        attn_list += get_var(self, 'attn_outs')
                    reg_var(self, 'pivot_hidden_states', [])
                    reg_var(self, 'attn_outs', [])
                    pbar_p.upd()
                # collect layers batch [16 tensors every step]
                lnum = len(pivot_hs_list) // bnum # count of layers for one pass = 16
                pivot_hs = [torch.cat([pivot_hs_list[b * lnum + l] for b in range(bnum)], dim=1) for l in range(lnum)]
                attns = [torch.cat([attn_list[b * lnum + l] for b in range(bnum)], dim=1) for l in range(lnum)]
                del pivot_hs_list, attn_list; clean_vram()
                if oncpu:
                    reg_var(self, 'pivot_hidden_states', [x.to(device) for x in pivot_hs])
                    reg_var(self, 'attn_outs', [x.to(device) for x in attns])
                else:
                    reg_var(self, 'pivot_hidden_states', pivot_hs)
                    reg_var(self, 'attn_outs', attns)
                reg_var(self, 'layer_idx', [0])
                del pivot_hs, attns; clean_vram()
            else:
                self.denoise_step(lats[pivot_ids], t, saved_lats[pivot_ids], self.cnet_conds[pivot_ids]) # NO BATCH
            reg_var(self, 'pivotal_pass', False)

            # make images
            pbar_b = progbar(int(math.ceil(mx / self.bs)))
            for i, b in enumerate(range(0, mx, self.bs)):
                reg_var(self, 'batch_idx', i)
                ids = [i for i in list(range(b, b+self.bs)) if i < mx]
                lats[ids] = self.denoise_step(lats[ids], t, saved_lats[ids], self.cnet_conds[ids])
                pbar_b.upd()

            pbar_t.upd(uprows = 4 if batch_pivot else 2)

        reg_var(self, 'pivot_hidden_states', None) # clean vram
        reg_var(self, 'attn_outs', None) # clean vram
        return lats

def seed_everything(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

def decode_images(sd, latents, save_dir, batch=8):
    os.makedirs(save_dir, exist_ok=True)
    pbar = progbar(int(math.ceil(len(latents) / batch)))
    for b in range(0, len(latents), batch):
        imgs = sd.vae.decode(latents[b : b+batch] / 0.18215).sample
        imgs = (imgs / 2. + .5).clamp(0,1)
        for j in range(len(imgs)):
            T.ToPILImage()(imgs[j]).save(f'{save_dir}/%05d.jpg' % (b + j))
        pbar.upd(str(b))

def save_video(dir, fps=10):
    frames = np.concatenate([np.array(load_img(path, tensor=False)[0])[None] for path in img_list(dir)])
    write_video(dir + '.mp4', frames, fps=fps, video_codec = "libx264", options = {"crf": "18", "preset": "slow"})


def invert(sd, lats, cnet_conds, a):
    toy_scheduler = DDIMScheduler.from_pretrained(os.path.join(a.maindir, 'v1/scheduler_config.json'))
    toy_scheduler.set_timesteps(a.steps)
    ts_to_save = get_timesteps(toy_scheduler, a.steps, strength=1.)

    sd.scheduler.set_timesteps(a.lat_steps)
    conds = get_conds(sd, a.src_txt, a.unprompt)[1].unsqueeze(0)
    print(' ddim inversion..')
    lats_inv = ddim_inversion(sd, lats, conds, a.batch_size, cnet_conds, ts_to_save, a.lat_dir)

    if a.recon:
        print(' reconstruction..')
        recon_lats = ddim_sample(sd, lats_inv, conds, a.batch_size, cnet_conds)
        print(' decoding..')
        decode_images(sd, recon_lats, os.path.join(a.out_dir, 'recon'))

def edit(sd, lats, cnet_conds, a):
    # get noise
    last_lat_path = file_list(a.lat_dir, ext='pt')[-1]
    last_lat_step = int(basename(last_lat_path).split('-')[-1])
    noisy_lats = torch.load(last_lat_path, weights_only=True)[range(len(lats))].to(device)
    alpha_prod_T = sd.scheduler.alphas_cumprod[last_lat_step]
    mu_T, sigma_T = alpha_prod_T ** 0.5, (1 - alpha_prod_T) ** 0.5
    ddim_eps = (noisy_lats - mu_T * lats) / sigma_T
    ddim_eps = ddim_eps.to(torch.float16).to(device)

    print(' editing..')
    editor = TokenFlow(sd, a.in_txt, a.src_txt, a.unprompt, cnet_conds, a.lat_dir, a.cfg_scale, a.batch_size)
    if a.edit_type == 'pnp':
        editor.init_pnp(conv_inject_t = int(a.steps * a.pnp_f_t), qk_inject_t = int(a.steps * a.pnp_attn_t))
    if a.edit_type == 'sde':
        sd.scheduler.timesteps = sd.scheduler.timesteps[int(1 - a.start * len(sd.scheduler.timesteps)):]
        editor.init_sde()
    noisy_lats = sd.scheduler.add_noise(lats, ddim_eps, sd.scheduler.timesteps[0])
    edited_lats = editor.sample_loop(noisy_lats, a.batch_pivot, a.cpu)
    print(' decoding..')
    clean_vram()
    decode_images(sd, edited_lats, os.path.join(a.out_dir, 'out'))

@torch.no_grad()
def main():
    a = get_args(main_args())
    assert os.path.isdir(a.in_img), "!! Images %s not found !!" % a.in_img
    if not isset(a, 'batch_size'): a.batch_size = len(img_list(a.in_img))

    sd = SDfu(a)
    a = sd.a

    a.lat_dir = os.path.join(a.out_dir, 'lats')
    a.lat_steps = a.steps # or can be 10 x steps
    os.makedirs(a.out_dir, exist_ok=True)
    save_cfg(a, a.out_dir)
    seed_everything(a.seed)

    count = int(math.ceil(len(img_list(a.in_img)) / a.max_len))
    print(count, 'chunks')
    for i in range(count):
        frames = [load_img(path)[0] for path in img_list(a.in_img)[i*a.max_len : (i+1)*a.max_len]]

        print(' encoding lats..')
        lats = torch.cat([sd.img_lat(x, deterministic=True) for x in frames])
        print(' encoded size', np.prod(list(lats.shape)), lats.shape)
        cnet_conds = (torch.cat([load_img(path)[0] for path in img_list(a.control_img)[i*a.max_len : (i+1)*a.max_len]]) + 1.) / 2. \
                      if sd.use_cnet else torch.zeros([len(frames)])

        if not os.path.exists(a.lat_dir) or len(file_list(a.lat_dir, ext='pt'))==0:
            invert(sd, lats, cnet_conds, a)
        if isset(a, 'in_txt'):
            edit(sd, lats, cnet_conds, a)

        try:
            save_video(os.path.join(a.out_dir, 'out'), fps=8)
        except:
            print("Cannot export video (pyav not installed?), exiting")


if __name__ == '__main__':
    main()
