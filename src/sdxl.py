
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import time
import numpy as np

import torch

from core.args import main_args, unprompt
from core.text import read_txt, txt_clean
from core.utils import load_img, lerp, slerp, makemask, blend, cvshow, calc_size, isok, isset, img_list, basename, progbar, save_cfg

device = torch.device('cuda')

def get_args(parser):
    parser.add_argument('-mdir', '--models_dir', default=None)
    parser.add_argument('-m',  '--model',   default='base', choices=['base', 'refiner'], help='base or refiner')
    parser.add_argument('-fs', '--fstep',   default=1, type=int, help="number of frames for each interpolation step (1 = no interpolation)")
    parser.add_argument(       '--curve',   default='linear', help="Interpolating curve: bezier, parametric, custom or linear")
    # UNUSED
    parser.add_argument('-sm', '--sampler', default=None, help='UNUSED')
    parser.add_argument(       '--vae',     default=None, help='UNUSED')
    parser.add_argument('-cg', '--cguide',  action=None, help='UNUSED')
    parser.add_argument('-lo', '--lowmem',  action=None, help='UNUSED')
    parser.add_argument('--precision',      default=None, help='UNUSED')
    parser.add_argument('-rt', '--load_token', default=None, help='UNUSED')
    parser.add_argument('-rd', '--load_custom', default=None, help='UNUSED')
    parser.add_argument('-rl', '--load_lora', default=None, help='UNUSED')
    return parser.parse_args()

def read_multitext(in_txt, prefix=None, postfix=None):
    if in_txt is None or len(in_txt)==0: return []
    lines = [tt.strip() for tt in read_txt(in_txt) if tt.strip()[0] != '#']
    prompts = [tt.split('|')[:2] for tt in lines] # 2 text encoders = 2 subprompts
    texts   = [txt_clean(tt) for tt in lines]
    return prompts, texts

@torch.no_grad()
def main():
    a = get_args(main_args())
    a.seed = a.seed or int((time.time()%1)*69696)
    g_ = torch.Generator("cuda").manual_seed(a.seed)
    do_cfg = a.cfg_scale not in [0,1]
    gendict = {}

    def get_model(name, dir=a.models_dir, pre='stabilityai/'):
        return pre + name if dir is None or not os.path.exists(dir) else os.path.join(dir, name)

    size = None if not isset(a, 'size') else calc_size(a.size, None)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.verbose: save_cfg(a, a.out_dir)
    if a.verbose: print(' sd xl ..', a.cfg_scale, '..', a.strength, '..', a.seed)
    
    if a.in_img is not None and a.fstep > 1: print('Interpolation with images not supported yet'); exit()

    pipe = None
    a.model = 'stable-diffusion-xl-%s-1.0' % a.model
    a.control_mod = 'controlnet-%s-sdxl-1.0-mid' % a.control_mod
    if isset(a, 'in_img'):
        if isset(a, 'mask'): # inpaint
            from diffusers import StableDiffusionXLInpaintPipeline as PP
        elif isset(a, 'control_img'): # img2img + controlnet
            from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline as PP
            cnet = ControlNetModel.from_pretrained(get_model(a.control_mod, a.models_dir, pre='diffusers/'), torch_dtype=torch.float16, variant="fp16")
            pipe = PP.from_pretrained(get_model(a.model, a.models_dir), controlnet=cnet, torch_dtype=torch.float16, variant="fp16").to("cuda")
        else: # img2img
            from diffusers import StableDiffusionXLImg2ImgPipeline as PP
    else:
        if isset(a, 'control_img'): # controlnet
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline as PP
            cnet = ControlNetModel.from_pretrained(get_model(a.control_mod, a.models_dir, pre='diffusers/'), torch_dtype=torch.float16, variant="fp16")
            pipe = PP.from_pretrained(get_model(a.model, a.models_dir), controlnet=cnet, torch_dtype=torch.float16, variant="fp16").to("cuda")
        else: # txt2img
            from diffusers import StableDiffusionXLPipeline as PP
    if pipe is None:
        pipe = PP.from_pretrained(get_model(a.model, a.models_dir), torch_dtype=torch.float16, variant="fp16").to("cuda")

    un = '' if a.unprompt=='no' else unprompt if a.unprompt is None else ', '.join([unprompt, a.unprompt])
    prompts, texts = read_multitext(a.in_txt, a.pretxt, a.postxt)
    count = len(prompts)

    if isset(a, 'in_img'):
        assert os.path.exists(a.in_img), "!! Image(s) %s not found !!" % a.in_img
        img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
        count = max(count, len(img_paths))
    if isset(a, 'mask'):
        masks = img_list(a.mask) if os.path.isdir(a.mask) else read_txt(a.mask)
        clipseg_path = os.path.join(a.maindir, 'clipseg/rd64-uni.pth')
    if isset(a, 'control_img'):
        assert os.path.exists(a.control_img), "!! ControlNet image(s) %s not found !!" % a.control_img
        cimg_paths = img_list(a.control_img) if os.path.isdir(a.control_img) else [a.control_img]
        count = max(count, len(cimg_paths))

    cs, pool_cs = [], []
    for prompt_ in prompts:
        p1, p2 = prompt_ if len(prompt_)==2 else prompt_ * 2
        c_, uc, pool_c, pool_uc = pipe.encode_prompt(p1, p2, device, a.num, do_cfg, un, un)
        cs.append(c_)
        pool_cs.append(pool_c)
    cs = torch.stack(cs)
    pool_cs = torch.stack(pool_cs)
    
    W, H = (1024,1024) if size is None else size
    if a.in_img is None:
        zs = torch.stack([torch.randn([a.batch * a.num, 4, H//8, W//8], generator=g_, device=device, dtype=torch.float16) for i in range(count)])

    def genmix(i=0, tt=0, prompt=None, img=None, cimg=None, mask=None):
        _ = None
        
        if img is None:
            z_ = slerp(zs[i % len(zs)], zs[(i+1) % len(zs)], tt)
        if prompt is None:
            c_ =  lerp(cs[i % len(cs)], cs[(i+1) % len(cs)], tt)
            pool_c = lerp(pool_cs[i % len(pool_cs)], pool_cs[(i+1) % len(pool_cs)], tt)
        
        if img is None:
            if cimg is not None: # txt2img + controlnet
                image = pipe(_, _, cimg, _, _, a.steps, a.cfg_scale, _, _, a.num, 0, g_, z_, c_, uc, pool_c, pool_uc, controlnet_conditioning_scale=a.control_scale).images[0]
            else: # txt2img
                image = pipe(_, _, _, _, a.steps, _, a.cfg_scale, _, _, a.num, 0, g_, z_, c_, uc, pool_c, pool_uc).images[0]
        else:
            W, H = img.size
            if mask is not None: # inpaint
                image = pipe(_, _, img, mask, _, H, W, a.strength, a.steps, _, _, a.cfg_scale, _, _, a.num, 0, g_, _, c_, uc, pool_c, pool_uc).images[0]
            elif cimg is not None: # img2img + controlnet
                image = pipe(_, _, img, cimg, H, W, a.strength, a.steps, a.cfg_scale, _, _, a.num, 0, g_, _, c_, uc, pool_c, pool_uc, controlnet_conditioning_scale=a.control_scale).images[0]
            else: # img2img
                image = pipe(_, _, img, a.strength, a.steps, _, _, a.cfg_scale, _, _, a.num, 0, g_, _, c_, uc, pool_c, pool_uc).images[0]
        return image
    
    pbar = progbar(count * a.fstep)
    for i in range(count):
        log = texts[i % len(texts)] if len(texts) > 0 else ''

        if isset(a, 'in_img'): # img2img
            img_path = img_paths[i % len(img_paths)]
            file_out = basename(img_path) if len(img_paths) == count else '%06d' % i
            log += ' .. %s' % os.path.basename(img_path)
            if len(img_paths) > 1 or i==0:
                init_img, (W,H) = load_img(img_path, size, tensor=False)
                if isset(a, 'mask'): # inpaint
                    log += ' / %s' % os.path.basename(masks[i % len(masks)])
                    gendict['mask'] = makemask(masks[i % len(masks)], init_img, a.invert_mask, tensor=False, model_path=clipseg_path)
                gendict['img'] = init_img

        else: # txt2img
            file_out = '%03d-%s' % (i, log)

        if isset(a, 'control_img'):
            gendict['cimg'], _ = load_img(cimg_paths[i % len(cimg_paths)], (W,H), tensor=False)

        if a.fstep <= 1: # single
            image = genmix(i, **gendict)
            if a.verbose: cvshow(np.array(image))
            image.save(os.path.join(a.out_dir, file_out + '.jpg'))
            pbar.upd(log, uprows=1)
        else: # interpolate
            for f in range(a.fstep):
                tt = blend(f / a.fstep, a.curve)
                file_out = '%05d' % (i * a.fstep + f)
                image = genmix(i, tt, **gendict)
                if a.verbose: cvshow(np.array(image))
                image.save(os.path.join(a.out_dir, file_out + '.jpg'))
                pbar.upd(log, uprows=1)


if __name__ == '__main__':
    main()
