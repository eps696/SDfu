
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import time
import numpy as np

import torch

from diffusers import DiffusionPipeline

from core.args import main_args
from core.text import read_txt, txt_clean
from core.utils import load_img, makemask, blend, cvshow, calc_size, isok, isset, img_list, basename, progbar, save_cfg

def get_args(parser):
    parser.add_argument('-mdir', '--models_dir', default=None)
    parser.add_argument('-fs', '--fstep',   default=1, type=int, help="number of frames for each interpolation step (1 = no interpolation)")
    parser.add_argument(       '--curve',   default='linear', help="Interpolating curve: bezier, parametric, custom or linear")
    # UNUSED
    parser.add_argument('-m',  '--model',   default='kandinsky 2.2', help='UNUSED')
    parser.add_argument('-cmod', '--control_mod', default=None, help='UNUSED')
    parser.add_argument('-sm', '--sampler', default=None, help='UNUSED')
    parser.add_argument(       '--vae',     default=None, help='UNUSED')
    parser.add_argument('-cg', '--cguide',  action=None, help='UNUSED')
    parser.add_argument('-lo', '--lowmem',  action=None, help='UNUSED')
    parser.add_argument('--precision',      default=None, help='UNUSED')
    parser.add_argument('-rt', '--load_token', default=None, help='UNUSED')
    parser.add_argument('-rd', '--load_custom', default=None, help='UNUSED')
    parser.add_argument('-rl', '--load_lora', default=None, help='UNUSED')
    parser.add_argument('-b',  '--batch',   default=None, help='UNUSED')
    return parser.parse_args()

unprompt = "low quality, poorly drawn, out of focus, blurry, tiled, segmented, oversaturated"
unprompt += ", ugly, deformed, disfigured, mutated, mutilated, bad anatomy, malformed hands, extra limbs"

def parse_line(txt):
    subs = []
    for subtxt in txt.split('|'):
        if ':' in subtxt:
            [subtxt, wt] = subtxt.split(':')
            wt = float(wt)
        else: wt = 1e-4 if len(subtxt.strip())==0 else 1.
        subs.append((subtxt.strip(), wt))
    return subs # list of tuples

def read_multitext(in_txt, prefix=None, postfix=None):
    if in_txt is None or len(in_txt)==0: return []
    lines = read_txt(in_txt)
    if len(prefix)  > 0: prefix  = read_txt(prefix)
    if len(postfix) > 0: postfix = read_txt(postfix)
    prompts = [parse_line(tt) for tt in lines if tt.strip()[0] != '#']
    texts   = [txt_clean(tt)  for tt in lines if tt.strip()[0] != '#']
    if len(prefix) > 0:
        prompts = [parse_line(prefix[i % len(prefix)]) + prompts[i] for i in range(len(prompts))]
    if len(postfix) > 0:
        prompts = [prompts[i] + parse_line(postfix[i % len(postfix)]) for i in range(len(prompts))]
    return prompts, texts # list of lists of tuples, list

def interweigh(prompts, i, tt):
    inputs1  = [t[0] for t in prompts[i % len(prompts)]]
    inputs2  = [t[0] for t in prompts[(i+1) % len(prompts)]]
    weights1 = [t[1] * (1.-tt) for t in prompts[i % len(prompts)]]
    weights2 = [t[1] * tt      for t in prompts[(i+1) % len(prompts)]]
    tuples = [(i,w) for i,w in zip(inputs1 + inputs2, weights1 + weights2)]
    return tuples

@torch.no_grad()
def main():
    a = get_args(main_args())
    a.seed = a.seed or int((time.time()%1)*69696)
    gen = torch.Generator("cuda").manual_seed(a.seed)
    gendict = {}

    def get_model(name, dir=a.models_dir):
        return 'kandinsky-community/' + name if dir is None or not os.path.exists(dir) else os.path.join(dir, name)

    size = None if not isset(a, 'size') else calc_size(a.size, None, quant=64)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.verbose: save_cfg(a, a.out_dir)
    if a.verbose: print(' kandinsky ..', a.cfg_scale, '..', a.strength, '..', a.seed)

    prompts, texts = read_multitext(a.in_txt, a.pretxt, a.postxt)
    un = '' if a.unprompt=='no' else unprompt if a.unprompt is None else ', '.join([unprompt, a.unprompt])
    count = len(prompts)

    # prior pipe
    if isset(a, 'in_img') and isset(a, 'control_img'): # img2img + controlnet
        from diffusers import KandinskyV22PriorEmb2EmbPipeline as PR
        pipe_prior = PR.from_pretrained(get_model('kandinsky-2-2-prior'), torch_dtype=torch.float16)
    else: # KandinskyV22PriorPipeline
        pipe_prior = DiffusionPipeline.from_pretrained(get_model('kandinsky-2-2-prior'), torch_dtype=torch.float16)

    # main pipe
    if isset(a, 'in_img'):
        if isset(a, 'mask'): # inpaint
            from diffusers import KandinskyV22InpaintPipeline as PP
            pipe = PP.from_pretrained(get_model('kandinsky-2-2-decoder-inpaint'), torch_dtype=torch.float16)
        elif isset(a, 'control_img'): # img2img + controlnet
            from diffusers import KandinskyV22ControlnetImg2ImgPipeline as PP
            pipe = PP.from_pretrained(get_model('kandinsky-2-2-controlnet-depth'), torch_dtype=torch.float16)
        else: # img2img
            from diffusers import KandinskyV22Img2ImgPipeline as PP
            pipe = PP.from_pretrained(get_model('kandinsky-2-2-decoder'), torch_dtype=torch.float16)
    else:
        if isset(a, 'control_img'): # controlnet
            from diffusers import KandinskyV22ControlnetPipeline as PP
            pipe = PP.from_pretrained(get_model('kandinsky-2-2-controlnet-depth'), torch_dtype=torch.float16)
        else:
            PP = DiffusionPipeline # txt2img KandinskyV22Pipeline
            pipe = PP.from_pretrained(get_model('kandinsky-2-2-decoder'), torch_dtype=torch.float16)
    pipe_prior.to("cuda")
    pipe.to("cuda")

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
        from transformers import pipeline
        depth_estimator = pipeline("depth-estimation")
        def make_depth(image):
            image = depth_estimator(image)["depth"]
            image = np.array(image)[:,:,None]
            depth = torch.from_numpy(image).repeat(1,1,3).permute(2,0,1).float() / 255.
            return depth

    def genmix(tuples, W, H, tt=0, img=None, cimg=None, mask=None):
        inputs  = [t[0] for t in tuples]
        weights = [t[1] for t in tuples]
        
        if img is not None and cimg is not None: # img2img + controlnet, NO INTERPOLATION !!!
            c_ = pipe_prior(inputs[0], img, a.strength, un, a.num, a.steps, gen, guidance_scale=a.cfg_scale)
            uc = pipe_prior(un, img, 1, None, a.num, a.steps, gen, guidance_scale=a.cfg_scale)
            image = pipe(c_.image_embeds, img, uc.image_embeds, cimg, H, W, a.steps, a.cfg_scale, a.strength, a.num, gen).images[0]
            return image

        c_, uc = pipe_prior.interpolate(inputs, weights, a.num, a.steps, gen, negative_prompt=un, guidance_scale=a.cfg_scale).to_tuple()

        if img is None:
            if cimg is None:
                image = pipe(c_, uc, H, W, a.steps, a.cfg_scale, a.num, gen).images[0] # txt2img
            else:
                image = pipe(c_, uc, cimg, H, W, a.steps, a.cfg_scale, a.num, gen).images[0] # txt2img + controlnet
        else:
            if mask is None:
                image = pipe(c_, img, uc, H, W, a.steps, a.cfg_scale, a.strength, a.num, gen)[0][0] # img2img
            else:
                image = pipe(c_, img, mask, uc, H, W, a.steps, a.cfg_scale, a.num, gen)[0][0] # inpaint
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
            W, H = (768,768) if size is None else size

        if isset(a, 'control_img'):
            cimg, _ = load_img(cimg_paths[i % len(cimg_paths)], (W,H), tensor=False)
            gendict['cimg'] = make_depth(cimg).unsqueeze(0).half().to("cuda")

        if a.fstep <= 1: # single image
            prompt = prompts[i % len(prompts)]
            printnum = len(prompt)
            image = genmix(prompt, W, H, **gendict)
            if a.verbose: cvshow(np.array(image))
            image.save(os.path.join(a.out_dir, file_out + '.jpg'))
            pbar.upd(log, uprows = printnum+2)
        else: # interpolate
            for f in range(a.fstep):
                tt = blend(f / a.fstep, a.curve)
                file_out = '%05d' % (i * a.fstep + f)
                tuples = interweigh(prompts, i, tt)
                printnum = len(tuples)
                image = genmix(tuples, W, H, **gendict)
                if a.verbose: cvshow(np.array(image))
                image.save(os.path.join(a.out_dir, file_out + '.jpg'))
                pbar.upd(log, uprows = printnum+2)


if __name__ == '__main__':
    main()
