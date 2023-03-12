
import os
import argparse

import torch

from util.setup import sd_setup, models, samplers, device
from util.text import read_txt, multiprompt
from util.utils import load_img, save_img, calc_size, isok, isset, img_list, basename, progbar

def get_args():
    parser = argparse.ArgumentParser()
    # inputs & paths
    parser.add_argument('-t',  '--in_txt',  default=None, help='Text string or file to process')
    parser.add_argument('-pre', '--pretxt', default='', help='Prefix for input text')
    parser.add_argument('-post','--postxt', default='', help='Postfix for input text')
    parser.add_argument('-im', '--in_img',  default=None, help='input image or directory with images (overrides width and height)')
    parser.add_argument('-M',  '--mask',    default=None, help='Path to input mask for inpainting mode (overrides width and height)')
    parser.add_argument('-un','--unprompt', default=None, help='Negative prompt to be used as a neutral [uncond] starting point')
    parser.add_argument('-o',  '--out_dir', default="_out", help="Output directory for generated images")
    parser.add_argument('-md', '--maindir', default='./models', help='Main SD models directory')
    # mandatory params
    parser.add_argument('-m',  '--model',   default='15', choices=models, help="model version")
    parser.add_argument('-sm', '--sampler', default='pndm', choices=samplers)
    parser.add_argument(       '--vae',     default='ema', help='orig, ema, mse')
    parser.add_argument('-C','--cfg_scale', default=13, type=float, help="prompt guidance scale")
    parser.add_argument('-f', '--strength', default=0.75, type=float, help="strength of image processing. 0 = preserve img, 1 = replace it completely")
    parser.add_argument('-di','--ddim_inv', action='store_true', help='Use DDIM inversion for image latent encoding')
    parser.add_argument(      '--ddim_eta', default=0., type=float)
    parser.add_argument('-s',  '--steps',   default=50, type=int, help="number of diffusion steps")
    parser.add_argument('--precision',      default='autocast')
    parser.add_argument('-b',  '--batch',   default=1, type=int, help="batch size")
    parser.add_argument('-S',  '--seed',    type=int, help="image seed")
    # finetuned stuff
    parser.add_argument('-tt', "--token_emb", default=None, help="path to the text inversion embeddings file")
    parser.add_argument('-dt', "--delta_ckpt", default=None, help="path to the custom diffusion delta checkpoint")
    # misc
    parser.add_argument('-sz', '--size',    default=None, help="image size, multiple of 8")
    parser.add_argument('-par', '--parens', action='store_true', help='Use modern prompt weighting with brackets (otherwise like a:1|b:2)')
    parser.add_argument('-inv', '--invert_mask', action='store_true')
    parser.add_argument('-v',  '--verbose', action='store_true')
    return parser.parse_args()

@torch.no_grad()
def main():
    a = get_args()
    [a, func, pipe, generate, uc] = sd_setup(a)

    posttxt = basename(a.in_txt) if isset(a, 'in_txt') and os.path.exists(a.in_txt) else ''
    postimg = basename(a.in_img) if isset(a, 'in_img') and os.path.isdir(a.in_img)  else ''
    if isok(posttxt) or isok(postimg):
        a.out_dir = os.path.join(a.out_dir, posttxt + '-' + postimg)
        a.out_dir += '-' + a.model
    os.makedirs(a.out_dir, exist_ok=True)

    size = None if not isset(a, 'size') else calc_size(a.size, a.model, a.verbose) 
    if a.verbose: print('.. model', a.model, '..', a.sampler, '..', a.cfg_scale, '..', a.strength)

    count = 0
    if isset(a, 'in_txt'):
        cs, texts = multiprompt(pipe, a.in_txt, a.pretxt, a.postxt, a.parens)
        count = max(count, len(cs))

    if isset(a, 'in_img'):
        assert os.path.exists(a.in_img), "!! Image(s) %s not found !!" % a.in_img
        img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
        count = max(count, len(img_paths))
    if isset(a, 'mask'):
        masks = img_list(a.mask) if os.path.isdir(a.mask) else read_txt(a.mask)

    pbar = progbar(count)
    for i in range(count):
        c_ = cs[i % len(cs)]
        log = texts[i % len(texts)][:88]

        if isset(a, 'in_img'):
            img_path = img_paths[i % len(img_paths)]
            file_out = basename(img_path)
            log += ' .. %s' % os.path.basename(img_path)
            init_image, (W,H) = load_img(img_path, size)

            if a.depth:
                gendict = func.prep_depth(init_image)
                z_ = func.img_z(init_image)
            elif isset(a, 'mask'):
                log += ' / %s' % os.path.basename(masks[i % len(masks)])
                gendict = func.prep_mask(masks[i % len(masks)], img_path, init_image)
                z_ = func.rnd_z(H, W) if a.inpaint else func.img_z(init_image)
            else: # standard img2img
                gendict = {}
                z_ = func.img_z(init_image)

            images = generate(z_, c_, **gendict)

        else: # txt2img, full sampler
            file_out = '%s-m%s-%s-%d' % (log, a.model, a.sampler, a.seed)
            W, H = [a.res]*2 if size is None else size
            z_ = func.rnd_z(H, W)
            images = generate(z_, c_)

        outcount = images.shape[0]
        if outcount > 1:
            for i in range(outcount):
                save_img(images[i], i, a.out_dir, filepath=file_out + '-%02d.jpg' % i)
        else:
            save_img(images[0], 0, a.out_dir, filepath=file_out + '.jpg')
        pbar.upd(log, uprows=2)

if __name__ == '__main__':
    main()
