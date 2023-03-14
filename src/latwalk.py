
import os
import pickle
import argparse

import torch

from util.setup import sd_setup
from util.text import multiprompt
from util.utils import load_img, save_img, slerp, lerp, blend, calc_size, isset, read_latents, img_list, basename, progbar, cvshow

samplers = ['klms', 'euler', 'ddim', 'pndm']

def get_args():
    parser = argparse.ArgumentParser()
    # this
    parser.add_argument('-il', '--in_lats', default=None, help='Directory or file with saved keypoints to interpolate between')
    parser.add_argument('-ol', '--out_lats', default=None, help='File to save keypoints for further interpolation')
    parser.add_argument('-fs', '--fstep',   default=25, type=int, help="number of frames for each interpolation step")
    parser.add_argument(       '--curve',   default='linear', help="Interpolating curve: bezier, parametric, custom or linear")
    parser.add_argument(       '--loop',    action='store_true', help='Loop inputs [or stop at the last one]')
    # inputs & paths
    parser.add_argument('-t',  '--in_txt',  default=None, help='Text string or file to process')
    parser.add_argument('-pre', '--pretxt', default='', help='Prefix for input text')
    parser.add_argument('-post', '--postxt', default='', help='Postfix for input text')
    parser.add_argument('-im', '--in_img',  default=None, help='input image or directory with images (overrides width and height)')
    parser.add_argument('-M',  '--mask',    default=None, help='Path to input mask for inpainting mode (overrides width and height)')
    parser.add_argument('-un','--unprompt', default='', help='Negative prompt to be used as a neutral [uncond] starting point')
    parser.add_argument('-o',  '--out_dir', default="_out", help="Output directory for generated images")
    parser.add_argument('-md', '--maindir', default='./models', help='Main SD models directory')
    # mandatory params
    parser.add_argument('-m',  '--model',   default='15', choices=['15','15i','2i','2d','21','21v'])
    parser.add_argument('-sm', '--sampler', default='klms', choices=samplers)
    parser.add_argument(       '--vae',     default='ema', help='orig, ema, mse')
    parser.add_argument('-C','--cfg_scale', default=13, type=float, help="prompt guidance scale")
    parser.add_argument('-f', '--strength', default=0.75, type=float, help="strength of image processing. 0 = preserve img, 1 = replace it completely")
    parser.add_argument('-di',  '--ddim_inv', action='store_true', help='Use DDIM inversion for image latent encoding (slower but exact)')
    parser.add_argument(      '--ddim_eta', default=0., type=float)
    parser.add_argument('-s', '--steps',    default=50, type=int, help="number of diffusion steps")
    parser.add_argument(     '--precision', default='autocast')
    parser.add_argument('-b',  '--batch',   default=1, type=int, help="batch size")
    parser.add_argument('-S', '--seed',     type=int, help="image seed")
    # finetuned stuff
    parser.add_argument('-tt', "--token_emb", default=None, help="path to the text inversion embeddings file")
    parser.add_argument('-dt', "--delta_ckpt", default=None, help="path to the custom diffusion delta checkpoint")
    # misc
    parser.add_argument('-sz', '--size',    default=None, help="image sizes, multiple of 8")
    parser.add_argument('-par', '--parens', action='store_true', help='Use modern prompt weighting with brackets (otherwise like a:1|b:2)')
    parser.add_argument('-inv', '--invert_mask', action='store_true')
    parser.add_argument('-v',  '--verbose', action='store_true')
    return parser.parse_args()

@torch.no_grad()
def main():
    a = get_args()
    if a.model[-1] in ['i', 'd']: a.sampler = 'ddim' # k-samplers don't work with concat-models
    if isset(a, 'in_img') and os.path.isdir(a.in_img): # images slerp = full ddim sampling with inversion
        a.ddim_inv = True

    [a, func, pipe, generate] = sd_setup(a)
    uc = multiprompt(pipe, a.unprompt)[0][0]

    os.makedirs(a.out_dir, exist_ok=True)
    size = None if a.size is None else calc_size(a.size, a.model, a.verbose) 
    gendict = {}
    if a.verbose: print('.. model', a.model, '..', a.sampler, '..', a.strength)

    if isset(a, 'in_img'):
        if os.path.isdir(a.in_img): # interpolation between images
            assert a.model[-1] not in ['i', 'd'], "!! Image interpolation doesn't work with inpaint/depth models !!"
            img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
            count = len(img_paths)
            cs, _ = multiprompt(pipe, a.in_txt, a.pretxt, a.postxt, a.parens)

            if isset(a, 'in_lats') and os.path.exists(a.in_lats):
                zs = read_latents(a.in_lats)
                count = len(zs)
                print(' loaded:', count, zs.shape)
            else:
                if isset(a, 'out_lats'):
                    lat_dir = 'lats/'
                    os.makedirs(os.path.join(a.out_dir, lat_dir), exist_ok=True)
                    a.out_lats = os.path.join(a.out_dir, lat_dir, os.path.basename(a.out_lats))

                print('.. inverting images ..')
                zs  = []
                pbar = progbar(count)
                for i in range(count):
                    zs += [func.img_z(load_img(img_paths[i], size)[0], uc)] # ddim inversion
                    if isset(a, 'out_lats'):
                        images = generate(zs[-1], uc, uc, verbose=False) # test sample with cfg_scale = 0 or 1
                        save_img(images[0], i, a.out_dir, prefix=lat_dir)
                        with open(a.out_lats, 'wb') as f:
                            pickle.dump((torch.stack(zs)), f)
                    pbar.upd()

            print('.. interpolating ..')
            pcount = count if a.loop else count-1
            pbar = progbar(pcount * a.fstep)
            for i in range(pcount):
                for j in range(a.fstep):
                    z_  = slerp(zs[i],           zs[(i+1) % count],   j / a.fstep)
                    c_  = slerp(cs[i % len(cs)], cs[(i+1) % len(cs)], j / a.fstep)
                    images = generate(z_, c_, uc)
                    if a.verbose: cvshow(images[0].detach().clone().permute(1,2,0))
                    save_img(images[0], i*a.fstep+j, a.out_dir)
                    pbar.upd(uprows=2)
            if not a.loop:
                images = generate(zs[count-1], c_, uc)
                save_img(images[0], pcount*a.fstep, a.out_dir)

            exit()

        elif os.path.isfile(a.in_img): # single image + text interpolation
            init_image, (W,H) = load_img(a.in_img, size)
            if a.depth:
                gendict = func.prep_depth(init_image)
            elif isset(a, 'mask'):
                gendict = func.prep_mask(a.mask, a.in_img, init_image)
            z_ = func.rnd_z(H, W) if a.inpaint else func.img_z(init_image)

        else:
            print("!! Image(s) %s not found !!" % a.in_img); exit()

    else: # only text interpolation
        W, H = [a.res]*2 if size is None else size
        z_ = None

    # prepare text interpolation
    if isset(a, 'in_lats') and os.path.exists(a.in_lats):
        zs, cs = read_latents(a.in_lats)
        count = len(cs)
        print(' loaded:', count, zs.shape, cs.shape)

    else:
        cs, _ = multiprompt(pipe, a.in_txt, a.pretxt, a.postxt, a.parens)
        count = len(cs)
        
        if isset(a, 'out_lats'):
            lat_dir = 'lats/'
            os.makedirs(os.path.join(a.out_dir, lat_dir), exist_ok=True)
            pbar = progbar(count)

        zs = [] if z_ is None else [z_]
        for i in range(len(cs)):
            if z_ is None: zs += [func.rnd_z(H, W)] # [1,4,64,64] noise

            if isset(a, 'out_lats'):
                images = generate(zs[-1], cs[i], uc)
                if a.verbose: cvshow(images[0].detach().clone().permute(1,2,0))
                save_img(images[0], i, a.out_dir, prefix=lat_dir)
                with open(a.out_lats, 'wb') as f:
                    pickle.dump((torch.stack(zs), torch.stack(cs)), f) # [n,4,64,64] [n,77,768]
                pbar.upd(uprows=2)

    # run interpolation
    lerp_z = lerp if isset(a, 'in_img') and not a.inpaint else slerp # img_z => lerp, rnd_z => slerp
    pcount = count if a.loop else count-1
    pbar = progbar(pcount * a.fstep)
    for i in range(pcount):

        for f in range(a.fstep):
            tt = blend(f / a.fstep, a.curve)
            z_ = lerp_z(zs[i % len(zs)], zs[(i+1) % len(zs)], tt)
            c_ =  slerp(cs[i % len(cs)], cs[(i+1) % len(cs)], tt)

            images = generate(z_, c_, uc, **gendict)
            if a.verbose: cvshow(images[0].detach().clone().permute(1,2,0))
            save_img(images[0], i * a.fstep + f, a.out_dir)
            pbar.upd(uprows=2)

    if not a.loop:
        images = generate(zs[pcount % len(zs)], cs[pcount % len(cs)], uc, **gendict)
        save_img(images[0], pcount * a.fstep, a.out_dir)


if __name__ == '__main__':
    main()
