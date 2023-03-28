
import os
import pickle

import torch

from util.sdsetup import SDfu
from util.args import main_args
from util.text import multiprompt
from util.utils import load_img, save_img, slerp, lerp, blend, calc_size, isset, read_latents, img_list, basename, progbar, cvshow

samplers = ['klms', 'euler', 'ddim', 'pndm']

def get_args(parser):
    parser.add_argument('-il', '--in_lats', default=None, help='Directory or file with saved keypoints to interpolate between')
    parser.add_argument('-ol', '--out_lats', default=None, help='File to save keypoints for further interpolation')
    parser.add_argument('-fs', '--fstep',   default=25, type=int, help="number of frames for each interpolation step")
    parser.add_argument(       '--curve',   default='linear', help="Interpolating curve: bezier, parametric, custom or linear")
    parser.add_argument('-lb', '--latblend', action='store_true', help='Use latent blending for smoother transitions')
    parser.add_argument('-n',  '--num',     default=1, type=int)
    parser.add_argument(       '--loop',    action='store_true', help='Loop inputs [or stop at the last one]')
    return parser.parse_args()

@torch.no_grad()
def main():
    a = get_args(main_args())
    # k-samplers don't work here with concat-models or latent blending 
    if a.model[-1] in ['i', 'd'] or a.latblend \
        or (isset(a, 'in_img') and os.path.isdir(a.in_img)): # images slerp = full ddim sampling with inversion
        a.sampler = 'ddim'

    sd = SDfu(a)

    os.makedirs(a.out_dir, exist_ok=True)
    size = None if a.size is None else calc_size(a.size, a.model, a.verbose) 
    gendict = {}
    if a.verbose: print('.. model', a.model, '..', a.sampler, '..', a.strength)

    if a.latblend:
        from util.latblend import LatentBlending
        lb = LatentBlending(sd, a.steps, a.cfg_scale)
        img_count = 0

    if isset(a, 'in_lats') and os.path.exists(a.in_lats): # get saved latents & conds
        zs, cs = read_latents(a.in_lats)
        count = max(len(zs), len(cs)) # cs or zs may be just 1-len list
        H, W = [sh * sd.vae_scale for sh in zs[0].shape[-2:]]
        print(' loaded:', count, 'zs', zs.cpu().numpy().shape, 'cs', cs.cpu().numpy().shape)

    else: # make new latents & conds
        cs, _ = multiprompt(sd, a.in_txt, a.pretxt, a.postxt, a.parens, a.text_norm)
        if a.num > 1: cs = cs * a.num
        count = len(cs)

        if isset(a, 'in_img'):
            if os.path.isdir(a.in_img): # interpolation between images
                assert a.model[-1] not in ['i', 'd'], "!! Image interpolation doesn't work with inpaint/depth models !!"

                img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
                count = len(img_paths)

                print('.. inverting images ..')
                zs  = []
                cs = [sd.uc]
                pbar = progbar(count)
                for i in range(count):
                    zs += [sd.ddim_inv(sd.img_lat(load_img(img_paths[i], size)[0]))] # ddim inversion
                    if i==0:
                        W, H = size = [sh * sd.vae_scale for sh in zs[-1].shape[-2:]] # fix size as the first image
                    pbar.upd()

            elif os.path.isfile(a.in_img): # single image + text interpolation
                init_image, (W,H) = load_img(a.in_img, size)
                if sd.depthmod:
                    gendict = sd.prep_depth(init_image)
                elif isset(a, 'mask'):
                    gendict = sd.prep_mask(a.mask, a.in_img, init_image)
                zs = [sd.rnd_z(H, W) if sd.inpaintmod else sd.img_z(init_image)]

            else:
                print("!! Image(s) %s not found !!" % a.in_img); exit()

        else: # only text interpolation
            W, H = [sd.res]*2 if size is None else size
            zs = [sd.rnd_z(H, W) for i in range(count)]

        # save key latents
        if isset(a, 'out_lats'):
            if sd.depthmod or isset(a, 'mask'):
                print('!! latents cannot be saved for inpaint/depth modes !!')
            else:
                print('.. exporting latents ..')
                lat_dir = 'lats/'
                os.makedirs(os.path.join(a.out_dir, lat_dir), exist_ok=True)
                a.out_lats = os.path.join(a.out_dir, lat_dir, os.path.basename(a.out_lats))
                pbar = progbar(count)
                with open(a.out_lats, 'wb') as f:
                    pickle.dump((torch.stack(zs), torch.stack(cs)), f)
                print('zs', torch.stack(zs).shape, 'cs', torch.stack(cs).shape)
                for i in range(count):
                    images = sd.generate(zs[i % len(zs)], cs[i % len(cs)], verbose=False)
                    if a.verbose: cvshow(images[0].detach().clone().permute(1,2,0), name='key lats')
                    save_img(images[0], i, a.out_dir, prefix=lat_dir)
                    pbar.upd()

    print('.. interpolating ..')
    pcount = count if a.loop else count-1
    pbar = progbar(pcount if a.latblend else pcount * a.fstep)
    for i in range(pcount):
        if a.latblend:
            lb.init_lats(zs[i % len(zs)], zs[(i+1) % len(zs)])
            lb.set_conds(cs[i % len(cs)], cs[(i+1) % len(cs)])
            lb.run_transition(W, H, max_branches = a.fstep, reuse = i>0)
            img_count += lb.save_imgs(a.out_dir, img_count)
            pbar.upd(uprows=2)
        else:
            for f in range(a.fstep):
                tt = blend(f / a.fstep, a.curve)
                z_ = slerp(zs[i % len(zs)], zs[(i+1) % len(zs)], tt)
                c_ =  lerp(cs[i % len(cs)], cs[(i+1) % len(cs)], tt)
                images = sd.generate(z_, c_, **gendict)

                if a.verbose: cvshow(images[0].detach().clone().permute(1,2,0))
                save_img(images[0], i * a.fstep + f, a.out_dir)
                pbar.upd(uprows=2)
            img_count = pcount*a.fstep

    if a.loop is not True:
        images = sd.generate(zs[pcount % len(zs)], cs[pcount % len(cs)], **gendict)
        save_img(images[0], img_count, a.out_dir)


if __name__ == '__main__':
    main()
