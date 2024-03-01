
import os
import pickle

import torch

from core.sdsetup import SDfu
from core.args import main_args, unprompt
from core.text import multiprompt
from core.utils import load_img, save_img, slerp, lerp, blend, calc_size, isset, read_latents, img_list, basename, progbar, cvshow, save_cfg

samplers = ['klms', 'euler', 'ddim', 'pndm']

def get_args(parser):
    parser.add_argument('-il', '--in_lats', default=None, help='Directory or file with saved keypoints to interpolate between')
    parser.add_argument('-ol', '--out_lats', default=None, help='File to save keypoints for further interpolation')
    parser.add_argument('-fs', '--fstep',   default=25, type=int, help="number of frames for each interpolation step")
    parser.add_argument('-lb', '--latblend', default=0, type=float, help="Strength of latent blending, if > 0: 0.1 ~ alpha-blend, 0.9 ~ full rebuild")
    parser.add_argument('-lg', '--lguide',  action='store_true', help='Use noise multiguidance for interpolation, instead of cond lerp')
    parser.add_argument(       '--curve',   default='linear', help="Interpolating curve: bezier, parametric, custom or linear")
    parser.add_argument(       '--loop',    action='store_true', help='Loop inputs [or stop at the last one]')
    parser.add_argument(       '--skiplast', action='store_true', help='Skip repeating last frame (for smoother animation)')
    return parser.parse_args()

def cond_mix(a, csb, cwb, i, tt=0):
    if a.lguide and a.cguide: # full cfg-multiguide (best, slow!)
        cs  = csb[i % len(csb)]
        cws = cwb[i % len(cwb)] * (1.- tt)
        if tt > 0:
            cs  = torch.cat([cs,  csb[(i+1) % len(csb)]])
            cws = torch.cat([cws, cwb[(i+1) % len(cwb)] * tt])
    elif a.cguide: # cfg-multiguide on batch + cond lerp between steps (less consistent)
        cs  = lerp(csb[i % len(csb)], csb[(i+1) % len(csb)], tt)
        cws = lerp(cwb[i % len(cwb)], cwb[(i+1) % len(cwb)], tt)
    elif a.lguide: # cond lerp on batch + cfg-multiguide between steps
        cs1  = csb[i % len(csb)]
        cws1 = cwb[i % len(cwb)]
        cs = sum([cws1[j] * cs1[j] for j in range(len(cs1))]).unsqueeze(0)
        cws = [1.- tt]
        if tt > 0:
            cs2  = csb[(i+1) % len(csb)]
            cws2 = cwb[(i+1) % len(cwb)]
            cs2 = sum([cws2[j] * cs2[j] for j in range(len(cs2))]).unsqueeze(0)
            cs  = torch.cat([cs, cs2])
            cws = [1.- tt, tt]
    else: # only cond lerp (incoherent for multi inputs)
        cwb = cwb[:, :, None, None]
        cs = lerp(csb[i % len(csb)] * cwb[i % len(cwb)], csb[(i+1) % len(csb)] * cwb[(i+1) % len(cwb)], tt).sum(0, keepdims=True)
        cws = [1.]
    return cs, cws

@torch.no_grad()
def main():
    a = get_args(main_args())
    if a.model[-1] in ['i', 'd'] or (isset(a, 'in_img') and os.path.isdir(a.in_img)): 
        a.sampler = 'ddim' # k-samplers don't work with concat-models; images slerp = full ddim sampling with inversion
    if a.latblend > 0: assert a.sampler in ['ddim', 'euler'], "Latent blending works only with euler or ddim samplers"

    sd = SDfu(a)
    a = sd.a

    a.model = basename(a.model)
    a.seed = sd.seed
    a.unprompt = '' if a.unprompt=='no' else unprompt if a.unprompt is None else ', '.join([unprompt, a.unprompt])
    os.makedirs(a.out_dir, exist_ok=True)
    if a.verbose: print('.. model', a.model, '..', a.sampler, '..', a.cfg_scale, '..', a.strength, '..', sd.seed)
    if a.verbose: save_cfg(a, a.out_dir)

    size = None if a.size is None else calc_size(a.size)
    gendict = {}
    cdict = {} # controlnet, ip adapter

    if a.latblend > 0:
        from core.latblend import LatentBlending
        lb = LatentBlending(sd, a.steps, a.cfg_scale)
        img_count = 0

    def genmix(zs, csb, cwb, uc, i, tt=0, c_img=None, verbose=True, **gendict):
        if c_img is not None:
            c_img = lerp(c_img[i % len(c_img)], c_img[(i+1) % len(c_img)], tt) # [1,1024]
        cs, cws = cond_mix(a, csb, cwb, i, tt)
        z_ = slerp(zs[i % len(zs)], zs[(i+1) % len(zs)], tt)
        images = sd.generate(z_, cs, uc, cws=cws, verbose=verbose, c_img=c_img, **gendict)
        return images

    if isset(a, 'in_lats') and os.path.exists(a.in_lats): # load saved latents & conds
        zs, csb, cwb, uc = read_latents(a.in_lats) # [num,1,4,h,w], [num,b,77,768], [num,b], [1,77,768]
        if os.path.isfile(a.in_lats.replace('.pkl', '-ic.pkl')): # ip adapter img conds
            img_conds, a.imgref_weight = read_latents(a.in_lats.replace('.pkl', '-ic.pkl'))
            cdict['c_img'] = img_conds
        uc = uc[:1] # eliminate multiplication if selected by indices
        zs *= sd.scheduler.init_noise_sigma # fit current sampler
        cwb /= a.cfg_scale
        count = max(len(zs), len(csb)) # csb or zs may be just one
        H, W = [sh * sd.vae_scale for sh in zs[0].shape[-2:]]
        print('.. loaded:', count, 'zs', zs.shape, 'csb', csb.shape, 'cwb', cwb.shape, 'uc', uc.shape)

    else: # make new latents & conds
        csb, cwb, texts = multiprompt(sd, a.in_txt, a.pretxt, a.postxt, a.num) # [num,b,77,768], [num,b], [..]
        uc = multiprompt(sd, a.unprompt)[0][0]
        count = len(csb)

        img_conds = []
        if isset(a, 'img_ref'):
            cdict['c_img'] = img_conds = sd.img_cus(a.img_ref, isset(a, 'allref')) # list of [2,1,1024]
            count = max(count, len(img_conds))

        if isset(a, 'in_img'):
            if os.path.isdir(a.in_img): # interpolation between images
                assert not (sd.inpaintmod or sd.depthmod), "!! Image interpolation doesn't work with inpaint/depth models !!"
                img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
                count = len(img_paths)

                print('.. inverting images ..')
                zs  = []
                pbar = progbar(count)
                for i in range(count):
                    zs += [sd.ddim_inv(sd.img_lat(load_img(img_paths[i], size)[0]), uc)] # ddim inversion
                    if i==0:
                        W, H = size = [sh * sd.vae_scale for sh in zs[-1].shape[-2:]][::-1] # set size as of the first image
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

    if sd.use_cnet and isset(a, 'control_img'):
        assert os.path.isfile(a.control_img), "!! ControlNet image %s not found !!" % a.control_img
        cdict['cnimg'] = (load_img(a.control_img, (W,H))[0] + 1) / 2

    # save key latents if needed
    if isinstance(zs, list): zs = torch.stack(zs)
    if isset(a, 'out_lats'):
        if sd.depthmod or isset(a, 'mask'):
            print('!! latents cannot be saved for inpaint/depth modes !!')
        else:
            lat_dir = 'lats/'
            os.makedirs(os.path.join(a.out_dir, lat_dir), exist_ok=True)
            a.out_lats = os.path.join(a.out_dir, lat_dir, os.path.basename(a.out_lats) + '.pkl')
            with open(a.out_lats, 'wb') as f:
                pickle.dump((zs / sd.scheduler.init_noise_sigma, csb, cwb * a.cfg_scale, uc), f) # forget sampler, remember scale
            if len(img_conds) > 0:
                with open(a.out_lats.replace('.pkl', '-ic.pkl'), 'wb') as f:
                    pickle.dump((img_conds, a.imgref_weight), f)
            if a.verbose:
                print('.. exporting :: zs', zs.shape, 'csb', csb.shape, 'cwb', cwb.shape)
                pbar = progbar(count)
                for i in range(count):
                    images = genmix(zs, csb, cwb, uc, i, **cdict, **gendict, verbose=False)
                    if a.verbose: cvshow(images[0].detach().clone().permute(1,2,0), name='key lats')
                    try:
                        file_out = '%03d-%s' % (i, texts[i % len(texts)][:80]) # , a.sampler, sd.seed
                        save_img(images[0], 0, a.out_dir, prefix=lat_dir, filepath=file_out + '.jpg')
                    except:
                        save_img(images[0], i, a.out_dir, prefix=lat_dir)
                    pbar.upd(uprows = 1 if a.sampler=='euler' else 0)

    print('.. interpolating ..')
    pcount = count if a.loop else count-1
    assert pcount > 0
    pbar = progbar(pcount if a.latblend > 0 else pcount * a.fstep)
    for i in range(pcount):
        if a.latblend > 0:
            if not a.cguide: # cond lerp (may be incoherent)
                csb = sum([csb[:,j] * cwb[:,j,None,None] for j in range(csb.shape[1])]).unsqueeze(1)
            if len(img_conds) > 0:
                cdict['c_img'] = [ img_conds[i % len(img_conds)], img_conds[(i+1) % len(img_conds)] ] # list 2 of lists? [2,1,..] 
            lb.set_conds(csb[i % len(csb)], csb[(i+1) % len(csb)], uc, cws=cwb[0], **cdict) # same weights for all multi conds, same cnet image for whole interpol
            lb.init_lats( zs[i % len(zs)],   zs[(i+1) % len(zs)])
            lb.run_transition(W, H, 1.- a.latblend, a.fstep, reuse = i>0)
            img_count += lb.save_imgs(a.out_dir, img_count, skiplast=a.skiplast)
            pbar.upd(uprows=2)
        else:
            for f in range(a.fstep):
                tt = blend(f / a.fstep, a.curve)
                images = genmix(zs, csb, cwb, uc, i, tt, **cdict, **gendict)
                if a.verbose: cvshow(images[0].detach().clone().permute(1,2,0))
                save_img(images[0], i * a.fstep + f, a.out_dir)
                pbar.upd(uprows=2)
            img_count = pcount * a.fstep

    if a.loop is not True:
        if len(img_conds) > 0:
            cdict['c_img'] = [ img_conds[(pcount) % len(img_conds)], img_conds[0] ]
        images = genmix(zs, csb, cwb, uc, pcount, **cdict, **gendict)
        save_img(images[0], img_count, a.out_dir)


if __name__ == '__main__':
    main()
