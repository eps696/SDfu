
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
from torchvision import transforms as T

from core.sdsetup import SDfu
from core.args import main_args, unprompt
from core.text import multiprompt
from core.utils import load_img, save_img, lerp, triblur, calc_size, isset, basename, progbar, cvshow, save_cfg, latent_anima

def get_args(parser):
    # override
    parser.add_argument('-sm', '--sampler', default='ddim')
    parser.add_argument('-f', '--strength', default=0.7, type=float, help="strength of image processing. 0 = preserve img, 1 = replace it completely")
    # main
    parser.add_argument(       '--unsharp', default=0.2, type=float)
    parser.add_argument(       '--renoise', default=0.1, type=float)
    parser.add_argument('-fs', '--fstep',   default=25, type=int, help="number of frames for each interpolation step")
    parser.add_argument('-lb', '--latblend', default=0.8, type=float, help="Add latent blending between frames (with this strength)")
    # animation
    parser.add_argument(       '--scale',   default=0., type=float)
    parser.add_argument(       '--shift',   default=0.02, type=float)
    parser.add_argument(       '--angle',   default=2, type=float)
    parser.add_argument('-as', '--astep',   default=None, type=int, help="number of frames for each animation step")
    parser.add_argument('-is', '--interstep', default=0, type=int, help="Add intermediate latent-blended transition with this number of frames")
    return parser.parse_args()

def frame_transform(img, size, angle, shift, scale):
    img = T.functional.affine(img, angle, tuple(shift), scale, 0., fill=0, interpolation=eval('T.InterpolationMode.BILINEAR'))
    img = T.functional.center_crop(img, size) # on 1.8+ also pads
    return img

def sharpen(img, unsharp_amt=0.2, threshold=0.015):
    img_blurred = triblur(img, k=5)
    img_sharp = (unsharp_amt + 1) * img - unsharp_amt * img_blurred
    if threshold > 0:
        low_contrast_mask = torch.abs(img - img_blurred) < threshold # source img if true
        img_sharp[low_contrast_mask] = img[low_contrast_mask]
    return img_sharp

def renorm(x, mean=0., std=1.):
    x = (x - x.mean()) * std / x.std() + mean
    return x

@torch.no_grad()
def main():
    a = get_args(main_args())
    if a.latblend > 0: assert a.sampler in ['ddim', 'euler'], "Latent blending works only with euler or ddim samplers"
    if a.interstep > 0: assert a.sampler=='ddim', "Intermediate transitions require DDIM sampler"
    sd = SDfu(a)
    a = sd.a

    a.model = basename(a.model)
    a.seed = sd.seed
    size = None if not isset(a, 'size') else calc_size(a.size)
    a.unprompt = '' if a.unprompt=='no' else unprompt if a.unprompt is None else ', '.join([unprompt, a.unprompt])
    os.makedirs(a.out_dir, exist_ok=True)
    if a.verbose: print('.. model', a.model, '..', a.sampler, '..', a.cfg_scale, '..', a.strength, '..', sd.seed)
    if a.verbose: save_cfg(a, a.out_dir)
    img_count = 0

    csb, cwb, texts = multiprompt(sd, a.in_txt, a.pretxt, a.postxt, a.num)
    uc = multiprompt(sd, a.unprompt)[0][0]
    count = len(csb)

    if a.latblend > 0 or a.interstep > 0:
        from core.latblend import LatentBlending
        lb = LatentBlending(sd, a.steps, a.cfg_scale, a.strength, verbose=False)

    if isset(a, 'in_img') and os.path.isfile(a.in_img):
        images, size = load_img(a.in_img, size)
    else:
        if size is None: size = [sd.res]*2
        sd.set_steps(a.steps, 1) # set full steps for txt2img
        z_init = sd.rnd_z(*size[::-1])
        images = sd.generate(z_init, csb[0], uc, cws=cwb[0], verbose=False)
        sd.set_steps(a.steps, a.strength) # set reduced steps for img2img
    save_img(images[0], 0, a.out_dir)
    z_ = sd.img_z(images)
    init_mean, init_std = images.mean(), images.std()
    W, H = size

    cdict = {} # for controlnet
    if sd.use_cnet and isset(a, 'control_img'):
        assert os.path.isfile(a.control_img), "!! ControlNet image %s not found !!" % a.control_img
        cdict['cnimg'] = (load_img(a.control_img, (W,H))[0] + 1) / 2

    glob_steps = count * a.fstep
    if a.astep is None: a.astep = a.fstep
    assert glob_steps % a.astep == 0, "Total frame count %d should be divisible by animation step %d" % (glob_steps, a.astep)

    if a.scale != 0:
        m_sh_ = -0.4
        m_scale = latent_anima([1], glob_steps, a.astep, uniform=True, seed=a.seed, verbose=False)
        m_shift = latent_anima([3], glob_steps, a.astep, uniform=True, seed=a.seed, verbose=False)
        m_angle = latent_anima([3], glob_steps, a.astep, uniform=True, seed=a.seed, verbose=False)
        m_scale = 1 + (m_scale - m_sh_) * a.scale # only zoom in for now
        m_shift = (m_shift-0.5) * a.shift * min(size) * abs(m_scale-1) / a.scale
        m_angle = (m_angle-0.5) * a.angle * abs(m_scale-1) / a.scale

    def c_mix(csb, cwb, i, tt=0):
        if a.cguide: # cfg-multiguide on batch
            cs  = lerp(csb[i % len(csb)], csb[(i+1) % len(csb)], tt)
            cws = lerp(cwb[i % len(cwb)], cwb[(i+1) % len(cwb)], tt)
        else: # cond lerp (incoherent for multi inputs)
            cwb = cwb[:, :, None, None]
            cs = lerp(csb[i % len(csb)] * cwb[i % len(cwb)], csb[(i+1) % len(csb)] * cwb[(i+1) % len(cwb)], tt).sum(0, keepdims=True)
            cws = [1.]
        return cs, cws

    cs = None
    if a.interstep > 0: glob_steps += count
    pbar = progbar(glob_steps)
    for i in range(count):
        for f in range(a.fstep):
            log = texts[i % len(texts)][:64]
            anim_step = a.fstep * i + f
            tt = f / a.fstep

            images = renorm(images, init_mean, init_std).clamp(-1., 1.)
            images = sharpen(images, a.unsharp)
            if a.scale != 0:
                scale = m_scale[anim_step]
                shift = m_shift[anim_step]
                angle = m_angle[anim_step]
                images = frame_transform(images, size[::-1], angle[2], shift[:2], scale)
            images = images + torch.randn(images.shape, device=images.device) * a.renoise

            z_prev = z_
            z_ = sd.img_z(images)
            if f==0:
                z_mean, z_std = z_.mean(), z_.std()
            else:
                z_ = renorm(z_, z_mean, z_std)

            if a.latblend > 0:
                cs_prev = cs
                cs, cws = c_mix(csb, cwb, i, tt)
                if cs_prev is None: cs_prev = cs
                lb.set_conds(cs_prev, cs, cws, uc)
                lb.init_lats(z_prev, z_, **cdict)
                lb.run_transition(W, H, 1.- a.latblend, max_branches = 3, reuse = f>0)
                img_count += lb.save_imgs(a.out_dir, img_count) - 1
                images = lb.tree_imgs[-1]
            else:
                cs, cws = c_mix(csb, cwb, i, tt)
                images = sd.generate(z_, cs, uc, cws=cws, verbose=False, **cdict)
                if a.verbose: cvshow(images[0].permute(1,2,0))
                save_img(images[0], img_count + 1, a.out_dir)
                img_count += 1
            pbar.upd()

        if a.interstep > 0:
            lb.set_steps(a.steps, 1) # set full steps for txt2img
            lb.verbose = True
            cs_new, cws = c_mix(csb, cwb, (i+1) % count)
            z_ = sd.ddim_inv(sd.img_lat(images), uc, **cdict) # last frame
            z_new = sd.rnd_z(*size[::-1]) if i < count-1 else z_init # new frame
            lb.set_conds(uc, cs_new, cws, uc)
            lb.init_lats(z_, z_new, **cdict)
            lb.run_transition(W, H, 2/a.steps, max_branches = a.interstep)
            img_count += lb.save_imgs(a.out_dir, img_count) - 1
            images = lb.tree_imgs[-1]
            init_mean, init_std = images.mean(), images.std()
            lb.set_steps(a.steps, a.strength) # set reduced steps for img2img
            z_ = sd.img_z(images)
            lb.verbose = False
            pbar.upd(uprows=2)

    if a.interstep == 0: # no loop
        z_ = sd.img_z(images)
        images = sd.generate(z_, csb[count-1], uc, cws=cwb[count-1], verbose=False, **cdict)
        save_img(images[0], img_count, a.out_dir)


if __name__ == '__main__':
    main()
