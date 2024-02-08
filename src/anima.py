
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import math
import numpy as np
import imageio

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from core.sdsetup import SDfu
from core.args import main_args, samplers, unprompt
from core.text import multiprompt, txt_clean
from core.utils import img_list, load_img, slerp, lerp, blend, basename, save_cfg, calc_size, isset
from core.unet_motion_model import animdiff_forward

def get_args(parser):
    parser.add_argument('-iv', '--in_vid',  default=None, help='input video or frame sequence (directory with images)')
    parser.add_argument('-vf', '--frames',  default=None, type=int, help="Frame count for generated video")
    parser.add_argument('-fs', '--fstep',   default=None, type=int, help="number of frames for each interpolation step")
    parser.add_argument('-cf', '--ctx_frames',  default=16, type=int, help="frame count to process at once with sliding window sampling")
    parser.add_argument('-ad', '--animdiff', default='models/anima', help="path to the Motion Adapter model")
    parser.add_argument(       '--curve',   default='bezier', help="Interpolating curve: bezier, parametric, custom or linear")
    parser.add_argument(       '--loop',    action='store_true')
    # override
    parser.add_argument('-b',  '--batch',   default=1, type=int, choices=[1])
    parser.add_argument('-s',  '--steps',   default=23, type=int, help="number of diffusion steps")
    return parser.parse_args()

def img_out(video):
    video = (video + 1.) / 2.
    video.clamp_(0, 1)
    images = video.permute(1,2,3,0).unbind(dim=0) # list of [h,w,c]
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images] # list of [h,w,c]
    return images

def frameset(ins, frames, loop=False, curve='linear'):
    if len(ins) == 1:
        return list(ins) * frames
    assert frames >= len(ins), "Frame count < input count"
    frames -= 1 # to include the last one
    if loop: ins = ins + [ins[0]]
    outs = [ins[0]]
    steps = len(ins) - 1
    for i in range(steps):
        curstep_count = frames // steps + (1 if i < frames % steps else 0)
        for j in range(1, curstep_count):
            alpha = blend(j / curstep_count, curve)
            outs += [(1 - alpha) * ins[i] + alpha * ins[i+1]]
        outs.append(ins[i+1])
    return outs

@torch.no_grad()
def main():
    a = get_args(main_args())
    sd = SDfu(a)
    a = sd.a
    os.makedirs(a.out_dir, exist_ok=True)
    gendict = {}

    # fix forward function of the motion model to allow batched/scheduled conditions
    setattr(sd.unet, 'forward', animdiff_forward.__get__(sd.unet, sd.unet.__class__))

    a.unprompt = '' if a.unprompt=='no' else unprompt if a.unprompt is None else ', '.join([unprompt, a.unprompt])
    uc = multiprompt(sd, a.unprompt)[0][0]
    if isset (a, 'in_txt'):
        csb, cwb, texts = multiprompt(sd, a.in_txt, a.pretxt, a.postxt, a.num) # [num,b,77,768], [num,b], [..]
    else:
        csb, cwb, texts = uc[None], torch.tensor([[1.]]), ['']
    count = len(csb)

    if isset(a, 'img_ref'):
        assert os.path.exists(a.img_ref), "!! Image ref %s not found !!" % a.img_ref
        img_refs = img_list(a.img_ref) if os.path.isdir(a.img_ref) else [a.img_ref]
        if isset(a, 'allref'):
            img_conds = sd.img_c([load_img(im, tensor=False)[0] for im in img_refs]) # all images at once
        else:
            img_conds = torch.cat([sd.img_c(load_img(im, tensor=False)[0]) for im in img_refs]) # [N,1024] # every image separately
            if len(csb) != len(img_conds): print('!! %d text prompts != %d img refs !!' % (len(csb), len(img_conds)))
            count = max(count, len(img_conds))

    if isset(a, 'in_vid'):
        assert os.path.exists(a.in_vid), "Not found %s" % a.in_vid
        if os.path.isdir(a.in_vid):
            frames = [imageio.imread(path) for path in img_list(a.in_vid)]
        else: 
            frames = imageio.mimread(a.in_vid, memtest=False)
        if isset(a, 'frames'): frames = frames[:a.frames]
        a.frames = len(frames)
        video = torch.from_numpy(np.stack(frames)).permute(0,3,1,2).cuda().half() / 127.5 - 1. # [f,c,h,w]
        size = list(video.shape[-2:])
    else:
        if isset(a, 'fstep') and not isset(a, 'frames'): 
            a.frames = count * a.fstep
        if isset(a, 'in_img') and os.path.isfile(a.in_img):
            size = load_img(a.in_img)[0].shape[-2:] # [:-3:-1] # last 2 reverted
        else:
            size = [sd.res, sd.res]
    if a.frames is None: a.frames = a.ctx_frames
    if not isset(a, 'size'): a.size = size[::-1]
    W, H = calc_size(a.size, pad=True)

    cs_frames = torch.stack(frameset(csb, a.frames, a.loop, a.curve)) # [f,b,77,768]
    cw_frames = torch.stack(frameset(cwb, a.frames, a.loop, a.curve)) # [f,b]
    if isset(a, 'img_ref'):
        gendict['c_img'] = torch.stack(frameset(img_conds, a.frames, a.loop, a.curve)) # [f,1,1024]
    uc_frames = uc.repeat(a.frames,1,1)

    if sd.use_cnet and isset(a, 'control_img'):
        assert os.path.exists(a.control_img), "!! ControlNet image(s) %s not found !!" % a.control_img
        if os.path.isdir(a.control_img):
            cn_imgs = [imageio.imread(path) for path in img_list(a.control_img)][:a.frames]
            assert len(cn_imgs) == a.frames, "!! Not enough ControlNet images: %d, total frame count %d !!" % (len(cn_imgs), a.frames)
        else: 
            cn_imgs = [imageio.imread(a.control_img)] * a.frames
        cn_imgs = torch.from_numpy(np.stack(cn_imgs)).cuda().half().permute(0,3,1,2) / 255. # [0..1] [f,c,h,w]
        if list(cn_imgs.shape[-2:]) != [H, W]:
            cn_imgs = F.interpolate(cn_imgs, (H, W), mode='bicubic', align_corners=True)
        gendict['cnimg'] = cn_imgs

    if a.verbose: 
        print('.. frames', a.frames, '.. model', a.model, '..', a.sampler, '..', '%dx%d' % (W,H), '..', a.cfg_scale, '..', a.strength, '..', sd.seed)
        save_cfg(a, a.out_dir)
    
    def genmix(z_, cs, cws, uc, **gendict):
        if a.cguide: # use noise lerp with cfg scaling (slower)
            video = sd.generate(z_, cs.squeeze(1), uc, cws=cws, **gendict)
        else: # use cond lerp (worse for multi inputs)
            c_ = sum([cs[:,j] * cws[:,j,None,None] for j in range(cs.shape[1])]) # [f,77,768]
            video = sd.generate(z_, c_, uc, **gendict)
        return video

    if isset(a, 'in_vid'):
        if list(video.shape[-2:]) != [H, W]:
            video = F.pad(video, (0, W - video.shape[-1], 0, H - video.shape[-2]), mode='reflect')
        sd.set_steps(a.steps, a.strength)
        z_ = sd.img_z(video) # [f,c,h,w]
        z_ = z_.permute(1,0,2,3)[None,:] # [1,c,f,h,w]
    else:
        sd.set_steps(a.steps, 1)
        z_ = sd.rnd_z(H, W, a.frames) # [1,c,f,h,w]

    video = genmix(z_, cs_frames, cw_frames, uc_frames, **gendict).squeeze(0) # [c,f,h,w]

    images = img_out(video)
    for i, image in enumerate(images):
        if isset(a, 'in_vid') and list(image.shape[:2]) != size: # if input padded
            image = image[:size[0], :size[1]]
        imageio.imsave(os.path.join(a.out_dir, '%04d.jpg' % i), image)


if __name__ == '__main__':
    main()
