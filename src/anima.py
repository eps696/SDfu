
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
from core.utils import file_list, basename, progbar, save_cfg, isset

def get_args(parser):
    parser.add_argument('-iv', '--in_vid',  default=None, help='input video or directory with images')
    parser.add_argument('-vf', '--frames',  default=16, type=int, help="Frame count for generated video")
    parser.add_argument('-ad', '--animdiff', default='models/anima', help="path to the Motion Adapter model")
    # override
    parser.add_argument('-b',  '--batch',   default=1, type=int, choices=[1])
    parser.add_argument('-sm', '--sampler', default='ddim', choices=samplers)
    parser.add_argument('-C','--cfg_scale', default=13, type=float, help="prompt guidance scale")
    return parser.parse_args()

def img_out(video):
    video = (video + 1.) / 2.
    video.clamp_(0, 1)
    images = video.permute(1,2,3,0).unbind(dim=0) # list of [h,w,c]
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]  # [f,h,w,c]
    return images

@torch.no_grad()
def main():
    a = get_args(main_args())
    sd = SDfu(a)

    csb, cwb, _ = multiprompt(sd, a.in_txt, a.pretxt, a.postxt, a.num) # [num,b,77,768], [num,b], [..]
    a.unprompt = '' if a.unprompt=='no' else unprompt if a.unprompt is None else ', '.join([unprompt, a.unprompt])
    uc = multiprompt(sd, a.unprompt)[0][0]
    
    videoin = []
    if a.in_vid is not None and os.path.exists(a.in_vid):
        if os.path.isdir(a.in_vid):
            frames = [imageio.imread(path) for path in img_list(a.in_vid)]
        else: 
            frames = imageio.mimread(a.in_vid, memtest=False)
        for i in range(math.ceil(len(frames) / a.frames)): # split to chunks
            subframes = frames[i*a.frames : (i+1)*a.frames]
            video = torch.from_numpy(np.stack(subframes)).permute(0,3,1,2) # [f,c,h,w]
            videoin += [video / 127.5 - 1.]
        name = basename(a.in_vid)
        if not isset(a, 'size'): a.size = list(video.shape[:-3:-1]) # last 2 reverted
    else:
        name = basename(a.in_txt) if os.path.exists(a.in_txt) else txt_clean(a.in_txt)
        if not isset(a, 'size'): a.size = [sd.res]
    W, H = calc_size(a.size)

    outdir = os.path.join(a.out_dir, '%s-%d' % (name, sd.seed))
    os.makedirs(outdir, exist_ok=True)
    if a.verbose: 
        print('.. model', a.model, '..', '%dx%d' % (W,H), '..', a.cfg_scale, '..', a.strength, '..', sd.seed)
        save_cfg(a, a.out_dir)
    
    def genmix(z_, cs, cws):
        if a.cguide: # use noise lerp with cfg scaling (slower)
            video = sd.generate(z_, cs, uc, cws=cws)
        else: # use cond lerp (worse for multi inputs)
            c_ = sum([cws[j] * cs[j] for j in range(len(cs))]).unsqueeze(0)
            video = sd.generate(z_, c_, uc)
        return video

    count = max(len(videoin), len(csb))
    pbar = progbar(count)
    for n in range(count):
        if len(videoin) > 0:
            video = videoin[n % len(videoin)].cuda()
            video = F.interpolate(video, (H, W), mode='bicubic', align_corners=True)
            sd.set_steps(a.steps, a.strength)
            z_ = sd.img_z(video) # [f,c,h,w]
            z_ = z_.permute(1,0,2,3)[None,:] # [1,c,f,h,w]
        else:
            sd.set_steps(a.steps, 1)
            z_ = sd.rnd_z(H, W, a.frames) # [1,c,f,h,w]

        video = genmix(z_, csb[n % len(csb)], cwb[n % len(cwb)]).squeeze(0) # [c,f,h,w]
        images = img_out(video)
        for i, image in enumerate(images):
            imageio.imsave(os.path.join(outdir, '%04d.jpg' % (n * a.frames + i)), image)
        pbar.upd(basename(outdir), uprows=2)


if __name__ == '__main__':
    main()