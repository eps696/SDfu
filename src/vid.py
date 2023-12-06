
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
from core.text import multiprompt
from core.utils import file_list, basename, progbar, save_cfg

def get_args(parser):
    parser.add_argument('-iv', '--in_vid',  default=None, help='input video or directory with images (overrides width and height)')
    parser.add_argument('-vf', '--frames', default=36, type=int, help="Frame count for generated video")
    parser.add_argument('-cf', '--ctx_frames',  default=30, type=int, help="frame count to process at once with sliding window sampling")
    parser.add_argument(       '--loop',    action='store_true')
    # override
    parser.add_argument('-m',  '--model',   default='vzs', help="Lo-res model")
    # parser.add_argument('-up', '--model_up', default=None, help="Hi-res model") # vpot
    parser.add_argument('-b',  '--batch',   default=1, type=int, choices=[1])
    parser.add_argument('-sm', '--sampler', default='euler', choices=samplers)
    return parser.parse_args()

def img_out(video):
    video = (video + 1.) / 2.
    video.clamp_(0, 1)
    images = video.permute(1,2,3,0).unbind(dim=0) # list of [h,w,c]
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]  # [f,h,w,c]
    return images

def save_mp4(frames, path):
    writer = imageio.get_writer(path, fps=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

@torch.no_grad()
def main():
    a = get_args(main_args())
    os.makedirs(a.out_dir, exist_ok=True)
    if a.verbose: save_cfg(a, a.out_dir)

    def genvid(videoin, model, W, H, strength=a.strength, out=False): # a
        if model is None:
            return videoin

        a.model = model
        sd = SDfu(a)
        a.seed = sd.seed # to keep it further
        csb, cwb, texts = multiprompt(sd, a.in_txt, a.pretxt, a.postxt, a.num) # [num,b,77,768], [num,b], [..]
        a.unprompt = '' if a.unprompt=='no' else unprompt if a.unprompt is None else ', '.join([unprompt, a.unprompt])
        uc = multiprompt(sd, a.unprompt)[0][0]
        if a.verbose: print('.. model', a.model, '..', a.sampler, '..', a.cfg_scale, '..', strength, '..', sd.seed)

        def genmix(z_, cs, cws):
            if a.cguide: # use noise lerp with cfg scaling (slower)
                video = sd.generate(z_, cs, uc, cws=cws)
            else: # use cond lerp (worse for multi inputs)
                c_ = sum([cws[j] * cs[j] for j in range(len(cs))]).unsqueeze(0)
                video = sd.generate(z_, c_, uc)
            return video

        videout = []
        count = max(len(videoin), len(csb))
        pbar = progbar(count)
        for i in range(count):
            if len(videoin) > 0:
                video = videoin[i % len(videoin)].cuda()
                video = F.interpolate(video, (H, W), mode='bicubic', align_corners=True)
                sd.set_steps(a.steps, strength)
                z_ = sd.img_z(video) # [f,c,h,w]
                z_ = z_.permute(1,0,2,3)[None,:] # [1,c,f,h,w]
            else:
                sd.set_steps(a.steps, 1)
                z_ = sd.rnd_z(H, W, a.frames) # [1,c,f,h,w]

            video = genmix(z_, csb[i % len(csb)], cwb[i % len(cwb)]).squeeze(0) # [c,f,h,w]
            images = img_out(video)

            log = '%03d-%s' % (i, texts[i % len(texts)][:80])
            file_out = '%s-%d-%s-%d-%s.mp4' % (log, a.frames, a.sampler, sd.seed, basename(a.model))
            save_mp4(images, os.path.join(a.out_dir, file_out))
            pbar.upd(log, uprows=2)
            if out:
                videout += [video.cpu().permute(1,0,2,3)]
        return videout

    # Input
    videoin = []
    if a.in_vid is not None and os.path.exists(a.in_vid):
        videolist = file_list(a.in_vid) if os.path.isdir(a.in_vid) else [a.in_vid]
        for path in videolist:
            video = torch.from_numpy(np.stack(imageio.mimread(path, memtest=False))).permute(0,3,1,2) # [f,c,h,w]
            videoin += [video / 127.5 - 1.]

    # Lo res
    videoin = genvid(videoin, a.model, 576, 320, out=True)

    # Hi res
    # torch.cuda.empty_cache()
    # genvid(videoin, a.model_up, 1024, 576)


if __name__ == '__main__':
    main()
