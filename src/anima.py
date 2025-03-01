
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import imageio

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from core.sdsetup import SDfu
from core.args import main_args, samplers
from core.text import multiprompt
from core.utils import img_list, load_img, framestack, basename, save_cfg, calc_size, isset

def get_args(parser):
    parser.add_argument('-iv', '--in_vid',  default=None, help='input video or frame sequence (directory with images)')
    parser.add_argument('-vf', '--frames',  default=None, type=int, help="Frame count for generated video")
    parser.add_argument('-fs', '--fstep',   default=None, type=int, help="number of frames for each interpolation step")
    parser.add_argument('-cf', '--ctx_frames',  default=16, type=int, help="frame count to process at once with sliding window sampling")
    parser.add_argument('-ad', '--animdiff', default='models/anima', help="path to the Motion Adapter model")
    parser.add_argument(       '--curve',   default='bezier', help="Interpolating curve: bezier, parametric, custom or linear")
    parser.add_argument(       '--loop',    action='store_true')
    # override
    parser.add_argument('-sm', '--sampler', default='euler', choices=samplers)
    parser.add_argument('-b',  '--batch',   default=1, type=int, choices=[1])
    parser.add_argument('-s',  '--steps',   default=23, type=int, help="number of diffusion steps")
    parser.add_argument('-cg', '--cguide',  default=False)
    return parser.parse_args()

def img_out(video):
    video = (video + 1.) / 2.
    video.clamp_(0, 1)
    images = video.movedim(0,-1).unbind(dim=0) # list of [h,w,c]
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images] # list of [h,w,c]
    return images

@torch.no_grad()
def main():
    a = get_args(main_args())
    sd = SDfu(a)
    a = sd.a
    os.makedirs(a.out_dir, exist_ok=True)
    gendict = {}

    uc = multiprompt(sd, a.unprompt)[0][0]
    if isset (a, 'in_txt'):
        csb, cwb, texts = multiprompt(sd, a.in_txt, a.pretxt, a.postxt, a.num) # [num,b,77,768], [num,b], [..]
    else:
        csb, cwb, texts = uc.unsqueeze(0), torch.tensor([[1.]], device=sd.device, dtype=sd.dtype), ['']
    count = len(csb)

    img_conds = []
    if isset(a, 'img_ref'):
        imrefs, iptypes, allrefs = a.img_ref.split('+'), a.ip_type.split('+'), a.allref.split('+')
        refcount = max([len(imrefs), len(iptypes), len(allrefs)])
        for i in range(refcount):
            img_conds += [sd.img_cus(imrefs[i % len(imrefs)], sd.ips.index(iptypes[i % len(iptypes)]), allref = 'y' in allrefs[i % len(allrefs)].lower())]
            count = max(count, len(img_conds[-1]))

    if isset(a, 'in_img') and os.path.isdir(a.in_img) and not isset(a, 'in_vid'):
        a.in_vid = a.in_img # probably confused invid and inimg
    if isset(a, 'in_vid'):
        assert os.path.exists(a.in_vid), "Not found %s" % a.in_vid
        if os.path.isdir(a.in_vid):
            frames = [imageio.imread(path) for path in img_list(a.in_vid)]
        else: 
            frames = imageio.mimread(a.in_vid, memtest=False)
        if isset(a, 'frames'): frames = frames[:a.frames]
        a.frames = len(frames)
        video = torch.from_numpy(np.stack(frames)).movedim(3,1).to(sd.device, dtype=sd.dtype) / 127.5 - 1. # [f,c,h,w]
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

    cs = sum([csb[:,j] * cwb[:,j,None,None] for j in range(csb.shape[1])]) # [num,77,768]
    cs_frames = framestack(cs, a.frames, a.curve, a.loop) # [f,77,768]
    uc_frames = uc.repeat(a.frames,1,1)
    if len(img_conds) > 0:
        gendict['c_img'] = [framestack(img_cond, a.frames, a.curve, a.loop, rejoin=True) for img_cond in img_conds]

    if sd.use_cnet and isset(a, 'control_img'):
        cn_imgs_all = []
        control_imgs = a.control_img.split('+')[:len(sd.cns)]
        assert len(control_imgs) == len(sd.cns), "Number of control images and models must match"
        for cn_img, cnet_mod in zip(control_imgs, sd.cns):
            assert os.path.exists(cn_img), f"!! ControlNet image(s) {cn_img} not found !!"
            dual8b = basename(cnet_mod) == 'deptha' # 16bit
            if os.path.isdir(cn_img):
                cn_imgs = [load_img(path, (W,H), dual8b=dual8b)[0] for path in img_list(cn_img)]
            else:
                cn_imgs = [load_img(cn_img, (W,H), dual8b=dual8b)[0]]
            if isset(a, 'frames'):
                cn_imgs = cn_imgs[:a.frames]
            else:
                a.frames = max(a.ctx_frames, len(cn_imgs))
            cn_imgs = torch.cat(cn_imgs) / 2. + 0.5  # [0..1] [n,c,h,w]
            if len(cn_imgs) == 1:
                cn_imgs = cn_imgs.repeat(a.frames,1,1,1)
            elif len(cn_imgs) < a.frames:
                print(f"!! Not enough ControlNet images for model {sd.cns[i]}: {len(cn_imgs)}, total frame count {a.frames} !!"); exit()
            cn_imgs_all.append(cn_imgs)
        gendict['cnimg'] = cn_imgs_all

    if a.verbose: 
        print('.. frames', a.frames, '.. model', a.model, '..', a.sampler, '..', '%dx%d' % (W,H), '..', a.cfg_scale, '..', a.strength, '..', sd.seed)
        save_cfg(a, a.out_dir)
    
    if isset(a, 'in_vid'):
        if list(video.shape[-2:]) != [H, W]:
            video = F.pad(video, (0, W - video.shape[-1], 0, H - video.shape[-2]), mode='reflect')
        sd.set_steps(a.steps, a.strength)
        z_ = sd.img_z(video) # [f,c,h,w]
        z_ = z_.movedim(1,0).unsqueeze(0) # [1,c,f,h,w]
    else:
        sd.set_steps(a.steps, 1)
        z_ = sd.rnd_z(H, W, a.frames) # [1,c,f,h,w]

    video = sd.generate(z_, cs_frames, uc_frames, **gendict).squeeze(0) # [c,f,h,w]

    images = img_out(video)
    for i, image in enumerate(images):
        if isset(a, 'in_vid') and list(image.shape[:2]) != size: # if input padded
            image = image[:size[0], :size[1]]
        imageio.imsave(os.path.join(a.out_dir, '%04d.jpg' % i), image)


if __name__ == '__main__':
    main()
