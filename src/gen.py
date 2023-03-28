
import os

import torch

from util.sdsetup import SDfu
from util.args import main_args, samplers
from util.text import read_txt, multiprompt
from util.utils import load_img, save_img, calc_size, isok, isset, img_list, basename, progbar

def get_args(parser):
    # override
    parser.add_argument('-sm', '--sampler', default='pndm', choices=samplers)
    parser.add_argument('-n',  '--num',     default=1, type=int)
    return parser.parse_args()

@torch.no_grad()
def main():
    a = get_args(main_args())
    sd = SDfu(a)

    posttxt = basename(a.in_txt) if isset(a, 'in_txt') and os.path.exists(a.in_txt) else ''
    postimg = basename(a.in_img) if isset(a, 'in_img') and os.path.isdir(a.in_img)  else ''
    if isok(posttxt) or isok(postimg):
        a.out_dir = os.path.join(a.out_dir, posttxt + '-' + postimg) + '-' + a.model
    os.makedirs(a.out_dir, exist_ok=True)

    size = None if not isset(a, 'size') else calc_size(a.size, a.model, a.verbose) 
    if a.verbose: print('.. model', a.model, '..', a.sampler, '..', a.cfg_scale, '..', a.strength)

    count = 0
    if isset(a, 'in_txt'):
        cs, texts = multiprompt(sd, a.in_txt, a.pretxt, a.postxt, a.parens)
        if a.num > 1: cs, texts = cs * a.num, texts * a.num
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

            if sd.depthmod:
                gendict = sd.prep_depth(init_image)
                z_ = sd.img_z(init_image)
            elif isset(a, 'mask'):
                log += ' / %s' % os.path.basename(masks[i % len(masks)])
                gendict = sd.prep_mask(masks[i % len(masks)], img_path, init_image)
                z_ = sd.rnd_z(H, W) if sd.inpaintmod else sd.img_z(init_image)
            else: # standard img2img
                gendict = {}
                z_ = sd.img_z(init_image)

            images = sd.generate(z_, c_, **gendict)

        else: # txt2img, full sampler
            if a.num > 1: 
                a.seed = i
            file_out = '%s-m%s-%s-%d' % (log, a.model, a.sampler, a.seed)
            W, H = [sd.res]*2 if size is None else size
            z_ = sd.rnd_z(H, W)
            images = sd.generate(z_, c_)

        outcount = images.shape[0]
        if outcount > 1:
            for j in range(outcount):
                save_img(images[j], j, a.out_dir, filepath=file_out + '-%02d.jpg' % j)
        else:
            save_img(images[0], 0, a.out_dir, filepath=file_out + '.jpg')
        pbar.upd(log, uprows=2)

if __name__ == '__main__':
    main()
