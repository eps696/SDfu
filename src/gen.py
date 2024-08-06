
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch

from core.sdsetup import SDfu, device
from core.args import main_args, samplers
from core.text import read_txt, multiprompt
from core.utils import load_img, save_img, calc_size, isok, isset, img_list, basename, progbar, save_cfg

def get_args(parser):
    # override
    parser.add_argument('-sm', '--sampler', default='pndm', choices=samplers)
    return parser.parse_args()

@torch.no_grad()
def main():
    a = get_args(main_args())
    sd = SDfu(a)
    a = sd.a

    a.model = basename(a.model)
    a.seed = sd.seed
    posttxt = basename(a.in_txt) if isset(a, 'in_txt') and os.path.exists(a.in_txt) else ''
    postimg = basename(a.in_img) if isset(a, 'in_img') and os.path.isdir(a.in_img)  else ''
    if isok(posttxt) or isok(postimg):
        a.out_dir = os.path.join(a.out_dir, posttxt + '-' + postimg) + '-' + a.model
    os.makedirs(a.out_dir, exist_ok=True)
    if a.verbose: save_cfg(a, a.out_dir)

    size = None if not isset(a, 'size') else calc_size(a.size)
    if a.verbose: print('.. model', a.model, '..', a.sampler, '..', a.cfg_scale, '..', a.strength, '..', sd.seed)

    uc = multiprompt(sd, a.unprompt)[0][0]
    if isset (a, 'in_txt'):
        csb, cwb, texts = multiprompt(sd, a.in_txt, a.pretxt, a.postxt, a.num) # [num,b,77,768], [num,b], [..]
    else:
        csb, cwb, texts = uc[None], torch.tensor([[1.]], device=device).half(), ['']
    count = len(csb)

    img_conds = []
    if isset(a, 'img_ref'):
        imrefs, iptypes, allrefs = a.img_ref.split('+'), a.ip_type.split('+'), a.allref.split('+')
        refcount = max([len(imrefs), len(iptypes), len(allrefs)])
        for i in range(refcount):
            img_conds += [sd.img_cus(imrefs[i % len(imrefs)], sd.ips.index(iptypes[i % len(iptypes)]), allref = 'y' in allrefs[i % len(allrefs)].lower())]
            count = max(count, len(img_conds[-1]))

    if isset(a, 'in_img'):
        assert os.path.exists(a.in_img), "!! Image(s) %s not found !!" % a.in_img
        img_paths = img_list(a.in_img) if os.path.isdir(a.in_img) else [a.in_img]
        count = max(count, len(img_paths))
    if isset(a, 'mask'):
        masks = img_list(a.mask) if os.path.isdir(a.mask) else read_txt(a.mask)

    cn_imgs = []
    if sd.use_cnet and isset(a, 'control_img'):
        control_imgs = a.control_img.split('+')[:len(sd.cns)]
        assert len(control_imgs) == len(sd.cns), "Number of control images %d and models %d must match" % (len(control_imgs), len(sd.cns))
        for cn_img in control_imgs:
            assert os.path.exists(cn_img), f"!! ControlNet image {cn_img} not found !!"
            cn_imgs += [img_list(cn_img) if os.path.isdir(cn_img) else [cn_img]]
            count = max(count, len(cn_imgs[-1]))
        if len(cn_imgs) != len(sd.cns): print('!! %d CNet inputs != %d CNet models !!' % (len(cn_imgs), len(sd.cns)))

    def genmix(z_, cs, cws, **gendict):
        if a.cguide: # use noise lerp with cfg scaling (slow!)
            images = sd.generate(z_, cs, uc, cws=cws, **gendict)
        else: # use cond lerp (worse for multi inputs)
            c_ = sum([cws[j] * cs[j] for j in range(len(cs))]).unsqueeze(0)
            images = sd.generate(z_, c_, uc, **gendict)
        return images
    
    pbar = progbar(count)
    for i in range(count):
        gendict = {}
        log = texts[i % len(texts)][:80] if len(texts) > 0 else ''

        if len(img_conds) > 0:
            gendict['c_img'] = [imcond[i % len(imcond)] for imcond in img_conds]

        if isset(a, 'in_img'):
            img_path = img_paths[i % len(img_paths)]
            file_out = basename(img_path) if len(img_paths) == count else '%06d' % i
            log += ' .. %s' % os.path.basename(img_path)
            if len(img_paths) > 1 or i==0:
                init_image, (W,H) = load_img(img_path, size)
                if sd.depthmod:
                    gendict = {**gendict, **sd.prep_depth(init_image)}
                    z_ = sd.img_z(init_image)
                elif isset(a, 'mask'): # inpaint
                    log += ' / %s' % os.path.basename(masks[i % len(masks)])
                    gendict = {**gendict, **sd.prep_mask(masks[i % len(masks)], img_path, init_image)}
                    z_ = sd.rnd_z(H, W) if sd.inpaintmod else sd.img_z(init_image)
                elif isset(a, 'img_scale'): # instruct pix2pix
                    ilat = sd.img_lat(init_image) / sd.vae.config.scaling_factor
                    gendict['ilat'] = torch.cat([torch.zeros_like(ilat)] + [ilat] * (csb.shape[1]+1), dim=0) # [unlat, ilat, ilat, ..]
                    z_ = sd.rnd_z(H, W)
                else: # standard img2img
                    z_ = sd.ddim_inv(sd.img_lat(init_image), uc) if a.sampler=='ddim' else sd.img_z(init_image)

        else: # txt2img
            file_out = '%03d-%s-m%s-%s-%d' % (i, log, a.model, a.sampler, sd.seed)
            W, H = [sd.res]*2 if size is None else size
            z_ = sd.rnd_z(H, W)

        if len(cn_imgs) > 0:
            gendict['cnimg'] = [(load_img(ci[i % len(ci)], (W,H), dual8b = ('deptha' in sd.cns[i]))[0] + 1) / 2 for i, ci in enumerate(cn_imgs)]

        images = genmix(z_, csb[i % len(csb)], cwb[i % len(cwb)], **gendict)

        postfix = a.load_custom or a.load_lora or a.load_token
        if postfix is not None: file_out += '-%s' % basename(postfix)
        outcount = images.shape[0]
        if outcount > 1:
            for j in range(outcount):
                save_img(images[j], j, a.out_dir, filepath=file_out + '-%02d.jpg' % j)
        else:
            save_img(images[0], 0, a.out_dir, filepath=file_out + '.jpg')
        pbar.upd(log, uprows=2)

if __name__ == '__main__':
    main()
