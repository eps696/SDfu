import os, sys
import argparse
import numpy as np
from imageio.v2 import imread, imsave
from skimage.transform import resize as imresize 

import logging
logging.getLogger('xformers').setLevel(logging.ERROR) # shutup triton, before torch!
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'xtra'))
from annotator.util import resize_image
from core.utils import img_list, basename, progbar

cn_methods = ['canny', 'depth', 'deptha', 'pose']

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='canny', choices=cn_methods, help="Annotation method")
parser.add_argument('-i', '--source', default='_in', help="source images")
parser.add_argument('-o', '--out_dir', default='_out', help="output dir")
parser.add_argument('-md', '--model_dir', default='models/control', help='Path to the models directory')
parser.add_argument('-sz', '--size', default=None, help="image size")
parser.add_argument(  '--float', action='store_true', help='Store 16-bit float as 2x8-bit channels')
a = parser.parse_args()

is_mac = torch.backends.mps.is_available() and torch.backends.mps.is_built() # M1/M2 chip?
is_cuda = torch.cuda.is_available()
device = 'mps' if is_mac else 'cuda' if is_cuda else 'cpu'
dtype = torch.float16 if is_cuda or is_mac else torch.float32

if a.type=='deptha':
    from core.utils import mono16_to_dual8
    a.float = True

model_canny = None
def canny(img, l=100, h=200):
    global model_canny
    if model_canny is None:
        from annotator.canny import CannyDetector
        model_canny = CannyDetector()
    return model_canny(img.astype(np.uint8), l, h)

model_midas = None
def midas(img):
    global model_midas
    if model_midas is None:
        from annotator.midas import MidasDetector
        model_midas = MidasDetector()
    return model_midas(resize_image(img))[0]

model_openpose = None
def openpose(img, has_hand=False):
    global model_openpose
    if model_openpose is None:
        from annotator.openpose import OpenposeDetector
        model_openpose = OpenposeDetector()
    return model_openpose(img, has_hand)[0]

model_deptha2 = None
def deptha(img):
    global model_deptha2
    if model_deptha2 is None:
        from annotator.deptha2.dpt import DepthAnythingV2
        model_deptha2 = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
        model_deptha2.load_state_dict(torch.load(os.path.join(a.model_dir, 'annote', f'depth_anything_v2_vitb.pth'), weights_only=True, map_location='cpu'))
        model_deptha2 = model_deptha2.to(device).eval()
    depth = np.expand_dims(model_deptha2.infer_image(img[:,:,::-1] / 255., 518, bgr=False), axis=0)
    return (depth - depth.min()) / (depth.max() - depth.min()) # [0..1] [1,h,w]

def fixshape(img):
    if len(img.shape) < 3:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    return img[:,:,:3]

annot_dict = {'depth': midas, 'deptha': deptha, 'pose': openpose, 'canny': canny}

def main():
    os.makedirs(a.out_dir, exist_ok=True)
    im_list = img_list(a.source) if os.path.isdir(a.source) else [a.source]
    
    annot_fn = annot_dict[a.type]
    annot_fn(np.zeros((256,256,3))) # warmup model

    pbar = progbar(len(im_list))
    for im_path in im_list:
        img = imread(im_path)
        img = fixshape(img)

        if a.size is not None:
            img = imresize(img, [int(s) for s in a.size.split('-')[::-1]], preserve_range=True, anti_aliasing=True)

        annotmap = annot_fn(img)

        if a.float:
            annotmap = mono16_to_dual8(annotmap)
            fname = os.path.join(a.out_dir, basename(im_path)+'.png')
        else:
            annotmap = fixshape(annotmap)
            fname = os.path.join(a.out_dir, os.path.basename(im_path))
        annotmap = np.clip(annotmap, 0, 255).astype(np.uint8)
        imsave(fname, annotmap, quality=95)

        pbar.upd()


if __name__ == "__main__":
    main()
