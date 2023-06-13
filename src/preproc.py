import os, sys
import argparse
import numpy as np
from imageio import imread, imsave

sys.path.append(os.path.join(os.path.dirname(__file__), 'xtra'))
from annotator.util import resize_image
from util.utils import img_list, basename, progbar

methods = ['canny', 'depth', 'pose']

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='canny', choices=methods, help="Annotation method")
parser.add_argument('-i', '--source', default='_in', help="source images")
parser.add_argument('-o', '--out_dir', default='_out', help="output dir")
parser.add_argument('-md', '--model_dir', default='models/control', help='Path to the models directory')
a = parser.parse_args()

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

def main():
    os.makedirs(a.out_dir, exist_ok=True)
    im_list = img_list(a.source) if os.path.isdir(a.source) else [a.source]
    
    annot_fn = openpose if a.type == 'pose' else midas if a.type == 'depth' else canny
    annot_fn(np.zeros((256,256,3))) # check model

    pbar = progbar(len(im_list))
    for im_path in im_list:
        img = imread(im_path)
        if len(img.shape) < 3:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = img[:,:,:3]
        
        annotmap = annot_fn(img)

        annotmap = np.clip(annotmap, 0, 255).astype(np.uint8)
        imsave(os.path.join(a.out_dir, os.path.basename(im_path)), annotmap)
        pbar.upd()


if __name__ == "__main__":
    main()
