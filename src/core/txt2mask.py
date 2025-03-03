'''Makes available the Txt2Mask class, which assists in the automatic
assignment of masks via text prompt using clipseg.

Here is typical usage:
    
    from txt2mask import Txt2Mask # SegmentedGrayscale
    from PIL import Image

    txt2mask = Txt2Mask(self.device)
    segmented = txt2mask.segment(Image.open('/path/to/img.png'),'a bagel')
    
    # this will return a grayscale Image of the segmented data
    grayscale = segmented.to_grayscale()

    # this will return a semi-transparent image in which the
    # selected object(s) are opaque and the rest is at various
    # levels of transparency
    transparent = segmented.to_transparent()

    # this will return a masked image suitable for use in inpainting:
    mask = segmented.to_mask(threshold=0.5)

The threshold used in the call to to_mask() selects pixels for use in
the mask that exceed the indicated confidence threshold. Values range
from 0.0 to 1.0. The higher the threshold, the more confident the
algorithm is. In limited testing, I have found that values around 0.5
work fine.
'''

import os, sys
import numpy as  np
from einops import rearrange, repeat
from PIL import Image, ImageOps

import torch
from torchvision import transforms

CLIP_VERSION = 'ViT-B/16'
CLIPSEG_SIZE = 352

RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../xtra'))

from clipseg.clipseg import CLIPDensePredT

class SegmentedGrayscale(object):
    def __init__(self, image:Image, heatmap:torch.Tensor):
        self.heatmap = heatmap
        self.image = image
        
    def to_grayscale(self,invert:bool=False)->Image:
        return self._rescale(Image.fromarray(np.uint8(255 - self.heatmap * 255 if invert else self.heatmap * 255)))

    def to_mask(self,threshold:float=0.5)->Image:
        discrete_heatmap = self.heatmap.lt(threshold).int()
        return self._rescale(Image.fromarray(np.uint8(discrete_heatmap*255),mode='L'))

    def to_transparent(self,invert:bool=False)->Image:
        transparent_image = self.image.copy()
        # For img2img, we want the selected regions to be transparent,
        # but to_grayscale() returns the opposite. Thus invert.
        gs = self.to_grayscale(not invert)
        transparent_image.putalpha(gs)
        return transparent_image

    # unscales and uncrops the 352x352 heatmap so that it matches the image again
    def _rescale(self, heatmap:Image)->Image:
        size = self.image.width if (self.image.width > self.image.height) else self.image.height
        resized_image = heatmap.resize((size,size), resample=RESAMPLE)
        return resized_image.crop((0,0,self.image.width,self.image.height))

class Txt2Mask(object):
    ''' Create new Txt2Mask object. The optional device argument can be one of 'cuda', 'mps' or 'cpu' '''
    def __init__(self, model_path='models/xtra/clipseg/rd64-uni.pth', device='cpu', refined=False):
        # print('>> Initializing clipseg model for text to mask inference')
        self.device = device
        self.model = CLIPDensePredT(version=CLIP_VERSION, reduce_dim=64, complex_trans_conv=refined)
        self.model.eval()
        # initially we keep everything in cpu to conserve space
        self.model.to('cpu')
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')), strict=False)

    @torch.no_grad()
    def segment(self, image, prompt:str) -> SegmentedGrayscale:
        '''
        Given a prompt string such as "a bagel", tries to identify the object in the
        provided image and returns a SegmentedGrayscale object in which the brighter
        pixels indicate where the object is inferred to be.
        '''
        self._to_device(self.device)
        prompts = [prompt]   # right now we operate on just a single prompt at a time

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((CLIPSEG_SIZE, CLIPSEG_SIZE)), # must be multiple of 64...
        ])

        if type(image) is str:
            image = Image.open(image).convert('RGB')

        image = ImageOps.exif_transpose(image)
        img = self._scale_and_crop(image)
        img = transform(img).unsqueeze(0)

        preds = self.model(img.detach().clone().repeat_interleave(len(prompts), dim=0), prompts)[0] # MPS-friendly 
        heatmap = torch.sigmoid(preds[0][0]).cpu()
        self._to_device('cpu')
        return SegmentedGrayscale(image, heatmap)

    def _to_device(self, device):
        self.model.to(device)

    def _scale_and_crop(self, image:Image)->Image:
        scaled_image = Image.new('RGB', (CLIPSEG_SIZE, CLIPSEG_SIZE))
        if image.width > image.height: # width is constraint
            scale = CLIPSEG_SIZE / image.width
        else:
            scale = CLIPSEG_SIZE / image.height
        scaled_image.paste(image.resize((int(scale * image.width), int(scale * image.height)), resample=RESAMPLE),box=(0,0))
        return scaled_image
