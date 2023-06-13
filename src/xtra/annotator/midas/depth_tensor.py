# Midas Depth Estimation
# From https://github.com/isl-org/MiDaS
# MIT LICENSE

import cv2
import numpy as np
import torch

from einops import rearrange
from .api import MiDaSInference

class MidasDetector:
    def __init__(self):
        self.model = MiDaSInference(model_type="dpt_hybrid").cuda()

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        image_depth = input_image
        with torch.no_grad():
            depth = self.model(image_depth)[0]
            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            return depth_pt
