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
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MiDaSInference(model_type="dpt_hybrid").to(self.device)

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().to(self.device)
            if self.device == 'mps':
                # Handle tensor expansion in MPS-friendly way
                image_depth = image_depth.unsqueeze(0)
                image_depth = image_depth.permute(0, 3, 1, 2).contiguous()
            else:
                image_depth = image_depth.unsqueeze(0).permute(0, 3, 1, 2)
            depth = self.model(image_depth)[0]
            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            return depth_pt
