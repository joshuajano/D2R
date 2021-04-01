import sys
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn

from loguru import logger
import torch.nn.functional as F

DEFAULT_FOCAL_LENGTH = 5000
def build_cam_proj():
    camera_scale_func = F.softplus
    mean_scale = 0.9
    mean_scale = np.log(np.exp(mean_scale) - 1)
    camera_mean = torch.tensor([mean_scale, 0.0, 0.0], dtype=torch.float32)
    camera = WeakPerspectiveCamera()
    camera_param_dim = 3
    return {
        'camera': camera,
        'mean': camera_mean,
        'scale_func': camera_scale_func,
        'dim': camera_param_dim
    }
class WeakPerspectiveCamera(nn.Module):
    ''' Scaled Orthographic / Weak-Perspective Camera
    '''

    def __init__(self):
        super(WeakPerspectiveCamera, self).__init__()

    def forward(
        self,
        points,
        scale,
        translation
    ):
        ''' Implements the forward pass for a Scaled Orthographic Camera

            Parameters
            ----------
                points: torch.tensor, BxNx3
                    The tensor that contains the points that will be projected.
                    If not in homogeneous coordinates, then
                scale: torch.tensor, Bx1
                    The predicted scaling parameters
                translation: torch.tensor, Bx2
                    The translation applied on the image plane to the points
            Returns
            -------
                projected_points: torch.tensor, BxNx2
                    The points projected on the image plane, according to the
                    given scale and translation
        '''
        assert translation.shape[-1] == 2, 'Translation shape must be -1x2'
        assert scale.shape[-1] == 1, 'Scale shape must be -1x1'

        projected_points = scale.view(-1, 1, 1) * (
            points[:, :, :2] + translation.view(-1, 1, 2))
        return projected_points