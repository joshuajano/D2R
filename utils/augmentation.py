import numpy as np
import cv2
import math
import utils as func
from utils import common
from utils import constants
from utils import keypoints
import torch
import albumentations as albu
from torchvision.transforms import Normalize
import random
class Pixel_augm(object):
    def __init__(self):
        self.augs = albu.Compose([
            albu.OneOf([
                albu.CLAHE(clip_limit=2),
                albu.IAASharpen(),
                albu.IAAEmboss(),
                albu.RandomBrightnessContrast(),
                albu.RandomGamma()
            ], p=0.5),
            albu.OneOf([
                albu.RGBShift(),
                albu.HueSaturationValue(),
            ], p=0.5)
        ])
        
        self.pipeline = albu.Compose(self.augs, additional_targets={'target': 'image'})
    def augm_img(self, blur, sharp):
        agm_im = self.pipeline(image=blur, target=sharp)
        # Agm_im = self.augs(image = img)
        return agm_im['image'], agm_im['target'] 
        # Agm_im = self.augs(image = img)
        # return Agm_im['image']

SIGN_FLIP = torch.tensor([6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17,
                          12, 13, 14, 18, 19, 20, 24, 25, 26, 21, 22, 23, 27,
                          28, 29, 33, 34, 35, 30, 31, 32,
                          36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51,
                          52, 53, 48, 49, 50, 57, 58, 59, 54, 55, 56, 63, 64,
                          65, 60, 61, 62],
                         dtype=torch.long) - 3
SIGN_FLIP = SIGN_FLIP.detach().numpy()

class Preprocessing_data(object):
    def __init__(self):
        super(Preprocessing_data, self).__init__()
        self.augm = Pixel_augm()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # mean = [0.5, 0.5, 0.5]
        # std = [0.5, 0.5, 0.5]
        self.normalize = Normalize(mean=mean, std=std)
        self.normalize_deblur = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.random_augm = self.get_transforms(256)
    def get_aug_config(self):
        scale_factor = 0.25
        rot_factor = 30
        color_factor = 0.2
        scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
        rot = np.clip(np.random.randn(), -2.0,
                      2.0) * rot_factor if random.random() <= 0.6 else 0
        c_up = 1.0 + color_factor
        c_low = 1.0 - color_factor
        color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
        if np.random.uniform() <= 0.6:
            rot = 0
        # rot = 0
        # scale = 1
        return scale, rot, color_scale
    def augm_color(self, img, pn):
        img = np.clip(img * pn[None,None,:], 0, 255)
        return img
    def rgb_preprocessing_deblur(self, blur, sharp, center, scale, rot, flip):
        a_bl, a_sh = self.augm.augm_img(blur, sharp)
        a_bl = a_bl.astype(np.float32)
        a_sh = a_sh.astype(np.float32)
        a_bl = common.crop(a_bl, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        a_sh = common.crop(a_sh, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        a_bl = np.transpose(a_bl.astype('float32'),(2,0,1))/255.0
        a_sh = np.transpose(a_sh.astype('float32'),(2,0,1))/255.0
        return a_bl, a_sh
    def rgb_preprocessing(self, blur, sharp, center, scale, rot, flip, pn, filename= 'ok'):
        #color augmentation
        a_bl = self.augm_color(blur, pn)
        a_sh = self.augm_color(sharp, pn)

        # a_bl, a_sh = self.augm.augm_img(blur, sharp)
        a_bl = a_bl.astype(np.float32)
        a_sh = a_sh.astype(np.float32)
        
        a_bl = common.crop(a_bl, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        a_sh = common.crop(a_sh, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # if flip:
        #     img = flip_img(img)
        a_bl = np.transpose(a_bl.astype('float32'),(2,0,1))/255.0
        a_sh = np.transpose(a_sh.astype('float32'),(2,0,1))/255.0
        return a_bl, a_sh
    def rgb_preprocessing_rand(self, blur, sharp):
        a_bl, a_sh = self.random_augm(blur, sharp)
        # a_bl, a_sh = self.augm.augm_img(blur, sharp)
        a_bl = a_bl.astype(np.float32)
        a_sh = a_sh.astype(np.float32)
        
        # if flip:
        #     img = flip_img(img)
        a_bl = np.transpose(a_bl.astype('float32'),(2,0,1))/255.0
        a_sh = np.transpose(a_sh.astype('float32'),(2,0,1))/255.0
        return a_bl, a_sh
    def get_transforms(self, size: int, scope: str = 'strong', crop='random'):
        augs = {'strong': albu.Compose([albu.HorizontalFlip(),
                                        albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=20, p=.4),
                                        albu.ElasticTransform(),
                                        albu.OpticalDistortion(),
                                        albu.OneOf([
                                            albu.CLAHE(clip_limit=2),
                                            albu.IAASharpen(),
                                            albu.IAAEmboss(),
                                            albu.RandomBrightnessContrast(),
                                            albu.RandomGamma()
                                        ], p=0.5),
                                        albu.OneOf([
                                            albu.RGBShift(),
                                            albu.HueSaturationValue(),
                                        ], p=0.5),
                                        ]),
                'weak': albu.Compose([albu.HorizontalFlip(),
                                      ]),
                'geometric': albu.OneOf([albu.HorizontalFlip(always_apply=True),
                                         albu.ShiftScaleRotate(always_apply=True),
                                         albu.Transpose(always_apply=True),
                                         albu.OpticalDistortion(always_apply=True),
                                         albu.ElasticTransform(always_apply=True),
                                         ])
                }

        aug_fn = augs[scope]
        geo_fn = augs['geometric']
        crop_fn = {'random': albu.RandomCrop(size, size, always_apply=True),
                   'center': albu.CenterCrop(size, size, always_apply=True)}[crop]
        pad = albu.PadIfNeeded(size, size)

        pipeline = albu.Compose([aug_fn, geo_fn, crop_fn, pad], additional_targets={'target': 'image'})

        def process(a, b):
            r = pipeline(image=a, target=b)
            return r['image'], r['target']

        return process
    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = common.transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        if f:
            kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp
    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S
    def norm_processing(self, img):
        norm_img = self.normalize(img)
        return norm_img
    def norm_processing_deblur(self, img):
        norm_img = self.normalize_deblur(img)
        return norm_img

        # in-plane rotation
        # flip the x coordinates
        # convert to normalized coordinates
        # flip the x coordinates

# in the rgb image we add pixel noise in a channel-wise manner
# img[:,:,0] = np.minimum(255.0, np.maximum(0.0, img[:,:,0]*pn[0]))
# img[:,:,1] = np.minimum(255.0, np.maximum(0.0, img[:,:,1]*pn[1]))
# img[:,:,2] = np.minimum(255.0, np.maximum(0.0, img[:,:,2]*pn[2]))









# def rot_aa(aa, rot):
#     """Rotate axis angle parameters."""
#     # pose parameters
#     R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
#                   [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
#                   [0, 0, 1]])
#     # find the rotation of the body in camera frame
#     per_rdg, _ = cv2.Rodrigues(aa)
#     # apply the global rotation to the global orientation
#     resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
#     aa = (resrot.T)[0]
#     return aa
# def flip_img(img):
#     """Flip rgb images or masks.
#     channels come last, e.g. (256,256,3).
#     """
#     img = np.fliplr(img)
#     return img

# def flip_kp(kp):
#     TO_REMOVE = 1
#     flipped_keypoints = kp.copy()
#     # flipped_keypoints[:, 0] = (constants.IMG_RES - flipped_keypoints[:, 0] - TO_REMOVE)
#     if kp.shape[0] ==24:
#         flipped_parts = constants.J24_FLIP_PERM
#         flipped_keypoints = flipped_keypoints[flipped_parts]
#         flipped_keypoints[:,0] = - flipped_keypoints[:,0]
#     else:
#         flipped_keypoints = flipped_keypoints[keypoints.FLIP_INDS]
#     return flipped_keypoints

# def augm_params(noise_factor, rot_factor, scale_factor):
#     """Get augmentation parameters."""
#     flip = 0            # flipping
#     pn = np.ones(3)  # per channel pixel-noise
#     rot = 0            # rotation
#     sc = 1   
#     if np.random.uniform() <= 0.5:
#         flip = 1
#     pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)

#     rot = min(2* rot_factor, max(-2* rot_factor, np.random.randn() * rot_factor))

#     sc = min(1 + scale_factor, max(1 - scale_factor, np.random.randn() * scale_factor + 1))
#     if np.random.uniform() <= 0.6:
#         rot = 0
#     return flip, pn, rot, sc

# def rgb_preprocessing(rgb_img, center, scale, rot, flip, pn):
#     img = func.crop(rgb_img, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
#     if flip:
#         img = flip_img(img)
#     # in the rgb image we add pixel noise in a channel-wise manner
#     img[:,:,0] = np.minimum(255.0, np.maximum(0.0, img[:,:,0]*pn[0]))
#     img[:,:,1] = np.minimum(255.0, np.maximum(0.0, img[:,:,1]*pn[1]))
#     img[:,:,2] = np.minimum(255.0, np.maximum(0.0, img[:,:,2]*pn[2]))
#     img = np.transpose(img.astype('float32'),(2,0,1))/255.0
#     return img

# def j2d_processing(kp, center, scale, r, f):
#     """Process gt 2D keypoints and apply all augmentation transforms."""
#     nparts = kp.shape[0]
#     for i in range(nparts):
#         kp[i,0:2] = func.transform(kp[i,0:2]+1, center, scale, 
#                               [constants.IMG_RES, constants.IMG_RES], rot=r)
#     # convert to normalized coordinates
#     kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
#     # flip the x coordinates
#     if f:
#         kp = flip_kp(kp)
#     kp = kp.astype('float32')
#     return kp

# def j3d_processing(S, r, f):
#     """Process gt 3D keypoints and apply all augmentation transforms."""
#     # in-plane rotation
#     rot_mat = np.eye(3)
#     if not r == 0:
#         rot_rad = -r * np.pi / 180
#         sn,cs = np.sin(rot_rad), np.cos(rot_rad)
#         rot_mat[0,:2] = [cs, -sn]
#         rot_mat[1,:2] = [sn, cs]
#     S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
#     # flip the x coordinates
#     if f:
#         S = flip_kp(S)
#     S = S.astype('float32')
#     return S
# def pose_smpl_processing(pose, r, f ):
#     pose[:3] = rot_aa(pose[:3], r)
#     """Flip pose.
#     The flipping is based on SMPL parameters.
#     """
#     if f:
#         flipped_parts = constants.SMPL_POSE_FLIP_PERM
#         pose = pose[flipped_parts]
#         # we also negate the second and the third dimension of the axis-angle
#         pose[1::3] = -pose[1::3]
#         pose[2::3] = -pose[2::3]
#     return pose
# def pose_body_processing(pose, r, f ):
#     if pose.shape[0] == 1:
#         pose = pose[0]
#     pose = pose.numpy()
#     pose[:3] = rot_aa(pose[:3], r)
#     pose = torch.from_numpy(pose) 
#     if f:
#         dim_flip = dim_flip = torch.tensor([1, -1, -1], dtype=pose.dtype)
#         pose = (pose.reshape(-1)[SIGN_FLIP].reshape(21, 3) * dim_flip).reshape(21 * 3)
    
#     return pose