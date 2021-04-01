import cv2
import numpy as np
import os
import torch.utils.data as dutils

from utils.keypoints import dset_to_body_model, get_part_idxs
import torch
import torchvision
import glob
import h5py
from  loguru import logger
from torchvision.transforms import Normalize
import utils as func
# from  utils import util
import smplx
from utils import augmentation
class InstaRand(dutils.Dataset):
    def __init__(self, cfg):
        super(InstaRand, self).__init__()
        self.data_path = cfg['npz_dir']
        self.root_dir = cfg['imgs_dir']
        self.data_path = cfg['npz_dir']
        data = np.load(self.data_path, allow_pickle=True)
        data = {key: data[key] for key in data.keys()}
        
        self.imgs_name = data['imgname']
        self.img_size = 256
        self.blur_dir = '/media/josh/Data/Datasets/InstaX/blur/'
        self.sharp_dir = '/media/josh/Data/Datasets/InstaX/sharp/'
        
        self.preprocess = augmentation.Preprocessing_data()
    def __len__(self):
        return len(self.imgs_name)
    def __getitem__(self, index):
        item = {}
        # img_fn = os.path.join(self.root_dir, self.imgs_name[index])
        img_fn = os.path.join(self.blur_dir, self.imgs_name[index])
        #get keypoints 
        try:
            img = cv2.imread(img_fn)[:,:,::-1].copy()
        except TypeError:
            print(img_fn)
        orig_shape = np.array(img.shape)[:2]
        #to cover blur region we add more reg value
        
        shrp_dir = os.path.join(self.sharp_dir, self.imgs_name[index])
        try:
            sh_img = cv2.imread(shrp_dir)[:,:,::-1].copy()
            orig_shape = np.array(sh_img.shape)[:2]
        except TypeError:
            print(shrp_dir)
        
        pn = np.ones(3) 
        rot = 0            
        sc = 1  
        bl_img, sh_img = self.preprocess.rgb_preprocessing_rand(img, sh_img)
        bl_img = torch.from_numpy(bl_img).float()
        sh_img = torch.from_numpy(sh_img).float()

        item['img'] = self.preprocess.norm_processing_deblur(bl_img)
        item['sharp_img'] = self.preprocess.norm_processing_deblur(sh_img)
        item['orig_shape'] = orig_shape
        item['dset_name'] = 'own_rand'
        
        return item