import cv2
import numpy as np
import os
import torch.utils.data as dutils
import utils as func
from torchvision.transforms import Normalize
import torch
from utils.keypoints import dset_to_body_model, get_part_idxs

class Threedpw(dutils.Dataset):
    def __init__(self, img_size= 256):
        super(Threedpw, self).__init__()
        
        self.data = np.load('3dpw.npz', allow_pickle= True)
        self.imgs_name = self.data['imgname']
        self.smplx = self.data['smplx'] 
        
        self.len_data = len(self.imgs_dir)
        mean = [0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        self.normalize = Normalize(mean=mean, std=std)
        self.img_size =img_size
    def __len__(self):
        return self.len_data
    def rgb_processing(self, img_name, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = func.crop(img_name, rgb_img, center, scale, [self.img_size, self.img_size], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img
    def __getitem__(self, index):
        item = {}
        #Without any augmentation
        flip = 0            
        pn = np.ones(3) 
        rot = 0            
        sc = 1 
        
        # output_keypoints2d = np.zeros([127 + 17,  3], dtype=np.float32)
        img_fn_data = self.imgs_name[index]
        img_fn = img_fn_data
        try:
            img = cv2.imread(img_fn)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(img_fn)
        full_img = img
        orig_shape = np.array(img.shape)[:2]
        
        img = self.rgb_processing(img_fn, img, center, scale * sc, rot, flip, pn)
        img = torch.from_numpy(img).float()
        smplx_param = self.smplx[index]
        vertices = torch.from_numpy(smplx_param.vertices).float()
        # item['full_img'] = full_img
        item['img'] = self.normalize(img)
        item['orig_shape'] = orig_shape
        item['vertices'] = vertices
        # scale = self.scale[index].copy()
        # center = self.center[index].copy()
        # pose = self.pose[index]
        # shape = self.shape[index]
        # item['orig_shape'] = orig_shape
        # item['pose'] = torch.from_numpy(pose).float()  
        # item['shape'] = torch.from_numpy(shape).float()
        # item['gender'] = self.gender[index]

        # #For saving propose
        # item['img_fn'] = img_fn_data
        # item['scale'] = scale
        # item['center'] = center
        # item['pose'] = pose
        # item['shape'] = shape
        return item 