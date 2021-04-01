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
from utils import augmentation
class Human_36m(dutils.Dataset):
    def __init__(self, cfg, batch_size = 24):
        # self.out_path = '/home/josh/Desktop/Paper/Dataset/Human3.6M/SPIN_extraction/'
        self.out_path = cfg['imgs_dir']
        self.data = np.load(cfg['npz_dir'])
        mean = [0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        self.normalize = Normalize(mean=mean, std=std)
        self.imgs_name = self.data['imgname']
        self.scale = self.data['scale']
        self.center = self.data['center']
        #get 3D GT, and 2D GT
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgs_name), 24, 3))
        
        self.keypoints = keypoints_gt
        self.batch_size = batch_size
        self.img_size = 256
        self.preprocess = augmentation.Preprocessing_data()
        self.use_face_contour = True
        #mapping
        source_idxs, target_idxs = dset_to_body_model(
            dset='h36m',
            model_type='smplx', use_hands=True, use_face=True,
            use_face_contour=self.use_face_contour,
            keyp_format='h36m')
        self.source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.target_idxs = np.asarray(target_idxs, dtype=np.int64)
        idxs_dict = get_part_idxs()

        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        left_hand_idxs = idxs_dict['left_hand']
        right_hand_idxs = idxs_dict['right_hand']
        face_idxs = idxs_dict['face']
        head_idxs = idxs_dict['head']
   
        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.left_hand_idxs = np.asarray(left_hand_idxs)
        self.right_hand_idxs = np.asarray(right_hand_idxs)
        self.face_idxs = np.asarray(face_idxs)
        self.head_idxs = np.asarray(head_idxs)

        self.body_dset_factor = 1.2
        self.head_dset_factor = 2.0
        self.hand_dset_factor = 2.0

        self.body_thresh = 0.1
        self.face_thresh = 0.4
        self.hand_thresh = 0.2
        self.binarization = True
    def __len__(self):
        return len(self.imgs_name)
    def get_keypoints_2d(self, keypoints2d):
        output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,  3], dtype=np.float32)
        output_keypoints2d[self.target_idxs] = keypoints2d[self.source_idxs]
        output_keypoints2d[output_keypoints2d[:, -1] < 0, -1] = 0

        body_conf = output_keypoints2d[self.body_idxs, -1]
            
        left_hand_conf = output_keypoints2d[self.left_hand_idxs, -1]
        right_hand_conf = output_keypoints2d[self.right_hand_idxs, -1]
        
        face_conf = output_keypoints2d[self.face_idxs, -1]
        
        body_conf[body_conf < self.body_thresh] = 0.0
        body_conf[body_conf > self.body_thresh] = 1.0
        left_hand_conf[left_hand_conf < self.hand_thresh] = 0.0
        right_hand_conf[right_hand_conf < self.hand_thresh] = 0.0
        face_conf[face_conf < self.face_thresh] = 0.0
        if self.binarization:
            body_conf = (
                body_conf >= self.body_thresh).astype(
                    output_keypoints2d.dtype)
            left_hand_conf = (
                left_hand_conf >= self.hand_thresh).astype(
                    output_keypoints2d.dtype)
            right_hand_conf = (
                right_hand_conf >= self.hand_thresh).astype(
                    output_keypoints2d.dtype)
            face_conf = (
                face_conf >= self.face_thresh).astype(
                    output_keypoints2d.dtype)
        output_keypoints2d[self.body_idxs, -1] = body_conf
        output_keypoints2d[self.left_hand_idxs, -1] = left_hand_conf
        output_keypoints2d[self.right_hand_idxs, -1] = right_hand_conf
        output_keypoints2d[self.face_idxs, -1] = face_conf
        return output_keypoints2d
    def get_keypoints_3d(self, keypoints3d):
        output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,  4], dtype=np.float32)
        output_keypoints2d[self.target_idxs] = keypoints3d[self.source_idxs]
        output_keypoints2d[output_keypoints2d[:, -1] < 0, -1] = 0

        body_conf = output_keypoints2d[self.body_idxs, -1]
            
        left_hand_conf = output_keypoints2d[self.left_hand_idxs, -1]
        right_hand_conf = output_keypoints2d[self.right_hand_idxs, -1]
        
        face_conf = output_keypoints2d[self.face_idxs, -1]
        
        body_conf[body_conf < self.body_thresh] = 0.0
        body_conf[body_conf > self.body_thresh] = 1.0
        left_hand_conf[left_hand_conf < self.hand_thresh] = 0.0
        right_hand_conf[right_hand_conf < self.hand_thresh] = 0.0
        face_conf[face_conf < self.face_thresh] = 0.0
        if self.binarization:
            body_conf = (
                body_conf >= self.body_thresh).astype(
                    output_keypoints2d.dtype)
            left_hand_conf = (
                left_hand_conf >= self.hand_thresh).astype(
                    output_keypoints2d.dtype)
            right_hand_conf = (
                right_hand_conf >= self.hand_thresh).astype(
                    output_keypoints2d.dtype)
            face_conf = (
                face_conf >= self.face_thresh).astype(
                    output_keypoints2d.dtype)
        output_keypoints2d[self.body_idxs, -1] = body_conf
        output_keypoints2d[self.left_hand_idxs, -1] = left_hand_conf
        output_keypoints2d[self.right_hand_idxs, -1] = right_hand_conf
        output_keypoints2d[self.face_idxs, -1] = face_conf
        return output_keypoints2d
    def test_visual(self, img, gt, conf):
        for i in range(gt.shape[0]):
            if conf[i] <1:
                continue
            else:
                x , y = int(gt[i][0]), int(gt[i][1])
                cv2.circle(img,(x,y), 4, (0,0,255), -1)
                cv2.imwrite('test.png', img)
        
        pass
    def __getitem__(self, index):
        item = {}
        # output_keypoints2d = np.zeros([127 + 17,  3], dtype=np.float32)
        img_fn = self.imgs_name[index]
        img_fn = os.path.join(self.out_path, img_fn)
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints2d = self.keypoints[index].copy()
        output_keypoints2d = self.get_keypoints_2d(keypoints2d) 
        conf_2d = output_keypoints2d[:, -1]
        S = self.pose_3d[index].copy()
        kp3d = self.get_keypoints_3d(S)
        _kp3d = kp3d[:, :3]
        conf = kp3d[:, -1]
        try:
            # img = cv2.imread(img_fn)[:,:,::-1].copy().astype(np.float32)
            img = cv2.imread(img_fn)[:,:,::-1].copy()
        except TypeError:
            print(img_fn)
        orig_shape = np.array(img.shape)[:2]
        #Test to check the location
        # self.test_visual(img, output_keypoints2d[:, :-1], conf)

        
        
        #augmentation 
        # flip, pn, rot, sc = augmentation.augm_params(0.4, 30, 0.25)
        flip = 0            
        pn = np.ones(3) 
        rot = 0            
        sc = 1 
        sc, rot, pn  = self.preprocess.get_aug_config()
        # item['j3d'] = torch.from_numpy(augmentation.j3d_processing(S, rot, flip)).float()
        # item['j3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
        
        item['j3d'] = torch.from_numpy(self.preprocess.j3d_processing(kp3d, rot, flip)).float()
        
        # output_keypoints2d[:24] = keypoints
        
        # img = self.preprocess.rgb_preprocessing(img, center, scale * sc, rot, flip, pn)
        try:
            bl_img, sh_img = self.preprocess.rgb_preprocessing(img, img, center, scale * sc, rot, flip, pn)
        except:
            print(img_fn)
        bl_img = torch.from_numpy(bl_img).float()
        sh_img = torch.from_numpy(sh_img).float()
        
        
        #put into item dict
        # item['j2d'] = torch.from_numpy(self.j2d_processing(keypoints, center, scale, 0, 0)).float()
        # img = self.rgb_processing(img_fn, img, center, scale * sc, rot, flip, pn)

        item['img'] = self.preprocess.norm_processing(bl_img)
        item['sharp_img'] = self.preprocess.norm_processing(sh_img)
        item['imgname'] = img_fn
        item['keypoints'] = torch.from_numpy(self.preprocess.j2d_processing(output_keypoints2d, center, sc*scale, rot, flip)).float()
        # item['keypoints'] = torch.from_numpy(self.j2d_processing(output_keypoints2d, center, sc*scale, rot, flip)).float()
        item['conf'] = torch.from_numpy(conf)
        
        item['betas'] = torch.zeros(10, dtype=torch.float32)
        item['expression'] = torch.zeros(10, dtype=torch.float32)
        item['global_orient'] = torch.zeros(1, 3, 3, dtype=torch.float32)
        item['body_pose'] = torch.zeros(21, 3, 3, dtype=torch.float32)
        item['leye_pose'] = torch.zeros(1, 3, 3, dtype=torch.float32)
        item['reye_pose'] = torch.zeros(1, 3, 3, dtype=torch.float32)
        item['jaw_pose'] = torch.zeros(1, 3, 3, dtype=torch.float32)
        item['jaw_pose'] = torch.zeros(1, 3, 3, dtype=torch.float32)
        item['left_hand_pose'] = torch.zeros(15, 3, 3, dtype=torch.float32)
        item['right_hand_pose'] = torch.zeros(15, 3, 3, dtype=torch.float32)


        item['camera'] = torch.zeros(1, 3).float()
        item['orig_shape'] = orig_shape
        #dataset with body only do not contain expressive  like Human3.6m

        item['dset_name'] = 'h36m'

        return item
