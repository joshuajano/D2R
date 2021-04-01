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
from utils import common
import smplx
from utils import augmentation
class Insta(dutils.Dataset):
    def __init__(self, cfg, data_path = '/home/josh/Desktop/Paper/Dataset/spin_dataset/', split = 'train', batch_size = 1):
        super(Insta, self).__init__()
        # args = parse_config()
        self.root_dir = cfg['imgs_dir']
        
        self.binarization = True
        self.use_face_contour = True
        self.num_betas = 10
        self.batch_size = batch_size
        # self.body_model = body_model
        mean = [0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        self.normalize = Normalize(mean=mean, std=std)
        
        #mapping
        self.keyp_format = 'coco25'
        source_idxs, target_idxs = dset_to_body_model(
            dset='openpose25+hands+face',
            model_type='smplx', use_hands=True, use_face=True,
            use_face_contour=self.use_face_contour,
            keyp_format=self.keyp_format)
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
        
        # self.data_path = os.path.join(data_path, 'train_own.npz')
        self.data_path = cfg['npz_dir']
        data = np.load(self.data_path, allow_pickle=True)
        data = {key: data[key] for key in data.keys()}
        self.imgs_name = data['imgname']
        self.keypoints2D = data['openpose'].astype(np.float32)
        self.smplx = data['smplx']
        self.img_size = 256

        self.body_thresh = 0.1
        self.face_thresh = 0.4
        self.hand_thresh=0.2
        self.blur_dir = '/media/josh/Data/Datasets/InstaX/blur/'
        self.sharp_dir = '/media/josh/Data/Datasets/InstaX/sharp/'
        
        self.preprocess = augmentation.Preprocessing_data()
    def __len__(self):
        return len(self.imgs_name)
    def get_keypoints(self, keypoints2d):
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
    def __getitem__(self, index):
        item = {}
        # img_fn = os.path.join(self.root_dir, self.imgs_name[index])
        img_fn = os.path.join(self.blur_dir, self.imgs_name[index])
        smplx_param = self.smplx[index]
        keypoints2d = self.keypoints2D[index]
        output_keypoints2d = self.get_keypoints(keypoints2d)
        #get keypoints 
        keypoints = output_keypoints2d[:, :-1]
        conf = output_keypoints2d[:, -1]
        try:
            img = cv2.imread(img_fn)[:,:,::-1].copy()
            orig_shape = np.array(img.shape)[:2]
        except TypeError:
            print(img_fn)
        orig_shape = np.array(img.shape)[:2]
        #to cover blur region we add more reg value
        
        reg = 20
        bbox = common.keyps_to_bbox(keypoints, conf, reg, img_size=orig_shape)
        split_text = img_fn.split('/')
        shrp_dir = os.path.join(self.sharp_dir, self.imgs_name[index])
        try:
            sh_img = cv2.imread(shrp_dir)[:,:,::-1].copy()
            orig_shape = np.array(sh_img.shape)[:2]
        except TypeError:
            print(shrp_dir)
        # sh_img = 
        full_imgs_bl = img
        full_imgs_sh = sh_img
        
        center, scale, bbox_size = common.bbox_to_center_scale(bbox, dset_scale_factor= self.body_dset_factor)
        flip = 0            
        pn = np.ones(3) 
        rot = 0            
        sc = 1  
        sc, rot, pn  = self.preprocess.get_aug_config()
        bl_img, sh_img = self.preprocess.rgb_preprocessing_deblur(img, sh_img, center, scale * sc, rot, flip)
        bl_img = torch.from_numpy(bl_img).float()
        sh_img = torch.from_numpy(sh_img).float()

        item['img'] = self.preprocess.norm_processing_deblur(bl_img)
        item['sharp_img'] = self.preprocess.norm_processing_deblur(sh_img)
        item['keypoints'] = torch.from_numpy(self.preprocess.j2d_processing(output_keypoints2d, center, sc*scale, rot, flip)).float() 
        item['imgname'] = split_text[5] + '_' + split_text[6] + '_' +  split_text[8]
        #SMPLX parameter
        item['betas'] = torch.from_numpy(smplx_param['shape'][0])
        item['expression'] = torch.from_numpy(smplx_param['expression'][0])
        item['global_orient'] = common.batch_rodrigues(torch.from_numpy( common.pose_processing(smplx_param['global_orient'], rot) ).view(-1, 3)).view(1, 3, 3)
        item['leye_pose'] = common.batch_rodrigues(torch.from_numpy(smplx_param['leye_pose']).view(-1, 3)).view(1, 3, 3)
        item['reye_pose'] = common.batch_rodrigues(torch.from_numpy(smplx_param['reye_pose']).view(-1, 3)).view(1, 3, 3)
        
        item['body_pose'] = common.batch_rodrigues(torch.from_numpy(smplx_param['body_pose'][0]).view(-1, 3))
        item['jaw_pose'] = common.batch_rodrigues(torch.from_numpy(smplx_param['jaw_pose']).view(-1, 3)).view(1, 3, 3)
        item['left_hand_pose'] = common.batch_rodrigues(torch.from_numpy(smplx_param['left_hand_pose']).view(-1, 3))
        item['right_hand_pose'] = common.batch_rodrigues(torch.from_numpy(smplx_param['right_hand_pose']).view(-1, 3))
        if flip:
            temp = item['left_hand_pose']
            item['left_hand_pose'] = item['right_hand_pose'] 
            item['right_hand_pose'] = temp

        item['orig_shape'] = orig_shape
        item['conf'] =   torch.from_numpy(conf)
        item['camera'] = torch.from_numpy(smplx_param['camera_translation'])
        # item['dset_name'] = 'own'
        item['dset_name'] = 'own'
        
        item['j3d'] = torch.zeros(144, 4, dtype=torch.float32)
        
        return item