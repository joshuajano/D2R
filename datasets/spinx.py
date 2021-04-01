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
# from  utils import util
class Spinx(dutils.Dataset):
    def __init__(self):
        data_path = '/home/josh/Desktop/Paper/3DHHM/train_spinx.npz'
        data = np.load(data_path, allow_pickle=True)
        data = {key: data[key] for key in data.keys()}

        self.betas = data['beta']
        self.keypoints2D = data['keypoints2D']
        self.pose = data['pose']
        self.camera = data['camera']
        self.imgname = data['imgname']
        self.centers = data['center']
        self.scales = data['scale']
        self.folds = data['fold']
        source_idxs, target_idxs = dset_to_body_model(
            model_type='smplx', use_hands=True, use_face=True,
            dset='spin', use_face_contour=True)
        self.spin_source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.spin_target_idxs = np.asarray(target_idxs, dtype=np.int64)
        source_idxs, target_idxs = dset_to_body_model(
            model_type='smplx', use_hands=True, use_face=True,
            dset='spin', use_face_contour=True)
        self.spinx_source_idxs = np.asarray(source_idxs, dtype=np.int64)
        self.spinx_target_idxs = np.asarray(target_idxs, dtype=np.int64)     
        self.num_betas = 10
        idxs_dict = get_part_idxs()
        body_idxs = idxs_dict['body']
        hand_idxs = idxs_dict['hand']
        face_idxs = idxs_dict['face']
        
        self.body_idxs = np.asarray(body_idxs)
        self.hand_idxs = np.asarray(hand_idxs)
        self.face_idxs = np.asarray(face_idxs)
        self.use_face_contour = True
        self.body_dset_factor = 1.2
        self.head_dset_factor = 2.0
        self.hand_dset_factor = 2.0

        self.body_thresh = 0.1
        self.face_thresh = 0.4
        self.hand_thresh=0.2
        self.binarization = True
        self.preprocess = augmentation.Preprocessing_data()
    def __len__(self):
        return len(self.betas)
    def __getitem__(self, index):
        item = {}
        fold = self.folds[index]
        if fold == 'coco':
            root_dir = '/media/josh/Data/Datasets/mscoco/coco/train2014/'
        elif fold == 'lsp':
            root_dir = '/media/josh/Data/expose_dataset/lsp/lsp_dataset_original/images/'
        elif fold == 'lspet':
            root_dir = '/media/josh/Data/expose_dataset/lsp/lspet/images/'
        elif fold == 'mpii':
            root_dir = '/media/josh/Data/expose_dataset/mpii/images/'
        
        flip = 0            
        pn = np.ones(3) 
        rot = 0            
        sc = 1
        sc, rot, pn  = self.preprocess.get_aug_config()

        img_fn = os.path.join(root_dir, self.imgname[index])
        pose = self.pose[index].copy()
        pose = np.array(pose).reshape(55, 3).astype(np.float32)
        betas = torch.from_numpy(self.betas[index, :self.num_betas])
        keypoints2d = self.keypoints2D[index]
        img = cv2.imread(img_fn)[:,:,::-1].copy()

        #pose rotation
        eye_offset = 2
        # pose_rot = util.pose_processing(pose, rot) 

        gp = util.pose_processing(pose[0], rot)
        # bp = util.pose_processing(pose[1:22, :], rot)
        # jp = util.pose_processing(pose[22], rot)
        # lhp = util.pose_processing(pose[23 + eye_offset:23 + eye_offset + 15], rot)
        # rhp = util.pose_processing(pose[23 + 15 + eye_offset:], rot)
        
        global_pose =  torch.from_numpy(gp)
        global_pose = global_pose.unsqueeze(dim=0)
        body_pose = torch.from_numpy(pose[1:22, :])
        jaw_pose = torch.from_numpy(pose[22])
        jaw_pose = jaw_pose.unsqueeze(dim=0)
        left_hand_pose = torch.from_numpy(pose[23 + eye_offset:23 + eye_offset + 15])
        right_hand_pose = torch.from_numpy(pose[23 + 15 + eye_offset:])

        # global_pose =  torch.from_numpy(gp)
        # global_pose = global_pose.unsqueeze(dim=0)
        # body_pose = torch.from_numpy(bp)
        # jaw_pose = torch.from_numpy(jp)
        # jaw_pose = jaw_pose.unsqueeze(dim=0)
        # left_hand_pose = torch.from_numpy(lhp)
        # right_hand_pose = torch.from_numpy(rhp)


        global_pose = func.batch_rodrigues(global_pose.view(-1, 3)).view(1, 3, 3)
        body_pose = func.batch_rodrigues(body_pose.view(-1, 3)).view(21, 3, 3)
        jaw_pose = func.batch_rodrigues(jaw_pose.view(-1, 3)).view(1, 3, 3)
        left_hand_pose = func.batch_rodrigues(left_hand_pose.view(-1, 3)).view(15, 3, 3)
        right_hand_pose = func.batch_rodrigues(right_hand_pose.view(-1, 3)).view(15, 3, 3)
        camera = self.camera[index]

        orig_shape = np.array(img.shape)[:2]
        output_keypoints2d = np.zeros([127 + 17 * self.use_face_contour,
                                       3], dtype=np.float32)

        output_keypoints2d[self.spin_target_idxs] = keypoints2d[self.spin_source_idxs]
        conf = output_keypoints2d[:, -1]
        # Remove joints with negative confidence
        output_keypoints2d[output_keypoints2d[:, -1] < 0, -1] = 0
        if self.body_thresh > 0:
            # Only keep the points with confidence above a threshold
            body_conf = output_keypoints2d[self.body_idxs, -1]
            hand_conf = output_keypoints2d[self.hand_idxs, -1]
            face_conf = output_keypoints2d[self.face_idxs, -1]

            body_conf[body_conf < self.body_thresh] = 0.0
            hand_conf[hand_conf < self.hand_thresh] = 0.0
            face_conf[face_conf < self.face_thresh] = 0.0
            if self.binarization:
                body_conf = (
                    body_conf >= self.body_thresh).astype(
                        output_keypoints2d.dtype)
                hand_conf = (
                    hand_conf >= self.hand_thresh).astype(
                        output_keypoints2d.dtype)
                face_conf = (
                    face_conf >= self.face_thresh).astype(
                        output_keypoints2d.dtype)

            output_keypoints2d[self.body_idxs, -1] = body_conf
            output_keypoints2d[self.hand_idxs, -1] = hand_conf
            output_keypoints2d[self.face_idxs, -1] = face_conf
        center = self.centers[index]
        scale = self.scales[index]
        
        try: 
            bl_img, sh_img = self.preprocess.rgb_preprocessing(img, img, center, scale * sc, rot, flip, pn)
        except:
            print(img_fn)
        bl_img = torch.from_numpy(bl_img).float()
        sh_img = torch.from_numpy(sh_img).float()

        item['img'] = self.preprocess.norm_processing(bl_img)
        item['sharp_img'] = self.preprocess.norm_processing(sh_img)
        item['imgname'] = img_fn
        item['keypoints'] = torch.from_numpy(self.preprocess.j2d_processing(output_keypoints2d, center, sc*scale, rot, flip)).float()
        item['betas'] = betas.float()
        item['expression'] = torch.zeros(10, dtype=torch.float32)
        item['global_orient'] = global_pose
        item['body_pose'] = body_pose
        item['jaw_pose'] = jaw_pose
        item['leye_pose'] = torch.zeros(1, 3, 3, dtype=torch.float32)
        item['reye_pose'] = torch.zeros(1, 3, 3, dtype=torch.float32)

        item['left_hand_pose'] = left_hand_pose
        item['right_hand_pose'] = right_hand_pose
        if flip:
            item['left_hand_pose'] =  right_hand_pose
            item['right_hand_pose'] = left_hand_pose
        
        item['camera'] = torch.from_numpy(camera).unsqueeze(dim=0).float()
        item['orig_shape'] = orig_shape
        item['conf'] = torch.from_numpy(conf) 

        item['dset_name'] = 'spinx'
        item['j3d'] = torch.zeros(144,4, dtype=torch.float32)
        return item
