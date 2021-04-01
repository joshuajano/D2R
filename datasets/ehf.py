import cv2
import numpy as np
import os

import torch.utils.data as dutils
import utils as func
from torchvision.transforms import Normalize
import torch
from utils.keypoints import dset_to_body_model, get_part_idxs
import utils.constants as constants

class EHF(dutils.Dataset):
    def __init__(self, imgs_path ='/media/josh/Data/EHF_v2/', img_size= 256):
        self.root_imgs_dir = imgs_path
        # data_path = os.path.join(imgs_path, 'ehf_val.npz')
        # self.data = np.load(os.path.join(imgs_path, 'ehf_val.npz'))
        self.data = np.load('ehf_val.npz')
        self.imgs_name = self.data['imgname']
        self.keypoints = self.data['keypoints']
        self.vertices = self.data['vertices']
        self.faces_name = self.data['faces']
        mean = [0.485, 0.456, 0.406]
        std =[0.229, 0.224, 0.225]
        self.keyp_format = 'coco25'
        self.normalize = Normalize(mean=mean, std=std)
        source_idxs, target_idxs = dset_to_body_model(
            dset='openpose25+hands+face',
            model_type='smplx', use_hands=True, use_face=True,
            use_face_contour= True,
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

        self.body_thresh = 0.1
        self.face_thresh = 0.4
        self.hand_thresh=0.2
        self.img_size = img_size
        self.binarization = True
    def get_keypoints(self, keypoints2d):
        output_keypoints2d = np.zeros([127 + 17 ,  3], dtype=np.float32)
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
    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = func.transform(kp[i,0:2]+1, center, scale, 
                                  [self.img_size, self.img_size], rot=r)
        # convert to normalized coordinates
        kp[:, 0:2] = 2. * kp[:, 0:2]/self.img_size - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp
    def rgb_processing(self, img_name, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = func.crop(rgb_img, center, scale, [self.img_size, self.img_size], rot=rot)
        
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
    def __len__(self):
        return len(self.imgs_name)
    def __getitem__(self, index):
        item = {}
        flip = 0            
        pn = np.ones(3) 
        rot = 0            
        sc = 1 

        keypoints = self.keypoints[index]
        output_keypoints2d = self.get_keypoints(keypoints)

        keypoints = output_keypoints2d[:, :-1]
        conf = output_keypoints2d[:, -1]

        vertices = self.vertices[index]
        img_fn = os.path.join('/media/josh/Data/EHF/', self.imgs_name[index]) 
        try:
            img = cv2.imread(img_fn)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(img_fn)
        orig_shape = np.array(img.shape)[:2]
        bbox = func.keyps_to_bbox(keypoints, conf, img_size=orig_shape)
        center, scale, bbox_size = func.bbox_to_center_scale(bbox, dset_scale_factor= self.body_dset_factor)
        full_img = np.transpose(img.astype('float32'),(2,0,1))/255.0
        full_img = torch.from_numpy(full_img).float()
        # output_keypoints2d = self.get_keypoints(keypoints)
        
        flip = 0            
        pn = np.ones(3) 
        rot = 0            
        sc = 1
        # bbox = func.keyps_to_bbox(keypoints, conf, img_size=orig_shape)
        # center, scale, bbox_size = func.bbox_to_center_scale(bbox, dset_scale_factor= 1.2)
        img = self.rgb_processing(img_fn, img, center, scale * sc, rot, flip, pn)
        img = torch.from_numpy(img).float()
        
        #additonal information for vertices
        transl = np.array([-0.03609917, 0.43416458, 2.37101226])
        camera_pose = np.array([-2.9874789618512025, 0.011724572107320893,
                                -0.05704686818955933])
        camera_pose = cv2.Rodrigues(camera_pose)[0]
        
        new_vert = vertices.dot(camera_pose.T) + transl.reshape(1, 3)
        item['joints'] = self.j2d_processing(output_keypoints2d, center, sc, rot, flip)
        item['img'] = self.normalize(img)
        item['full_img'] = self.normalize(full_img)
        item['orig_shape'] = orig_shape
        item['vertices'] = torch.from_numpy(new_vert).float()
        # item['vertices'] = torch.from_numpy(vertices).float()
        return item