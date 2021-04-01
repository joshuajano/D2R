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
from utils import common
# from  utils import util
class CuratedFittings(dutils.Dataset):
    def __init__(self, cfg, data_path = 'data/curated_fits/', split = 'train', batch_size = 1):
        super(CuratedFittings, self).__init__()
        self.binarization = True
        self.use_face_contour = True
        self.batch_size = batch_size
        self.num_betas = 10
        # self.data_path = os.path.join(data_path, split + '.npz')
        self.data_path = cfg['npz_dir']
        self.root_dir = cfg['imgs_dir']
        data = np.load(self.data_path, allow_pickle=True)
        data = {key: data[key] for key in data.keys()}

        self.betas = data['betas'].astype(np.float32)
        self.expression = data['expression'].astype(np.float32)
        self.keypoints2D = data['keypoints2D'].astype(np.float32)
        self.pose = data['pose'].astype(np.float32)
        self.camera = data['translation'].astype(np.float32)
        self.img_fns = np.asarray(data['img_fns'], dtype=np.string_)
        self.num_items = self.pose.shape[0]
        self.keyp_format = 'coco25'

        #mapping
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

        self.body_thresh = 0.1
        self.face_thresh = 0.4
        self.hand_thresh=0.2
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
       
        # self.normalize = Normalize(mean=mean, std=std)
        
        self.img_size = 256
        self.preprocess = augmentation.Preprocessing_data()
    def __len__(self):
        return self.num_items
    
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
        
    def rgb_processing(self, img_name, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        
        # rgb_img = func.crop(img_name, rgb_img, center, scale, [self.img_size, self.img_size], rot=rot)
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
    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = func.transform(kp[i,0:2]+1, center, scale, 
                                  [self.img_size, self.img_size], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.img_size - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp
    
    def __getitem__(self, index):
        item = {}
        #extract pose betas expression dan camera
        pose = self.pose[index].copy()
        betas = torch.from_numpy(self.betas[index, :self.num_betas])
        expression = torch.from_numpy(self.expression[index])
        camera = self.camera[index]
        keypoints2d = self.keypoints2D[index]
        '''Image preprocessing'''
        img_fn = self.img_fns[index].decode('utf-8')
        img_fn = os.path.join(self.root_dir, img_fn)
        try:
            # img = cv2.imread(img_fn)[:,:,::-1].copy().astype(np.float32)
            img = cv2.imread(img_fn)[:,:,::-1].copy()
        except TypeError:
            print(img_fn)
        orig_shape = np.array(img.shape)[:2]

        '''Keypoints preprocessing, keep in mind, we only start for body
        For extracting scale, center
        '''

        output_keypoints2d = self.get_keypoints(keypoints2d)
        #get keypoints 
        keypoints = output_keypoints2d[:, :-1]
        conf = output_keypoints2d[:, -1]
        #from keypoints get bbox
        bbox = common.keyps_to_bbox(keypoints, conf, img_size=orig_shape)
        center, scale, bbox_size = common.bbox_to_center_scale(bbox, dset_scale_factor= self.body_dset_factor)

        #next preprocessing image by cropping
        flip = 0            
        pn = np.ones(3) 
        rot = 0            
        sc = 1 
        sc, rot, pn  = self.preprocess.get_aug_config() 
        # flip, pn, rot, sc = augmentation.augm_params(0.4, 30, 0.25)
        # img = augmentation.rgb_preprocessing(img, center, scale * sc, rot, flip, pn)

        eye_offset = 0 if pose.shape[0] == 53 else 2

        #pose rotation
        gp = common.pose_processing(pose[0], rot)

        #decoder        
        global_pose =  torch.from_numpy(gp)
        global_pose = global_pose.unsqueeze(dim=0)
        body_pose = torch.from_numpy(pose[1:22, :])
        jaw_pose = torch.from_numpy(pose[22])
        jaw_pose = jaw_pose.unsqueeze(dim=0)
        left_hand_pose = torch.from_numpy(pose[23 + eye_offset:23 + eye_offset + 15])
        right_hand_pose = torch.from_numpy(pose[23 + 15 + eye_offset:])

        # body_pose = augmentation.pose_body_processing(body_pose, rot, flip)

        #change to 6 rotation degree bf.out_path = '/home/josh/Desktop/Paper/Dataset/Human3.6M/SPIN_extraction/'y batch rodrigues
        global_pose = common.batch_rodrigues(global_pose.view(-1, 3)).view(1, 3, 3)
        body_pose = common.batch_rodrigues(body_pose.view(-1, 3)).view(21, 3, 3)
        jaw_pose = common.batch_rodrigues(jaw_pose.view(-1, 3)).view(1, 3, 3)
        left_hand_pose = common.batch_rodrigues(left_hand_pose.view(-1, 3)).view(15, 3, 3)
        right_hand_pose = common.batch_rodrigues(right_hand_pose.view(-1, 3)).view(15, 3, 3)
        
        root_ext =  img_fn.split('/')
        #root_ext[5] + '_' + root_ext[7]
        #augmentation
         
        # img = self.rgb_processing(img_fn, img, center, scale * sc, rot, flip, pn)
        try:
            bl_img, sh_img = self.preprocess.rgb_preprocessing(img, img, center, scale * sc, rot, flip, pn)
        except:
            print(img_fn)
        bl_img = torch.from_numpy(bl_img).float()
        sh_img = torch.from_numpy(sh_img).float()

        #return item
        item['img'] = self.preprocess.norm_processing(bl_img)
        item['sharp_img'] = self.preprocess.norm_processing(sh_img)
        item['imgname'] = img_fn
        item['keypoints'] = torch.from_numpy(self.preprocess.j2d_processing(output_keypoints2d, center, sc*scale, rot, flip)).float()
        # item['keypoints'] = torch.from_numpy(augmentation.j2d_processing(output_keypoints2d, center, sc*scale, rot, flip)).float()
        
        item['betas'] = betas
        item['expression'] = expression

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

        item['dset_name'] = 'curated'
        item['j3d'] = torch.zeros(144,4, dtype=torch.float32)
        return item