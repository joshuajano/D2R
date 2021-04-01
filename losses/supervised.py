import numpy as np
import torch
import torch.nn as nn
from utils.keypoints import get_part_idxs, KEYPOINT_NAMES
import random 
from torch.autograd import Variable
import torch.autograd as autograd
from losses.common import PerceptualLoss, PixelDistanceLoss
class Supervised(nn.Module):
    def __init__(self, smplx_layer):
        super(Supervised, self).__init__()

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
        self.left_hip_idx = KEYPOINT_NAMES.index('left_hip')
        self.right_hip_idx = KEYPOINT_NAMES.index('right_hip')
        self.smplx_layer = smplx_layer
        self.device = 'cuda'
        self.criterion_keypoints = nn.L1Loss(reduction='none').to('cuda')
        self.criterion_regr = nn.MSELoss().to('cuda')
        self.LAMBDA = 10
        self.percp_loss = PerceptualLoss()
        self.pixDis_loss = PixelDistanceLoss()
        self.adv_lambda = 0.001
        self.dist_lambda = 50
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty
    def compute_loss_WGANGP(self, x, y, net):
        D_fake = net.forward(x.detach())
        D_fake = D_fake.mean()

        D_real = net.forward(y)
        D_real = D_real.mean()

        # Combined loss
        loss_D = D_fake - D_real
        gradient_penalty = self.calc_gradient_penalty(net, x.data, y.data)
        return loss_D + gradient_penalty
    def get_loss_D(self, x, y, net):
        batch = x.shape[0]
        crop_idx = random.sample(range(0, 192), batch)
        patch_x = torch.zeros((batch, 3, 64 ,64)).cuda()
        patch_y = torch.zeros((batch, 3, 64 ,64)).cuda()
        #get patch 
        for i in range(batch):
            patch_x[i, :, :, :] = x[i, :,  crop_idx[i]: crop_idx[i] + 64, crop_idx[i]: crop_idx[i] + 64]
            patch_y[i, :, :, :] = y[i, :,  crop_idx[i]: crop_idx[i] + 64, crop_idx[i]: crop_idx[i] + 64]
        D_Full_loss = self.compute_loss_WGANGP(x, y, net.module.deb_D_Full)
        D_Patch_loss = self.compute_loss_WGANGP(patch_x, patch_y, net.module.deb_D_Patch)
        
        loss = (D_Full_loss + D_Patch_loss)/2

        return {'loss_D_Full': D_Full_loss, 'loss_D_Patch': D_Patch_loss, 'loss_D': loss}
    def compute_loss_G(self, x, y, net):
        loss_percep = self.percp_loss(x, y)
        loss_pixdist = self.pixDis_loss(x, y)
        
        loss_full_Adv = net.module.deb_D_Full(x)
        loss_full_Adv = -loss_full_Adv.mean()

        loss_patch_Adv = net.module.deb_D_Patch(x)
        loss_patch_Adv = -loss_patch_Adv.mean()

        loss_Adv = (loss_full_Adv + loss_patch_Adv)/2
        loss_G = (loss_Adv * self.adv_lambda) + (loss_pixdist * self.dist_lambda) + loss_percep
        return {'loss_percep': loss_percep, 'loss_pixdist': loss_pixdist, 'loss_Adv': loss_Adv, 'loss_G':loss_G}
    def get_deblur_loss(self, input, pred_params, pred_imgs, sharp_imgs, net):
        loss_deblur = self.compute_loss_G(pred_imgs, sharp_imgs, net)
        loss_body = self.get_body_loss(input, pred_params, return_raw = False)
        loss_comb = loss_deblur['loss_G'] + loss_body['loss_FULL']
        return {'loss_deb': loss_deblur, 'loss_comb': loss_comb}
    def get_body_loss(self, input, pred_params, return_raw = False):
        GT_params = self.get_GT_params(input)
        gt_body = self.smplx_layer.gen_smplx(GT_params, name_type='gt')
        pred_body, raw_pred = self.smplx_layer.gen_smplx(pred_params, name_type='pred')
        conf = input['conf'].to(self.device)
        dset_name = input['dset_name']
        h36m_idxs, cur_idxs, own_idxs = self.get_dset_idxs(dset_name)
        
        #2D information
        GT_kp_2D = input['keypoints'].cuda()
        PRED_kp_2D = self.smplx_layer.get_2d_pred(pred_body.joints, raw_pred)
        loss_2D = self.get_2D_loss(PRED_kp_2D, GT_kp_2D, conf, h36m_idxs, cur_idxs, own_idxs)

        #3D information
        PRED_kp_3D = pred_body.joints
        GT_kp_3D = gt_body.joints
        #For H36M use the 3D from GT
        h36m_j3d = input['j3d'].to('cuda')
        GT_kp_3D[h36m_idxs] = h36m_j3d[h36m_idxs, :, :3]
        loss_3D = self.get_3D_loss(PRED_kp_3D, GT_kp_3D, conf, h36m_idxs, cur_idxs, own_idxs)

        #SMPLX loss
        loss_smplx = self.get_smplx_loss(pred_body, gt_body, cur_idxs, own_idxs)
        if return_raw:
            return  {'loss_2D': loss_2D, 'loss_3D': loss_3D, 'loss_SMPLX': loss_smplx, 'loss_FULL': 
                    loss_2D['face_2D'] + loss_2D['body_2D'] + loss_2D['hand_2D'] + \
                    loss_3D['body_3D'] + loss_smplx['betas'] + loss_smplx['pose'] + loss_smplx['exp'] + \
                    ((torch.exp(-raw_pred['camera'][:,0]*10)) ** 2 ).mean()}, {'raw_pred' : raw_pred, 'gt_param': gt_body, 'pred_param': pred_body}
        else:
            return  {'loss_2D': loss_2D, 'loss_3D': loss_3D, 'loss_SMPLX': loss_smplx, 'loss_FULL': 
                        loss_2D['face_2D'] + loss_2D['body_2D'] + loss_2D['hand_2D'] + \
                        loss_3D['body_3D'] + loss_smplx['betas'] + loss_smplx['pose'] + loss_smplx['exp'] + \
                        ((torch.exp(-raw_pred['camera'][:,0]*10)) ** 2 ).mean()}
        

    def forward(self, input, pred_imgs, pred_params, is_body = True):
        GT_params = self.get_GT_params(input)
        gt_body = self.smplx_layer.gen_smplx(GT_params, name_type='gt')
        pred_body, raw_pred = self.smplx_layer.gen_smplx(pred_params, name_type='pred')
        conf = input['conf'].to(self.device)
        dset_name = input['dset_name']
        h36m_idxs, cur_idxs, own_idxs = self.get_dset_idxs(dset_name)
        
        #2D information
        GT_kp_2D = input['keypoints'].cuda()
        PRED_kp_2D = self.smplx_layer.get_2d_pred(pred_body.joints, raw_pred)
        loss_2D = self.get_2D_loss(PRED_kp_2D, GT_kp_2D, conf, h36m_idxs, cur_idxs, own_idxs)

        #3D information
        PRED_kp_3D = pred_body.joints
        GT_kp_3D = gt_body.joints
        #For H36M use the 3D from GT
        h36m_j3d = input['j3d'].to('cuda')
        GT_kp_3D[h36m_idxs] = h36m_j3d[h36m_idxs, :, :3]
        loss_3D = self.get_3D_loss(PRED_kp_3D, GT_kp_3D, conf, h36m_idxs, cur_idxs, own_idxs)

        #SMPLX loss
        loss_smplx = self.get_smplx_loss(pred_body, gt_body, cur_idxs, own_idxs)

        if is_body:
            return loss_2D['face_2D'] + loss_2D['body_2D'] + loss_2D['hand_2D'] + \
                    loss_3D['body_3D'] + loss_smplx['betas'] + loss_smplx['pose'] + loss_smplx['exp']
        else:
            sharp_imgs = input['sharp_img'].to(self.device)
            pass
    def get_smplx_loss(self, x, y, cur_idxs, own_idxs):
        batch = len(cur_idxs) + len(own_idxs)
        loss_cur = self.compute_smplx_loss(x, y, cur_idxs)
        loss_own = self.compute_smplx_loss(x, y, own_idxs)

        loss = {'betas' : (loss_cur[0] + loss_own[0]) /batch,
                'pose' : (loss_cur[1] + loss_own[1]) /batch,
                'exp' : (loss_cur[2]  + loss_own[2]) /batch,
        }
        return loss
    def compute_smplx_loss(self, input, target, idxs):
        pred_betas = input.betas.clone()
        gt_betas = target.betas.clone()

        pred_betas = input.betas[idxs].clone()
        gt_betas = target.betas[idxs].clone()

        pred_global_orient = input.global_orient[idxs].clone()
        gt_global_orient = target.global_orient[idxs].clone()

        pred_body_pose = input.body_pose[idxs].clone()
        gt_body_pose =  target.body_pose[idxs].clone()
        
        pred_jaw_pose = input.jaw_pose[idxs].clone()
        gt_jaw_pose =  target.jaw_pose[idxs].clone()
        pred_hand_left_pose = input.left_hand_pose[idxs].clone()
        gt_hand_left_pose =  target.left_hand_pose[idxs].clone()
        pred_hand_right_pose = input.right_hand_pose[idxs].clone()
        gt_hand_right_pose =  target.right_hand_pose[idxs].clone()

        pred_all_pose = torch.cat((pred_global_orient, pred_body_pose, pred_jaw_pose, pred_hand_left_pose, pred_hand_right_pose), dim=1)
        gt_all_pose = torch.cat((gt_global_orient, gt_body_pose, gt_jaw_pose, gt_hand_left_pose, gt_hand_right_pose), dim=1)
        #get expression
        pred_expression = input.expression[idxs].clone()
        gt_expression = target.expression[idxs].clone()

        diff_betas = self.criterion_regr(pred_betas, gt_betas)
        diff_expression = self.criterion_regr(pred_expression, gt_expression)
        diff_pose = self.criterion_regr(pred_all_pose, gt_all_pose)
     
        return diff_betas, diff_pose, diff_expression
    def compute_3d_loss(self, pred, gt, conf):
        pred = pred.clone()
        gt = gt.clone()
        weights = conf.clone().unsqueeze(dim=-1)
        pred_pelvis = pred[:, [self.left_hip_idx, self.right_hip_idx], :].mean(dim=1, keepdim=True)
        centered_pred_joints = pred - pred_pelvis
        gt_pelvis = gt[:, [self.left_hip_idx, self.right_hip_idx], :].mean(dim=1, keepdim=True)
        centered_gt_joints = gt - gt_pelvis
        raw_diff = centered_pred_joints - centered_gt_joints
        diff = raw_diff.abs()
        weighted_diff = diff * weights
        return torch.sum(weighted_diff)
    def get_2D_loss(self, x, y, conf, h36m_idxs, cur_idxs, own_idxs):
        batch_body = len(h36m_idxs) + len(cur_idxs) + len(own_idxs)
        batch_fh = len(cur_idxs) + len(own_idxs)
        
        loss_cur = self.compute_2d_loss(x[cur_idxs], y[cur_idxs][:, :, :-1], conf[cur_idxs])
        loss_h36m = self.compute_2d_loss(x[h36m_idxs], y[h36m_idxs][:, :, :-1], conf[h36m_idxs])
        loss_own = self.compute_2d_loss(x[own_idxs], y[own_idxs][:, :, :-1], conf[own_idxs])

        loss = {'face_2D' : (loss_cur[0] + loss_h36m[0] + loss_own[0]) / batch_fh,
                'body_2D' : (loss_cur[1] + loss_h36m[1] + loss_own[1]) / batch_body,
                'hand_2D' : (loss_cur[2] + loss_h36m[2] + loss_own[2]) / batch_fh,
        }
        return loss
    def get_3D_loss(self, x, y, conf, h36m_idxs, cur_idxs, own_idxs):
        batch = len(h36m_idxs) + len(cur_idxs) + len(own_idxs)

        loss_cur = self.compute_3d_loss(x[cur_idxs], y[cur_idxs], conf[cur_idxs])
        loss_h36m = self.compute_3d_loss(x[h36m_idxs], y[h36m_idxs], conf[h36m_idxs])
        loss_own = self.compute_3d_loss(x[own_idxs], y[own_idxs], conf[own_idxs])

        loss = {'body_3D' : (loss_cur + loss_h36m + loss_own) / batch}
        return loss
    def get_GT_params(self, gt):
        gt_body_param = { 
                    'global_orient': gt['global_orient'].to(self.device),
                    'body_pose': gt['body_pose'].to(self.device),
                    'jaw_pose': gt['jaw_pose'].to(self.device),
                    'left_hand_pose': gt['left_hand_pose'].to(self.device),
                    'right_hand_pose': gt['right_hand_pose'].to(self.device),
                    'expression': gt['expression'].to(self.device),
                    'left_eye_pose': gt['leye_pose'].to(self.device),
                    'right_eye_pose': gt['reye_pose'].to(self.device),
                    'betas':gt['betas'].to(self.device),
                    'camera': gt['camera'].to(self.device)
                }
        return gt_body_param
    def get_dset_idxs(self, dset_name):
        h36m_idxs = []
        insta_idxs = []
        curated_idxs = []
        
        for i in range(len(dset_name)):
            if dset_name[i] == 'h36m':
                h36m_idxs.append(i)
            elif dset_name[i] == 'curated':
                curated_idxs.append(i)
            elif dset_name[i] == 'own':
                insta_idxs.append(i)
        return torch.from_numpy(np.asarray(h36m_idxs)), torch.from_numpy(np.asarray(curated_idxs)), torch.from_numpy(np.asarray(insta_idxs))
    def compute_2d_loss(self, pred, gt, conf):
        weights = conf.clone().unsqueeze(dim=-1)
        weights[weights==-1] = 0
        
        body_pred = pred.clone()[:, self.body_idxs]
        body_gt = gt.clone()[:, self.body_idxs]
        body_conf = weights.clone()[:, self.body_idxs]
        
        hand_pred = pred[:, self.hand_idxs]
        hand_gt = gt[:, self.hand_idxs]
        hand_conf = weights[:, self.hand_idxs] 

        face_pred = pred[:, self.face_idxs]
        face_gt = gt[:, self.face_idxs]
        face_conf = weights[:, self.face_idxs] 

        body_loss = self.criterion_keypoints(body_pred, body_gt)
        hand_loss = self.criterion_keypoints(hand_pred, hand_gt)
        face_loss = self.criterion_keypoints(face_pred, face_gt)

        wface_loss = face_loss * face_conf
        wbody_loss = body_loss * body_conf
        whand_loss = hand_loss * hand_conf

        return torch.sum(wface_loss), torch.sum(wbody_loss), torch.sum(whand_loss)
