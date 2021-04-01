import logging
from collections import defaultdict

import numpy as np
from tensorboardX import SummaryWriter
WINDOW_SIZE = 100

class Tensorboard_vis():
    def __init__(self, exp_name = 'logs'):
        self.writer = SummaryWriter(exp_name)
        logging.basicConfig(filename='logs/{}.log'.format(exp_name), level=logging.DEBUG)
        self.metrics = defaultdict(list)
        self.images = defaultdict(list)
        self.best_metric = 0
    def write_body_loss(self, losses, epoch_num):
        self.writer.add_scalar('Loss_body/face_2D', losses['loss_2D']['face_2D'], global_step=epoch_num)
        self.writer.add_scalar('Loss_body/body_2D', losses['loss_2D']['body_2D'], global_step=epoch_num)
        self.writer.add_scalar('Loss_body/hand_2D', losses['loss_2D']['hand_2D'], global_step=epoch_num)
        self.writer.add_scalar('Loss_body/body_3D', losses['loss_3D']['body_3D'], global_step=epoch_num)
        self.writer.add_scalar('Loss_body/SMPLX', losses['loss_SMPLX']['betas'], global_step=epoch_num)
        self.writer.add_scalar('Loss_body/SMPLX', losses['loss_SMPLX']['pose'], global_step=epoch_num)
        self.writer.add_scalar('Loss_body/SMPLX', losses['loss_SMPLX']['exp'], global_step=epoch_num)
        self.writer.add_scalar('Loss_body/Full_body', losses['loss_FULL'], global_step=epoch_num)
    def write_deblur_G_loss(self, losses, epoch_num):
        self.writer.add_scalar('Loss_deblur_G/perceptual', losses['loss_deb']['loss_percep'], global_step=epoch_num)
        self.writer.add_scalar('Loss_deblur_G/pixel_distance', losses['loss_deb']['loss_pixdist'], global_step=epoch_num)
        self.writer.add_scalar('Loss_deblur_G/adversarial', losses['loss_deb']['loss_Adv'], global_step=epoch_num)
        self.writer.add_scalar('Loss_deblur_G/Full_G', losses['loss_deb']['loss_G'], global_step=epoch_num)
        self.writer.add_scalar('Loss_body_and_deblur/Full', losses['loss_comb'], global_step=epoch_num)
    def write_deblur_D_loss(self, losses, epoch_num):
        self.writer.add_scalar('Loss_deblur_D/Full', losses['loss_D_Full'], global_step=epoch_num)
        self.writer.add_scalar('Loss_deblur_D/Patch', losses['loss_D_Patch'], global_step=epoch_num)
        self.writer.add_scalar('Loss_deblur_D/All', losses['loss_D'], global_step=epoch_num)
        
    # def write_to_tensorboard(self, value, epoch_num):
    #     # all_loss = { 
    #     #     'face_2d' : loss_2d_face,
    #     #     'body_2d' : loss_2d_body,
    #     #     'hand_2d' : loss_2d_hand,
    #     #     'all_3d': loss_3d_keyps,
    #     #     'beta' : loss_betas,
    #     #     'pose' : loss_pose,
    #     #     'exp': loss_expression,
    #     #     'all' : loss_all
    #     # }
    #     self.writer.add_scalar('Loss/face_2d', value['face_2d'], global_step=epoch_num)
    #     self.writer.add_scalar('Loss/body_2d', value['body_2d'], global_step=epoch_num)
    #     self.writer.add_scalar('Loss/hand_2d', value['hand_2d'], global_step=epoch_num)
    #     self.writer.add_scalar('Loss/Loss_3D', value['all_3d'], global_step=epoch_num)
    #     self.writer.add_scalar('Loss/Loss_ALL', value['all' ], global_step=epoch_num)
    #     self.writer.add_scalar('Loss/Loss_Betas', value['beta'], global_step=epoch_num)
    #     self.writer.add_scalar('Loss/Loss_Pose', value['pose'], global_step=epoch_num)
    #     self.writer.add_scalar('Loss/Loss_Exp', value['exp'], global_step=epoch_num)

    #     self.writer.add_scalar('Loss_Deblur/Loss_percep', value['Perceptual_Loss'], global_step=epoch_num)
    #     self.writer.add_scalar('Loss_Deblur/Loss_PixDis', value['PixDIs_MSE_Loss'], global_step=epoch_num)
    #     self.writer.add_scalar('Loss_Deblur/Adv_loss', value['Adv_loss'], global_step=epoch_num)
    #     self.writer.add_scalar('Loss_Deblur/Disc_loss', value['Disc_loss'], global_step=epoch_num)
        
    #     self.writer.add_scalar('Validation/PA_MPJPE_BODY', value['PA_MPJPE_BODY'], global_step=epoch_num)
    #     self.writer.add_scalar('Validation/PA_V2V_all', value['PA_V2V_all'], global_step=epoch_num)
    #     self.writer.add_scalar('Validation/PA_V2V_body', value['PA_V2V_body'], global_step=epoch_num)
    #     # self.writer.add_scalar('Loss/Loss_VGG', value[6], global_step=epoch_num)
    #     # self.writer.add_scalar('Loss/Loss_MSE_pixdis', value[7], global_step=epoch_num)
    #     # self.writer.add_scalar('PSNR_SSIM/psnr_db_sh', value[7], global_step=epoch_num)
    #     # self.writer.add_scalar('PSNR_SSIM/ssim_bl_sh', value[8], global_step=epoch_num)
    #     # self.writer.add_scalar('PSNR_SSIM/ssim_db_sh', value[9], global_step=epoch_num)
    # def write_to_tensorboard_validation(self, value, epoch_num):
    #     self.writer.add_scalar('Validation/MPJPE', value[0], global_step=epoch_num)
    #     self.writer.add_scalar('Validation/PA-MPJPE', value[1], global_step=epoch_num)