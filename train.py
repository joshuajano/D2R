import torch
import numpy as np
from loguru import logger
import torch.optim as optim
from tqdm import tqdm
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import os

from networks.joint import JointNet
from utils.scheduler import LinearDecay
from datasets.mixed import Mixed_Dataset
from datasets.insta_rand import InstaRand
from utils import common
from losses.supervised import Supervised
from utils.smplxLayer import SMPLXLayer
from utils.plot import Tensorboard_vis
from utils.renderer import Visualization_render
class Train(object):
    def __init__(self, cfg, pose_desc_dict, param_mean):
        logger.warning('TRAINING MODE')
        self.conf = cfg
        self.model = JointNet(cfg, param_mean)
        if cfg['is_parallel']:
            self.model = DataParallel(self.model).cuda()
            self.model.train()
        self.get_optimizer()
        self.get_scheduler()
        self.get_datasets()
        self.smplx = SMPLXLayer(cfg['body_param'], pose_desc_dict)
        self.losses = Supervised(self.smplx)
        self.adv_lambda = 0.001
        self.plot = Tensorboard_vis()
        self.step = 0
        self.render = Visualization_render(cfg['focal_length'], cfg['render']['img_size'], self.smplx.faces)
        self.weights_dir = 'weights'
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir )
        self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').reshape(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').reshape(1,3,1,1)
    def get_optimizer(self):
        D_params = list(self.model.module.deb_D_Patch.parameters()) + list(self.model.module.deb_D_Full.parameters())
        G_params = list(self.model.module.deb_G.parameters())
        R_params = list(self.model.module.bodyNet.parameters()) + list(self.model.module.struct_feat.parameters()) 
        self.optim_deb_D = optim.Adam(D_params, lr = self.conf['deblur']['lr'])
        self.optim_deb_G = optim.Adam(G_params, lr = self.conf['deblur']['lr'])
        self.optim_body_struct = optim.Adam(R_params, lr = self.conf['recon']['lr'])
        logger.debug('ADAM optimizer is used')
    def get_scheduler(self):
        self.sched_body_struct = LinearDecay(self.optim_body_struct, min_lr=self.conf['min_lr'], num_epochs= self.conf['num_epoch'], start_epoch = self.conf['start'])
        self.sched_deb_G = LinearDecay(self.optim_deb_G, min_lr=self.conf['min_lr'], num_epochs= self.conf['num_epoch'], start_epoch = self.conf['start'])
        self.sched_deb_D = LinearDecay(self.optim_deb_D, min_lr=self.conf['min_lr'], num_epochs= self.conf['num_epoch'], start_epoch = self.conf['start'])
        logger.debug('Linear decay scheduling is used')
    def get_datasets(self):
        self.main_dset = Mixed_Dataset(self.conf['datasets'])
        self.extra_dset = InstaRand(self.conf['datasets']['insta'])
    def save_model(self):
        '''Save model'''
        torch.save({
            'recon_net': self.model.module.bodyNet.state_dict(),
            'restore_net': self.model.module.deb_G.state_dict(),
            'struct2feat': self.model.module.struct_feat.state_dict(),
            'comb_feat': self.model.module.comb_feat.state_dict(),
            'recon_opt': self.optim_body_struct.state_dict(),
            'restore_opt': self.optim_deb_G.state_dict(),
            
        }, '{}/joint_model.h5'.format(self.weights_dir))
    
    def forward(self):
        for i in range(self.conf['num_epoch']):
            self.save_viz = True
            main_data = DataLoader(self.main_dset, batch_size = self.conf['batch_size'], shuffle= True, num_workers= 6, pin_memory= True)
            extra_data = DataLoader(self.extra_dset, batch_size = self.conf['extra_batch_size'], shuffle= True, num_workers= 2, pin_memory= True)
            tq = tqdm(main_data, total = len(extra_data) - 1)
            extra_data = iter(extra_data)
            self.total_data = len(extra_data) - 1
            raw_pred = self.train_per_epoch(tq, extra_data, i)
            self.save_model()
            self.sched_body_struct.step()
            self.sched_deb_G.step() 
            self.sched_deb_D.step()

    def update_D(self, x , y):
        self.optim_deb_D.zero_grad()
        loss_d = self.losses.get_loss_D(x, y, self.model)
        loss_d['loss_D'].backward(retain_graph=True)
        self.optim_deb_D.step()
        return loss_d
    def update_body_struct(self, x, y):
        self.optim_body_struct.zero_grad()
        loss_body, body_dict = self.losses.get_body_loss(x, y, return_raw= True)
        loss_body['loss_FULL'].backward(retain_graph=True)
        self.optim_body_struct.step()
        return loss_body, body_dict
    def update_deblur(self, x, y, pred_imgs, sharp_imgs):
        #Update only the discriminator
        loss_D = self.update_D(pred_imgs, sharp_imgs)
        #Update only the generator
        self.optim_deb_G.zero_grad()
        loss_all = self.losses.get_deblur_loss(x, y, pred_imgs, sharp_imgs, self.model)
        loss_all['loss_comb'].backward()
        self.optim_deb_G.step()
        return loss_all, loss_D
    def train_per_epoch(self, input, extra_input, epoch):
        num_iter = 0
        for data in input:
            extra_data = extra_input.next()
            gt_dset_name = data['dset_name']
            own_idxs, own_names = common.get_dset_idxs(gt_dset_name, 'own')

            main_sharp_imgs = data['sharp_img'].cuda()
            extra_sharp_imgs = extra_data['sharp_img'].cuda()

            sharp_imgs = torch.cat((main_sharp_imgs[own_idxs], extra_sharp_imgs), 0)
            """Update only the body network"""
            deb_imgs, params = self.model(data, extra_data, own_idxs, is_deblur= False)
            loss_body, body_dict = self.update_body_struct(data, params)
            del deb_imgs
            del params

            """Update only the deblur network"""
            deb_imgs, params = self.model(data, extra_data, own_idxs, is_deblur= True)
            loss_comb, loss_d = self.update_deblur(data, params, deb_imgs, sharp_imgs)
            
            """Plot all the losses in tensorboard"""
            if self.conf['tensorboard']:
                self.plot.write_body_loss(loss_body, self.step)
                self.plot.write_deblur_G_loss(loss_comb, self.step)
                self.plot.write_deblur_D_loss(loss_d, self.step)
            del loss_comb
            del loss_d
            del loss_body
            """Save the images"""
            if self.save_viz:
                #Save deblur result
                main_blur_imgs = data['img'].cuda()
                extra_blur_imgs = extra_data['img'].cuda()
                blur_imgs = torch.cat((main_blur_imgs[own_idxs], extra_sharp_imgs), 0)
                self.render.save_deblur(blur_imgs, sharp_imgs, deb_imgs.detach(), epoch)

                #save render
                pred_cam = torch.stack([body_dict['raw_pred']['camera'][:,1],
                                body_dict['raw_pred']['camera'][:,2],
                                2*self.conf['focal_length']/(self.conf['img_size'] * body_dict['raw_pred']['camera'][:,0] +1e-9)],dim=-1)
                
                self.render.save_overlay(pred_cam.detach().cpu(), body_dict['pred_param'].vertices.detach(), body_dict['gt_param'].vertices.detach(), main_blur_imgs, epoch)
                self.save_viz = False
            if num_iter == self.total_data:
                break
            num_iter+=1
            self.step+=1

            

            