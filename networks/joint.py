import torch
import torch.nn as nn
import numpy as np
import math
from loguru import logger
import functools
from collections import OrderedDict
from networks.hmr import Bottleneck, HMR
from networks.fpn import FPNInception
from networks.discriminator import NLayerDiscriminator
from networks.structfeat import StructureFeat
from utils import common
class JointNet(nn.Module):
    def __init__(self, cfg, param_mean):
        self.conf = cfg
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
        super().__init__()
        #======= Model Initialization =======
        self.bodyNet = HMR(Bottleneck, [3, 4, 6, 3],  param_mean)
        self.deb_G = FPNInception()
        self.deb_D_Patch = NLayerDiscriminator(n_layers=3, norm_layer=norm_layer, use_sigmoid=False)
        self.deb_D_Full = NLayerDiscriminator(n_layers=5, norm_layer=norm_layer, use_sigmoid=False)
        
        logger.debug('Body network, DeblurGAN-V2 with 2 discriminator have been initialized')
        #======= Load the pretrain network =======
        if cfg['pretrained_weights']:
            logger.warning('Pretrained are loaded')
            deblur_weights = torch.load(cfg['deblur']['weight_path'])

            deb_G_w = self.load_weights(deblur_weights['model'])
            self.deb_G.load_state_dict(deb_G_w)
            logger.warning('Deblur generator  sucessfully loaded')

            deb_D_Full_w = self.load_weights(deblur_weights['disc_full'])
            self.deb_D_Full.load_state_dict(deb_D_Full_w)
            logger.warning('Deblur full discriminator sucessfully loaded')
            
            deb_D_Patch_w = self.load_weights(deblur_weights['disc_patch'])
            self.deb_D_Patch.load_state_dict(deb_D_Patch_w)
            logger.warning('Deblur patch discriminator sucessfully loaded')
            
            body_weights = torch.load(cfg['recon']['weight_path'])
            bodyNet_w = self.load_weights(body_weights['model'])
            self.bodyNet.load_state_dict(bodyNet_w)
            logger.warning('Body network sucessfully loaded')
        #========================================
        #Structure 
        self.struct_feat = StructureFeat()
        self.comb_feat = self.make_conv_layers([128, 64], max_pool=False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').reshape(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').reshape(1,3,1,1)
    def freeze_and_unfreeze(self, is_deblur=True):
        if is_deblur:
            for param in self.deb_G.parameters():
                param.requires_grad = True
            for param in self.bodyNet.parameters():
                param.requires_grad = False
            for param in self.struct_feat.parameters():
                param.requires_grad = False
            for param in self.comb_feat.parameters():
                param.requires_grad = False
        else:
            for param in self.deb_G.parameters():
                param.requires_grad = False
            for param in self.bodyNet.parameters():
                param.requires_grad = True
            for param in self.struct_feat.parameters():
                param.requires_grad = True
            for param in self.comb_feat.parameters():
                param.requires_grad = True
    def forward(self, x, extra_x, hmb_idxs, is_deblur =True):
        self.freeze_and_unfreeze(is_deblur)
        batch_human = len(hmb_idxs)
        imgs = x['img'].to('cuda')
        extra_imgs = extra_x['img'].to('cuda')

        # blur_imgs = imgs[hmb_idxs]
        blur_imgs = torch.cat((imgs[hmb_idxs], extra_imgs), 0)
        
        body_input = imgs.clone()
        
        deb_imgs = self.deb_G(blur_imgs)
        pred_imgs = deb_imgs.clone()

        blur_imgs = blur_imgs[:batch_human].clone() + 1 /2.0
        #exclude the random cropping

        deb_imgs =  deb_imgs[:batch_human].clone() + 1 /2.0
        strfeat = self.struct_feat(blur_imgs, deb_imgs)
        deb_imgs = deb_imgs.clone() - self.mean
        deb_imgs = deb_imgs / self.std

        body_input[hmb_idxs] = deb_imgs
        pred_params = self.bodyNet(body_input, strfeat, self.comb_feat, hmb_idxs, is_struct=True)
        
        return pred_imgs, pred_params 
    def load_weights(self, weights):
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    def make_conv_layers(self, feat_dims, kernel=3, stride=1, padding=1, max_pool = True, bnrelu_final=True):
        layers = []
        for i in range(len(feat_dims)-1):
            layers.append(
                nn.Conv2d(
                    in_channels=feat_dims[i],
                    out_channels=feat_dims[i+1],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding
                    ))
            # Do not use BN and ReLU for final estimation
            if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
                layers.append(nn.BatchNorm2d(feat_dims[i+1]))
                layers.append(nn.ReLU(inplace=True))
            if max_pool:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)