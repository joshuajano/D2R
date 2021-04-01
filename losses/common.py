import pickle
import numpy as np
from torchvision.models import vgg19
import torch.nn as nn
import torchvision.transforms as transforms
import torch
def compute_mse(pred, gt):
    return nn.MSELoss()(pred, gt)
def compute_mae(pred, gt):
    return nn.L1Loss()(pred, gt)
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        conv_3_3_layer = 14
        feat = vgg19(pretrained=True).features
        feat.cuda()
        self.model = nn.Sequential()
        self.model = self.model.cuda()
        self.model = self.model.eval()
        for i, layer in enumerate(list(feat)):
            self.model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def forward(self, pred, gt):
        predIm = (pred + 1) / 2.0
        gtIm = (gt + 1) / 2.0
        l_pc = torch.zeros(gtIm.shape[0]).cuda()
        # with torch.no_grad():
        for i in range(gtIm.shape[0]):
            gtIm_n = self.transform(gtIm[i]).unsqueeze(dim=0)
            predIm_n = self.transform(predIm[i]).unsqueeze(dim=0)
            f_pred = self.model(predIm_n)
            f_gt = self.model(gtIm_n)
            l_pc[i] = compute_mse(f_pred, f_gt)
        return torch.mean(l_pc) 

class PixelDistanceLoss(nn.Module):
    def __init__(self):
        super(PixelDistanceLoss, self).__init__()
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def forward(self, pred, gt):
        predIm = (pred + 1) / 2.0
        gtIm = (gt + 1) / 2.0
        l_pd = torch.zeros(gtIm.shape[0]).cuda()
        # with torch.no_grad():
        for i in range(gtIm.shape[0]):
            gtIm_n = self.transform(gtIm[i]).unsqueeze(dim=0)
            predIm_n = self.transform(predIm[i]).unsqueeze(dim=0)
            l_pd[i] = compute_mse(predIm_n, gtIm_n)
        return torch.mean(l_pd) 