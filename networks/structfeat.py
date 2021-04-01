import torch
import torch.nn as nn
from utils.structure import SSIM
class StructureFeat(nn.Module):
    def __init__(self):
        super().__init__()
        #Structure to feature space
        self.struct2feat_conv = self.make_conv_layers([3, 64, 64]) 
        self.img2struct = SSIM()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').reshape(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').reshape(1,3,1,1)
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
    def get_structure_info(self, img1, img2):
        inp1 = img1 
        inp2 = img2 
        _, structure_map = self.img2struct(inp1.clone(), inp2.clone())
        dssim = 1 - structure_map
        return dssim
    def forward(self, x, y):
        with torch.no_grad():
            struct_img = self.get_structure_info(x.clone(), y.clone())
        feat_deb = self.struct2feat_conv(struct_img)
        return feat_deb