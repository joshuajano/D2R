import cv2
import numpy as np
import os

import torch.utils.data as dutils
import torch
import torchvision

import glob
import h5py
from  loguru import logger
from torchvision.transforms import Normalize

from .curated import CuratedFittings
from .human36m import Human_36m
from .insta import Insta
from .spinx import Spinx
from datasets.insta_rand import InstaRand
from loguru import logger

# from dataloader import Human_36m_revised, CuratedFittings_revised, Own_dataset

class Mixed_Dataset(dutils.Dataset):
    def __init__(self, cfg, batch_size = 1):
        super(Mixed_Dataset, self).__init__()
        '''Batch size setting'''
        self.batch_size = batch_size

        '''List of datasets'''
        self.human_36m_dset = Human_36m(cfg['human36m'], batch_size)
        self.curated_dset = CuratedFittings(cfg['curated'], batch_size= batch_size)
        self.own_dset = Insta(cfg['insta'], batch_size= batch_size)
        '''Get all total dataset'''
        # self.human_36m_dset
        self.sum_dataset =  self.own_dset.__len__() + self.human_36m_dset.__len__()  + self.curated_dset.__len__()
        '''Dataset splits into based on portion''' 
        self.partition = [0.3, 0.3, 0.4]
        logger.debug('Mixed dataset with 30 H3.6M, 30 Curated, 40 Ours')
        self.partition = np.array(self.partition).cumsum()
    def __len__(self):
        return self.human_36m_dset.__len__()
    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(3):
            if p <= self.partition[0]:
                return self.human_36m_dset[index % len(self.human_36m_dset)]
            elif p <= self.partition[1]:
                return self.curated_dset[index % len(self.curated_dset)]
            else:
                return self.own_dset[index % len(self.own_dset)]
            
            
    