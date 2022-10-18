import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import os
import shutil
import torch.optim as optim
import random
import numpy as np
from torch.autograd import Variable

from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.append('code/utils')
from measurement import mmd_from_generated_test, coverage_from_generated_test, JSD_generated_test, p2p_mmd_from_generated, p2p_coverage_from_generated, p2p_jsd_from_generated
from basic_modules import PointNetfeat
from loss_function import Calus

sys.path.append('code/TreeGAN')
from model.gan_network import Generator

sys.path.append('code/primitive_fitting')
from nets import SegNets
from sampler import EqualDistanceSamplerSQ
from loss import DualLoss
from p_utils import param2surface
from primitives import deform, quaternions_to_rotation_matrices

import subprocess

def create_identities(dim=64, n_parts=4):
    identities = []
    for i in range(n_parts):
        tmp = []
        for j in range(n_parts):
            tmp.append(torch.zeros([dim, dim]))
        tmp[i] = torch.eye(dim)
        identities.append(torch.cat(tmp, dim=0))
        
    return identities
identities = create_identities()

## KL divergence
def KL(mu, logvar, bias=0):
    tmp = - 0.5 * torch.mean(1 + logvar - (mu-bias).pow(2) - logvar.exp())
    return tmp

## sample latents from learned mu and logvar
def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    eps = Variable(eps)
    return mu + eps*std

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

class TrainVAE(nn.Module):
    def __init__(self, beta, batch_size, train_loader, test_loader, category, device, n_parts=8, n_latents=256, epoch=1000):
        super().__init__()
        
        self.n_parts = n_parts
        self.n_latents = n_latents
        self.epoch = epoch
        self.device = device
        self.batch_size = batch_size
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.beta = beta
        self.category = category
        
        self.train_len = len(self.train_loader)
        self.test_len = len(self.test_loader)
        
        self.colors = [[255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255],
                       [0, 255, 255],
                       [255, 0, 255],
                       [255, 255, 0],
                       [128, 128, 128],
                       [128, 128, 0]]
        
        self.sampler = EqualDistanceSamplerSQ(n_samples=200, D_eta=0.05, D_omega=0.05)
       
    def save_model(self, addition=''):
        torch.save(self.model.state_dict(), self.path+'model/'+self.md_name+addition+'.pth')
        
    def load(self, addition=''):
        self.model = self.model.cpu()
        self.model.load_state_dict(torch.load(self.path+'model/'+self.md_name+addition+'.pth', map_location='cpu'))
        self.model = self.model.to(self.device)
        
    def mmd_generate(self, train=False, distance='CD', parts_only = False, per_parts = False, sp = None, device=None, symmetry=False):
        if train:
            dt = self.train_loader
        else:
            dt = self.test_loader
        if per_parts:
            return p2p_mmd_from_generated(dt, self.path+'generated/', self.device if device is None else device, self.train_len+self.test_len*self.n_parts, self.train_len, train, distance, self.n_parts, sp)
        else:
            return mmd_from_generated_test(dt, self.path+'generated/', self.device if device is None else device, self.train_len+self.test_len if not parts_only else self.train_len+self.test_len*self.n_parts, self.train_len, train, distance, parts_based=parts_only, symmetry=symmetry)
        
    def cov_generate(self, train=False, distance='CD', per_parts = False, sp = None, device=None, symmetry=False):
        if train:
            dt = self.train_loader
        else:
            dt = self.test_loader
        if per_parts:
            return p2p_coverage_from_generated(dt, self.path+'generated/', self.device if device is None else device, self.train_len+self.test_len*self.n_parts, self.train_len, self.batch_size, train, distance, self.n_parts, sp)
        else:
            return coverage_from_generated_test(dt, self.path+'generated/', self.device if device is None else device, self.train_len+self.test_len, self.train_len, self.batch_size, train, distance, symmetry=symmetry)
        
    def jsd_generate(self, train=False, per_parts = False, sp = None, device=None, symmetry=False, n_parts=None):
        if n_parts is None:
            n_parts = self.n_parts
        if train:
            dt = self.train_loader
        else:
            dt = self.test_loader
        if per_parts:
            return p2p_jsd_from_generated(dt, self.path+'generated/', self.train_len+self.test_len*n_parts, self.train_len, train, n_parts, sp, self.device if device is None else device)
        else:
            return JSD_generated_test(dt, self.path+'generated/', self.train_len+self.test_len, self.train_len, train, symmetry=symmetry)
    
    def baseline_pt_list(self, point, seg, rg=None, bias=None):
        pt_size = point.size()
        seg_size = seg.size()
        assert pt_size[0] == seg_size[0]
        assert pt_size[1] == seg_size[1]
        
        if rg is None:
            rg = self.n_parts
        if bias is None:
            bias = 0

        pt_list = []
        for b in range(pt_size[0]):
            pt = point[b,:,:]
            sg = seg[b,:]
            
            sub_pt_list = []
            for n in range(0+bias, rg+bias):
                idx = (sg==n).nonzero(as_tuple=True)[0]
                sub_pt = torch.index_select(pt, 0, idx)
#                 if sub_pt.size(0) == 0:
#                     continue
                sub_pt_list.append(sub_pt)
            pt_list.append(sub_pt_list)

        pt_list = [pad_sequence(item).transpose(0, 1) for item in zip(*pt_list)]

        return pt_list
        
    def sample_from_encoder(self):
        self.records = []
        with torch.no_grad():
            for _iter, data in enumerate(self.train_loader):
                point, _ = data
                point = point.to(self.device)

                _, mu, logvar, _ = self.model(point.transpose(1,2))
                self.records.append([mu, logvar])
            
    def M_Gaussian(self):
        mu, logvar = random.choice(self.records)
        return reparameterize(mu, logvar)
    
    def _single_opt(self, opt, idx):
        out = [None]*len(opt)
        for i in range(len(opt)):
            out[i] = opt[i][idx,...]
        return out   