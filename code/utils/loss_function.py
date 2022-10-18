import os
import os.path as osp

import torch
import torch.nn as nn
# import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from numpy.random import choice

import numpy as np

alpha1 = 0.05
alpha2 = 0.05

def BatchedChamferDistance(x, y):
    # compute Chamfer distance between two point clouds
    # x, y: batchsize * n_pts * 3
    
    x_size = x.size()
    y_size = y.size()
    assert( x_size[0] == y_size[0])
    assert( x_size[2] == y_size[2]) # batchsize and coordinate dimensions should be equal
    x = torch.unsqueeze(x, 1) # batchsize * 1 * n_pts_x * 3
    y = torch.unsqueeze(y, 2) # batchsize * n_pts_y * 1 * 3
    
    x = x.repeat(1, y_size[1], 1, 1) # batchsize * n_pts_y * n_pts_x * 3
    y = y.repeat(1, 1, x_size[1], 1) # batchsize * n_pts_y * n_pts_x * 3
    # print(x.element_size() * x.nelement())
    # print(y.element_size() * y.nelement())
    
    ## calculate the min square Euclidean distance of each point to another cloud
    distance = x - y # difference between each point in x and y
    distance = torch.pow(distance, 2) # power of difference
    distance = torch.sum(distance, 3, keepdim=True) # square of Euclidean distance in x, y, z axis
    distance = torch.squeeze(distance, 3) # batchsize * n_pts_y * n_pts_x
    x_min_distance, _ = torch.min(distance, 1, keepdim=True) # batchsize * 1 * n_pts_x, for each point in x, calculate the min distance to y
    y_min_distance, _ = torch.min(distance, 2, keepdim=True) # same, now with respect to y
    
    # the mean of the distance, batchsize * 1 * 1
    x_mean_distance = torch.mean(x_min_distance, 2, keepdim=True)
    y_mean_distance = torch.mean(y_min_distance, 1, keepdim=True)
    
    # the max of mean distance
#     mean_distance = torch.cat((x_mean_distance, y_mean_distance), 1) # 1 * 2
#     chamfer_distance = torch.sum(mean_distance)
    chamfer_distance = (x_mean_distance + y_mean_distance).mean()

    return chamfer_distance

def _remove_zero_rows(x):
    index = x.sum(1).nonzero(as_tuple=True)[0]
    return torch.index_select(x, 0, index)

def _point_level_calus(x, y):
    # compute Chamfer distance, coverage, quality between two point clouds
    # x, y: n_pts * 3
    # only works for single batch, different batchs may have different number of unique index, impossible to construct such a tensor.matrix with different row length
    
#     print(x.size())
#     print(x.sum())
#     print(x.size())

#     x = _remove_zero_rows(x)
#     y = _remove_zero_rows(y)
    
#     ## zero size segmentation is not encouraged
#     if x.size(0) == 0 or y.size(0) == 0:
#         return torch.tensor(100, dtype=torch.float32).cuda(), 1.0, 1.0
# #     print(x.size())

    x_size = x.size()
    y_size = y.size()
#     print(x.size())
    
    x = torch.unsqueeze(x, 0) # 1 * n_pts_x * 3
    y = torch.unsqueeze(y, 1) # n_pts_x * 1 * 3


    x = x.repeat(y_size[0], 1, 1) # n_pts_y * n_pts_x * 3
    y = y.repeat(1, x_size[0], 1) # n_pts_y * n_pts_x * 3
    
    ## calculate the min square Euclidean distance of each point to another cloud
    distance = x - y # difference between each point in x and y
    distance = torch.pow(distance, 2) # power of difference
    distance = torch.sum(distance, 2, keepdim=True) # square of Euclidean distance in x, y, z axis
#     distance = torch.sqrt(distance)
    distance = torch.squeeze(distance, 2) # n_pts_y * n_pts_x
    
    ## for chamfer distance
    x_min_distance, x_min_index = torch.min(distance, 0, keepdim=True) # 1 * n_pts_x, for each point in x, calculate the min distance to y
    y_min_distance, y_min_index = torch.min(distance, 1, keepdim=True) # same, now with respect to y
    
    # the mean of the distance, batchsize * 1 * 1
    x_mean_distance = torch.mean(x_min_distance, 1, keepdim=True)
    y_mean_distance = torch.mean(y_min_distance, 0, keepdim=True)
    
    # the max of mean distance
    mean_distance = torch.cat((x_mean_distance, y_mean_distance), 1) # 1 * 2
    chamfer_distance = torch.sum(mean_distance)
    
    # for coverage
    cov = torch.unique(x_min_index).size(0)/y_size[0]
    
    # for quality
    quality = torch.unique(y_min_index).size(0)/x_size[0]
    
    return chamfer_distance, cov, quality

def Calus(x, y, device='cuda:0', probs=None):
    # compute Chamfer distance, Coverage, quality within original point cloud (x) and the generated one (y)
    # x, y: batchsize * n_pts * 3
    # probs: batchsize
    x_size = x.size()
    y_size = y.size()
    
    if x_size[0] != y_size[0]:
        return torch.tensor(-1)
#     assert( x_size[0] == y_size[0])
    assert( x_size[2] == y_size[2] == 3) # batchsize and coordinate dimensions should be equal
    
    if device is None:
        cds = torch.tensor(0.0).cuda()
    else:
        cds = torch.tensor(0.0).to(device)
    covs = []
    quals = []
    x_list = x.split(1)
    y_list = y.split(1)
    for i in range(len(x_list)):
        sub_x = x_list[i].squeeze(0)
        sub_y = y_list[i].squeeze(0)
        
        sub_x = _remove_zero_rows(sub_x)
        sub_y = _remove_zero_rows(sub_y)
        if sub_x.size(0) == 0 or sub_y.size(0) == 0:
            continue
            
        cd, cov, qual = _point_level_calus(sub_x, sub_y)
        
        if probs is not None:
            cd = cd * probs[i]
            cov = cov * probs[i].item()
            qual = qual * probs[i].item()
        
        cds = torch.cat((cds.view(1,-1), cd.view(1,-1)), dim=1)[0]
        covs.append(cov)
        quals.append(qual)
    
    if cds.view(1, -1).size(1) == 1:
        return None, None, None
    else:
        chamfer_distance = torch.mean(cds[1:])
        mean_cov = sum(covs)/len(covs)
        mean_qual = sum(quals)/len(quals)
        return chamfer_distance, mean_cov, mean_qual

def _single_chamfer_distance(x, y):
    # compute Chamfer distance between two point clouds
    # x, y: n_pts * 3
    
#     print(y.size())
    x = _remove_zero_rows(x)
    y = _remove_zero_rows(y)

    x_size = x.size()
    y_size = y.size()
#     print(y.size())
    
    x = torch.unsqueeze(x, 0) # 1 * n_pts_x * 3
    y = torch.unsqueeze(y, 1) # n_pts_x * 1 * 3


    x = x.repeat(y_size[0], 1, 1) # n_pts_y * n_pts_x * 3
    y = y.repeat(1, x_size[0], 1) # n_pts_y * n_pts_x * 3
    
    ## calculate the min square Euclidean distance of each point to another cloud
    distance = x - y # difference between each point in x and y
    distance = torch.pow(distance, 2) # power of difference
    distance = torch.sum(distance, 2, keepdim=True) # square of Euclidean distance in x, y, z axis
#     distance = torch.sqrt(distance)
    distance = torch.squeeze(distance, 2) # n_pts_y * n_pts_x
    x_min_distance, _ = torch.min(distance, 0, keepdim=True) # 1 * n_pts_x, for each point in x, calculate the min distance to y
    y_min_distance, _ = torch.min(distance, 1, keepdim=True) # same, now with respect to y
    
    # the mean of the distance, batchsize * 1 * 1
    x_mean_distance = torch.mean(x_min_distance, 1, keepdim=True)
    y_mean_distance = torch.mean(y_min_distance, 0, keepdim=True)
    
    # the max of mean distance
    mean_distance = torch.cat((x_mean_distance, y_mean_distance), 1) # 1 * 2
#     chamfer_distance = torch.max(mean_distance) # 1 * 1
#     chamfer_distance = torch.mean(mean_distance)
    chamfer_distance = torch.sum(mean_distance)
    
    return chamfer_distance

def ChamferDistance(x, y, take_mean=True, device = 'cuda:0'):
    # compute Chamfer distance between two point clouds
    # x, y: batchsize * n_pts * 3
    
    x_size = x.size()
    y_size = y.size()
    
    if x_size[0] != y_size[0]:
        return torch.tensor(-1)
#     assert( x_size[0] == y_size[0])
    assert( x_size[2] == y_size[2] == 3) # batchsize and coordinate dimensions should be equal
    
    cds = torch.tensor(0.0).to(device)
    x_list = x.split(1)
    y_list = y.split(1)
    for i in range(len(x_list)):
#         print(i)
        sub_x = x_list[i].squeeze(0)
        sub_y = y_list[i].squeeze(0)
        cd = _single_chamfer_distance(sub_x, sub_y)
        cds = torch.cat((cds.view(1,-1), cd.view(1,-1)), dim=1)[0]
    # finally, the mean of Chamfer distance for each patch
    chamfer_distance = torch.mean(cds[1:])
    
#     torch.cuda.empty_cache()
    return chamfer_distance


class ChamferLoss(nn.Module):
    def __init__(self, device = 'cuda:0'):
        super(ChamferLoss, self).__init__()
        self.device = device
        
    def forward(self, x, y):
        return ChamferDistance(x, y, device = self.device)

class NormalizedLoss(nn.Module):
    def __init__(self, device = 'cuda:0', wcd = 1, wcov = 0.0001, wqual = 0.0001):
        super(NormalizedLoss, self).__init__()
        self.device = device
        self.wcd = wcd
        self.wcov = wcov
        self.wqual = wqual
        
    def forward(self, x, y):
        cd, cov, qual = Calus(x, y, device = self.device)
        val = self.wcd * cd - self.wcov * cov - self.wqual * qual
        return val, cd, cov, qual

# assume 1 is the label of real and 0 is the label of fake
def crossEntropy(p, label):
    ## the input of p is batch * 1
    
    return torch.mean(- label * torch.log(p) - (1-label) * torch.log(1-p))



# KL divergence
# details see Appendix B from VAE paper: https://arxiv.org/abs/1312.6114
# KLD = 0.5 * sum(1+log(sigma^2) - mu^2 - sigma^2)
# argmax KLD = argmin -KLD
def KLDivergence(mu, logvar):
    return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class encoderGANLoss(nn.Module):
    def __init__(self):
        super(encoderGANLoss, self).__init__()
        
    def forward(self, raw, generated, real_p, real_label, fake_p, fake_label):
        ce_real = crossEntropy(real_p, real_label)
        ce_fake = crossEntropy(fake_p, fake_label)
        cd = ChamferDistance(raw, generated)
        
        return alpha1 * (ce_real + ce_fake) + cd


class softmax_BCELoss(nn.Module):
    def __init__(self):
        super(softmax_BCELoss, self).__init__()
        self.BCELoss = nn.BCELoss()
        
    def forward(self, pred, label):
        bce = self.BCELoss(pred, label)
        
        return bce


class VAELoss(nn.Module):
	def __init__(self):
		super(VAELoss, self).__init__()

	def forward(self, raw, generated, mu, logvar):
		cd = ChamferDistance(raw, generated)
		kld = KLDivergence(mu, logvar)

		return cd + alpha2 * kld

