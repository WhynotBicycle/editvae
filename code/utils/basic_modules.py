import os
import os.path as osp

import torch
import torch.nn as nn
# import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


## 3D spatial transformation network (STN)
## a mini-PointNet that takes raw point cloud as input and regress to a 3 * 3 matrix
## trained as regularization loss with weight 0.001
class STN3D(nn.Module):
    def __init__(self, device):
        super(STN3D, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1) # batch * n_pts * 3 -> batch * n_pts * 64
        self.conv2 = torch.nn.Conv1d(64, 128, 1) # batch * n_pts * 64 -> batch * n_pts * 128
        self.conv3 = torch.nn.Conv1d(128, 1024, 1) # batch * n_pts * 128 -> batch * n_pts * 1024
        
        ## shared MLP on each points
        ## also act as full connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.device = device
        
    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
#         x = torch.mean(x, 2, keepdim = True)
        x = torch.max(x, 2, keepdim = True)[0]
        x = x.view(-1, 1024)
#         print(x.size())
        
        # fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) # x is 9*1
        
        ##################################### add identity matrix ###################################
        ## the output matrix is initizlized as identity matrix (for each patch)
        # to size batchsize * 9, each row is the identity 'matrix'
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 
                                                   0, 1, 0, 
                                                   0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        
#         if x.is_cuda:
        iden = iden.to(self.device)
            
        x = x + iden
        x = x.view(-1, 3, 3)
        
        return x

## PointNet to get local feature and global feature
## no 64 * 64 T-Net
## the input of point clouds batchsize * 3 * n_pts
## the False of global feature will still calculate the global feature, and will also return local feature
class PointNetfeat(nn.Module):
    def __init__(self, size=1024, trans=True, layers = [3, 64, 128, 128, 256], device='cuda:0', bias=True):
        super(PointNetfeat, self).__init__()
        self.device = device
        self.bias = bias
        if trans:
            self.stn = STN3D(device)
#         print(layers)
        self.midconvs = nn.ModuleList([nn.Conv1d(layers[idx], layers[idx+1], 1, bias=self.bias) for idx in range(len(layers)-1)])
        self.endconv = nn.Conv1d(layers[-1], size, 1, bias=self.bias)
        
        self.bns = nn.ModuleList([nn.BatchNorm1d(layers[idx]) for idx in range(1, len(layers))])
        self.endbn = nn.BatchNorm1d(size)
#         self.global_feat = global_feat
        self.trans = trans
        self.size = size
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self, x, global_feat=True):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        if self.trans:
            trans = self.stn(x) ## 3*3 T-Net for each batch in PointNet paper
            x = x.transpose(2, 1) ## input become batchsize * n_pts * 3
            x = torch.bmm(x, trans) ## matrix multiplication within each patch (n_pts * 3) * (3 * 3) -> (n_pts * 3)
            x = x.transpose(2, 1) ## become batchsize * 3 * n_pts
        
        for idx in range(len(self.midconvs)):
#             x = x.cuda()
            x = self.relu(self.bns[idx](self.midconvs[idx](x)))
        pointfeat = x ## as the point feature, batchsize * layers[-1] * n_pts
        x = self.endbn(self.endconv(x)) ## batchsize * size * n_pts
        
        if not global_feat:
            l_feat = x.transpose(2, 1) ## batchsize * size * n_pts
        
        ## get global feature
        g_feat = torch.max(x, 2, keepdim=True)[0]
        g_feat = g_feat.view(-1, self.size) # batchsize * size, global feature in PointNet
        

        if self.trans:
            if global_feat:
                return g_feat, trans
            else:
                return g_feat, l_feat, trans
        else:
            if global_feat:
                return g_feat
            else:
                return g_feat, l_feat
