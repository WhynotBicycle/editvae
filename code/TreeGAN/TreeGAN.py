#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
#%matplotlib inline
import os
import os.path as osp
import subprocess
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torch.utils.tensorboard import SummaryWriter


# In[2]:


from model.gan_network import Generator, Discriminator
from model.gradient_penalty import GradientPenalty

import sys
sys.path.append('code/utils')
from measurement import coverage_from_generator
from measurement import quality_from_generator
from measurement import coverage_from_batch
from loss_function import ChamferDistance
from loss_function import ChamferLoss

from encoder_decoder import Encoder


# In[3]:


batch_size = 6
workers = 6
point_num = 2048
G_FEAT=[96, 256, 256, 256, 128, 128, 128, 3]
D_FEAT=[3,  64,  128, 256, 512, 1024]
DEGREE=[1,  2,   2,   2,   2,   2,   64]
support=10
lambdaGP=10
g_lr=1e-4
d_lr=1e-4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
epochs=1000


# In[4]:


############################## Data loader ####################################
print('Loading data.........')
# Root directory for dataset
dataroot = '/home/shidi/3d-generate/data/shapeNet/ShapeNetCore.v2/'

category = 'Airplane'
path = osp.join(osp.dirname(osp.abspath('')), '..', 'data', 'shapeNet', 'ShapeNetCore.v2')
# when in .py file
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'shapeNet', 'ShapeNetCore.v2')

transform = T.Compose([
#     T.RandomTranslate(0.01), #translate node positions by randomly sampled translation value, within (-0.01, 0.01)
#     T.RandomRotate(15, axis = 0), # rotate axis 0 with degree sampled in (-15, 15)
#     T.RandomRotate(15, axis = 1),
#     T.RandomRotate(15, axis=2),
    T.FixedPoints(point_num)
])
pre_transform = T.NormalizeScale() # centers and normalizes node positions to the interval (-1, 1)
train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
                        pre_transform=pre_transform)
test_dataset = ShapeNet(path, category, split='test',
                        pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=workers)


# In[5]:


class TreeGAN():
    def __init__(self, train_loader, test_loader):
        # ------------------------------------------------Dataset---------------------------------------------- #
        self.test_loader = test_loader
        self.train_loader = train_loader
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        self.G = Generator(batch_size=batch_size, features=G_FEAT, degrees=DEGREE, support=support).to(device)
        self.D = Discriminator(batch_size=batch_size, features=D_FEAT).to(device)
        self.E = Encoder(dim=G_FEAT[0]).to(device)
        
        self.optimizerG = optim.Adam(self.G.parameters(), lr=g_lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=d_lr, betas=(0, 0.99))
        self.optimizerE = optim.Adam(self.E.parameters(), lr=g_lr, betas=(0, 0.99))

        self.GP = GradientPenalty(lambdaGP, gamma=1, device=device)
        self.EncoderLoss = ChamferLoss()
        print("Network prepared.")
        # ----------------------------------------------------------------------------------------------------- #
        
        self.d_losses = []
        self.g_losses = []
        self.c_losses = []
        self.save_path = '../../data/TreeGAN/baseline/'
    
    def _save_losses(self, filename, data):
        with open(self.save_path+'losses/'+filename,'w') as f:
            for line in data:
                f.write(str(line) + '\n')
        
    def save_losses(self):
        self._save_losses('d_losses.txt', self.d_losses)
        self._save_losses('g_losses.txt', self.g_losses)
        self._save_losses('c_losses.txt', self.c_losses)
    
    def save_model(self):
        torch.save(self.G.state_dict(), self.save_path+'model/g'+category+str(epochs)+'.pth')
        torch.save(self.D.state_dict(), self.save_path+'model/d'+category+str(epochs)+'.pth')
    
    def load(self):
        self.G.load_state_dict(torch.load(self.save_path+'model/g'+category+str(epochs)+'.pth'))
        self.D.load_state_dict(torch.load(self.save_path+'model/d'+category+str(epochs)+'.pth'))
    
    def show_examples(self):
        z = torch.randn(batch_size, 1, 96).to(device)
        tree = [z]
        fake_point = self.G(tree)
        for i in range(batch_size):
            out = fake_point[i].unsqueeze(0)
            out_color = torch.as_tensor(torch.tensor([0, 0, 255]).repeat(out.size()[1], 1), dtype=torch.int).unsqueeze(0)
            writer.add_mesh('generated samples '+str(i), vertices=out, colors=out_color)
            
    def save_generated(self):
        for i in range(392):
            z = torch.randn(batch_size, 1, 96).to(device)
            tree = [z]
            fake_point = self.G(tree)
            torch.save(fake_point, self.save_path+'generated/'+str(i)+'.pt')
        print('All saved !')
    
    def coverage(self):
        return coverage_from_generator(self.test_loader, self.save_path+'generated/', device)
#         return quality_from_generator(self.train_loader, self.save_path+'generated/', device)
    def coverage_recon(self):
        return coverage_from_batch(self.test_loader, self.G, device, True, self.E)

    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):  
        writer = SummaryWriter('../../runs/WGAN_'+category+str(epochs))
        for epoch in range(0, epochs):
            for _iter, data in enumerate(self.train_loader):
                point = pad_sequence([inp['pos'] for inp in data.to_data_list()], batch_first=True)
                point = point.to(device)
#                 print(point.size())
                # -------------------- Discriminator -------------------- #
                for d_iter in range(5):
                    self.D.zero_grad()
                    
                    z = torch.randn(batch_size, 1, 96).to(device)
                    tree = [z]
                    
                    with torch.no_grad():
                        fake_point = self.G(tree)         
                        
                    D_real = self.D(point)
                    D_realm = D_real.mean()

                    D_fake = self.D(fake_point)
                    D_fakem = D_fake.mean()

                    gp_loss = self.GP(self.D, point.data, fake_point.data)
                    
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    d_loss_gp.backward()
                    self.optimizerD.step()
                # ---------------------- Generator ---------------------- #
                self.G.zero_grad()
                
                z = torch.randn(batch_size, 1, 96).to(device)
                tree = [z]
                
                fake_point = self.G(tree)
                G_fake = self.D(fake_point)
                G_fakem = G_fake.mean()
                
                g_loss = -G_fakem
                g_loss.backward()
                self.optimizerG.step()
                
                gmcd = ChamferDistance(point, fake_point)
                # --------------------- Encoder & Generator -------------- #
                z, _ = self.E(point.transpose(2,1))
                z = z.unsqueeze(1)
                if z.size(0) == batch_size: ## because code for TreeGAN only deal with fixed number of batch size
                    self.E.zero_grad()
                    self.G.zero_grad()
                    tree = [z]
                    fake_point = self.G(tree)
                    mcd = self.EncoderLoss(point, fake_point)
                    mcd.backward()
                    self.optimizerE.step()
                    self.optimizerG.step()
                else: # only for ploting
                    mcd = ChamferDistance(point, fake_point)
                # --------------------- Chamfer distance ----------------- #
                self.d_losses.append(d_loss.item())
                self.g_losses.append(g_loss.item())
                self.c_losses.append(mcd.item())
                # --------------------- Visualization -------------------- #
#                 print(_iter)
#                 if _iter % 30 == 29:
                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                      "[ G_Loss ] ", "{: 7.6f}".format(g_loss),
                      "[ Chamfer Loss] ", "{: 7.6f}".format(mcd),
                      "[ Generated Chamfer Loss] ", "{: 7.6f}".format(gmcd))

            # ---------------------- Plot on the tensorboard ------------- #
            ## print the first point cloud in the last batch each epoch
            if epoch % 50 == 0:
    #             ipt = point[0].unsqueeze(0)
                out = fake_point[0].unsqueeze(0)
                out_color = torch.as_tensor(torch.tensor([0, 0, 255]).repeat(out.size()[1], 1), dtype=torch.int).unsqueeze(0)
                writer.add_mesh('output '+ str(epoch), vertices=out, colors=out_color)
                
            # ---------------------- Save losses & model ----------------- #
            self.save_losses()
            self.save_model()
        writer.close()


# In[6]:


model = TreeGAN(train_loader, test_loader)
# print(model.G)
model.run()
# model.load()
model.save_generated()
model.coverage()
# model.coverage_recon()
# model.show_examples()


# In[ ]:





# In[ ]:




