#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
#get_ipython().run_line_magic('matplotlib', 'notebook')
#%matplotlib inline
import numpy as np
import math
import random

import torch
import torch.nn.parallel


# In[2]:


import sys

from sVAEs_module import TrainSVAE

sys.path.append('code/TreeGAN/dataloader')
from dataset_benchmark import BenchmarkDataset, BenchmarkDatasetOnTheFly


# In[3]:


SEED = 0 
random.seed(SEED) 
np.random.seed(SEED) 
torch.manual_seed(SEED) 


# In[4]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
# device = torch.device('cpu')
print(device)


# In[5]:


sp_reg_terms = {"regularizer_type": ['bernoulli_regularizer', 'parsimony_regularizer', 'overlapping_regularizer'],
                 "bernoulli_regularizer_weight": 1,
                 "entropy_bernoulli_regularizer_weight": 0,
                 "parsimony_regularizer_weight": 1e-3,
                 "sparsity_regularizer_weight": 0,
                 "overlapping_regularizer_weight": 1e-5,
                 "minimum_number_of_primitives": 4,
                 "maximum_number_of_primitives": 4,
                 "diversity_regularizer_weight": 0,
                 "w1": 0.005,
                 "w2": 0.005}


# In[6]:


############################## Data loader ####################################
print('Loading data.........')
# Root directory for dataset
dataroot = 'data/datasetTreeGAN/shapenetcore_partanno_segmentation_benchmark_v0/'
# dataroot = '../../../scripts/neural_parts/data/ShapeNet'

# labs = False
# duplicated = True
# down_sample = False
labs = False
duplicated = False
down_sample = True
category = 'chair'
#data = BenchmarkDataset(dataroot, npoints=2048, uniform=False, classification=False, down_sample = down_sample,
#                          duplicated = duplicated, class_choice=category, device=device)
data = BenchmarkDatasetOnTheFly(dataroot, npoints=2048, uniform=False, classification=False, down_sample=down_sample,
                                 duplicated = duplicated, labs = labs, class_choice=category, device=device)
# data = BenchmarkDatasetOnTheFly(dataroot, npoints=2048, class_choice=category, device=device)

split = 0.9
total_num = data.__len__()
train_num = math.floor(total_num*split)
test_num = total_num - train_num
train_set, test_set = torch.utils.data.random_split(data, [train_num, test_num])

batch_size = 30
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=0, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0, drop_last=True)

print(total_num)


# In[7]:


vae = TrainSVAE(beta=5e-3, batch_size=batch_size, epoch=2,
                train_loader=train_loader, test_loader=test_loader,
                category = category, device=device,
                sp_reg_terms=sp_reg_terms, use_realNVP=True)


# In[8]:


vae.run()
# vae.load()
# vae.run()
# vae.compare()
# vae.duplicate()
# vae.reconstruction()
# vae.train_prim()
# vae.vis()
# vae.vis_generated(distribution='S_Gaussian')
# flow = vae.realNVP()
# vae.load_realNVP()
# vae.shape_sampling_test(from_local=False)
# vae.shape_mixing(rescale=True)
# vae.shape_resize()
# vae.fig_plot()


# In[ ]:


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[ ]:


# print(count_parameters(vae.model))


# In[ ]:


# vae.load()
# print('======= Standard Gaussian results =========')
# st = vae.param_stat()
# print('======= Mix Gaussian results =========')
# st_m = vae.param_stat('M_Gaussian')
# print('======= Difference =========')
# names = ['R mean diff: ', 'R var diff: ',
#          'T mean diff: ', 'T var diff: ',
#          'Size mean diff: ', 'Size var diff: ',
#          'Shape mean diff: ', 'Shape var diff: ',
#          'D mean diff: ', 'D var diff: ']
# for i in range(len(st)):
#     print(names[i], (st[i]-st_m[i]).detach().cpu().numpy())


# In[ ]:


# print(st[0].size())


# In[ ]:


# vae.estimate_latent()


# In[ ]:


# vae.vis_generated(distribution='S_Gaussian')
# vae.vis_reconstruction()


# In[ ]:


# vae.vis()


# In[ ]:


vae.generate('M_Gaussian')
print('Mixed Gaussian, MMD-CD: ', vae.mmd_generate(device=device))
print('Mixed Gaussian, COV-CD: ', vae.cov_generate(device=device))
# # # print('Mixed Gaussian, COV-EMD: ', vae.cov_generate(distance='EMD', device=device))
print('Mixed Gaussian, JSD: ', vae.jsd_generate())
# vae.generate('M_Gaussian', parts=True)
# print('Mixed Gaussian, MMD-CD per-Parts: ', vae.mmd_generate(per_parts=True, sp=vae.get_seg, device=device))
# # print('Mixed Gaussian, COV-CD per-Parts: ', vae.cov_generate(per_parts=True, sp=vae.get_seg, device=device))
# # print('Mixed Gaussian, COV-EMD per-Parts: ', vae.cov_generate(distance='EMD', per_parts=True, device=device))
# # print('Mixed Gaussian, JSD per-Parts: ', vae.jsd_generate(per_parts=True, sp=vae.get_seg, device=device))

vae.generate('S_Gaussian')
# # # # # # # vae.generate_from_NVP()
print('Standard Gaussian, MMD-CD: ', vae.mmd_generate(device=device, symmetry=False))
print('Standard Gaussian, COV-CD: ', vae.cov_generate(device=device, symmetry=False))
print('Standard Gaussian, JSD: ', vae.jsd_generate(symmetry=False))
print('Standard Gaussian, MMD-EMD: ', vae.mmd_generate(distance='EMD', device=device, symmetry=False))
print('Standard Gaussian, COV-EMD: ', vae.cov_generate(distance='EMD', device=device, symmetry=False))
# vae.generate('S_Gaussian', parts=True)
# # # # vae.generate_from_NVP(parts=True)
# # print('Standard Gaussian, MMD-CD per-Parts: ', vae.mmd_generate(per_parts=True, sp=vae.get_seg, device=device))
# # # # print('Standard Gaussian, COV-CD per-Parts: ', vae.cov_generate(per_parts=True, sp=vae.get_seg, device=device))
# # # # print('Standard Gaussian, COV-EMD per-Parts: ', vae.cov_generate(distance='EMD', per_parts=True, device=device))
# print('Standard Gaussian, JSD per-Parts: ', vae.jsd_generate(per_parts=True, sp=vae.get_seg, device=device))
# vae.generate_from_NVP()
# print('Standard Gaussian, MMD-CD: ', vae.mmd_generate(device=device))
# vae.generate_from_NVP(parts=True)
# print('Standard Gaussian, MMD-CD per-Parts: ', vae.mmd_generate(per_parts=True, sp=vae.get_seg, device=device))


# In[ ]:


# vae.generate('S_Gaussian', equal_number=True)
# # # # # # vae.generate_from_NVP()
# print('Standard Gaussian, MMD-CD: ', vae.mmd_generate(device=device, symmetry=False))
# print('Standard Gaussian, COV-CD: ', vae.cov_generate(device=device, symmetry=False))
# print('Standard Gaussian, JSD: ', vae.jsd_generate(symmetry=False))
# print('Standard Gaussian, MMD-EMD: ', vae.mmd_generate(distance='EMD', device=device, symmetry=False))
# print('Standard Gaussian, COV-EMD: ', vae.cov_generate(distance='EMD', device=device, symmetry=False))

