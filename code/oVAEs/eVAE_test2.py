#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
#get_ipython().run_line_magic('matplotlib', 'notebook')
#%matplotlib inline
import numpy as np
import math
import random

import torch
import torch.nn.parallel


# In[ ]:


import sys

from sVAEs_module_geo2size_deform import TrainSVAE

sys.path.append('code/TreeGAN/dataloader')
from dataset_benchmark import BenchmarkDataset, BenchmarkDatasetOnTheFly


# In[ ]:


SEED = 0 
random.seed(SEED) 
np.random.seed(SEED) 
torch.manual_seed(SEED) 


# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
# device = torch.device('cpu')
print(device)


# In[ ]:


############################## Data loader ####################################
print('Loading data.........')
# Root directory for dataset
dataroot = '../../data/datasetTreeGAN/shapenetcore_partanno_segmentation_benchmark_v0/'

category = 'chair'
data = BenchmarkDataset(dataroot, npoints=2048, uniform=False, classification=False, class_choice=category, device=device)
# data = BenchmarkDatasetOnTheFly(dataroot, npoints=2048, uniform=False, classification=False, class_choice=category, device=device)

split = 0.9
total_num = data.__len__()
train_num = math.floor(total_num*split)
test_num = total_num - train_num
train_set, test_set = torch.utils.data.random_split(data, [train_num, test_num])

batch_size = 30
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=0, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0, drop_last=True)

print(total_num)


# In[ ]:


sp_reg_terms = {"regularizer_type": ['bernoulli_regularizer', 'parsimony_regularizer',
                                     'entropy_bernoulli_regularizer', 'overlapping_regularizer'],
                 "bernoulli_regularizer_weight": 1,
                 "entropy_bernoulli_regularizer_weight": 1e-3,
                 "parsimony_regularizer_weight": 1e-3,
                 "sparsity_regularizer_weight": 0,
                 "overlapping_regularizer_weight": 1e-5,
                 "minimum_number_of_primitives": 3,
                 "maximum_number_of_primitives": 8,
                 "w1": 0.005,
                 "w2": 0.005}
# sp_reg_terms = {"regularizer_type": ['bernoulli_regularizer', 'parsimony_regularizer',
#                                      'overlapping_regularizer'],
#                  "bernoulli_regularizer_weight": 1,
#                  "entropy_bernoulli_regularizer_weight": 0,
#                  "parsimony_regularizer_weight": 1e-3,
#                  "sparsity_regularizer_weight": 0,
#                  "overlapping_regularizer_weight": 1e-5,
#                  "minimum_number_of_primitives": 3,
#                  "maximum_number_of_primitives": 8,
#                  "w1": 0.005,
#                  "w2": 0.005}


# In[ ]:


vae = TrainSVAE(beta=1e-2, batch_size=batch_size, epoch=1000,
                train_loader=train_loader, test_loader=test_loader,
                category = category, device=device, ppl=True, 
                sp_reg_terms=sp_reg_terms, m_samples=False,
                train_rate=1, use_realNVP=True, multi_NVP=True)


# In[ ]:


vae.run(threshold_epoch=2000)
# vae.load()

vae.reconstruction()
# vae.vis()
# vae.vis_generated(distribution='S_Gaussian')
# flow = vae.realNVP()
# vae.load_realNVP()
# vae.shape_sampling(duplicates=0)
# vae.shape_mixing()


# In[ ]:


# vae.estimate_latent()


# In[ ]:


# vae.vis_generated(distribution='M_Gaussian')
# vae.reconstruction()


# In[ ]:


vae.generate('M_Gaussian')
print('Mixed Gaussian, MMD-CD: ', vae.mmd_generate(device=device))
print('Mixed Gaussian, COV-CD: ', vae.cov_generate(device=device))
print('Mixed Gaussian, JSD: ', vae.jsd_generate())
vae.generate('M_Gaussian', parts=True)
print('Mixed Gaussian, MMD-CD per-Parts: ', vae.mmd_generate(per_parts=True, sp=vae.get_seg, device=device))

vae.generate('S_Gaussian')
print('Standard Gaussian, MMD-CD: ', vae.mmd_generate(device=device))
print('Standard Gaussian, MMD-EMD: ', vae.mmd_generate(distance='EMD', device=device))
print('Standard Gaussian, COV-CD: ', vae.cov_generate(device=device))
print('Standard Gaussian, COV-EMD: ', vae.cov_generate(distance='EMD', device=device))
print('Standard Gaussian, JSD: ', vae.jsd_generate())
vae.generate('S_Gaussian', parts=True)
print('Standard Gaussian, MMD-CD per-Parts: ', vae.mmd_generate(per_parts=True, sp=vae.get_seg, device=device))

vae.generate_from_NVP()
print('Standard Gaussian, MMD-CD: ', vae.mmd_generate(device=device))
vae.generate_from_NVP(parts=True)
print('Standard Gaussian, MMD-CD per-Parts: ', vae.mmd_generate(per_parts=True, sp=vae.get_seg, device=device))


# In[ ]:


# vae.device = torch.device('cpu')
# vae.latent_stat()

