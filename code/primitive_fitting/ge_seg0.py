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

from model import trainPrimitives

sys.path.append('/home/shidi/3d-generate/code/TreeGAN/dataloader')
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


# In[5]:


############################## Data loader ####################################
print('Loading data.........')
# Root directory for dataset
dataroot = '../../data/datasetTreeGAN/shapenetcore_partanno_segmentation_benchmark_v0/'

labs = False

category = 'chair'
data = BenchmarkDataset(dataroot, npoints=2048, uniform=False, classification=False, labs = labs, class_choice=category, device=device)
# data = BenchmarkDatasetOnTheFly(dataroot, npoints=2500, uniform=False, classification=False, labs = labs, class_choice=category, device=device)

split = 0.9
total_num = data.__len__()
train_num = math.floor(total_num*split)
test_num = total_num - train_num
train_set, test_set = torch.utils.data.random_split(data, [train_num, test_num])

batch_size = 30
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=0, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0, drop_last=True)

print(total_num)


# In[6]:


# regs_term = {"regularizer_type": ['bernoulli_regularizer', 'parsimony_regularizer',
#                                   'entropy_bernoulli_regularizer', 'overlapping_regularizer'],
#              "bernoulli_regularizer_weight": 1,
#              "entropy_bernoulli_regularizer_weight": 1e-3,
#              "parsimony_regularizer_weight": 1e-3,
#              "sparsity_regularizer_weight": 0,
#              "overlapping_regularizer_weight": 1e-5,
#              "minimum_number_of_primitives": 3,
#              "maximum_number_of_primitives": 8,
#              "w1": 0.005,
#              "w2": 0.005}
# regs_term = {"regularizer_type": ['bernoulli_regularizer', 'parsimony_regularizer',
#                                   'overlapping_regularizer'],
#              "bernoulli_regularizer_weight": 1,
#              "entropy_bernoulli_regularizer_weight": 1e-3,
#              "parsimony_regularizer_weight": 1e-3,
#              "sparsity_regularizer_weight": 0,
#              "overlapping_regularizer_weight": 1e-6,
#              "minimum_number_of_primitives": 3,
#              "maximum_number_of_primitives": 8,
#              "w1": 0.005,
#              "w2": 0.005}
regs_term = {"regularizer_type": ['overlapping_regularizer'],
             "bernoulli_regularizer_weight": 0,
             "entropy_bernoulli_regularizer_weight": 0,
             "parsimony_regularizer_weight": 0,
             "sparsity_regularizer_weight": 0,
             "overlapping_regularizer_weight": 10,
             "minimum_number_of_primitives": 3,
             "maximum_number_of_primitives": 8,
             "w1": 0.005,
             "w2": 0.005}


# In[7]:


model = trainPrimitives(train_loader=train_loader, 
                        test_loader=test_loader,
                        device=device, regularizer_terms=regs_term, category=category,
                        NVP=False, beta=1e-2, epoch=300, symm=True)


# In[8]:


model.run()
# model.run_with_NVP()
# model.gt_seg()

# model.load()
# model.vis()


# In[ ]:


# model.save_unsupervised_seg()


# In[9]:


# model.load()
# model.vis()


# In[ ]:





# In[ ]:




