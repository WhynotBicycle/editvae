import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torch.nn.utils.rnn import pad_sequence
from loss import DualLoss, seg_labels, SymmLoss
from nets import KLoss, SymmSegNets
from sampler import EqualDistanceSamplerSQ
from functools import partial
from p_utils import param2surface, sampling_from_superquadric
from primitives import quaternions_to_rotation_matrices, deform, transform_to_primitives_centric_system
import time

def batched_params(primitive_params):
    translations = []; rotations = []; size = []; shape = []; deformations = []; probabilities = []
    for p in primitive_params:
        translations.append(p[0].unsqueeze(1))
        rotations.append(p[1].unsqueeze(1))
        size.append(p[2].unsqueeze(1))
        shape.append(p[3].unsqueeze(1))
        deformations.append(p[4].unsqueeze(1))
        probabilities.append(p[5])
    translations = torch.cat(translations, dim=1)
    rotations = torch.cat(rotations, dim=1)
    size = torch.cat(size, dim=1)
    shape = torch.cat(shape, dim=1)
    deformations = torch.cat(deformations, dim=1)
    probabilities = torch.cat(probabilities, dim=1)
    
    return translations, rotations, size, shape, deformations, probabilities

def nn_labels(ipt, primitive_params, sampler):
    translations, rotations, size, shape, deformations, probabilities = primitive_params
    
    B = translations.size(0)
    M = translations.size(1)
    N = ipt.size(1)
    S = sampler.n_samples
    
    ipt_transformed = transform_to_primitives_centric_system(ipt, translations, rotations) ## B*N*M*3
    
    primitive_points, _ = sampling_from_superquadric(size, shape, sampler)
    primitive_points = deform(primitive_points, size, deformations)
    assert primitive_points.size() == (B,M,S,3)
    assert ipt_transformed.size() == (B,N,M,3)
    
    diff = (primitive_points.unsqueeze(3) - (ipt_transformed.permute(0, 2, 1, 3)).unsqueeze(2))
    assert diff.size() == (B, M, S, N, 3)
    dist = torch.sum(diff**2, -1)
    assert dist.size() == (B, M, S, N)
    
    return seg_labels(dist, probabilities)
    
class trainPrimitives(nn.Module):
    def __init__(self, train_loader, test_loader, device, regularizer_terms=None, category=None, 
                 n_shapes=3, n_pos=8, beta=0.01, epoch=1000):
        super().__init__()
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.category = category
        self.n_pos = n_pos
        self.beta = beta
        self.epoch = epoch
        self.n_shapes = n_shapes
        
        self.train_len = len(self.train_loader)
        self.test_len = len(self.test_loader)
        
        self.sampler = EqualDistanceSamplerSQ(n_samples=200, D_eta=0.05, D_omega=0.05)
        self.model = SymmSegNets(latent_size=256, n_prims=n_shapes, n_trans=self.n_pos).to(self.device)
        self.loss = SymmLoss(self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)
        
        if regularizer_terms is None:
            regs_term = {"regularizer_type": ['overlapping_regularizer'],
                         "bernoulli_regularizer_weight": 0,
                         "entropy_bernoulli_regularizer_weight": 0,
                         "parsimony_regularizer_weight": 0,
                         "sparsity_regularizer_weight": 0,
                         "overlapping_regularizer_weight": 1e-2,
                         "minimum_number_of_primitives": 0,
                         "maximum_number_of_primitives": 0,
                         "w1": 0.005,
                         "w2": 0.005}
        else:
            self.regularizer_terms = regularizer_terms
            
#         self.md_name = 'test'
        self.md_name = 'simiPrim_' + (self.category if self.category is not None else '')+\
                       '_beta'+str(self.beta).replace('.','')+\
                       '_epoch'+str(epoch)+\
                       '_nShape'+str(n_shapes)+\
                       '_nPos'+str(n_pos)+\
                       '_regOverlap'+str(self.regularizer_terms['overlapping_regularizer_weight']).replace('.','')
        print(self.md_name)
        self.path = '../../models/' + self.md_name + '/baseline57k/'
        if not os.path.exists(self.path):
            os.mkdir('../../models/'+self.md_name)
            os.mkdir(self.path)
            os.mkdir(self.path+'model/')
            os.mkdir(self.path+'losses/')
            os.mkdir(self.path+'generated/')
            os.mkdir(self.path+'reconstructed/')
            
            
    def save_model(self):
        torch.save(self.model.state_dict(), self.path+'model/'+self.md_name+'.pth')
        
    def load(self):
        self.model = self.model.cpu()
        self.model.load_state_dict(torch.load(self.path+'model/'+self.md_name+'.pth', map_location='cpu'))
        self.model = self.model.to(self.device)
    
    def run(self):
        for epoch in range(self.epoch):
            t = time.time()
            e_ls = 0; e_pcl2prim = 0; e_prim2pcl = 0; e_regs = 0; e_emb_regs = 0
            length = len(self.train_loader)
            for _iter, data in enumerate(self.train_loader):
                point, seg = data
                
                self.optimizer.zero_grad()
                _, primitive_params = self.model(point)
                ls, pcl_to_prim, prim_to_pcl, regs, emb_regs = self.loss(point, 
                                                                         primitive_params, 
                                                                         self.sampler, 
                                                                         self.regularizer_terms)
                ls.backward()
                self.optimizer.step()
                
                e_ls += ls.item()
                e_pcl2prim += pcl_to_prim.item(); e_prim2pcl += prim_to_pcl.item()
                e_regs += regs.item(); e_emb_regs += emb_regs.item()
            t2 = time.time()
            print("[Epoch] ", "{:3}".format(epoch+1),
                  "[ Loss ] ", "{: 7.6f}".format(e_ls/length),
                  "[ point clouds 2 primitives ] ", "{: 7.6f}".format(e_pcl2prim/length),
                  "[ primitives 2 point clouds ] ", "{: 7.6f}".format(e_prim2pcl/length),
                  "[ regularizer ] ", "{: 7.6f}".format(e_regs/length),
                  "[ emb reg ] ", "{: 7.6f}".format(e_emb_regs/length),
                  "[ time ]", "{: 7.6f}".format(t2 - t))
            self.save_model()
      
    def baseline_pt_list(self, point, seg, return_list=False):
        pt_size = point.size()
        seg_size = seg.size()
        assert pt_size[0] == seg_size[0]
        assert pt_size[1] == seg_size[1]

        sub_pt_list = []
        max_n = 0
        for b in range(pt_size[0]):
            pt = point[b,:,:]
            sg = seg[b,:]

            idx = (sg==3).nonzero(as_tuple=True)[0]
            sub_pt = torch.index_select(pt, 0, idx)
            if sub_pt.size(0) == 0:
                sub_pt = torch.zeros(1, 3, device=self.device)
            sub_pt_list.append(sub_pt)
            if max_n < sub_pt.size(0):
                max_n = sub_pt.size(0)
                
        if not return_list:
            for i in range(len(sub_pt_list)):
                num = sub_pt_list[i].size(0)
                if num < max_n:
                    sub_pt_list[i] = torch.cat([sub_pt_list[i], sub_pt_list[i][torch.randint(0, num, (max_n-num, ), device=self.device)] ], dim=0).unsqueeze(0)
                elif num == max_n:
                    sub_pt_list[i] = sub_pt_list[i].unsqueeze(0)

            sub_pt_list = torch.cat(sub_pt_list, dim=0)
        
        return sub_pt_list
    
    def gt_seg(self):
        for epoch in range(20000):
            for _iter, data in enumerate(self.train_loader):
                point, seg = data
#                 point = point.to(self.device)
#                 seg = seg.to(self.device)
                legs = self.baseline_pt_list(point, seg)
                
                self.optimizer.zero_grad()
                _, primitive_params = self.model(legs)
                ls, pcl_to_prim, prim_to_pcl, regs, _ = self.loss(legs, primitive_params, self.sampler, self.regularizer_terms)
                ls.backward()
                self.optimizer.step()

                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ Loss ] ", "{: 7.6f}".format(ls.item()),
                      "[ point clouds 2 primitives ] ", "{: 7.6f}".format(pcl_to_prim.item()),
                      "[ primitives 2 point clouds ] ", "{: 7.6f}".format(prim_to_pcl.item()),
                      "[ regularizer ] ", "{: 7.6f}".format(regs.item()))

            self.save_model()
            
    def save_seg(self):
        root = '../../data/datasetTreeGAN/shapenetcore_partanno_segmentation_benchmark_v0/'
        catfile = root + 'synsetoffset2category.txt'
        self.cat = {}
        with open(catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0].lower()] = ls[1]
        save_root = root + self.cat[self.category] + '/legs/'
        lab_save_root = root + self.cat[self.category] + '/legs_label/'
        
        for d in [self.train_loader, self.test_loader]:
            for _iter, data in enumerate(d):
                point, seg, name = data
                legs = self.baseline_pt_list(point, seg)
                _, primitive_params = self.model(legs)
                
                idx = nn_labels(legs, primitive_params, self.sampler)

                for b in range(legs.size(0)):
                    pt = legs[b]
                    lab = idx[b]
                    np.savetxt(save_root+name[b]+'.pts', pt.detach().cpu().numpy(), fmt='%.5f')
                    np.savetxt(lab_save_root+name[b]+'.seg', lab.detach().cpu().numpy().astype(int), fmt="%d")
                    
    ## save whole points
    def save_unsupervised_seg(self):
        root = '../../data/datasetTreeGAN/shapenetcore_partanno_segmentation_benchmark_v0/'
        catfile = root + 'synsetoffset2category.txt'
        self.cat = {}
        with open(catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0].lower()] = ls[1]
        lab_save_root = root + self.cat[self.category] + '/points_unsup_label/'
        save_root = root + self.cat[self.category] + '/points_unsup/'
        
        for d in [self.train_loader, self.test_loader]:
            for _iter, data in enumerate(d):
                point, _, name = data
                point = point.to(self.device)
                
                if self.NVP:
                    primitive_params, _, _ = self.model(point)
                    primitive_params = batched_params(primitive_params)
                else:
                    _, primitive_params = self.model(point)
                
                worked_idx = [i for i in range(primitive_params[5].size(1)) if primitive_params[5][:,i].mean()>0.5]
                primitive_params = [p[:,worked_idx,...] for p in primitive_params]
                
                idx = nn_labels(point, primitive_params, self.sampler)+1

                for b in range(point.size(0)):
                    lab = idx[b]
                    np.savetxt(lab_save_root+name[b]+'.seg', lab.detach().cpu().numpy().astype(int), fmt="%d")
                    
                    sub_p = point[b]
                    np.savetxt(save_root+name[b]+'.pts', sub_p.detach().cpu().numpy(), fmt='%.5f')

            
    def vis(self, n_pcl=None, gt_parts=False):
        for _iter, data in enumerate(self.train_loader):
            if _iter == 10:
                point, seg = data
                point = point.to(self.device)
                seg = seg.to(self.device)
                if gt_parts:
                    point = self.baseline_pt_list(point, seg)
                
                _, primitive_params = self.model(point)
                    
                _, idx = torch.max(primitive_params[6], 1)
                for i in [2,3,4]:
                    primitive_params[i] = torch.gather(primitive_params[i], 
                                                       1, idx.unsqueeze(2).repeat(1,1,primitive_params[i].size(2))
                                                      )
                primitive_params = primitive_params[:-1]
                
                idx = nn_labels(point, primitive_params, self.sampler)
                if n_pcl is None or n_pcl > point.size(0):
                    n_pcl = point.size(0)
                
                for j in range(n_pcl):
                    tmp = [None]*len(primitive_params)
                    for i in range(len(primitive_params)):
                        tmp[i] = primitive_params[i][j,...].unsqueeze(0)
                    self._vis_fitted_primitive(tmp, point[j,...], idx[j,...])
                
                return
            
    ## x, y, z: 1*M*S
    ## transitions: 1*M*3
    ## rotations: 1*M*4
    def _transform2global_coordinate(self, x, y, z, transitions, rotations):
        B = x.size(0)
        M = x.size(1)
        S = x.size(2)
        assert B == y.size(0) and M == y.size(1) and S == y.size(2)
        assert B == z.size(0) and M == z.size(1) and S == z.size(2)
        assert B == transitions.size(0) and M == transitions.size(1) and 3 == transitions.size(2)
        assert B == rotations.size(0) and M == rotations.size(1) and 4 == rotations.size(2)
        
        R = quaternions_to_rotation_matrices(rotations.view(-1, 4)).view(B, M, 3, 3)
        R = torch.transpose(R, 2, 3)
        
        coords = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=3)
        coords = R.unsqueeze(2).matmul(coords.unsqueeze(-1)).squeeze(-1) ## (B, M, 1, 3, 3) * (B, M, S, 3, 1)
        
        x = coords[:,:,:,0] + transitions[:,:,0].unsqueeze(2)
        y = coords[:,:,:,1] + transitions[:,:,1].unsqueeze(2)
        z = coords[:,:,:,2] + transitions[:,:,2].unsqueeze(2)
        
        return x, y, z
            
    ## the primitive_params: dict of params, each param is 1*M*dim_params
    def _vis_fitted_primitive(self, primitive_params, gt_point, seg_idx, save_path=None):
        trans = primitive_params[0]
        rotations = primitive_params[1]
        size = primitive_params[2]
        shape = primitive_params[3]
        deformations = primitive_params[4]
        probs = primitive_params[5]
        
        B = trans.size(0)
        M = trans.size(1)
        S = self.sampler.n_samples
        
        a1 = size[:, :, 0].unsqueeze(-1) # M
        a2 = size[:, :, 1].unsqueeze(-1)
        a3 = size[:, :, 2].unsqueeze(-1)
        e1 = shape[:, :, 0].unsqueeze(-1)
        e2 = shape[:, :, 1].unsqueeze(-1)
        
        etas, omegas = self.sampler.sample_on_batch(size.detach().cpu().numpy(), shape.detach().cpu().numpy())
        etas = size.new_tensor(etas)
        omegas = size.new_tensor(omegas)
        x, y, z = param2surface(a1, a2, a3, e1, e2 ,etas, omegas)
        
        primitive_points = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)], dim=-1)
        primitive_points = deform(primitive_points, size, deformations)
        x = primitive_points[:,:,:,0]; y = primitive_points[:,:,:,1]; z = primitive_points[:,:,:,2]
        
        x, y, z = self._transform2global_coordinate(x, y, z, trans, rotations)    
        
        color = ['red', 'blue', 'green', 'black', 'yellow', 'orange', 'fuchsia', 'crimson']
        
#         color = ['dimgray', 'rosybrown', 'red', 'sienna', 'saddlebrown',
#                  'darkorange', 'darkgoldenrod', 'olive', 'yellowgreen', 'darkseagreen',
#                  'darkgreen', 'turquoise', 'darkcyan', 'deepskyblue', 'lightslategrey',
#                  'royalblue', 'navy', 'blueviolet', 'fuchsia', 'crimson']
        
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        z = z.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        print(probs)
        
        gt_x = gt_point[:,0].detach().cpu().numpy()
        gt_y = gt_point[:,1].detach().cpu().numpy()
        gt_z = gt_point[:,2].detach().cpu().numpy()
        
        
        seg_clr = np.take(np.array(color), np.reshape(seg_idx.detach().cpu().numpy(), -1))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ## vis the fitted_params
        for i in range(M):
            ax.scatter3D(x[:,i,:], y[:,i,:], z[:,i,:], c=color[i], s=1, alpha=probs[:,i])
            
        ## vis gt_points
        ax.scatter3D(gt_x+0.5, gt_y+0.5, gt_z+0.5, c=seg_clr, s=1)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.grid(False)
        plt.title('Superquadric representations of point cloud')
        
        if save_path is not None:
            plt.subplots_adjust()
            plt.savefig(save_path)
        else:
            plt.show()
    
    
    
    
    
    
    
    
    
    
    
    