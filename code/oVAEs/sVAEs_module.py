import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math
import time
import torch
import torch.nn as nn
import os
import shutil
import torch.optim as optim
import random
import numpy as np

from torch.nn.utils.rnn import pad_sequence
# from torch.utils.tensorboard import SummaryWriter
from torch import distributions

from pVAEs_module import TrainVAE, reparameterize, KL

import sys
sys.path.append('code/utils')
from basic_modules import PointNetfeat
from loss_function import _point_level_calus, _remove_zero_rows, ChamferDistance, Calus

sys.path.append('code/TreeGAN')
from model.gan_network import Generator

sys.path.append('code/primitive_fitting')
from sampler import EqualDistanceSamplerSQ
from nets import TransNN, RotaNN, SizeNN, ShapeNN, DeformNN, ProbNN
from p_utils import PrimitiveParams, sampling_from_superquadric, param2surface
from primitives import transform_to_primitives_centric_system, deform, inside_outside_function, quaternions_to_rotation_matrices
from loss import seg_labels, pcl_to_prim_loss, prim_to_pcl_loss, get_regularizer_term, get_regularizer_weights

from realNVP_module import RealNVP

# # import visualization
# sys.path.append('/home/shidi/3d-generate/code/visualization')
# from realguisuperquad import MayaviVisualization

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

## inver of deformation
## ori: B*N*3
## size: B*3
## deform: B*2
def inverse_deform(ori, size, deform):
    B = ori.size(0)
    N = ori.size(1)
    K = deform/size[:,-1].unsqueeze(-1)
    f = K.unsqueeze(1) * ori[:,:,-1].unsqueeze(-1) + 1.0
    f = torch.cat([f, f.new_ones(B, N, 1)], -1)

    return ori/f


def parts_distance(ipt, opt, primitive_params, idx):
    chamfer_d = 0; coverage = 0; quality = 0
    num = 0
    
    for i in range(len(opt)):
        p = opt[i]
        tran = primitive_params[i][0]
        rota = primitive_params[i][1]
        prob = primitive_params[i][5]
        size = primitive_params[i][2]
        df = primitive_params[i][4]
        
        n = 0
        cds = 0; covs = 0; quals = 0
        for b in range(p.size(0)):
            y = p[b,...] ## (N/n_parts) * 3
            x = ipt[b,...]
            x = torch.index_select(x, 0, (idx[b,:] == i).nonzero(as_tuple=True)[0])
            
            x = _remove_zero_rows(x)
            y = _remove_zero_rows(y)
            if x.size(0) != 0 and y.size(0) != 0:
                n += 1
                ## transform the gt point cloud
                x = transform_to_primitives_centric_system(x.unsqueeze(0), 
                                                           tran[b,:].unsqueeze(0).unsqueeze(1),
                                                           rota[b,:].unsqueeze(0).unsqueeze(1)).squeeze(0).squeeze(1)
                

                cd, cov, qual = _point_level_calus(x, y)
                cds += cd*prob[b,:]; covs += cov*prob[b,:]; quals += qual*prob[b,:]
        if n is not 0:
            num += 1
            chamfer_d += cds/n; coverage += covs/n; quality += quals/n
    return chamfer_d/num, coverage/num, quality/num

def nn_labels(ipt, primitive_params, sampler):
    translations, rotations, size, shape, deformations, probabilities = batched_params(primitive_params)
    bp = [translations, rotations, size, shape, deformations, probabilities]
    
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

    
def fitting_performance(ipt, primitive_params, sampler, regularizer_terms):
    translations, rotations, size, shape, deformations, probabilities = batched_params(primitive_params)
    bp = [translations, rotations, size, shape, deformations, probabilities]
    
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
    
    idx = seg_labels(dist, probabilities) ## for all primitives regardless it's label
    
    ## loss for fitting primitive to ground truth points
    pcl_to_prim, F = pcl_to_prim_loss(bp, ipt_transformed, dist)
    assert F is None or F.shape == (B, N, M)
    prim_to_pcl = prim_to_pcl_loss(bp, diff, dist)
    
    ## loss for regularizer only
    regularizers = get_regularizer_term(bp, F, regularizer_terms)
    reg_values = get_regularizer_weights(regularizers, regularizer_terms)    
    regs = sum(reg_values.values())
    
    return pcl_to_prim + prim_to_pcl, regs, idx

def assigned_calus(ipt, opt, primitive_params, sampler, regularizer_terms):
    prim_loss, regs, idx = fitting_performance(ipt, primitive_params, sampler, regularizer_terms)
    ## geometrical loss
    cd, cov, qual = parts_distance(ipt, opt, primitive_params, idx)
    
    return cd, cov, qual, prim_loss, regs, idx

class BasicLoss(nn.Module):
    def __init__(self, beta, sampler, regularizer_terms, device):
        super(BasicLoss, self).__init__()
        self.device = device
        self.regularizer_terms = regularizer_terms
        self.sampler = sampler
        self.beta = beta
        
    def forward(self):
        return None

class orderedLossSp(BasicLoss):
    def __init__(self, beta, sampler, regularizer_terms, device):
        super().__init__(beta, sampler, regularizer_terms, device)
        
    def forward(self, ipts, primitive_params, mu, logvar):
        prims, regs, _ = fitting_performance(ipts, primitive_params, self.sampler, self.regularizer_terms)
        kld = KL(mu, logvar)
        return prims + regs + self.beta*kld, prims, regs, kld
    
class orderedLossGeo(BasicLoss):
    def __init__(self, beta, sampler, regularizer_terms, device):
        super().__init__(beta, sampler, regularizer_terms, device)
        
    ## here the latent is geo_latent
    def forward(self, ipt, opt, primitive_params, idx, mu, logvar):
        cds, covs, quals = parts_distance(ipt, opt, primitive_params, idx)
        kld = KL(mu, logvar)
        return cds + self.beta*kld, cds, covs, quals, kld
        
class realNVPLoss(nn.Module):
    def __init__(self, device):
        super(realNVPLoss, self).__init__()
        self.device = device
        
    def forward(self, latent, logp, prior):
        # print(prior.log_prob(latent.cpu()).to(self.device).get_device())
        return - (prior.log_prob(latent.cpu()).to(self.device) + logp).mean()

## pointNet as encoder, for decoder, first use fully connect layer to compass the latent dimension, than use TreeGAN decoder
class shareVAE(nn.Module):
    def __init__(self, batch_size, n_parts, dim=256, geo_dim=32, prim_dim=16, use_realNVP=False, device='cpu'):
        super().__init__()
        
        self.dim = dim
        self.geo_dim = geo_dim
        self.prim_dim = prim_dim
        self.n_parts = n_parts
        self.use_realNVP = use_realNVP
        self.device = device
        
        encoder_layer_dim = {512: [3, 64, 128, 256, 512],
                             256: [3, 64, 128, 128, 256],
                             128: [3, 64, 64, 128]}
        
        ## dim from input latent size to output coordinate size(=3)
        ## here assume n_parts is fixed
        ## total_dim == n_parts * parts_latent_dim
        geo_decoder_features_dim = {64: [64,64,32,32,16,16,3],
                                    32: [32,32,16,16,3],
                                    16: [16,8,8,3]}
#         geo_decoder_features_dim = {64: [32,32,16,16,3],
#                                     32: [32,32,16,16,3],
#                                     16: [16,8,8,3]}
        
        ## cumprod is the generated number
        ## totally 2048 num of points, each part has 2048/n_parts num of points
        ## len_degrees == len_features - 1
        geo_decoder_degrees_dim = {8: [1,2,4,32]}
#         geo_decoder_degrees_dim = {4: [1,2,4,32]}
        
        assert len(geo_decoder_features_dim[self.dim/self.n_parts]) == len(geo_decoder_degrees_dim[self.n_parts]) + 1
        assert dim in encoder_layer_dim.keys()
        
        ## dim of latent space, 256
        self.encoder = nn.Sequential(
            PointNetfeat(size=self.dim, trans=False, layers=encoder_layer_dim[self.dim]),
            nn.Linear(self.dim, self.dim),
            nn.BatchNorm1d(self.dim),
            nn.ReLU()
        )
        
        self.fc_geo = nn.Linear(self.dim, self.geo_dim)
        self.fc_prim = nn.Linear(self.dim, self.prim_dim)
        
        self.geo_decoders = nn.ModuleList([Generator(batch_size = batch_size,
                                                     features = geo_decoder_features_dim[self.dim/self.n_parts],
                                                     degrees = geo_decoder_degrees_dim[self.n_parts],
                                                     support = 10) for i in range(self.n_parts)])
        
        self.prims_decoders = nn.ModuleList([nn.ModuleList([TransNN(1, prim_dim),
                                                            RotaNN(1, prim_dim),
                                                            SizeNN(1, prim_dim),
                                                            ShapeNN(1, prim_dim),
                                                            DeformNN(1, prim_dim),
                                                            ProbNN(1, prim_dim)]) for i in range(self.n_parts)])
        
        self.fc_logvar = nn.Linear(self.dim, self.dim)
        self.fc_mu = nn.Linear(self.dim, self.dim)
        
        if self.use_realNVP:
            self.geo_nf = RealNVP(self.geo_dim, device=self.device)
            self.prim_nf = RealNVP(self.prim_dim, device=self.device)
        
    def to_dist(self, x):
        code = self.encoder(x.transpose(1,2))
        mu = self.fc_mu(code)
        logvar = self.fc_logvar(code)
        
        return mu, logvar
    
    def to_prim(self, x):
        B = x.size(0)
        mu, logvar = self.to_dist(x)
        
        latent = reparameterize(mu, logvar)
        
        prim_latent = self.fc_prim(latent)
        if self.use_realNVP:
            prim_latent, _ = self.prim_nf.f(prim_latent)
            prim_latent = self.prim_nf.g(prim_latent)

        params = []
        for i in range(self.n_parts):
            p = PrimitiveParams()
            for j in range(len(self.prims_decoders[i])):
                p[j] = self.prims_decoders[i][j](prim_latent).view(B, -1)
            params.append(p)
            
        return params, mu, logvar, latent, prim_latent
    
    def to_opt(self, x):
        B = x.size(0)
        mu, logvar = self.to_dist(x)
        
        latent = reparameterize(mu, logvar)
        geo_latent = self.fc_geo(latent)
        
        if self.use_realNVP:
            geo_latent, _ = self.geo_nf.f(geo_latent)
            geo_latent = self.geo_nf.g(geo_latent)
        
        geo_opts = []
        for i in range(self.n_parts):
            geo_opts.append(self.geo_decoders[i](geo_latent.unsqueeze(1)))
            
        return geo_opts, mu, logvar, latent, geo_latent
    
    def local_lt2pt(self, geo_latent, prim_latent):
        B = geo_latent.size(0)
        params = []
        geo_opts = []
        for i in range(self.n_parts):
            sub_geo = geo_latent[:,i,:]
            sub_geo = self.geo_nf[i].g(sub_geo)
            
            sub_prim = prim_latent[:,i,:]
            sub_prim = self.prim_nf[i].g(sub_prim)
            
            geo_opts.append(self.geo_decoders[i](sub_geo.unsqueeze(1)))
            p = PrimitiveParams()
            for j in range(len(self.prims_decoders[i])):
                p[j] = self.prims_decoders[i][j](sub_prim).view(B, -1)
            params.append(p)
            
        return geo_opts, params
    
    def local_lt2pt_back(self, geo_latent, prim_latent):
        B = geo_latent.size(0)
        params = []
        geo_opts = []
        for i in range(self.n_parts):
            sub_geo = geo_latent[:,i,:]
            sub_geo, _ = self.geo_nf[i].f(sub_geo)
            sub_geo = self.geo_nf[i].g(sub_geo)
            
            sub_prim = prim_latent[:,i,:]
            sub_prim, _ = self.prim_nf[i].f(sub_prim)
            sub_prim = self.prim_nf[i].g(sub_prim)
            
            geo_opts.append(self.geo_decoders[i](sub_geo.unsqueeze(1)))
            p = PrimitiveParams()
            for j in range(len(self.prims_decoders[i])):
                p[j] = self.prims_decoders[i][j](sub_prim).view(B, -1)
            params.append(p)
            
        return geo_opts, params
    
    def lt2pt(self, latent):
        B = latent.size(0)
        
        geo_latent = self.fc_geo(latent)
        prim_latent = self.fc_prim(latent)

        if self.use_realNVP:
            geo_latent, _ = self.geo_nf.f(geo_latent)
            geo_latent = self.geo_nf.g(geo_latent)
            prim_latent, _ = self.prim_nf.f(prim_latent)
            prim_latent = self.prim_nf.g(prim_latent)

        params = []
        geo_opts = []
        for i in range(self.n_parts):
            geo_opts.append(self.geo_decoders[i](geo_latent.unsqueeze(1)))
            p = PrimitiveParams()
            for j in range(len(self.prims_decoders[i])):
                p[j] = self.prims_decoders[i][j](prim_latent).view(B, -1)
            params.append(p)
                
        return geo_opts, params, geo_latent, prim_latent
    
    def forward(self, x):
        mu, logvar = self.to_dist(x)
        latent = reparameterize(mu, logvar)
        geo_opts, params, geo_latent, prim_latent = self.lt2pt(latent)
        
        return geo_opts, params, mu, logvar, latent, geo_latent, prim_latent
    
    def to_nf(self, x):
        mu, logvar = self.to_dist(x)
        
        latent = reparameterize(mu, logvar)
        geo_latent = self.fc_geo(latent)
        prim_latent = self.fc_prim(latent)
        
        if self.use_realNVP:
            geo_latent, geo_logp = self.geo_nf.f(geo_latent)
            geo_latent = self.geo_nf.g(geo_latent)
            prim_latent, prim_logp = self.prim_nf.f(prim_latent)
            prim_latent = self.prim_nf.g(prim_latent)
            
        return geo_latent, geo_logp, prim_latent, prim_logp

class TrainSVAE(TrainVAE):
    def __init__(self, beta, batch_size, train_loader, test_loader, category, device, n_parts=8, n_latents=256, epoch=1000, sp_reg_terms=None, use_realNVP=False):
        super().__init__(beta, batch_size, train_loader, test_loader, category, device, n_parts, n_latents, epoch)
        
        self.use_realNVP = use_realNVP
        if sp_reg_terms is None:
            self.sp_reg_terms = {"regularizer_type": ['bernoulli_regularizer', 'parsimony_regularizer'],
                                 "bernoulli_regularizer_weight": 1,
                                 "entropy_bernoulli_regularizer_weight": 0,
                                 "parsimony_regularizer_weight": 1e-3,
                                 "sparsity_regularizer_weight": 0,
                                 "overlapping_regularizer_weight": 0,
                                 "minimum_number_of_primitives": 3,
                                 "maximum_number_of_primitives": 8,
                                 "w1": 0.005,
                                 "w2": 0.005}
        else:
            self.sp_reg_terms = sp_reg_terms
        
        self.md_name = 'sVAE_'+self.category+\
                       '_beta'+str(self.beta).replace('.','')+\
                       '_nparts'+str(self.n_parts)+\
                       '_nlatents'+str(self.n_latents)+\
                       ('_regBern'+str(sp_reg_terms['bernoulli_regularizer_weight']).replace('.','')+\
                        '_regParsimony'+str(sp_reg_terms['parsimony_regularizer_weight']).replace('.','')+\
                        '_regSparsity'+str(sp_reg_terms['sparsity_regularizer_weight']).replace('.','')+\
                        '_regOverlap'+str(sp_reg_terms['overlapping_regularizer_weight']).replace('.','')+\
                        '_min'+str(sp_reg_terms['minimum_number_of_primitives']).replace('.','')+\
                        '_max'+str(sp_reg_terms['maximum_number_of_primitives']).replace('.','') if sp_reg_terms is not None else '')+\
                       ('_realNVP' if self.use_realNVP else '')

        print('path: ', self.md_name)
        self.path = 'models/'+ self.md_name +'/baseline57k/'
        if not os.path.exists(self.path):
            os.mkdir('models/'+ self.md_name)
            os.mkdir(self.path)
            os.mkdir(self.path+'model/')
            os.mkdir(self.path+'losses/')
            os.mkdir(self.path+'generated/')
            os.mkdir(self.path+'reconstructed/')
            
        self.sampler = EqualDistanceSamplerSQ(n_samples=200, D_eta=0.05, D_omega=0.05)
        self.model = shareVAE(self.batch_size, 
                              self.n_parts, 
                              dim=self.n_latents, 
                              use_realNVP=self.use_realNVP, device = self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-4)

        self.geo_loss = orderedLossGeo(self.beta, 
                                       self.sampler, 
                                       self.sp_reg_terms, 
                                       self.device).to(self.device)
        self.trans_loss = orderedLossSp(self.beta, self.sampler, self.sp_reg_terms, self.device).to(self.device)
            
        if self.use_realNVP:
            self.geo_nfLoss = realNVPLoss(self.device).to(self.device)
            self.prim_nfLoss = realNVPLoss(self.device).to(self.device)
    
    ## train the realNVP only
    def on_off(self, on=True):
        for p in self.model.encoder.parameters():
            p.requires_grad = on
        for p in self.model.fc_mu.parameters():
            p.requires_grad = on
        for p in self.model.fc_logvar.parameters():
            p.requires_grad = on
     
    def self_similarity_measurement(self):
        total_cd = 0; bn = 0
        with torch.no_grad():
            for _iter, data in enumerate(self.train_loader):
                point, _ = data
                opts, params, mu, logvar, _, _, _ = self.model(point)
                opts = [opts[i] for i in range(len(params)) if params[i][5].mean()>0.5]
                M = len(opts)
                
                cds = []
                for i in range(M):
                    for j in range(i,M):
                        if i != j:
                            p1 = opts[i]
                            p2 = opts[j]
                            cd = ChamferDistance(p1, p2, device=self.device)
                            cds.append(cd)

                total_cd += sum(cds)/len(cds)
                bn += 1
                print(cds)
        print(total_cd/bn)
        
    def sampling_semantic_meaningful(self):
        with torch.no_grad():
            total_prob = 0
            n = 0
            for _iter, data in enumerate(self.train_loader):
                point, seg = data
                opts, params, mu, logvar, _, _, _ = self.model(point)
                params = [params[i] for i in range(len(params)) if params[i][5].mean()>0.5]
                infer_seg = nn_labels(point, params, self.sampler)
                
                count = 0
                for b in range(self.batch_size):
                    if_s = infer_seg[b]
                    gt_s = seg[b]
                
                    # airplane
                    if_s1 = (if_s==1).nonzero(as_tuple=True)[0]
                    if_s2 = (if_s==2).nonzero(as_tuple=True)[0]
                    
                    s1 = if_s1[torch.randint(if_s1.size(0), (1,))]
                    s2 = if_s2[torch.randint(if_s2.size(0), (1,))]
                    
                    if gt_s[s1] == gt_s[s2]:
                        count += 1
                count = count/self.batch_size
                print(count)
                total_prob += count
                n += 1
            total_prob = total_prob/n
            print('===============')
            print(total_prob)
        
    def semantic_meaningful(self):
        total_cd = []
        with torch.no_grad():
            for _iter, data in enumerate(self.train_loader):
                point, seg = data
                opts, params, mu, logvar, _, _, _ = self.model(point)
                gt_parts = self.baseline_pt_list(point, seg, rg=4, bias=1) ## airplane chair
                # gt_parts = self.baseline_pt_list(point, seg, rg=3, bias=1) ## table

                params = [params[i] for i in range(len(params)) if params[i][5].mean()>0.5]
                infer_seg = nn_labels(point, params, self.sampler)
                infer_parts = self.baseline_pt_list(point, infer_seg, rg=3, bias=0)
    
                batch_d = []
                for b in range(self.batch_size):
                    sample_d = []
                    for infer_p in infer_parts:
                        ip = infer_p[b]
                        # ip = _remove_zero_rows(ip)
                        inf_cds = []
                        for gt_p in gt_parts:
                            gp = gt_p[b]
                            # gp = _remove_zero_rows(gp)
                            
                            if gp.size(0)==0 or ip.size(0)==0:
                                cd = torch.tensor(0.0).to(self.device)
                            else:
                                cd, _, _ = _point_level_calus(gp, ip)
                            inf_cds.append(cd)
                        inf_cds = min(inf_cds)
                        sample_d.append(inf_cds)
                    sample_d = sum(sample_d)/len(sample_d)
                    batch_d.append(sample_d)
                batch_d = sum(batch_d)/len(batch_d)
                total_cd.append(batch_d)
                print(batch_d)
        print('=================')
        print(sum(total_cd)/len(total_cd))
    
    def run(self, threshold_epoch=1000):
        for epoch in range(0, self.epoch):
            t = time.time()
            for _iter, data in enumerate(self.train_loader):
                point, _ = data
                
                ## to keep the same sampled latent
                ## train the primitive autoencoder parts
                self.optimizer.zero_grad()
                primitive_params, mu, logvar, _, _  = self.model.to_prim(point)
                if epoch > threshold_epoch:
                    ## update probabilities only
                    primitive_params = [[ps[i].clone().detach() if i is not 5 else ps[i] for i in range(len(ps))]\
                                        for ps in primitive_params]
                sp_ls, prim, regs, sp_kld = self.trans_loss(point, primitive_params, mu, logvar)
                sp_ls.backward()
                self.optimizer.step()

                ## train the geometrical autoencoder parts
                self.optimizer.zero_grad()
                opts, primitive_params, mu, logvar, _, _, _ = self.model(point)
                idx = nn_labels(point, primitive_params, self.sampler).clone().detach()
                primitive_params = [[p.clone().detach() for p in ps] for ps in primitive_params]
                geo_ls, cd, covs, quals, geo_kld = self.geo_loss(point, opts, primitive_params, idx, mu, logvar)
                geo_ls.backward()
                self.optimizer.step()
                ls = geo_ls + sp_ls

                ## train the realNVP for primitive latent space
                if self.use_realNVP:
                    self.on_off(False)
                    mu, logvar = self.model.to_dist(point)
                    latent = reparameterize(mu, logvar)
                    latent = latent.clone().detach()

                    prim_latent = self.model.fc_prim(latent)
                    prim_latent = prim_latent.clone().detach()
                    prim_latent, prim_logp = self.model.prim_nf.f(prim_latent)
                    ## update prim realNVP
                    self.optimizer.zero_grad()
                    # print(prim_latent.get_device())
                    # print(prim_logp.get_device())
                    prim_nvp_ls = self.prim_nfLoss(prim_latent, prim_logp, self.model.prim_nf.prior)
                    prim_nvp_ls.backward(retain_graph=True)
                    self.optimizer.step()

                    ## update geo realNVP
                    geo_latent = self.model.fc_geo(latent)
                    geo_latent = geo_latent.clone().detach()
                    geo_latent, geo_logp = self.model.geo_nf.f(geo_latent)
                    self.optimizer.zero_grad()
                    geo_nvp_ls = self.geo_nfLoss(geo_latent, geo_logp, self.model.geo_nf.prior)
                    geo_nvp_ls.backward(retain_graph=True)
                    self.optimizer.step()
                    self.on_off(True) ## reactive all other nets for next batch
                
            print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                  "[ Loss ] ", "{: 7.6f}".format(ls.item()),
                  "[ Chamfer distance ] ", "{: 7.6f}".format(cd.item()),
                  "[ Coverage ] ", "{: 7.6f}".format(covs.item()),
                  "[ Quality ] ", "{: 7.6f}".format(quals.item()),
#                       "[ KL divergence ] ", "{: 7.6f}".format(kld),
                  "[ Geometric KL divergence ] ", "{: 7.6f}".format(geo_kld),
                  "[ Primitive KL divergence ] ", "{: 7.6f}".format(sp_kld),
                  "[ primitive loss ] ", "{: 7.6f}".format(prim.item()),
                  "[ regularizer ] ", "{: 7.6f}".format(regs.item()))
            if self.use_realNVP:
                print("[ geo realNVP ] ", "{: 7.6f}".format(geo_nvp_ls.item()),
                      "[ primitive realNVP ] ", "{: 7.6f}".format(prim_nvp_ls.item()))
            t1 = time.time()
#             print(t1 - t)
            self.save_model()               
    
    #point: N*3
    def _vis_whole(self, point):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = point[:, 0].detach().cpu().numpy()
        y = point[:, 1].detach().cpu().numpy()
        z = point[:, 2].detach().cpu().numpy()
        ax.scatter3D(x, y, z, s=20, depthshade=False)
        ax.grid(False)
        plt.axis('off')
        plt.title('Input')
        
    def _rotate_pt(self, pt, Rx, Ry, Rz):
        pt = Rx.matmul(pt.T).T
        pt = Ry.matmul(pt.T).T
        pt = Rz.matmul(pt.T).T
        return pt
        
    def rotate_points(self, pc_list, theta=None, phi=None, gamma=None):
        if theta is None:
            theta = torch.tensor(1/2 * math.pi) # x
        if phi is None:
            phi = torch.tensor(-1/18 * math.pi) # y
        if gamma is None:
            gamma = torch.tensor(-1/4 * math.pi) # z
            
        Rx = torch.tensor([[1, 0, 0],
                           [0, torch.cos(theta), -torch.sin(theta)],
                           [0, torch.sin(theta), torch.cos(theta)]]).to(self.device)
        Ry = torch.tensor([[torch.cos(phi), 0, torch.sin(phi)],
                           [0, 1, 0],
                           [-torch.sin(phi), 0, torch.cos(phi)]]).to(self.device)
        Rz = torch.tensor([[torch.cos(gamma), -torch.sin(gamma), 0],
                           [torch.sin(gamma), torch.cos(gamma), 0],
                           [0, 0, 1]]).to(self.device)
        
        pc_list = [self._rotate_pt(pc, Rx, Ry, Rz) for pc in pc_list]
        
        return pc_list
        
    # def fig_plot(self):
    #     theta = torch.tensor(1/6 * math.pi) # x
    #     phi = torch.tensor(-1/4 * math.pi) # y
    #     gamma = torch.tensor(-1/12 * math.pi) # z
        
    #     gt_red = (162/255, 153/255, 136/255)
    #     gt_blue = (134/255, 150/255, 167/255)
    #     gt_orange = (173/255, 83/255, 72/255)
    #     gt_yellow = (241/255, 182/255, 26/255)
    #     gt_green = (130/255, 144/255, 96/255)
    #     ## defind colors
    #     pink = (208/255, 193/255, 198/255)
    #     light = (216/255, 202/255, 175/255)
    #     blue = (134/255, 150/255, 167/255)
    #     brown = (162/255, 153/255, 136/255)
    #     dark = (191/255, 191/255, 191/255)
    #     green = (181/255, 196/255, 177/255)
    #     red = (150/255, 84/255, 84/255)
    #     pc_clr = [blue]
    #     background_clr = [dark, dark, dark, dark, dark, dark, dark, dark]
    #     sq_clr = pink
    #     parts_clr = [gt_red, gt_blue, gt_green, dark, pink, dark, light, red]
    #     # pc_opa = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    #     pc_opa = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
        
    #     with torch.no_grad():
    #         for _iter, data in enumerate(self.train_loader):
    #             if _iter == 0:
    #                 point, seg = data
    #                 point = point.to(self.device)
    #                 seg = seg.to(self.device)
                    
    #                 opts, params, _, _, _, _, _ = self.model(point)
                    
    #                 # idx = 12
    #                 # idx = 20
    #                 idx = 2
    #                 pt = point[idx]

    #                 ## assemble outputs
    #                 opts = [self._pt_trans(opts[i], params[i][0], params[i][1]) for i in range(len(opts)) if params[i][5][idx]>0.5]
    #                 opts = [opts[i][idx] for i in range(len(opts))]
    #                 opts = [torch.index_select(tmp, 0, tmp.sum(1).nonzero(as_tuple=True)[0]) for tmp in opts]

    #                 ## primitives
    #                 superquads = [[sub_p[idx,:] for sub_p in p] for p in params if p[5][idx]>0.5]

    #                 ## vis assembled generated point cloud (use input)
    #                 vis = MayaviVisualization(superquads, opts, pc_colors=[dark, dark, dark, dark, dark, dark, dark, dark], vis_r=[theta, phi, gamma], vis_sq=False, pt_size=.045)
    #                 vis.configure_traits()

    #                 pt_seg = self.baseline_pt_list(point, seg, rg=4, bias=1)
    #                 pt_seg = [pt_seg[i][idx] for i in range(len(pt_seg))]
    #                 pt_seg = [torch.index_select(tmp, 0, tmp.sum(1).nonzero(as_tuple=True)[0]) for tmp in pt_seg] #remove zero points
    #                 ## parts point cloud
    #                 pt_seg = [tmp for tmp in pt_seg if tmp.size(0)!=0] # remove empty point clouds
                    
    #                 ## vis assembled generated point cloud (use input)
    #                 vis = MayaviVisualization(superquads, pt_seg, pc_colors=parts_clr, vis_r=[theta, phi, gamma], vis_sq=False)
    #                 vis.configure_traits()
                    
    #                 ## vis assembled superquads
    #                 vis = MayaviVisualization(superquads, vis_r=[theta, phi, gamma], sq_colors=parts_clr)
    #                 vis.configure_traits()
                    
    #                 ## input point cloud
    #                 vis = MayaviVisualization(point_clouds = [pt], pc_colors=[background_clr[0]], vis_r=[theta, phi, gamma])
    #                 vis.configure_traits()
    
    #                 ## for each primitives with point clouds
    #                 for i in range(len(superquads)):
    #                     sq = superquads[i]
    #                     tmp_clr = background_clr.copy()
    #                     tmp_clr[i] = parts_clr[i]
    #                     tmp_op = pc_opa.copy()
    #                     tmp_op[i] = 1
    #                     vis = MayaviVisualization([sq], pt_seg, sq_colors = [parts_clr[i]], sq_opa=1, 
    #                                               pc_colors=tmp_clr, pc_opa=tmp_op, vis_r=[theta, phi, gamma])
    #                     vis.configure_traits()
                        
    #                 ## for each reconstructed point clouds, per parts
    #                 for i in range(len(superquads)):
    #                     sq = superquads[i]
    #                     tmp_clr = background_clr.copy()
    #                     tmp_clr[i] = parts_clr[i]
    #                     tmp_op = pc_opa.copy()
    #                     tmp_op[i] = 1
    #                     vis = MayaviVisualization([sq], pt_seg, sq_colors = sq_clr, sq_opa=.95, 
    #                                               pc_colors=tmp_clr, pc_opa=tmp_op, vis_r=[theta, phi, gamma], 
    #                                               vis_sq=False, origin_coord=False)
    #                     vis.configure_traits()
                        
    #                 ## vis single part in original coordinate
    #                 origin_coord = True
    #                 superquads[0][0] -= torch.tensor([0.03, 0, 0.0]).to(self.device)
    #                 vis_r = [torch.tensor(1/4 * math.pi),
    #                          torch.tensor(-1/4 * math.pi),
    #                          torch.tensor(0 * math.pi)]
    #                 for i in range(len(superquads)):
    #                     sq = superquads[i]
    #                     sub_pt = pt_seg[i]
    #                     vis = MayaviVisualization(superquads=[sq], sq_colors = [parts_clr[i]], sq_opa=.95, 
    #                                               pc_colors=[parts_clr[i]], pc_opa=[1], origin_coord=origin_coord,
    #                                               vis_r=vis_r)
    #                     vis.configure_traits()
                        
    #                     vis = MayaviVisualization([sq], [sub_pt], sq_colors = [parts_clr[i]], sq_opa=.95, 
    #                                               pc_colors=[parts_clr[i]], pc_opa=[1], origin_coord=origin_coord,
    #                                               vis_r=vis_r, vis_sq = False)
    #                     vis.configure_traits()
                
    #                 break
        
    def get_local_latents(self, latent):
        geo_latent = []
        prim_latent = []
        for i in range(self.n_parts):
            tmp = self.model.fc_geo[i](latent)
            geo_latent.append(self.model.geo_nf[i].f(tmp)[0].unsqueeze(1))
            tmp = self.model.fc_prim[i](latent)
            prim_latent.append(self.model.prim_nf[i].f(tmp)[0].unsqueeze(1))

        geo_latent = torch.cat(geo_latent, dim=1)
        prim_latent = torch.cat(prim_latent, dim=1)
        return geo_latent, prim_latent
 
    # def shape_interpolation(self, rescale=False):
    #     theta = torch.tensor(1/6 * math.pi) # x
    #     phi = torch.tensor(-1/4 * math.pi) # y
    #     gamma = torch.tensor(-1/12 * math.pi) # z
        
    #     pt_size = .045
        
    #     pink = (208/255, 193/255, 198/255)
    #     light = (216/255, 202/255, 175/255)
    #     blue = (134/255, 150/255, 167/255)
    #     brown = (162/255, 153/255, 136/255)
    #     dark = (191/255, 191/255, 191/255)
    #     green = (130/255, 144/255, 96/255)
    #     red = (150/255, 84/255, 84/255)
        
    #     gt_red = (162/255, 153/255, 136/255)
    #     gt_blue = (134/255, 150/255, 167/255)
    #     gt_orange = (173/255, 83/255, 72/255)
    #     gt_yellow = (240/255, 223/255, 167/255)
    #     gt_green = (130/255, 144/255, 96/255)
        
    #     hightlight_clr = [red, blue, green]
    #     background_clr = [dark, dark, dark, dark, dark, dark, dark, dark]
    #     gt_clr = [red, red, red, red, red, red, red]
    #     ref_clr = [blue, blue, blue, blue, blue, blue, blue, blue]
    #     clr = [gt_yellow, red, blue]
        
    #     with torch.no_grad():
    #         for _iter, data in enumerate(self.train_loader):
    #             if _iter == 0:
    #                 point, _ = data
    #                 point = point.to(self.device)
                    
    #                 opts, params, _, _, _, geo_latent, prim_latent = self.model(point)
                    
    #                 ## for chair 24, 15, 22
    #                 # ref 1
    #                 gt_idx = 24
    #                 point = [tmp[gt_idx,:,:].unsqueeze(0) for tmp in opts]
    #                 superquads = [[sub_p[gt_idx,:].unsqueeze(0) for sub_p in p] for p in params]
    #                 valid_parts = [i for i in range(len(superquads)) if superquads[i][5]>0.5]
    #                 ## remove inactivated parts
    #                 point = [point[i] for i in range(len(point)) if i in valid_parts]
    #                 superquads = [superquads[i] for i in range(len(superquads)) if i in valid_parts]

    #                 gt_ori = point.copy()
    #                 ## vis ground truth
    #                 gt = [self._pt_trans(point[i], superquads[i][0], superquads[i][1]).squeeze(0) for i in range(len(point))]
    #                 vis = MayaviVisualization(superquads, gt, pc_colors=clr, vis_r=[theta, phi, gamma],\
    #                                           pt_size=pt_size, vis_sq=False)
    #                 vis.configure_traits()
                    
    #                 # for idx in [24, 15, 22]:# for chair
    #                 ## ref 2
    #                 idx = 15
    #                 point = [tmp[idx,:,:].unsqueeze(0) for tmp in opts]
    #                 rt_sq = [[sub_p[idx,:].unsqueeze(0) for sub_p in p] for p in params]
    #                 ## remove inactivated parts
    #                 point = [point[i] for i in range(len(point)) if i in valid_parts]
    #                 rt_sq = [rt_sq[i] for i in range(len(rt_sq)) if i in valid_parts]

    #                 rt_ori = point.copy()
    #                 # vis
    #                 rt = [self._pt_trans(point[i], rt_sq[i][0], rt_sq[i][1]).squeeze(0) for i in range(len(point))]
    #                 vis = MayaviVisualization(rt_sq, rt, pc_colors=clr, vis_r=[theta, phi, gamma],\
    #                                           pt_size=pt_size, vis_sq=False)
    #                 vis.configure_traits()
                    
    #                 tmp_idx = 2
    #                 for prob in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #                     mixed_latent = geo_latent.clone()
    #                     mixed_latent = (1-prob)*geo_latent + prob*geo_latent[idx,...].unsqueeze(0)
    #                     mx_prim_latent = (1-prob)*prim_latent + prob*prim_latent[idx,...].unsqueeze(0)
    #                     opts, params = self.model.local_lt2pt_back(mixed_latent, mx_prim_latent)

    #                     point = [tmp[gt_idx,:,:].unsqueeze(0) for tmp in opts]
    #                     superquads = [[sub_p[gt_idx,:].unsqueeze(0) for sub_p in p] for p in params]
    #                     valid_parts = [i for i in range(len(superquads)) if superquads[i][5]>0.5]
    #                     print('valid idx ', valid_parts)
    #                     ## remove inactivated parts
    #                     point = [point[i] for i in range(len(point)) if i in valid_parts]
    #                     superquads = [superquads[i] for i in range(len(superquads)) if i in valid_parts]

    #                     ## vis mixed
    #                     gt = [self._pt_trans(point[i], superquads[i][0], superquads[i][1]).squeeze(0) for i in range(len(point))]
    #                     vis = MayaviVisualization(superquads, gt, pc_colors=clr, vis_r=[theta, phi, gamma],\
    #                                               pt_size=pt_size, vis_sq=False)
    #                     vis.configure_traits()
                    
    #                 break
        
    def shape_resize(self):
        with torch.no_grad():
            for _iter, data in enumerate(self.train_loader):
                if _iter == 0:
                    point, _ = data
                    point = point.to(self.device)
                    
                    opts, params, _, _, _, _, _ = self.model(point)
                    
                    idx_pt = 24
                    sample = [tmp[idx_pt,:,:].unsqueeze(0) for tmp in opts]
                    sample_p = [[sub_p[idx_pt,:].unsqueeze(0) for sub_p in p] for p in params]
                    self._vp(out_list=[self._pt_trans(sample[i], sample_p[i][0], sample_p[i][1], 
                                                      sample_p[i][2], sample_p[i][4]).unsqueeze(0) for i in range(len(sample)) if sample_p[i][5]>0.5], title='Original 1')
                    
                    idx_pt = 28
                    sample2 = [tmp[idx_pt,:,:].unsqueeze(0) for tmp in opts]
                    sample_p2 = [[sub_p[idx_pt,:].unsqueeze(0) for sub_p in p] for p in params]
                    self._vp(out_list=[self._pt_trans(sample2[i], sample_p2[i][0], sample_p2[i][1],
                                                     sample_p2[i][2], sample_p2[i][4]).unsqueeze(0) for i in range(len(sample2)) if sample_p2[i][5]>0.5], title='Original 2')
                    
                    valid_parts = [i for i in range(len(sample_p)) if sample_p[i][5]>0.5]
                    idx = valid_parts[0]
                    tmp = sample[idx]
                    tmp = inverse_deform(tmp, sample_p[idx][2], sample_p[idx][4])
                    tmp = tmp/sample_p[idx][2].unsqueeze(0)
                    tmp = tmp*sample_p2[idx][2].unsqueeze(0)
                    tmp = deform(tmp.unsqueeze(1), sample_p[idx][2].unsqueeze(1), sample_p[idx][4].unsqueeze(1)).squeeze(1)
                    sample2[idx] = tmp
                    self._vp(out_list=[self._pt_trans(sample2[i], sample_p2[i][0], sample_p2[i][1],
                                                     sample_p2[i][2], sample_p2[i][4]).unsqueeze(0) for i in range(len(sample2)) if sample_p2[i][5]>0.5], title='parts resized and re-deformed')
                    
                    valid_parts = [i for i in range(len(sample_p)) if sample_p[i][5]>0.5]
                    idx = valid_parts[0]
                    tmp = sample[idx]
                    tmp = tmp/sample_p[idx][2].unsqueeze(0)
                    tmp = tmp*sample_p2[idx][2].unsqueeze(0)
                    sample2[idx] = tmp
                    self._vp(out_list=[self._pt_trans(sample2[i], sample_p2[i][0], sample_p2[i][1],
                                                     sample_p2[i][2], sample_p2[i][4]).unsqueeze(0) for i in range(len(sample2)) if sample_p2[i][5]>0.5], title='parts resized')
                    
                    # no resize
                    tmp = sample[idx]
                    sample2[idx] = tmp
                    self._vp(out_list=[self._pt_trans(sample2[i], sample_p2[i][0], sample_p2[i][1],
                                                     sample_p2[i][2], sample_p2[i][4]).unsqueeze(0) for i in range(len(sample2)) if sample_p2[i][5]>0.5], title='parts no resize')
                
                    break
            
    # def shape_mixing(self, rescale=False):
    #     theta = torch.tensor(1/6 * math.pi) # x
    #     phi = torch.tensor(-1/4 * math.pi) # y
    #     gamma = torch.tensor(-1/12 * math.pi) # z
        
    #     pt_size = .045
        
    #     pink = (208/255, 193/255, 198/255)
    #     light = (216/255, 202/255, 175/255)
    #     blue = (134/255, 150/255, 167/255)
    #     brown = (162/255, 153/255, 136/255)
    #     dark = (191/255, 191/255, 191/255)
    #     green = (130/255, 144/255, 96/255)
    #     red = (150/255, 84/255, 84/255)
        
    #     gt_red = (162/255, 153/255, 136/255)
    #     gt_blue = (134/255, 150/255, 167/255)
    #     gt_orange = (173/255, 83/255, 72/255)
    #     gt_yellow = (241/255, 182/255, 26/255)
    #     gt_green = (130/255, 144/255, 96/255)
        
    #     hightlight_clr = [red, blue, green]
    #     background_clr = [dark, dark, dark, dark, dark, dark, dark, dark]
    #     gt_clr = [red, red, red, red, red, red, red]
    #     ref_clr = [blue, blue, blue, blue, blue, blue, blue, blue]
        
    #     with torch.no_grad():
    #         for _iter, data in enumerate(self.train_loader):
    #             if _iter == 0:
    #                 point, _ = data
    #                 point = point.to(self.device)
                    
    #                 opts, params, _, _, _, _, _ = self.model(point)
                    
    #                 ## for airplane 0, 6, 7, 8, 9, 10, 11, 12, 13, 26, 28
    #                 ## for chair 24, 15, 22
    #                 ## for table 1, 2, 3, 6, 7, 9, 10, 15, 27
    #                 # for gt_idx in range(self.batch_size):
    #                 #     print(gt_idx)
    #                 gt_idx = 20
    #                 point = [tmp[gt_idx,:,:].unsqueeze(0) for tmp in opts]
    #                 superquads = [[sub_p[gt_idx,:].unsqueeze(0) for sub_p in p] for p in params]
    #                 valid_parts = [i for i in range(len(superquads)) if superquads[i][5]>0.5]
    #                 ## remove inactivated parts
    #                 point = [point[i] for i in range(len(point)) if i in valid_parts]
    #                 superquads = [superquads[i] for i in range(len(superquads)) if i in valid_parts]

    #                 gt_ori = point.copy()
    #                 ## vis ground truth
    #                 gt = [self._pt_trans(point[i], superquads[i][0], superquads[i][1]).squeeze(0) for i in range(len(point))]
    #                 vis = MayaviVisualization(superquads, gt, pc_colors=gt_clr, vis_r=[theta, phi, gamma],\
    #                                           pt_size=pt_size, vis_sq=False)
    #                 vis.configure_traits()
                    
    #                 # for idx in [24, 15, 22]:# for chair
    #                 # for idx in [6, 7, 13, 26]: # for airplane 3
    #                 # for idx in [0, 1, 3, 5, 9, 11, 13, 15, 17, 20, 22, 24, 25, 26]:
    #                 for idx in [2, 5, 6, 8, 12, 16, 26]:
    #                     print(idx)
    #                     point = [tmp[idx,:,:].unsqueeze(0) for tmp in opts]
    #                     rt_sq = [[sub_p[idx,:].unsqueeze(0) for sub_p in p] for p in params]
    #                     ## remove inactivated parts
    #                     point = [point[i] for i in range(len(point)) if i in valid_parts]
    #                     rt_sq = [rt_sq[i] for i in range(len(rt_sq)) if i in valid_parts]
                        
    #                     rt_ori = point.copy()
    #                     # vis
    #                     rt = [self._pt_trans(point[i], rt_sq[i][0], rt_sq[i][1]).squeeze(0) for i in range(len(point))]
    #                     vis = MayaviVisualization(rt_sq, rt, pc_colors=ref_clr, vis_r=[theta, phi, gamma],\
    #                                               pt_size=pt_size, vis_sq=False)
    #                     vis.configure_traits()
                        
    #                     if self.category is 'table' and self.sp_reg_terms['minimum_number_of_primitives']==6:
    #                         rg = [[1, 4], [0, 2, 3]]
    #                     else:
    #                         rg = range(len(gt))
    #                     for pt_i in rg:
    #                         tmp = rt_ori.copy()
    #                         clr = ref_clr.copy()
                            
    #                         if isinstance(pt_i, list):
    #                             for sub_i in pt_i:
    #                                 tmp[sub_i] = gt_ori[sub_i]
    #                                 clr[sub_i] = gt_clr[sub_i]
    #                         else:
    #                             tmp[pt_i] = gt_ori[pt_i]
    #                             clr[pt_i] = gt_clr[pt_i]
                            
    #                         if rescale:
    #                             # size, resize
    #                             ## inverse transform
    #                             tmp[pt_i] = inverse_deform(tmp[pt_i], superquads[pt_i][2], superquads[pt_i][4])
    #                             tmp[pt_i] = tmp[pt_i]/superquads[pt_i][2].unsqueeze(1)

    #                             # transform
    #                             tmp[pt_i] = tmp[pt_i]*rt_sq[pt_i][2].unsqueeze(0)
    #                             tmp[pt_i] = deform(tmp[pt_i].unsqueeze(1), rt_sq[pt_i][2].unsqueeze(0),\
    #                                                                        rt_sq[pt_i][4].unsqueeze(0)).squeeze(0)
                                
    #                         rt = [self._pt_trans(tmp[i], rt_sq[i][0], rt_sq[i][1]).squeeze(0) for i in range(len(point))]
    #                         vis = MayaviVisualization(rt_sq, rt, pc_colors=clr, vis_r=[theta, phi, gamma],\
    #                                                   pt_size=pt_size, vis_sq=False)
    #                         vis.configure_traits()
    
    #                 break
                
    # def shape_sampling(self, from_local=True, repeat=1):
    #     theta = torch.tensor(1/6 * math.pi) # x
    #     phi = torch.tensor(-1/4 * math.pi) # y
    #     gamma = torch.tensor(-1/12 * math.pi) # z

    #     pt_size = .045
    #     # pt_size = .02
        
    #     pink = (208/255, 193/255, 198/255)
    #     light = (216/255, 202/255, 175/255)
    #     blue = (134/255, 150/255, 167/255)
    #     brown = (162/255, 153/255, 136/255)
    #     dark = (191/255, 191/255, 191/255)
    #     green = (134/255, 145/255, 115/255)
    #     red = (150/255, 84/255, 84/255)
    #     orange = (237/255, 120/255, 83/255)
    #     purple = (181/255, 122/255, 174/255)
    #     # hightlight_clr = [green, blue, orange]
    #     hightlight_clr = [red, red, red, red, red, red]
    #     background_clr = [dark, dark, dark, dark, dark, dark, dark, dark]
    #     # background_clr = [pink, purple, blue, brown, dark, green, red, light]
    #     ## valid index for table: 1, 4
        
        
    #     with torch.no_grad():
    #         for _iter, data in enumerate(self.train_loader):
    #             if _iter == 0:
    #                 point, _ = data
    #                 point = point.to(self.device)
                    
    #                 opts, params, _, _, _, _, _ = self.model(point)
                    
    #                 idx_pt = 2 ## for table
    #                 ## ground truth point cloud
    #                 point = [tmp[idx_pt,:,:].unsqueeze(0) for tmp in opts]
    #                 superquads = [[sub_p[idx_pt,:].unsqueeze(0) for sub_p in p] for p in params]
    #                 valid_parts = [i for i in range(len(superquads)) if superquads[i][5]>0.5]
    #                 point = [point[i] for i in range(len(point)) if i in valid_parts]
    #                 superquads = [superquads[i] for i in range(len(superquads)) if i in valid_parts]
    #                 gt = [self._pt_trans(point[i], superquads[i][0], superquads[i][1]).squeeze(0) for i in range(len(point))]

    #                 ## vis ground truth
    #                 vis = MayaviVisualization(superquads, gt, pc_colors=background_clr, vis_r=[theta, phi, gamma],\
    #                                           pt_size=pt_size, vis_sq=False)
    #                 vis.configure_traits()
                    
    #                 # idx = [0, 2, 3]
    #                 # idx = [0,1,2,3,4]
    #                 # idx = [1]
    #                 idx = [0, 2]
    #                 for _r in range(repeat):
    #                     if from_local:
    #                         geo_latent = []
    #                         for m in idx:
    #                             geo_latent.append(torch.randn(self.batch_size, 32).to(self.device))
    #                     else:
    #                         latent = torch.randn(self.batch_size, self.n_latents).to(self.device)
    #                         geo_latent = []
    #                         for m in idx:
    #                             geo_latent.append(self.model.fc_geo[valid_parts[m]](latent).to(self.device))

                       
    #                     for sub_i in range(len(idx)):
    #                         geo_latent[sub_i] = self.model.geo_nf.g(geo_latent[sub_i])
                        
    #                     parts = []
    #                     for sub_i in range(len(idx)):
    #                         parts.append(self.model.geo_decoders[valid_parts[idx[sub_i]]](geo_latent[sub_i].unsqueeze(1)))

    #                     for b in range(self.batch_size):

    #                         tmp = point
    #                         for sub_i in range(len(idx)):
    #                             tmp_idx = idx[sub_i]
    #                             tmp[tmp_idx] = parts[sub_i][b,:,:].unsqueeze(0)
    #                         tmp = [self._pt_trans(tmp[i], superquads[i][0], superquads[i][1]).squeeze(0)\
    #                                for i in range(len(tmp))]

    #                         tmp_clr = background_clr.copy()
    #                         for sub_i in range(len(idx)):
    #                             tmp_idx = idx[sub_i]
    #                             tmp_clr[tmp_idx] = hightlight_clr[tmp_idx]

    #                         vis = MayaviVisualization(superquads, tmp, pc_colors=tmp_clr, vis_r=[theta, phi, gamma],\
    #                                               pt_size=pt_size, vis_sq=False)
    #                         vis.configure_traits()

    #                         # break
                        
    #                 return
    
    def _vp(self, in_list=None, out_list=None, title=None, c_list=None):
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        batch_size = out_list[0].size(0)
        n_parts = len(out_list)
        for b in range(batch_size):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(n_parts):
                # plot reconstructed
                if out_list is not None:
                    x = out_list[i][b,:,0].detach().cpu().numpy()
                    y = out_list[i][b,:,1].detach().cpu().numpy()
                    z = out_list[i][b,:,2].detach().cpu().numpy()
                    x = x[x!=0]; y = y[y!=0]; z = z[z!=0]
                    ax.scatter3D(x, y, z, c = color[i], s = 5, depthshade=True)
                    
                if in_list is not None:
                    # plot baselines
                    x = in_list[i][b,:,0].detach().cpu().numpy()
                    y = in_list[i][b,:,1].detach().cpu().numpy()
                    z = in_list[i][b,:,2].detach().cpu().numpy()
                    x = x[x!=0]; y = y[y!=0]; z = z[z!=0]
                    ax.scatter3D(x+0.5, y+0.5, z+0.5, c = color[i], s = 5, depthshade=True)
                    
                if c_list is not None:
                    # plot baselines
                    x = c_list[i][b,:,0].detach().cpu().numpy()
                    y = c_list[i][b,:,1].detach().cpu().numpy()
                    z = c_list[i][b,:,2].detach().cpu().numpy()
                    x = x[x!=0]; y = y[y!=0]; z = z[z!=0]
                    ax.scatter3D(x-0.5, y-0.5, z-0.5, c = color[i], s = 5, depthshade=True)
            ax.set_xlabel("x-axis")
            ax.set_ylabel("y-axis")
            ax.set_zlabel("z-axis")
            ax.grid(False)
            plt.axis('off')
            plt.title('Generated point cloud' if title is None else title)   
    
    def reconstruction(self, train=True, separately=False):
        data_loader = self.train_loader if train else self.test_loader
        loss = []; chamfer_distance = []; kldivergence = []; coverage = []; quality = []
        primitive = []; regularizer = []
            
        probs = [[], [], [], [],
                 [], [], [], []]
            
        with torch.no_grad():
            for _iter, data in enumerate(data_loader):
                point, _ = data
                if self.symmetry:
                    point = self.symmetry_pt(point)
                point = point.to(self.device)

                opts, primitive_params, mu, logvar, _, _, _ = self.model(point)
                print((batched_params(primitive_params)[5].mean(dim=0) > 0.5).nonzero(as_tuple=True)[0])
                idx = nn_labels(point, primitive_params, self.sampler)
                sp_ls, prim, regs, kld = self.trans_loss(point, primitive_params, mu, logvar)
                geo_ls, cd, covs, quals, _ = self.geo_loss(point, opts, primitive_params, idx, mu, logvar)

                ls = (sp_ls + geo_ls)/2
                
                for i in range(len(primitive_params)):
                    probs[i].append(primitive_params[i][5])
                
                print("[ Iter ] ", "{:3}".format(_iter),
                      "[ Loss ] ", "{: 7.6f}".format(ls.item()),
                      "[ Chamfer distance ] ", "{: 7.6f}".format(cd.item()),
                      "[ Coverage ] ", "{: 7.6f}".format(covs.item()),
                      "[ Quality ] ", "{: 7.6f}".format(quals.item()),
                      "[ KL Divergence ] ", "{: 7.6f}".format(kld),
                      "[ Primitive loss ] ", "{: 7.6f}".format(prim.item()),
                      "[ regularizer ] ", "{: 7.6f}".format(regs.item()))
                loss.append(ls.item()); chamfer_distance.append(cd.item()); kldivergence.append(kld)
                coverage.append(covs.item()); quality.append(quals.item())
                primitive.append(prim.item()); regularizer.append(regs.item())
            
        for i in range(len(primitive_params)):
            probs[i] = torch.cat(probs[i], dim=0)    
        probs = torch.cat(probs, dim=1).detach().cpu().numpy()
            
        fig, axes = plt.subplots(8, 1, figsize=(10, 10))
        for i in range(len(axes.flatten())):
            a = axes.flatten()[i]
            p = probs[:, i]
            a.hist(p, 200, density = True)
        fig.tight_layout()
        plt.show()
            
            
        mean_ls = sum(loss)/float(len(loss)); mean_cd = sum(chamfer_distance)/float(len(chamfer_distance))
        mean_kld = sum(kldivergence)/float(len(kldivergence))
        mean_cov = sum(coverage)/float(len(coverage)); mean_qual = sum(quality)/float(len(quality))
        mean_prim = sum(primitive)/float(len(primitive)); mean_reg = sum(regularizer)/float(len(regularizer))
        
        print("[ Mean loss ] ", "{: 7.6f}".format(mean_ls),
              "[ Mean Chamfer distance ] ", "{: 7.6f}".format(mean_cd),
              "[ Mean Coverage ] ", "{: 7.6f}".format(mean_cov),
              "[ Mean Quality ] ", "{: 7.6f}".format(mean_qual),
              "[ Mean KL Divergence ] ", "{: 7.6f}".format(mean_kld),
              "[ Mean Primitive reconstruction loss ] ", "{: 7.6f}".format(mean_prim),
              "[ Mean Regularizer ] ", "{: 7.6f}".format(mean_reg))
        return mean_ls, mean_cd, mean_cov, mean_qual, mean_kld, mean_prim, mean_reg
    
    def generate(self, distribution, parts=False, balanced=False):     
        if parts:
            rg = range(self.train_len, self.train_len+self.test_len*self.n_parts, self.n_parts)
        else:
            rg = range(self.train_len, self.train_len+self.test_len)
            
        if distribution is 'M_Gaussian':
            self.sample_from_encoder()
               
        with torch.no_grad():
            for _iter in rg:
                if distribution is 'M_Gaussian':
                    latent = self.M_Gaussian()
                elif distribution is 'S_Gaussian':
                    latent = torch.randn(self.batch_size, self.n_latents)
                latent = latent.to(self.device)
                
                opts, params = self.g_parts(latent)
                opts = [self._pt_trans(opts[i], params[i][0], params[i][1], params[i][2], params[i][4]) for i in range(len(opts))]
                
                print((batched_params(params)[5].mean(dim=0) > 0.5).nonzero(as_tuple=True)[0])
                
                if balanced:
                    # choice0 = np.random.choice(opts[0].size(1), size=int(500), replace=False)
                    choice1 = np.random.choice(opts[0].size(1), size=int(160), replace=False)
                    choice2 = np.random.choice(opts[0].size(1), size=int(233), replace=False)
                    # choice3 = np.random.choice(opts.size(1), size=int(700), replace=True)
                    opts[0] = opts[0][:,choice1,:]
                    opts[6] = opts[6][:,choice2,:]
                    # opts[0] = opts[0][:,choice0,:]
                    # opts[2] = opts[2][:,choice2,:]
                    opts = torch.cat(opts, dim=1)
                
                gt = []
                for i in range(len(opts)):
                    pt = opts[i]
                    if parts:
                        torch.save(pt, self.path+'generated/'+str(_iter+i)+'.pt')
                    else:
                        # if params[i][5].mean() > 0.8:
                        #     gt.append(pt)
                        gt.append(pt)
                
                if not parts:
                    out = []
                    for b in range(latent.size(0)):
                        pt = []
                        for j in range(len(gt)):
                            if params[j][5][b] > 0.5:
                                pt.append(gt[j][b,...])
                        if len(pt) != 0:
                            pt = torch.cat(pt, dim=0)
                            out.append(pt)
                    out = pad_sequence(out).permute(1, 0, 2)
                        
                    # gt = torch.cat(gt, dim=1)
                    torch.save(out, self.path+'generated/'+str(_iter)+'.pt')
                    
        print('All generated saved!')

    def g_parts(self, latent):
        B = latent.size(0)
        
        geo_latent = self.model.fc_geo(latent)
        prim_latent = self.model.fc_prim(latent)
        if self.use_realNVP:
            geo_latent, _ = self.model.geo_nf.f(geo_latent)
            geo_latent = self.model.geo_nf.g(geo_latent)
            prim_latent, _ = self.model.prim_nf.f(prim_latent)
            prim_latent = self.model.prim_nf.g(prim_latent)

        params = []
        geo_opts = []
        for i in range(self.n_parts):
            geo_opts.append(self.model.geo_decoders[i](geo_latent.unsqueeze(1)))
            p = PrimitiveParams()
            for j in range(len(self.model.prims_decoders[i])):
                p[j] = self.model.prims_decoders[i][j](prim_latent).view(B, -1)
            params.append(p)
        
        return geo_opts, params
    
    def generate_from_NVP(self, parts=False):     
        if parts:
            rg = range(self.train_len, self.train_len+self.test_len*self.n_parts, self.n_parts)
        else:
            rg = range(self.train_len, self.train_len+self.test_len)
                
        with torch.no_grad():
            for _iter in rg:
                geo_latent = torch.randn(self.batch_size, 32).to(self.device)
                geo_latent = self.model.geo_nf.g(geo_latent)
                prim_latent = torch.randn(self.batch_size, 16).to(self.device)
                prim_latent = self.model.prim_nf.g(prim_latent)
                
                params = []
                geo_opts = []
                for i in range(self.n_parts):
                    geo_opts.append(self.model.geo_decoders[i](geo_latent.unsqueeze(1)))
                    p = PrimitiveParams()
                    for j in range(len(self.model.prims_decoders[i])):
                        p[j] = self.model.prims_decoders[i][j](prim_latent).view(self.batch_size, -1)
                    params.append(p)
                
                opts = [self._pt_trans(geo_opts[i], params[i][0], params[i][1], params[i][2], params[i][4]) for i in range(len(geo_opts))]
                
                gt = []
                for i in range(len(opts)):
                    pt = opts[i]
                    if parts:
                        torch.save(pt, self.path+'generated/'+str(_iter+i)+'.pt')
                    else:
                        gt.append(pt)
                
                if not parts:
                    out = []
                    for b in range(self.batch_size):
                        pt = []
                        for j in range(len(gt)):
                            if params[j][5][b] > 0.5:
                                pt.append(gt[j][b,...])
                        if len(pt) != 0:
                            pt = torch.cat(pt, dim=0)
                            out.append(pt)
                    out = pad_sequence(out).permute(1, 0, 2)
                        
                    torch.save(out, self.path+'generated/'+str(_iter)+'.pt')
                    
        print('All generated saved!')
        
        
    # def vis_NVP(self, parts=False):
    #     theta = torch.tensor(1/6 * math.pi) # x
    #     phi = torch.tensor(-1/4 * math.pi) # y
    #     gamma = torch.tensor(-1/12 * math.pi) # z
    
    #     pt_size = .045
        
    #     pink = (208/255, 193/255, 198/255)
    #     light = (216/255, 202/255, 175/255)
    #     blue = (134/255, 150/255, 167/255)
    #     brown = (162/255, 153/255, 136/255)
    #     dark = (191/255, 191/255, 191/255)
    #     green = (134/255, 145/255, 115/255)
    #     red = (150/255, 84/255, 84/255)
    #     orange = (237/255, 120/255, 83/255)
    #     purple = (181/255, 122/255, 174/255)
    #     yellow = (240/255, 223/255, 167/255)
    #     background_clr = [blue, red, green, purple, green, light, orange, dark]
        
    #     with torch.no_grad():
    #         for t in range(3):
    #             geo_latent = torch.randn(self.batch_size, 32).to(self.device)
    #             geo_latent = self.model.geo_nf.g(geo_latent)
    #             prim_latent = torch.randn(self.batch_size, 16).to(self.device)
    #             prim_latent = self.model.prim_nf.g(prim_latent)

    #             params = []
    #             geo_opts = []
    #             for i in range(self.n_parts):
    #                 geo_opts.append(self.model.geo_decoders[i](geo_latent.unsqueeze(1)))
    #                 p = PrimitiveParams()
    #                 for j in range(len(self.model.prims_decoders[i])):
    #                     p[j] = self.model.prims_decoders[i][j](prim_latent).view(self.batch_size, -1)
    #                 params.append(p)

    #             opts = [self._pt_trans(geo_opts[i], params[i][0], params[i][1]) for i in range(len(geo_opts))]

    #             for b in range(self.batch_size):
    #                 pt = [tmp[b,...] for tmp in opts]

    #                 ## remove in activate part
    #                 valid_parts = [i for i in range(len(params)) if params[i][5][b]>0.5]
    #                 pt = [pt[i] for i in range(len(pt)) if i in valid_parts]

    #                 vis = MayaviVisualization(point_clouds=pt, pc_colors=background_clr, vis_r=[theta, phi, gamma],\
    #                                                   pt_size=pt_size, vis_sq=False)
    #                 vis.configure_traits()
        
    # def vis_generated(self, distribution='M_Gaussian', n_pcl=None, repeat=1):
    #     theta = torch.tensor(1/6 * math.pi) # x
    #     phi = torch.tensor(-1/4 * math.pi) # y
    #     gamma = torch.tensor(-1/12 * math.pi) # z
        
    #     ## defind colors
    #     pink = (208/255, 193/255, 198/255)
    #     light = (216/255, 202/255, 175/255)
    #     blue = (134/255, 150/255, 167/255)
    #     brown = (162/255, 153/255, 136/255)
    #     dark = (191/255, 191/255, 191/255)
    #     green = (130/255, 144/255, 96/255)
    #     red = (150/255, 84/255, 84/255)
    #     yellow = (240/255, 223/255, 167/255)
    #     orange = (173/255, 83/255, 72/255)
        
    #     if self.category is 'chair':
    #         # parts_clr = [green, yellow, blue, light, pink, red, dark, brown] ## color for chair7
    #         parts_clr = [yellow, red, blue, light, pink, brown, dark, red] ## color for chair3
    #         pt_size = .025 # point size for chair
    #     elif self.category is 'table':
    #         parts_clr = [blue, yellow, green, light, red, dark, brown, pink] ## color for table
    #         pt_size = .045 # point size for table
            
    #         # M=3
    #         # parts_clr = [blue, red, green, light, yellow, dark, brown, pink] ## color for table
    #         # pt_size = .045 # point size for table
            
    #     elif self.category is 'airplane':
    #         # M=6
    #         parts_clr = [yellow, blue, green, light, pink, red, dark, brown] ## color for airplane6 & 3
    #         pt_size = .045 # point size for airplane
            
    #         # # M=3
    #         # parts_clr = [red, blue, green, light, pink, red, dark, brown] ## color for airplane6 & 3
    #         # pt_size = .045 # point size for airplane
        
    #     if distribution is 'M_Gaussian':
    #         self.sample_from_encoder()
            
    #     for i in range(repeat):
    #         with torch.no_grad():
    #             if distribution is 'M_Gaussian':
    #                 latent = self.M_Gaussian()
    #             elif distribution is 'S_Gaussian':
    #                 latent = torch.randn(self.batch_size, self.n_latents)
    #             latent = latent.to(self.device)

    #             opts, params = self.g_parts(latent)
    #             opts = [self._pt_trans(opts[i], params[i][0], params[i][1], params[i][2], params[i][4]) for i in range(len(opts))]

    #             for b in range(self.batch_size):
    #                 # if b > 0:
    #                 #     break
    #                 valid_parts = [i for i in range(len(params)) if params[i][5][b]>0.5]
    #                 parts = [opts[i][b,...] for i in range(len(opts)) if i in valid_parts]
    #                 sq = [[p[b,...] for p in params[i]] for i in range(len(params)) if i in valid_parts]
    #                 if self.category is 'chair':
    #                     parts[2] += torch.tensor([0.0, -0.03, 0.0]).unsqueeze(0).to(self.device) # for chair 3 only
    #                 vis = MayaviVisualization(sq, parts, pc_colors = parts_clr, sq_colors = parts_clr, pt_size=pt_size, vis_r=[theta, phi, gamma], vis_sq=False)
    #                 vis.configure_traits()

    #                 vis = MayaviVisualization(sq, pc_colors = parts_clr, sq_colors = parts_clr, pt_size=pt_size, vis_r=[theta, phi, gamma])
    #                 vis.configure_traits()

    #             # for b in range(self.batch_size):
    #             #     parts = [p[b,...] for p in opts]
    #             #     probs = [p[5].mean().detach().cpu().numpy() for p in params]
    #             #     self._vis_parts(parts, probs)
                
    def _vis_parts(self, parts, probs=[1, 1, 0.1, 1, 1, 1, 1, 0.1], save_path=None):
        ## modify the probabilities
        M = len(parts)
        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        parts = [p.detach().cpu().numpy() for p in parts]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ## vis the fitted_params
        for i in range(M):
            x = parts[i][:,0]; y = parts[i][:,1]; z = parts[i][:,2]
            ax.scatter3D(x, y, z, c = color[i], s=1, alpha=probs[i])
            # ax.scatter3D(x, y, z, c = color[i], s=1, alpha=1)
            
        ## vis gt_points
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.grid(False)
        plt.title('Generated point cloud')
        
        if save_path is not None:
            plt.subplots_adjust()
            plt.savefig(save_path)
        else:
            plt.show()
       
    def get_seg(self, point):
        _, primitive_params, _, _, _, _, _ = self.model(point)
        seg = nn_labels(point, primitive_params, self.sampler)
        
        return seg
    
    def sample_from_encoder(self):
        self.records = []
        with torch.no_grad():
            for _iter, data in enumerate(self.train_loader):
                point, _ = data
                point = point.to(self.device)
                _, _, mu, logvar, _, _, _ = self.model(point)
                self.records.append([mu, logvar])
                
    def vis(self, n_pcl=None):
        with torch.no_grad():
            for _iter, data in enumerate(self.train_loader):
                if _iter == 10:
                    point, _ = data
                    point = point.to(self.device)
                    
                    opts, primitive_params, mu, logvar, _, _, _ = self.model(point)
                    idx = nn_labels(point, primitive_params, self.sampler)
                    primitive_params = batched_params(primitive_params)
                    
                    if n_pcl is None or n_pcl > point.size(0):
                        n_pcl = point.size(0)

                    for j in range(n_pcl):
                        tmp = [None]*len(primitive_params)
                        for i in range(len(primitive_params)):
                            tmp[i] = primitive_params[i][j,...].unsqueeze(0)
                        self._vis_fitted_primitive(tmp, point[j,...], self._single_opt(opts, j), idx[j,...])

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
      
    ## transform from original coordinates to global one, for each part
    ## ori: B*N*3
    ## trans: B*3
    ## rotas: B*4
    ## size: B*3
    ## deform: B*2
    def _pt_trans(self, ori, trans, rotas, size=None, df=None):
        B = ori.size(0)
        N = ori.size(1)
        assert B == trans.size(0) and 3 == trans.size(1)
        assert B == rotas.size(0) and 4 == rotas.size(1)
        
        R = quaternions_to_rotation_matrices(rotas)
        R = torch.transpose(R, 1, 2)
        
        ori = R.unsqueeze(1).matmul(ori.unsqueeze(-1)).squeeze(-1) ## (B, 1, 3, 3) * (B, N, 3, 1)
        ori = ori + trans.unsqueeze(1)
        
        
        return ori.squeeze(0)
    
    ## for a single point cloud
    def _vis_fitted_primitive(self, primitive_params, gt_point, opt, seg_idx, save_path=None):
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
        print(probs)

        color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        z = z.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        
        gt_x = gt_point[:,0].detach().cpu().numpy()
        gt_y = gt_point[:,1].detach().cpu().numpy()
        gt_z = gt_point[:,2].detach().cpu().numpy()
        
        seg_clr = np.take(np.array(color), np.reshape(seg_idx.detach().cpu().numpy(), -1))
        
        opt = [self._pt_trans(opt[i].unsqueeze(0), trans[:,i,:], rotations[:,i,:], 
                              size[:,i,:], deformations[:,i,:]) for i in range(len(opt))]
        opt = [op.detach().cpu().numpy() for op in opt]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ## vis the fitted_params
        for i in range(M):
            ax.scatter3D(x[:,i,:], y[:,i,:], z[:,i,:], c=color[i], s=1, alpha=probs[:,i])
            # ax.scatter3D(x[:,i,:], y[:,i,:], z[:,i,:], c=color[i], s=1)
            
            opt_x = opt[i][:,0]; opt_y = opt[i][:,1]; opt_z = opt[i][:,2]
            ax.scatter3D(opt_x+1, opt_y+1, opt_z+1, c = color[i], s=1, alpha=probs[:,i])
            
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        