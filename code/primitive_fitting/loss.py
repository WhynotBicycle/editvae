import torch
import torch.nn as nn
import numpy as np

from primitives import transform_to_primitives_centric_system, deform, inside_outside_function
from regularizers import get as get_regularizer
from regularizers import emb_prim_reg, emb_discrete, emb_tran_reg
from p_utils import sampling_from_superquadric

class SymmLoss(nn.Module):
    def __init__(self, device):
        super(SymmLoss, self).__init__()
        self.device = device
        
    def forward(self, pcl, primitives, sampler, regularizer_terms, seg_lab=False):
        ## pcl: point cloud with batch_size * n_points * 3
        ## primitives: instance of PrimitiveParams, each is batch_size * n_primitives * 3
        loss_weights = {"pcl_to_prim_weight": 1.0, "prim_to_pcl_weight": 1.0}
        
        translations = primitives[0]
        rotations = primitives[1]
        size = primitives[2]
        shape = primitives[3]
        deformations = primitives[4]
        prob = primitives[5]
        embeddings = primitives[6]
        
        B = translations.size(0)
        M_t = translations.size(1)
        M_p = size.size(1)
        N = pcl.size(1)
        S = sampler.n_samples
        
        ## embedding loss
        emb_reg = emb_prim_reg(embeddings) + 10*emb_tran_reg(embeddings) + emb_discrete(embeddings)
        
        pcl_transformed = transform_to_primitives_centric_system(pcl, translations, rotations) ## B*N*M*3
        
        primitive_points, primitive_normals = sampling_from_superquadric(size, shape, sampler)
        
        primitive_points = deform(primitive_points, size, deformations)
        
        ## make normals in unit vectors
        primitive_normals_norm = primitive_normals.norm(dim=-1).view(B,M_p,S,1)
        primitive_normals = primitive_normals/primitive_normals_norm
        
        assert primitive_points.size() == (B,M_p,S,3)
        assert primitive_normals.size() == (B,M_p,S,3)
        assert pcl_transformed.size() == (B,N,M_t,3)
        ## make sure the unit vector
        assert torch.sqrt(torch.sum(primitive_normals**2, -1)).sum() == B*M_p*S
        
        ## duplicate primitives, change primitive_points
        _, idx = torch.max(embeddings, 1)
        primitive_points = torch.gather(primitive_points, 1, idx.unsqueeze(2).unsqueeze(3).repeat(1,1,S,3))
        for i in [2,3,4]:
            primitives[i] = torch.gather(primitives[i], 1, idx.unsqueeze(2).repeat(1,1,primitives[i].size(2)))
        
        ## compute the Euclidean distance between point cloud and the surface of primitive
        diff = (primitive_points.unsqueeze(3) - (pcl_transformed.permute(0, 2, 1, 3)).unsqueeze(2))
        assert diff.size() == (B, M_t, S, N, 3)
        dist = torch.sum(diff**2, -1)
        assert dist.size() == (B, M_t, S, N)
        
        pcl_to_prim, F = pcl_to_prim_loss(primitives, pcl_transformed, dist)
        assert F is None or F.shape == (B, N, M_t)
        prim_to_pcl = prim_to_pcl_loss(primitives, diff, dist)
              
        regularizers = get_regularizer_term(primitives, F, regularizer_terms)
        reg_values = get_regularizer_weights(regularizers, regularizer_terms)
        
        regs = sum(reg_values.values())
        w1 = loss_weights["pcl_to_prim_weight"]
        w2 = loss_weights["prim_to_pcl_weight"]
        
        return w1 * pcl_to_prim + w2 * prim_to_pcl + regs + emb_reg, pcl_to_prim, prim_to_pcl, regs, emb_reg
#         return emb_reg

class DualLoss(nn.Module):
    def __init__(self, device):
        super(DualLoss, self).__init__()
        self.device = device
        
    def forward(self, pcl, primitives, sampler, regularizer_terms, seg_lab=False):
        ## pcl: point cloud with batch_size * n_points * 3
        ## primitives: instance of PrimitiveParams, each is batch_size * n_primitives * 3
        loss_weights = {"pcl_to_prim_weight": 1.0, "prim_to_pcl_weight": 1.0}
        
        translations = primitives[0]
        rotations = primitives[1]
        size = primitives[2]
        shape = primitives[3]
        deformations = primitives[4]
        probabilities = primitives[5]
        
        B = translations.size(0)
        M = translations.size(1)
        N = pcl.size(1)
        S = sampler.n_samples
        
        pcl_transformed = transform_to_primitives_centric_system(pcl, translations, rotations) ## B*N*M*3
        
        primitive_points, primitive_normals = sampling_from_superquadric(size, shape, sampler)
        
        primitive_points = deform(primitive_points, size, deformations)
        
        ## make normals in unit vectors
        primitive_normals_norm = primitive_normals.norm(dim=-1).view(B,M,S,1)
        primitive_normals = primitive_normals/primitive_normals_norm
        
        assert primitive_points.size() == (B,M,S,3)
        assert primitive_normals.size() == (B,M,S,3)
        assert pcl_transformed.size() == (B,N,M,3)
        ## make sure the unit vector
        assert torch.sqrt(torch.sum(primitive_normals**2, -1)).sum() == B*M*S
        
        ## compute the Euclidean distance between point cloud and the surface of primitive
        diff = (primitive_points.unsqueeze(3) - (pcl_transformed.permute(0, 2, 1, 3)).unsqueeze(2))
        assert diff.size() == (B, M, S, N, 3)
        dist = torch.sum(diff**2, -1)
        assert dist.size() == (B, M, S, N)
        
        if seg_lab:
            idx = seg_labels(dist, probabilities)
        else:
            idx = None
        
        pcl_to_prim, F = pcl_to_prim_loss(primitives, pcl_transformed, dist)
        assert F is None or F.shape == (B, N, M)
        prim_to_pcl = prim_to_pcl_loss(primitives, diff, dist)
              
        regularizers = get_regularizer_term(primitives, F, regularizer_terms)
        reg_values = get_regularizer_weights(regularizers, regularizer_terms)
        
        regs = sum(reg_values.values())
        w1 = loss_weights["pcl_to_prim_weight"]
        w2 = loss_weights["prim_to_pcl_weight"]
        
        return w1 * pcl_to_prim + w2 * prim_to_pcl + regs, pcl_to_prim, prim_to_pcl, regs, idx
    
def seg_labels(dist, probs):
    dist = torch.min(dist, 2)[0].permute(0,2,1)
    _, idx = torch.min(dist, 2)
    return idx
    
# def seg_labels(dist, probs):
#     probs = probs.mean(0)
#     dist = torch.min(dist, 2)[0].permute(0,2,1)
#     ## mash the low probability primitives
#     tmp = (probs<0.5).nonzero().squeeze(-1).detach().cpu().numpy()
#     dist[:,:,tmp] = torch.full_like(dist[:,:,tmp], 1000)

#     _, idx = torch.min(dist, 2)
#     return idx

def pcl_to_prim_loss_parts_based(params, pcl_transformed, dist, want_inout=True):
    B, N, M, _ = pcl_transformed.size()
    
    Np = int(M/N)
    
    size = params[2]
    shape = params[3]
    
    ## inside-outside function, indicate the valid point
    F = None
    if want_inout:
        F = inside_outside_function(pcl_transformed, size, shape)
    
    dist = torch.min(dist, 2)[0].permute(0,2,1) # B*N*M
    assert dist.size() == (B, N, M)
    
    dist = torch.min(dist, 2)[0]
    
    loss = dist.sum() / B / N
    return loss, F

# def prim_to_pcl_loss_parts_based(params, diff, dist):
#     B, M, S, N, _ = diff.size()
    
#     assert dist.size() == (B, M, S, N)
    
#     dist = dist.min(-1)[0]
#     dist[dist >= 1e30] = 0.0
#     assert dist.shape == (B, M, S)
    
#     ## compute an approximate area of the superellipsoid
#     size = params[2]
#     area = 4 * np.pi * (
#         (size[:,:,0] * size[:,:,1])**1.6/3 +
#         (size[:,:,0] * size[:,:,2])**1.6/3 +
#         (size[:,:,1] * size[:,:,2])**1.6/3)**0.625
#     area = M*area/area.sum(dim=-1, keepdim=True)
    
#     loss = torch.einsum("ij,ij->", [torch.mean(dist, -1), area])
#     loss = loss / B / M

#     return loss

def pcl_to_prim_loss(params, pcl_transformed, dist, want_inout=True):
    B, N, M, _ = pcl_transformed.size()
    
    size = params[2]
    shape = params[3]
    probs = params[5]
    
    ## inside-outside function, indicate the valid point
    F = None
    if want_inout:
        F = inside_outside_function(pcl_transformed, size, shape)
    
    dist = torch.min(dist, 2)[0].permute(0,2,1) # B*N*M
    assert dist.size() == (B, N, M)
    
    distance, idx = torch.sort(dist, dim=-1)
    
    ## weighted cumulative product
    probs = torch.cat([probs[i].take(idx[i]).unsqueeze(0) for i in range(len(idx))])
    neg_cumprod = torch.cumprod(1-probs, dim=-1)
    neg_cumprod = torch.cat(
        [neg_cumprod.new_ones((B, N, 1)), neg_cumprod[:,:,:-1]],
        dim=-1
    )
    minprob = probs.mul(neg_cumprod)
    
    loss = torch.einsum("ijk,ijk->", [distance, minprob])
    loss = loss / B / N
    
    return loss, F
  
def prim_to_pcl_loss(params, diff, dist):
    B, M, S, N, _ = diff.size()
    
    assert dist.size() == (B, M, S, N)
    
    dist = dist.min(-1)[0]
    dist[dist >= 1e30] = 0.0
    assert dist.shape == (B, M, S)
    
    ## compute an approximate area of the superellipsoid
    size = params[2]
    area = 4 * np.pi * (
        (size[:,:,0] * size[:,:,1])**1.6/3 +
        (size[:,:,0] * size[:,:,2])**1.6/3 +
        (size[:,:,1] * size[:,:,2])**1.6/3)**0.625
    area = M*area/area.sum(dim=-1, keepdim=True)
    
    loss = torch.einsum("ij,ij->", [torch.mean(dist, -1), area])
    loss = loss / B / M

    return loss
    
# def prim_to_pcl_loss(params, diff, dist):
#     B, M, S, N, _ = diff.size()
    
#     probs = params[5]
    
#     assert dist.size() == (B, M, S, N)
    
#     dist = dist.min(-1)[0]
#     dist[dist >= 1e30] = 0.0
#     assert dist.shape == (B, M, S)
    
#     ## compute an approximate area of the superellipsoid
#     size = params[2]
#     area = 4 * np.pi * (
#         (size[:,:,0] * size[:,:,1])**1.6/3 +
#         (size[:,:,0] * size[:,:,2])**1.6/3 +
#         (size[:,:,1] * size[:,:,2])**1.6/3)**0.625
#     area = M*area/area.sum(dim=-1, keepdim=True)
    
#     loss = torch.einsum("ij,ij,ij->", [torch.mean(dist, -1), probs, area])
#     loss = loss / B / M

#     return loss
    
def get_regularizer_term(params, F, regularizer_terms):
    regularizers = [
        "sparsity_regularizer",
        "bernoulli_regularizer",
        "entropy_bernoulli_regularizer",
        "parsimony_regularizer",
        "overlapping_regularizer",
        "diversity_regularizer"
    ]
    if regularizer_terms["regularizer_type"] is None:
        regularizer_terms["regularizer_type"] = []

    return {
        r: get_regularizer(
            r if r in regularizer_terms["regularizer_type"] else "",
            params,
            F,
            regularizer_terms
        )
        for r in regularizers
    }
    
def get_regularizer_weights(regularizers, regularizer_terms):
    # Ensures that the expected number of primitives lies between a minimum and
    # a maximum number of primitives.
    bernoulli_reg = regularizers["bernoulli_regularizer"] *\
        regularizer_terms["bernoulli_regularizer_weight"]
    # Ensures that the bernoullis will be either 1.0 or 0.0 and not 0.5
    entropy_bernoulli_reg = regularizers["entropy_bernoulli_regularizer"] *\
        regularizer_terms["entropy_bernoulli_regularizer_weight"]
    # Minimizes the expected number of primitives
    parsimony_reg = regularizers["parsimony_regularizer"] *\
        regularizer_terms["parsimony_regularizer_weight"]
    # Ensures that primitves do not intersect with each other using the F
    # function
    overlapping_reg = regularizers["overlapping_regularizer"] *\
        regularizer_terms["overlapping_regularizer_weight"]
    # Similar to the bernoulli_regularizer. Again we want to ensure that the
    # expected number of primitives will be between a minimum an a maximum
    # number of primitives.
    sparsity_reg = regularizers["sparsity_regularizer"] *\
        regularizer_terms["sparsity_regularizer_weight"]
    
    diversity_reg = regularizers["diversity_regularizer"] *\
        regularizer_terms["diversity_regularizer_weight"]

    reg_values = {
        "sparsity_regularizer": sparsity_reg,
        "overlapping_regularizer": overlapping_reg,
        "parsimony_regularizer": parsimony_reg,
        "entropy_bernoulli_regularizer": entropy_bernoulli_reg,
        "bernoulli_regularizer": bernoulli_reg,
        "diversity_regularizer": diversity_reg
    }

    return reg_values
    
    
    
    
    
    