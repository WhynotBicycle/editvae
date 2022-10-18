import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
sys.path.append('code/utils')
from basic_modules import PointNetfeat

from p_utils import PrimitiveParams

sys.path.append('code/oVAEs')
from realNVP_module import RealNVP

## output batch_size * n_primitives * 3
## lie inside the unit cube
class TransNN(nn.Module):
    def __init__(self, n_primitives, input_channels):
        super(TransNN, self).__init__()
        self.n_primitives = n_primitives
        self.translation_layer = nn.Linear(input_channels, self.n_primitives*3)
        self.tanh = nn.Tanh()

    def forward(self, lt):
        ## input latent: batch_size * input_channels
        trans = self.tanh(self.translation_layer(lt)) * 0.51
        return trans.view(-1, self.n_primitives, 3)

## output batch_size * n_primitives * 4
## with non-linearity as L2-norm to ensure the unit norm constrain
class RotaNN(nn.Module):
    def __init__(self, n_primitives, input_channels):
        super(RotaNN, self).__init__()
        self.n_primitives = n_primitives
        self.rotation_layer = nn.Linear(input_channels, self.n_primitives*4)

    def forward(self, lt):
        ## input latent: batch_size * input_channels
        quats = self.rotation_layer(lt)
        quats = quats.view(-1, self.n_primitives, 4)
        rotations = quats / torch.norm(quats, 2, -1, keepdim=True)
        return rotations

## output batch_size * n_primitives * 3
## size in [1e-2, 0.51] to avoid numerical instabilities with the inside-outside function
class SizeNN(nn.Module):
    def __init__(self, n_primitives, input_channels):
        super(SizeNN, self).__init__()
        self.n_primitives = n_primitives
        self.size_layer = nn.Linear(input_channels, self.n_primitives*3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lt):
        ## input latent: batch_size * input_channels
        sizes = self.sigmoid(self.size_layer(lt)) * 0.5 + 0.03
        return sizes.view(-1, self.n_primitives, 3)

    
## output batch_size * n_primitives * 2
## size in [0.4, 1.5] to avoid numerical instabilities with the inside-outside function
class ShapeNN(nn.Module):
    def __init__(self, n_primitives, input_channels):
        super(ShapeNN, self).__init__()
        self.n_primitives = n_primitives
        self.shape_layer = nn.Linear(input_channels, self.n_primitives*2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lt):
        ## input latent: batch_size * input_channels
        shapes = self.sigmoid(self.shape_layer(lt))*1.1 + 0.4
        return shapes.view(-1, self.n_primitives, 2)

## output batch_size * n_primitives * 2
class DeformNN(nn.Module):
    def __init__(self, n_primitives, input_channels):
        super(DeformNN, self).__init__()
        self.n_primitives = n_primitives
        self.deform_layer = nn.Linear(input_channels, self.n_primitives*2)
        self.tanh = nn.Tanh()

    def forward(self, lt):
        ## input latent: batch_size * input_channels
        taperings = self.tanh(self.deform_layer(lt))*0.9
        return taperings.view(-1, self.n_primitives, 2)
    
## batch_size * n_primitives
class ProbNN(nn.Module):
    def __init__(self, n_primitives, input_channels):
        super(ProbNN, self).__init__()
        self.n_primitives = n_primitives
        self.probability_layer = nn.Linear(input_channels, self.n_primitives)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lt):
        probs = self.sigmoid(self.probability_layer(lt))
        return probs

class embeddingNN(nn.Module):
    def __init__(self, latent_size=256, n_prims=4, n_trans=8):
        super(embeddingNN, self).__init__()
        
        self.latent_size = latent_size
        self.n_prims = n_prims
        self.n_trans = n_trans
        self.linear = nn.Linear(latent_size, n_prims*n_trans)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, lt):
        ot = self.sigmoid(self.linear(lt))
        
        return ot.view(-1, self.n_prims, self.n_trans)
    
class SegNets(nn.Module):
    def __init__(self, latent_size=256, n_primitive=10):
        super(SegNets, self).__init__()
        
        self.latent_size = latent_size
        self.n_primitives = n_primitive
        self.encoder = nn.Sequential(
            PointNetfeat(size=self.latent_size, trans=False, layers=[3, 64, 128, 128, 256]),
            nn.Linear(self.latent_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU()
        )
        
        self.decoders = nn.ModuleList(
            [TransNN(n_primitive, latent_size),
             RotaNN(n_primitive, latent_size),
             SizeNN(n_primitive, latent_size),
             ShapeNN(n_primitive, latent_size),
             DeformNN(n_primitive, latent_size),
             ProbNN(n_primitive, latent_size)]
        )
        
    def forward(self, x):
        latent = self.encoder(x.transpose(1,2))
        
        params = PrimitiveParams()
        for i in range(len(self.decoders)):
            params[i] = self.decoders[i](latent)
            
        return latent, params
    
def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    eps = Variable(eps)
    return mu + eps*std

def KL(mu, logvar, bias=0):
    tmp = - 0.5 * torch.mean(1 + logvar - (mu-bias).pow(2) - logvar.exp())
    return tmp

class NVPSegNets(nn.Module):
    def __init__(self, latent_size=256, prim_size=16, n_parts=8, device='cpu'):
        super(NVPSegNets, self).__init__()
        
        self.latent_size = latent_size
        self.n_parts = n_parts
        self.prim_size = prim_size
        self.device = device
        
        self.encoder = nn.Sequential(
            PointNetfeat(size=self.latent_size, trans=False, layers=[3, 64, 128, 128, 256]),
            nn.Linear(self.latent_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU()
        )
        self.fc_logvar = nn.Linear(self.latent_size, self.latent_size)
        self.fc_mu = nn.Linear(self.latent_size, self.latent_size)
        
        ## for group of parts
        self.fc_prim = nn.ModuleList([nn.Linear(self.latent_size, self.prim_size) for i in range(self.n_parts)])
        self.nf = nn.ModuleList([RealNVP(self.prim_size, device=self.device) for i in range(self.n_parts)])
        
        ## decoders
        self.decoders = nn.ModuleList([nn.ModuleList([TransNN(1, self.prim_size),
                                                      RotaNN(1, self.prim_size),
                                                      SizeNN(1, self.prim_size),
                                                      ShapeNN(1, self.prim_size),
                                                      DeformNN(1, self.prim_size),
                                                      ProbNN(1, self.prim_size)]) for i in range(self.n_parts)])
        
    def forward(self, x):
        B = x.size(0)
        
        code = self.encoder(x.transpose(1,2))
        mu = self.fc_mu(code)
        logvar = self.fc_logvar(code)
        latent = reparameterize(mu, logvar)
        
        params = []
        for i in range(self.n_parts):
            prim_latent = self.fc_prim[i](latent)
            prim_latent, _ = self.nf[i].f(prim_latent)
            prim_latent = self.nf[i].g(prim_latent)
            p = PrimitiveParams()
            for j in range(len(self.decoders[i])):
                p[j] = self.decoders[i][j](prim_latent).view(B, -1)
            params.append(p)
            
        return params, mu, logvar
        
class KLoss(nn.Module):
    def __init__(self):
        super(KLoss, self).__init__()
        
    def forward(self, mu, logvar):
        return KL(mu, logvar)
        
        
        
    
class SymmSegNets(nn.Module):
    def __init__(self, latent_size=256, n_prims=4, n_trans=8):
        super(SymmSegNets, self).__init__()
        
        self.latent_size = latent_size
        self.n_prims = n_prims
        self.n_trans = n_trans
        
        self.encoder = nn.Sequential(
            PointNetfeat(size=self.latent_size, trans=False, layers=[3, 64, 128, 128, 256]),
            nn.Linear(self.latent_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU()
        )
        
        self.decoders = nn.ModuleList(
            [TransNN(n_trans, latent_size),
             RotaNN(n_trans, latent_size),
             SizeNN(n_prims, latent_size),
             ShapeNN(n_prims, latent_size),
             DeformNN(n_prims, latent_size),
             embeddingNN(latent_size, n_prims, n_trans)]
        )
        
    def forward(self, x):
        B = x.size(0)
        latent = self.encoder(x.transpose(1,2))
        
        params = PrimitiveParams()
        for i in range(len(self.decoders)-1):
            params[i] = self.decoders[i](latent)
        params[-1] = params[0].new_ones((B, self.n_trans))
        params.append(self.decoders[-1](latent))
            
#         for p in params:
#             print(p.size())
            
        return latent, params
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    