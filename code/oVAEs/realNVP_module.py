import torch
from torch import nn
import numpy as np
from torch import distributions


class RealNVP(nn.Module):
    def __init__(self, dim=32, device='cpu'):
        super(RealNVP, self).__init__()
        self.device = device
        mask = nn.Parameter(torch.from_numpy(np.array([np.repeat([0, 1], dim/2, axis=0),
                                                       np.repeat([1, 0], dim/2, axis=0)] * 3).astype(np.float32)),
                            requires_grad=False).to(device)
        nets = lambda: nn.Sequential(nn.Linear(dim, 256), nn.LeakyReLU(), 
                                     nn.Linear(256, 256), nn.LeakyReLU(), 
                                     nn.Linear(256, dim), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(dim, 256), nn.LeakyReLU(), 
                                     nn.Linear(256, 256), nn.LeakyReLU(), 
                                     nn.Linear(256, dim))
        
        # self.prior = distributions.MultivariateNormal(torch.zeros(dim).to(device),
        #                                               torch.eye(dim).to(device))
        self.prior = distributions.MultivariateNormal(torch.zeros(dim),
                                                      torch.eye(dim))

        self.mask = nn.Parameter(mask, requires_grad=False).to(device)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))]).to(device)
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))]).to(device)
        
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z).to(self.device) + logp
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1)).to(self.device)
        logp = self.prior.log_prob(z).to(self.device)
        x = self.g(z)
        return x