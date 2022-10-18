import torch

def fexp(x, p):
    return torch.sign(x)*(torch.abs(x)**p)

def param2surface(a1, a2, a3, e1, e2, eta, omega):
    x = a1 * fexp(torch.cos(eta), e1) * fexp(torch.cos(omega), e2)
    y = a2 * fexp(torch.cos(eta), e1) * fexp(torch.sin(omega), e2)
    z = a3 * fexp(torch.sin(eta), e1)
    
    return x, y, z

## size of the superquadric, control the rate of x,y,z, B*M*3
## shape of the superquadric, epsilon, B*M*2
def sampling_from_superquadric(size, shape, sampler):
    B = size.size(0)
    M = size.size(1)
    S = sampler.n_samples
    
    etas, omegas = sampler.sample_on_batch(size.detach().cpu().numpy(), shape.detach().cpu().numpy())
    etas[etas==0] += 1e-6
    omegas[omegas==0] += 1e-6
    
    etas = size.new_tensor(etas)
    omegas = size.new_tensor(omegas)
#     print(etas.size())
#     print(omegas.size())
    
    a1 = size[:,:,0].unsqueeze(-1)
    a2 = size[:,:,1].unsqueeze(-1)
    a3 = size[:,:,2].unsqueeze(-1)
    e1 = shape[:,:,0].unsqueeze(-1)
    e2 = shape[:,:,1].unsqueeze(-1)
    
    x, y, z = param2surface(a1, a2, a3, e1, e2, etas, omegas)
    
    ## make sure no close to 0
    x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
    y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
    z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))
    
    ## the normals
    nx = (torch.cos(etas)**2) * (torch.cos(omegas)**2)/x
    ny = (torch.cos(etas)**2) * (torch.sin(omegas)**2)/y
    nz = (torch.sin(etas)**2)
    
    return torch.stack([x, y, z], -1), torch.stack([nx, ny, nz], -1)


class PrimitiveParams(object):
    def __init__(self, trans=None, rota=None, size=None, shape=None, deform=None, prob=None):
        self.trans = trans
        self.rota = rota
        self.size = size
        self.shape = shape
        self.deform = deform
        self.prob = prob
        
        self.members = [self.trans, self.rota, self.size, self.shape, self.deform, self.prob]
    
    def __len__(self):
        return len(self.members)
    
    def __getitem__(self, i):
        return self.members[i]
    
    def __setitem__(self, i, ipt):
        self.members[i] = ipt
        
    def append(self, it):
        self.members.append(it)

    
class ReducedPrimitiveParams(object):
    def __init__(self, trans=None, rota=None, size=None, shape=None, deform=None, prob=None):
        self.trans = trans
        self.rota = rota
        self.size = size
        self.shape = shape
        self.deform = deform
        
        self.members = [self.trans, self.rota, self.size, self.shape, self.deform]
    
    def __len__(self):
        return len(self.members)
    
    def __getitem__(self, i):
        return self.members[i]
    
    def __setitem__(self, i, ipt):
        self.members[i] = ipt
        
    def append(self, it):
        self.members.append(it)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    