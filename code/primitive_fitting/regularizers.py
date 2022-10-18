from functools import partial

import torch

# N*N_p*N_t
def emb_prim_reg(embeddings):
    lower_bound = 1
    prim_sum = embeddings.sum(2)
    zero = prim_sum.new_tensor(0)

    l = torch.max(lower_bound - prim_sum, zero).mean()
    
    return l

# N*N_p*N_t
def emb_tran_reg(embeddings):
    lower_bound = 1
    prim_sum = embeddings.sum(1)
    zero = prim_sum.new_tensor(0)
    
    l = torch.max(lower_bound - prim_sum, zero).mean() + torch.max(prim_sum - lower_bound, zero).mean()
    
    return l

def emb_discrete(embeddings):
    """Minimize the entropy of each bernoulli variable pushing them to either 1
    or 0"""
    sm = embeddings.new_tensor(1e-3)

    t1 = torch.log(torch.max(embeddings, sm))
    t2 = torch.log(torch.max(1 - embeddings, sm))

    return torch.mean((-embeddings * t1 - (1-embeddings) * t2).sum(-1))
    

def bernoulli(parameters, minimum_number_of_primitives,
              maximum_number_of_primitives, w1, w2):
    """Ensure that we have at least that many primitives in expectation"""
    expected_primitives = parameters[5].sum(-1)

    lower_bound = minimum_number_of_primitives - expected_primitives
    upper_bound = expected_primitives - maximum_number_of_primitives
    zero = expected_primitives.new_tensor(0)

    t1 = torch.max(lower_bound, zero)
    t2 = torch.max(upper_bound, zero)
    
    return (w1*t1 + w2*t2).mean()


def sparsity(parameters, minimum_number_of_primitives,
             maximum_number_of_primitives, w1, w2):
    expected_primitives = parameters[5].sum(-1)
    lower_bound = minimum_number_of_primitives - expected_primitives
    upper_bound = expected_primitives - maximum_number_of_primitives
    zero = expected_primitives.new_tensor(0)

    t1 = torch.max(lower_bound, zero) * lower_bound**4
    t2 = torch.max(upper_bound, zero) * upper_bound**2

    return (w1*t1 + w2*t2).mean()


def parsimony(parameters):
    """Penalize the use of more primitives"""
    expected_primitives = parameters[5].sum(-1)

    return expected_primitives.mean()


def entropy_bernoulli(parameters):
    """Minimize the entropy of each bernoulli variable pushing them to either 1
    or 0"""
    probs = parameters[5]
    sm = probs.new_tensor(1e-3)

    t1 = torch.log(torch.max(probs, sm))
    t2 = torch.log(torch.max(1 - probs, sm))

    return torch.mean((-probs * t1 - (1-probs) * t2).sum(-1))


def overlapping(F, threshold=1):
    """Penalize primitives that are inside other primitives
        Arguments:
        -----------
        F: Tensor of shape BxNxM for the X points
    """
    assert len(F.shape) == 3
    B, N, M = F.shape

#     loss = F.new_tensor(0)
#     for j in range(M):
#         f = F[F[:, :, j] > 0.9]
#         if len(f) > 0:
#             f[:, j] = f.new_tensor(0)
#             loss += torch.max(f - f.new_tensor(0.9), f.new_tensor(0)).mean()

    loss = F.new_tensor(0)
    for j in range(M):
        f = F[F[:, :, j] > 0]
#         f = F.reshape(-1, M)
        f[:, j] = f.new_tensor(0)
        loss += torch.max(f.new_tensor(threshold)-f, f.new_tensor(0)).mean()

    return loss

def diversity(params):
    ## range [0, 0.5]
    size = params[2]
    M = size.size(1)
    size1 = size.unsqueeze(2).repeat(1, 1, M, 1)
    size2 = size.unsqueeze(1).repeat(1, M, 1, 1)
    dist_s = size1 - size2
    dist_s = torch.sum(dist_s**2, -1)
    dist_s = torch.tanh(-dist_s.mean()*4)+1
    
#     ## range [-0.5, 0.5]
#     trans = params[0] #B*M*3
#     trans1 = trans.unsqueeze(2).repeat(1, 1, M, 1)
#     trans2 = trans.unsqueeze(1).repeat(1, M, 1, 1)
#     dist_t = trans1 - trans2
#     dist_t = torch.sum(dist_t**2, -1)
#     dist_t = torch.tanh(-dist_t.mean())+1
    
    return dist_s

def get(regularizer, parameters, F, arguments):
    n_primitives = parameters[1].shape[1]
    regs = {
        "bernoulli_regularizer": partial(
            bernoulli,
            parameters,
            arguments.get("minimum_number_of_primitives", None),
            arguments.get("maximum_number_of_primitives", None),
            arguments.get("w1", None),
            arguments.get("w2", None)
        ),
        "parsimony_regularizer": partial(parsimony, parameters),
        "entropy_bernoulli_regularizer": partial(
            entropy_bernoulli,
            parameters
        ),
        "overlapping_regularizer": partial(
            overlapping, 
            F,
            arguments.get("overlapping_threshold", 1)
        ),
        "sparsity_regularizer": partial(
            sparsity,
            parameters,
            arguments.get("minimum_number_of_primitives", None),
            arguments.get("maximum_number_of_primitives", None),
            arguments.get("w1", None),
            arguments.get("w2", None)
        ),
        "diversity_regularizer": partial(diversity, parameters),
    }

    # Call the regularizer or return 0
    return regs.get(regularizer, lambda: parameters[0].new_tensor(0))()