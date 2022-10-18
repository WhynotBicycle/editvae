from functools import partial

import torch

def bernoulli(parameters, minimum_number_of_primitives,
              maximum_number_of_primitives, w1, w2):
    """Ensure that we have at least that many primitives in expectation"""
    expected_primitives = parameters[4].sum(-1)

    lower_bound = minimum_number_of_primitives - expected_primitives
    upper_bound = expected_primitives - maximum_number_of_primitives
    zero = expected_primitives.new_tensor(0)

    t1 = torch.max(lower_bound, zero)
    t2 = torch.max(upper_bound, zero)
    
    return (w1*t1 + w2*t2).mean()


def sparsity(parameters, minimum_number_of_primitives,
             maximum_number_of_primitives, w1, w2):
    expected_primitives = parameters[4].sum(-1)
    lower_bound = minimum_number_of_primitives - expected_primitives
    upper_bound = expected_primitives - maximum_number_of_primitives
    zero = expected_primitives.new_tensor(0)

    t1 = torch.max(lower_bound, zero) * lower_bound**4
    t2 = torch.max(upper_bound, zero) * upper_bound**2

    return (w1*t1 + w2*t2).mean()


def parsimony(parameters):
    """Penalize the use of more primitives"""
    expected_primitives = parameters[4].sum(-1)

    return expected_primitives.mean()


def entropy_bernoulli(parameters):
    """Minimize the entropy of each bernoulli variable pushing them to either 1
    or 0"""
    probs = parameters[4]
    sm = probs.new_tensor(1e-3)

    t1 = torch.log(torch.max(probs, sm))
    t2 = torch.log(torch.max(1 - probs, sm))

    return torch.mean((-probs * t1 - (1-probs) * t2).sum(-1))


def overlapping(F):
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
        loss += torch.max(f.new_tensor(1)-f, f.new_tensor(0)).mean()

    return loss


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
        "overlapping_regularizer": partial(overlapping, F),
        "sparsity_regularizer": partial(
            sparsity,
            parameters,
            arguments.get("minimum_number_of_primitives", None),
            arguments.get("maximum_number_of_primitives", None),
            arguments.get("w1", None),
            arguments.get("w2", None)
        ),
    }

    # Call the regularizer or return 0
    return regs.get(regularizer, lambda: parameters[0].new_tensor(0))()