import numpy as np
from scipy.stats import entropy
from general_utils import unit_cube_grid_point_cloud
from sklearn.neighbors import NearestNeighbors
import warnings

def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    '''Computes the JSD between two sets of point-clouds, as introduced in the paper ```Learning Representations And Generative Models For 3D Point Clouds```.    
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    '''   
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)
    
def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False):
    '''Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    '''
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        p = 0.0
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)      # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    '''another way of computing JSD'''
    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


def test():
    p1 = np.random.random_sample((1, 2, 3))/3.5
    p2 = np.random.random_sample((6, 30, 3))/3.5
#     print(p1.max())
#     print(p2.max())
    
    return jsd_between_point_cloud_sets(p1, p2)
    
# print(test())
    