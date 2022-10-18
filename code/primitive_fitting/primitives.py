import torch
import torch.nn as nn

## quaternions: n_primitives * 4
## rotation_matrices: n_primitives * 3 * 3
def quaternions_to_rotation_matrices(quaternions):
    K = quaternions.shape[0]
    # Allocate memory for a Tensor of size Kx3x3 that will hold the rotation
    # matrix along the x-axis
    R = quaternions.new_zeros((K, 3, 3))

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[:, 1]**2
    yy = quaternions[:, 2]**2
    zz = quaternions[:, 3]**2
    ww = quaternions[:, 0]**2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = quaternions.new_zeros((K, 1))
    s[n != 0] = 2 / n[n != 0]

    xy = s[:, 0] * quaternions[:, 1] * quaternions[:, 2]
    xz = s[:, 0] * quaternions[:, 1] * quaternions[:, 3]
    yz = s[:, 0] * quaternions[:, 2] * quaternions[:, 3]
    xw = s[:, 0] * quaternions[:, 1] * quaternions[:, 0]
    yw = s[:, 0] * quaternions[:, 2] * quaternions[:, 0]
    zw = s[:, 0] * quaternions[:, 3] * quaternions[:, 0]

    xx = s[:, 0] * xx
    yy = s[:, 0] * yy
    zz = s[:, 0] * zz

    idxs = torch.arange(K).to(quaternions.device)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw

    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw

    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R

## transform point clouds into centric coordinate system
## pcl: point cloud with batch_size * n_points * 3
## translations: batch_size * n_primitives * 3
## rotations: batch_size * n_primitives * 4
## return transformed point cloud: batch_size * n_points * n_primitives * 3
def transform_to_primitives_centric_system(pcl, translations, rotations):
    # Make sure that all tensors have the right shape
    assert pcl.size(0) == translations.size(0)
    assert translations.size(0) == rotations.size(0) ## batch size
    assert translations.size(1) == rotations.size(1) ## primitive number
    assert pcl.size(2) == 3 ## point cloud coordinates
    assert translations.size(2) == 3
    assert rotations.size(2) == 4
    
    B = pcl.size(0)
    N = pcl.size(1)
    M = translations.size(1)

    pcl_transformed = pcl.unsqueeze(2) - translations.unsqueeze(1) ## batch_size * n_points * n_primitives * 3
    
    R = quaternions_to_rotation_matrices(rotations.view(-1, 4)).view(B, M, 3, 3)

    # Let as denote a point x_p in the primitive-centric coordinate system and
    # its corresponding point in the world coordinate system x_w. We denote the
    # transformation from the point in the world coordinate system to a point
    # in the primitive-centric coordinate system as x_p = R(x_w - t)
    pcl_transformed = R.unsqueeze(1).matmul(pcl_transformed.unsqueeze(-1)) ## (B*1*M*3*3) and (B*N*M*3*1)
    
    # if points' coordinate less than 1e-5, than get the value as 1e-5
    pcl_signs = (pcl_transformed > 0).float() * 2 - 1
    pcl_abs = pcl_transformed.abs()
    pcl_transformed = pcl_signs * torch.max(pcl_abs, pcl_abs.new_tensor(1e-5))

    return pcl_transformed.squeeze(-1)


## p_pcl: point cloud for primitive surface, B*M*S*3
## size: size parameter B*M*3
## deformations: deformation parameter B*M*2
def deform(p_pcl, size, deformations, bending=None):
    B, M, S, _ = p_pcl.size()
    
    assert size.size(0) == B ## batch size
    assert deformations.size(0) == B
    assert size.size(2) == 3 ## dim of paramters
    assert deformations.size(2) == 2
    assert size.size(1) == M ## number of primitives
    assert deformations.size(1) == M
    
    ## compute the two linear tapering function
    K = deformations/size[:,:,-1].unsqueeze(-1)
    
    f = K.unsqueeze(2) * p_pcl[:,:,:,-1].unsqueeze(-1) + 1.0
    assert f.size() == (B, M, S, 2)
    f = torch.cat([f, f.new_ones(B,M,S,1)], -1)
    
    p_pcl_d = p_pcl * f
    
    return p_pcl_d
    
    
    
def inside_outside_function(pcl_transformed, size, shape):
    """
    Arguments:
    ----------
        pcl_transformed: Tensor with size BxNxMx3, containing the 3D points, where B is the
           batch size and N is the number of points
        size: Tensor with size BxMx3, containing the shape along each
                      axis for the M primitives
        shape: Tensor with size BxMx2, containing the shape along the
                  longitude and the latitude for the M primitives
    Returns:
    ---------
        F: Tensor with size BxNxM, containing the values of the
           inside-outside function
    """
    B, N, M, _ = pcl_transformed.size()
    
    # Make sure that both tensors have the right shape
    assert size.size(0) == B  # batch size
    assert shape.size(0) == B  # batch size
    assert size.size(1) == M  # number of primitives
    assert size.size(1) == shape.size(1)
    assert size.size(-1) == 3  # number of size parameters
    assert shape.size(-1) == 2  # number of shape parameters
    assert pcl_transformed.size(-1) == 3  # 3D points

    # Declare some variables
    a1 = size[:, :, 0].unsqueeze(1)  # size Bx1xM
    a2 = size[:, :, 1].unsqueeze(1)  # size Bx1xM
    a3 = size[:, :, 2].unsqueeze(1)  # size Bx1xM
    e1 = shape[:, :, 0].unsqueeze(1)  # size Bx1xM
    e2 = shape[:, :, 1].unsqueeze(1)  # size Bx1xM

    # Add a small constant to points that are completely dead center to avoid
    # numerical issues in computing the gradient
    pcl_transformed = ((pcl_transformed > 0).float() * 2 - 1) * torch.max(torch.abs(pcl_transformed), pcl_transformed.new_tensor(1e-6))

    F = (((pcl_transformed[:, :, :, 0] / a1)**2)**(1./e2) + \
         ((pcl_transformed[:, :, :, 1] / a2)**2)**(1./e2) )**(e2 / e1) + \
         ((pcl_transformed[:, :, :, 2] / a3)**2)**(1./e1)
    # Sanity check to make sure that we have the expected size
    assert F.shape == (B, N, M)
    return F**e1
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    