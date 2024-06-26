import torch
import numpy as np

# def huber_loss(error, delta=1.0):
#     """
#     Args:
#         error: Torch tensor (d1,d2,...,dk)
#     Returns:
#         loss: Torch tensor (d1,d2,...,dk)
#     x = error = pred - gt or dist(pred,gt)
#     0.5 * |x|^2                 if |x|<=d
#     0.5 * d^2 + d * (|x|-d)     if |x|>d
#     Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
#     """
#     abs_error = torch.abs(error)
#     quadratic = torch.clamp(abs_error, max=delta)
#     linear = (abs_error - quadratic)
#     loss = 0.5 * quadratic**2 + delta * linear
#     return loss


# def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
#     """
#     Input:
#         pc1: (B,N,C) torch tensor
#         pc2: (B,M,C) torch tensor
#         l1smooth: bool, whether to use l1smooth loss
#         delta: scalar, the delta used in l1smooth loss
#     Output:
#         dist1: (B,N) torch float32 tensor
#         idx1: (B,N) torch int64 tensor
#         dist2: (B,M) torch float32 tensor
#         idx2: (B,M) torch int64 tensor
#     """
#     N = pc1.shape[1]
#     M = pc2.shape[1]
#     pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
#     pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
#     pc_diff = pc1_expand_tile - pc2_expand_tile

#     if l1smooth:
#         pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)  # (B,N,M)
#     elif l1:
#         pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # (B,N,M)
#     else:
#         pc_dist = torch.sum(pc_diff**2, dim=-1)  # (B,N,M)


#     dist1, idx1 = torch.min(pc_dist, dim=2)  # (B,N)
#     dist2, idx2 = torch.min(pc_dist, dim=1)  # (B,M)
#     return dist1, idx1, dist2, idx2

from scipy.spatial import cKDTree

def closest_point_distance(points, other_points):
    tree = cKDTree(other_points)
    dists, _ = tree.query(points, k=1)
    return torch.from_numpy(dists)  # Convert back to Torch tensor for further computations

def nn_distance(pc1, pc2):
    """
    Input:
        pc1, pc2: (B,N,C) and (B,M,C) torch tensors
    Output:
        dist1, idx1, dist2, idx2
    """
    batch_size = pc1.shape[0]
    dist1 = torch.empty(pc1.shape[:2])
    dist2 = torch.empty(pc2.shape[:2])

    # Compute for each batch
    for i in range(batch_size):
        pc1_np = pc1[i].cpu().numpy()  # Convert to NumPy for cKDTree
        pc2_np = pc2[i].cpu().numpy()

        dist1[i] = closest_point_distance(pc1_np, pc2_np)
        dist2[i] = closest_point_distance(pc2_np, pc1_np)

    # Dummy indices (not used but maintained for compatibility)
    idx1 = torch.zeros_like(dist1, dtype=torch.int64)
    idx2 = torch.zeros_like(dist2, dtype=torch.int64)

    return dist1, idx1, dist2, idx2

def chamfer_raw(pc_a, pc_b, swap_axes=False):
    """ Compute the chamfer loss for batched pointclouds.
    :param pc_a: torch.Tensor B x Na-points per point-cloud x 3
    :param pc_b: torch.Tensor B x Nb-points per point-cloud x 3
    :return: dist_a: torch.Tensor, dist_b: torch.Tensor
    # Note: this is 10x slower than the chamfer_loss in losses/chamfer.py BUT this plays also in CPU (the
    other does not).
    """
    if swap_axes:
        pc_a = pc_a.transpose(-1, -2).contiguous()
        pc_b = pc_b.transpose(-1, -2).contiguous()
    dist_a, _, dist_b, _ = nn_distance(pc_a, pc_b)

    return dist_a, dist_b