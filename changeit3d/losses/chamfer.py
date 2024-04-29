import torch
import warnings

from changeit3d.losses.nn_distance import chamfer_raw


def chamfer_loss(pc_a, pc_b, swap_axes=False, reduction='mean'):
    """Compute the chamfer loss for batched pointclouds.
        :param pc_a: torch.Tensor B x Na-points per point-cloud x 3
        :param pc_b: torch.Tensor B x Nb-points per point-cloud x 3
        :return: B floats, indicating the chamfer distances when reduction is mean, else un-reduced distances
    """

    n_points_a = pc_a.shape[1]
    n_points_b = pc_b.shape[1]

    if swap_axes:
        pc_a = pc_a.transpose(-1, -2).contiguous()
        pc_b = pc_b.transpose(-1, -2).contiguous()

    
    dist_a, dist_b = chamfer_raw(pc_a, pc_b)

    if reduction == 'mean':
        # Reduce separately, sizes of points can be different

        chamfer_a = (dist_a.mean(dim=0) * n_points_a).sum(dim=0)
        chamfer_b = (dist_b.mean(dim=1) * n_points_b).sum(dim=0)
        chamfer_distance = (chamfer_a + chamfer_b) / (n_points_a + n_points_b)
    
    elif reduction is None:
        return dist_a, dist_b
    else:
        raise ValueError('Unknown reduction rule.')

    return chamfer_distance


if __name__ == '__main__':
    pca = torch.rand(10, 2048, 3).cuda()
    pcb = torch.rand(10, 4096, 3).cuda()
    a, b = chamfer_loss(pca, pcb, reduction=None)
    print(a.shape, b.shape)
    l = chamfer_loss(pca, pcb)
    print(l.shape)