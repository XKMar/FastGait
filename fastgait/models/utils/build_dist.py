import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "build_dist",
    "compute_euclidean_distance",
]


@torch.no_grad()
def build_dist(
    cfg, feat_1, feat_2=None, dist_m=None
):
    """Computes distance.

    Args:
        input1 (torch.Tensor): 3-D feature matrix.
        input2 (torch.Tensor): 3-D feature matrix. (optional)

    Returns:
        numpy.ndarray: distance matrix.
    """

    if dist_m is None:
        dist_m = cfg.dist_metric

    if dist_m == "euclidean":

        if feat_2 is not None:
            final_dist = compute_euclidean_distance(feat_1, feat_2)
        else:
            final_dist = compute_euclidean_distance(feat_1, feat_1)

        final_dist = final_dist.cpu().numpy()

        return final_dist

    elif dist_m == "cosine":

        if feat_2 is not None:
            final_dist =  compute_cosine_distance(feat_1, feat_2)
        else:
            final_dist =  compute_cosine_distance(feat_1, feat_1)

        final_dist = final_dist.cpu().numpy()

        return final_dist

    else:
        assert "Unknown distance metric: {}".format(dist_m)

@torch.no_grad()
def compute_euclidean_distance(
    x, y
):

    if x.dim() > 2:
        # Calculate the Euclidean distance for each part
        # TODO: rethinking the Integrated approach
        x = x.permute(1, 0, 2).contiguous()
        y = y.permute(1, 0, 2).contiguous()
        dist = torch.sum(x ** 2, 2).unsqueeze(2) + torch.sum(y ** 2, 2).unsqueeze(
            2).transpose(1, 2) - 2 * torch.matmul(x, y.transpose(1, 2))
        dist = torch.mean(torch.sqrt(F.relu(dist)), 0)
    else:
        dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
                1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
        dist = torch.sqrt(F.relu(dist))

    return dist


@torch.no_grad()
def compute_cosine_distance(
    x, y
):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    if x.dim() > 2:
        # Calculate the Cosine similarity for each part
        # TODO: rethinking the Integrated approach
        x = x.permute(1, 0, 2).contiguous()
        y = y.permute(1, 0, 2).contiguous()
        simi =  1 - torch.mean(torch.matmul(x, y.transpose(1, 2)), 0)
    else:
        simi =  1 - torch.matmul(x, y.transpose(0, 1))

    return simi