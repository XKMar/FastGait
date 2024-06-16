import torch
from torch import nn
import torch.nn.functional as F

from ...utils.dist_utils import get_dist_info
from ..utils.loss_utils import GatherLayer, all_gather


def euclidean_dist(x):
    """ Computate the euclidean distance """
    x2 = torch.sum(x ** 2, 2)
    dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

class TripletLoss(nn.Module):   
    r"""Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
        Loss for Person Re-Identification'."""

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        self.rank, self.world_size, self.dist = get_dist_info()

    def forward(self, results, targets):
        # feature: [P, B, C], label: [P, B]
        emb = results["feature"]
        lab = targets["label"]
        
        if self.dist:
            all_emb = torch.cat(GatherLayer.apply(emb), dim=0)
            all_lab = torch.cat(all_gather(lab), dim=0)
        else:
            all_emb = emb
            all_lab = lab

        feature = all_emb.permute(1, 0, 2).contiguous()
        label = all_lab.unsqueeze(0).repeat(feature.size(0), 1)

        P, B, C = feature.size() # [P B C]
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)

        mat_dist = euclidean_dist(feature)
        mat_dist = mat_dist.view(-1)
        mean_dist = torch.mean(mat_dist)

        # # hard
        # hard_hp_dist = torch.max(torch.masked_select(mat_dist, hp_mask).view(P, B, -1), 2)[0]
        # hard_hn_dist = torch.min(torch.masked_select(mat_dist, hn_mask).view(P, B, -1), 2)[0]
        # hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(P, -1)

        # full
        full_hp_dist = torch.masked_select(mat_dist, hp_mask).view(P, B, -1, 1)
        full_hn_dist = torch.masked_select(mat_dist, hn_mask).view(P, B, 1, -1)

        if self.margin > 0:
            full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(P, -1)
        else:
            full_loss_metric = F.softplus(full_hp_dist - full_hn_dist).view(P, -1)  

        loss_num = (full_loss_metric != 0).sum(1).float()

        loss_avg = full_loss_metric.sum(1) / (loss_num + 1e-6)
        loss_avg[loss_num == 0] = 0
        
        loss_info = {
            'tr_loss': loss_avg.mean().cpu().detach().numpy(),
            'loss_num': loss_num.mean().cpu().detach().numpy(),
            'mean_dist': mean_dist.cpu().detach().numpy()
        }

        return loss_avg.mean() * self.world_size, loss_info