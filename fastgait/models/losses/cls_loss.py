import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.dist_utils import get_dist_info
from ..utils.loss_utils import GatherLayer, all_gather

class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
        Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
        Equation: y = (1 - epsilon) * y + epsilon / K.
        Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, factor=16, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.factor = factor
        self.epsilon = epsilon

        self.logsoftmax = nn.LogSoftmax(dim=-1)
        assert self.num_classes > 0

        self.rank, self.world_size, self.dist = get_dist_info()

    def forward(self, results, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, parts, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        emb = results["logits"]
        lab = targets["label"]
        if self.dist:
            all_emb = torch.cat(GatherLayer.apply(emb), dim=0)
            all_lab = torch.cat(all_gather(lab), dim=0)
        else:
            all_emb = emb
            all_lab = lab

        n, p, c = all_emb.size()
        
        # compute the logsoftmax value
        all_emb = all_emb[:,:,:self.num_classes].permute(1, 0, 2).contiguous() # [p, n, c]
        log_probs = self.logsoftmax(all_emb * self.factor) # [p, n, c]

        # compute the onehot label
        with torch.no_grad():
            label = torch.zeros(n, self.num_classes).to(all_lab.device)
            label = label.scatter(1, all_lab.unsqueeze(-1), 1) #[n, c]
            label = label.unsqueeze(0).repeat(p, 1, 1) # [p, n, c]
        # label = (1 - self.epsilon) * label + self.epsilon / self.num_classes

        loss = (-label * log_probs).sum(-1) # [p, n]
        loss = torch.mean(loss.mean(-1))

        # compute the CE accuracy
        pred = all_emb.argmax(dim=-1) # [p, n]
        accu = (pred == all_lab.unsqueeze(0)).float().mean()

        loss_info = {
            'ce_loss': loss.cpu().detach().numpy(),
            'ACC@R1': accu.cpu().detach().numpy(),
        }

        return loss * self.world_size, loss_info
