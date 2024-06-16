import torch.nn as nn
import torch.distributed as dist

from .cls_loss import CrossEntropyLoss
from .triplet import TripletLoss


def build_loss(
    cfg,
    num_classes=None,
    margin=None,
    cuda=False,
):

    criterions = {}
    for loss_name in cfg.losses.keys():

        if loss_name == "cross_entropy":
            assert num_classes is not None
            criterion = CrossEntropyLoss(num_classes)

        elif loss_name == "triplet":
            if margin is None:
                margin = 0.2
            criterion = TripletLoss(margin)

        else:
            raise KeyError("Unknown loss:", loss_name)

        criterions[loss_name] = criterion

    if cuda:
        for key in criterions.keys():
            criterions[key].cuda()

    return criterions