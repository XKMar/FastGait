import copy
import warnings

import torch
import torch.nn as nn
from torch.nn import init

from fastgait.utils.dist_utils import get_dist_info
from fastgait.utils.torch_utils import copy_state_dict, load_checkpoint
from fastgait.models.meta_arch import build_arch

__all__ = ["build_model",]

def build_model(
    cfg,
    init=None,
):
    """
    Build the gait model
    with domain-specfic BNs (optional)
    """

    rank, world_size, dist = get_dist_info()

    # build the gait model
    model = build_arch(
                    cfg.MODEL.meta_arch,
                    cfg.MODEL.num_parts,
                    cfg.MODEL.num_classes,
                    cfg.MODEL.set_channel,
                    cfg.MODEL.embd_feature,
                    back_name = cfg.MODEL.back_bone,
                    head_name = cfg.MODEL.head_name,
                    drop_out = cfg.MODEL.dropout,
                    with_glob = cfg.MODEL.with_glob,
                    pretrained = init,
                )

    # load source-domain pretrain (optional)
    if init is not None:
        state_dict = load_checkpoint(init)
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        copy_state_dict(state_dict, model, strip="module.")

    # convert to sync bn (optional)
    if cfg.MODEL.sync_bn and dist:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        if cfg.MODEL.sync_bn and not dist:
            warnings.warn(
                "Sync BN is switched off, since the program is running without DDP"
            )
        cfg.MODEL.sync_bn = False

    model.cuda()

    return model