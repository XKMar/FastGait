import torch
import torch.nn as nn

from ..backbones import build_backbone
from ..heads import build_head

class Basic(nn.Module):
    r""" 
        A basic gait recognition framework structure for some mainstream gait networks, 
        e.g., 'GaitSet', 'GaitPart', 'GaitGL', 'DANet', etc. which contains:
            + one backbone: 'shiftnet',
            + one neck: 'mpa',
            + one head: 'part'.

        Args:
            mum_parts    (list[int]): The number of horizontal split parts (default: 16).
            num_classes       (int) : The number of classes for cross-entropy loss.
            set_channels (list[int]): The dimensions of backbone.
            embd_feature      (int) : The dimensions of the output features.
            dropout         (float) : The dropout ratios.
            with_glob        (bool) : Whether or not it contains a global branch.
            pretrained        (str) : The path of source-domain pretrain model.

        outputs:
            dict[str, tensor]: A dictionary of gait feature components.

    """

    def __init__(
        self,
        num_parts,
        num_classes,
        set_channels,
        embd_feature,
        back_name,
        head_name,
        drop_out=0.0,
        with_glob=False,
        pretrained=None,
        ):
        super(Basic, self).__init__()

        # horizontal split parts
        self.num_bin = sum(num_parts)
        if with_glob:
            self.num_bin = sum(num_parts) * 2 # global branch

        # build backbone layer
        self.backbone = build_backbone(back_name, set_channels)

        # build head layer
        self.head = build_head(
                    head_name,
                    self.num_bin,
                    num_parts, 
                    num_classes,
                    set_channels[-1], 
                    embd_feature,)

        # initialization the parameters
        if pretrained is None:
            self.reset_params()

    def reset_params(self):
        # Random initialization parameters
        print("=== Random initialization the Basic framework parameters ===")
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        r"""
            Forward function.

            Args:
                >>> Inputs: x (Tensor), Input images. [B C T H W]
                (B: Batch, C: Channel, T: Temporal, H, W: Hight and Width)
            
            Returns:
                >>> Outputs: dict[str, tensor]: A dictionary of output features.
        """

        # networks architecture
        x = self.backbone(x.unsqueeze(1)) # [B C T H W]
        feature, logits = self.head(x)       # [B P num_classes]

        # return the results
        results = {}
        results["feature"] = feature
        results["logits"] = logits

        return results

 