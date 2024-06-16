import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.map_utils import GeMPartMapping

class GeMHead(nn.Module):
    def __init__(
        self,
        num_bins,
        num_parts,
        num_classes,
        in_channels,
        out_channels,
    ):
        super().__init__()
        
        self.num_bins = num_bins
        self.num_channels = out_channels

        # Horizontal mapping layer
        self.GMapping = GeMPartMapping(num_parts)

        # Feed forward layer
        self.FForward = nn.Parameter(nn.init.xavier_uniform_(
                    torch.zeros(num_bins, in_channels, out_channels)))
        
        # Separate BN layer
        self.bn1d = nn.ModuleList([copy.deepcopy(nn.BatchNorm1d(out_channels)) 
                                    for _ in range(num_bins)])

        # Separate Fc layer
        self.fc1d = nn.Parameter(nn.init.xavier_uniform_(
                    torch.zeros(num_bins, out_channels, num_classes)))

    def forward(self, x):
        r"""Forward function.
            Args:
                >>> Inputs: [B C T H W]
                >>> Output: [B P C]
        """
        # splitting feature map into strips
        x = self.GMapping(x) # [B T C P]

        # max and mean pooling
        feature = x.max(1)[0] + x.mean(1)                       # [B C P]
        feature = feature.permute(2, 0, 1).contiguous()         # [P B C]

        # feed forward for each strip
        outputs = feature.matmul(self.FForward)                 # [P B C]
        outputs = outputs.permute(1, 0, 2).contiguous()         # [B P C]

        # operate on each part
        bn_feat = torch.cat([bn(_.squeeze(1)).unsqueeze(1) 
                for _, bn in zip(outputs.split(1, 1), self.bn1d)], 1) # [B P C]
        bn_feat = F.normalize(bn_feat, dim=-1)

        bn_feat = bn_feat.permute(1, 0, 2).contiguous()         # [P B C]
        logits = bn_feat.matmul(F.normalize(self.fc1d, dim=1))  # [P B C]

        # permute the bn feature and logits
        # bn_feat = bn_feat.permute(1, 0, 2).contiguous()       # [B P C]
        logits = logits.permute(1, 0, 2).contiguous()           # [B P num_classes]

        return outputs, logits