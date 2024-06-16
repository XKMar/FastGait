import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.map_utils import HorizontalMapping

class TFAHead(nn.Module):
    ''' Motion Pattern Aggregator neck for gait recognition

    Args:
        num_bin (List[int]): Number of horizontal split parts.
        in_planes (int): The dimension of the gait features.

    Example:
        >>> input = [torch.rand(128, 30, 256, 16, 11)] #[b, s, c, h, w]
        >>> output = MultiHeadAttention(input)
        >>> print(f'output.shape = {output.shape}')
        outputs.shape = torch.Size([128, 256, 16]) #[b, c, p] 
    '''
    def __init__(
            self,
            num_bins,
            num_parts,
            num_classes,
            in_channels,
            out_channels,
            squeeze=4
            ):
        super(TFAHead, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.num_bin = num_bins
        self.num_channels = in_channels

        # Horizontal mapping layer
        self.HMapping = HorizontalMapping(num_parts)

        # MTB1
        conv3x1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(hidden_dim, in_channels, kernel_size=1))
        self.conv1d3x1 = nn.ModuleList([copy.deepcopy(conv3x1) for _ in range(self.num_bin)])
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)

        # MTB2
        conv3x3 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(hidden_dim, in_channels, kernel_size=3, padding=1))
        self.conv1d3x3 = nn.ModuleList([copy.deepcopy(conv3x3) for _ in range(self.num_bin)])
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

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
        x = self.HMapping(x) # [B T C P]

        b, t, c, p = x.size()
        x = x.permute(3, 0, 2, 1).contiguous()  # [P B C T]
        feature = x.split(1, 0)  # [[B C T], ...]
        x = x.view(-1, c, t)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, b, c, t)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, b, c, t)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        feature = torch.max(feature3x1 + feature3x3, dim=-1)[0]  # [P B C]

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