import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.map_utils import HorizontalMapping

class GMPAHead(nn.Module):
    r''' Motion Pattern Aggregator head for gait recognition

        Args:
            num_bin        (int) : The total number of horizontal split parts (include global).
            num_parts (List[int]): The number of horizontal split parts.
            in_channels    (int) : The input channels.
            out_channels   (int) : The output channels.

        Returns:
            >>> Outputs: Tensor: the gait features.
    '''
    def __init__(
        self,
        num_bins,
        num_parts,
        num_classes,
        in_channels,
        out_channels,
        reduction=16
    ):
        super(GMPAHead, self).__init__()
        
        self.num_bins = num_bins
        self.num_channels = out_channels

        # Horizontal mapping layer
        self.HMapping = HorizontalMapping(num_parts)

        # Motion Aggregation layer
        self.MPABlock = nn.ModuleList([GMPABlock(in_channels, reduction)
                        for i in range(num_bins)])

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
        r"""
            Forward function.

            Args:
                >>> Inputs: x (Tensor), Input images. [B C T H W]
                (B: Batch, C: Channel, T: Temporal, H, W: Hight and Width)
            
            Returns:
                >>> Outputs: Tensor: the gait features.
        """
        # Split the feature map into strips
        x = self.HMapping(x) # [B T C P]

        # motion aggregate for each strip
        feature = list()
        for i in range(self.num_bins):
            align_output = self.MPABlock[i](x[:,:,:,i])         # [B C]
            feature.append(align_output.unsqueeze(0))           # [P B C]

        feature = torch.cat(feature, dim=0).contiguous()        # [P B C]

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
    
class GMPABlock(nn.Module):
    r''' 
        Motion Pattern Aggregator block module 

        Args:
            d_model   (int): The dimension of input feature.
            reduction (int): Dimensionality reduction hyperparameters.

        Example:
            >>> input = [torch.rand(128, 30, 256)] # [B T C]
            >>> output = MPABlock(input)
            >>> print(f'output.shape = {output.shape}')
            outputs.shape = torch.Size([128, 30, 256]) # [B T C]
    '''

    def __init__(self, d_model, reduction=16):
        super().__init__()

        # context Modeling
        self.fc_mask = nn.Linear(d_model, reduction, bias=False)
        self.softmax = nn.Softmax(dim=1)
        # remapping
        self.to_out = nn.Sequential(
                    nn.Linear(d_model * reduction, d_model, bias=False),
                    nn.LeakyReLU(inplace=True))

    def forward(self, x):
        r"""Forward function.
            Args:
                >>> Inputs: [B T C]
                >>> Output: [B T C]
        """
        b, t, c = x.size()
        res = x

        # pass through the mask projection: [b, t, n_head]
        # seprate different heads: [b, t, n_head]
        context_mask = self.fc_mask(x)
        context_mask = self.softmax(context_mask)

        # scale dot-product attention: [b, c * n_head]
        # build the global features: [b, 1, c]
        out = torch.matmul(x.transpose(1, 2), context_mask).view(b, -1)
        out = self.to_out(out).view(b, 1, c)

        # add the global features to the residual features
        output = res + out

        # obtain global features by taking max for the temporal dimension: b x c
        output = torch.max(output, dim=1)[0]

        return output