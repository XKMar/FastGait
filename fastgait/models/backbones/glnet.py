import torch
import torch.nn as nn

from ..utils.net_utils import BasicConv3d, GLConv3d

class GLNet(nn.Module):
    def __init__(
        self,
        set_channels
    ):
        super(GLNet, self).__init__()
        _set_in_channels = 1
        _set_channels = set_channels

        # base conv
        self.set_layer1 = BasicConv3d(_set_in_channels, _set_channels[0], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.set_layer2 = BasicConv3d(_set_channels[0], _set_channels[0], kernel_size=(3,1,1), stride=(3,1,1), padding=(0,0,0))
        # focal conv
        self.set_layer3 = GLConv3d(_set_channels[0], _set_channels[1], kernel_size=(3,3,3), halving=3, fm_sign=False, stride=(1,1,1), padding=(1,1,1))
        self.set_layer4 = GLConv3d(_set_channels[1], _set_channels[1], kernel_size=(3,3,3), halving=3, fm_sign=False, stride=(1,1,1), padding=(1,1,1))
        self.set_layer5 = GLConv3d(_set_channels[1], _set_channels[2], kernel_size=(3,3,3), halving=3, fm_sign=True,  stride=(1,1,1), padding=(1,1,1))

        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x):
        r"""Forward function.
            Args:
                >>> Inputs: [B C T H W]
                >>> Output: [B C 1 H W]
        """

        # base stage
        out = self.set_layer1(x)
        out = self.set_layer2(out) # Local Temporal Aggregation

        # focal stage
        out = self.set_layer3(out) # GLConvA
        out = self.pool1(out)

        out = self.set_layer4(out) # GLConvA
        out = self.set_layer5(out) # GLConvB

        # set pooling 
        out = torch.max(out, dim=2)[0]  # [B C H W]
        out = out.unsqueeze(2)          # [B C 1 H W]
        
        return out