import torch
import torch.nn as nn
from einops import rearrange

from ..utils.net_utils import BasicConv2d, FocalConv2d

class PartNet(nn.Module):
    def __init__(
        self,
        set_channels
    ):
        super().__init__()
        _set_in_channels = 1
        _set_channels = set_channels
        self.num_channels = set_channels[-1]

        # base conv
        self.set_layer1 = BasicConv2d(_set_in_channels, _set_channels[0], kernel_size=5, stride=1, padding=2)
        self.set_layer2 = BasicConv2d(_set_channels[0], _set_channels[0], kernel_size=3, stride=1, padding=1)
        # focal conv
        self.set_layer3 = FocalConv2d(_set_channels[0], _set_channels[1], kernel_size=3, halving=2, stride=1, padding=1)
        self.set_layer4 = FocalConv2d(_set_channels[1], _set_channels[1], kernel_size=3, halving=2, stride=1, padding=1)
        self.set_layer5 = FocalConv2d(_set_channels[1], _set_channels[2], kernel_size=3, halving=3, stride=1, padding=1)
        self.set_layer6 = FocalConv2d(_set_channels[2], _set_channels[2], kernel_size=3, halving=3, stride=1, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        r"""Forward function.
            Args:
                >>> Inputs: [B C T H W]
                >>> Output: [B C T H W]
        """

        # reshape the input size
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        # base stage [B C T H W]
        out = self.set_layer1(x)
        out = self.set_layer2(out)
        out = self.pool1(out)

        out = self.set_layer3(out)
        out = self.set_layer4(out)
        out = self.pool2(out)

        out = self.set_layer5(out)
        out = self.set_layer6(out)

        # reshape the output size
        out = rearrange(out, '(b t) c h w -> b c t h w', b=b, t=t)

        return out