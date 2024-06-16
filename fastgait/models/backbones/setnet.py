import torch
import torch.nn as nn
from einops import rearrange

from ..utils.net_utils import BasicConv2d, SetBlock

class SetNet(nn.Module):
    r"""Create a SetNet instance from config.
        Args:
            set_channels (List[int]): the backbone model feature dimension;
            bin_num (List[int]): Number of horizontal split parts;
            pretrained (str): the pretrained model path;
        Returns:
            SetNet (class): SetNet instance.
    """
    def __init__(
        self,
        set_channels
    ):
        super(SetNet, self).__init__()
        _set_in_channels = 1
        _set_channels = set_channels
        self.num_channels = set_channels[-1]

        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], kernel_size=5, stride=1, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], kernel_size=3, stride=1, padding=1), pooling=True)
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], kernel_size=3, stride=1, padding=1))
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], kernel_size=3, stride=1, padding=1), pooling=True)
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], kernel_size=3, stride=1, padding=1))
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], kernel_size=3, stride=1, padding=1))

        self.gl_layer1 = BasicConv2d(_set_channels[0], _set_channels[1], kernel_size=3, stride=1, padding=1)
        self.gl_layer2 = BasicConv2d(_set_channels[1], _set_channels[1], kernel_size=3, stride=1, padding=1)
        self.gl_layer3 = BasicConv2d(_set_channels[1], _set_channels[2], kernel_size=3, stride=1, padding=1)
        self.gl_layer4 = BasicConv2d(_set_channels[2], _set_channels[2], kernel_size=3, stride=1, padding=1)
        self.gl_pooling = nn.MaxPool2d(2)

    def forward(self, x):
        r"""Forward function.
            Args:
                >>> Inputs: [B C T H W]
                >>> Output: [B C 1 H W]
        """
        x = rearrange(x, 'b c s h w-> b s c h w')

        # Forward function
        # local block1
        x = self.set_layer1(x)
        x = self.set_layer2(x)
        # global block1
        g = self.gl_layer1(torch.max(x, dim=1)[0])
        g = self.gl_layer2(g)
        g = self.gl_pooling(g)

        # local block2
        x = self.set_layer3(x)
        x = self.set_layer4(x)
        # global block2
        g = self.gl_layer3(torch.max(x, dim=1)[0] + g)
        g = self.gl_layer4(g)

        # local block3
        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = torch.max(x, dim=1)[0]

        # global block3
        g = g + x

        # Resize the output tensor
        x = x.unsqueeze(2) # [B C 1 H W]
        g = g.unsqueeze(2) # [B C 1 H W]

        return [x, g]