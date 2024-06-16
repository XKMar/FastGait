import torch
import torch.nn as nn

from ..utils.net_utils import BasicConv3d, Residual, STMixedConv3d

class MixedNet(nn.Module):
    r"""The backbone of DANet consit of several LCMB blocks.

        Args:
            set_channels (List[int]): the backbone model feature dimension.
        Returns:
            output (tensor): the gait features.
    """
    def __init__(
        self,
        set_channels
    ):
        super(MixedNet, self).__init__()
        _set_in_channels = 1
        _set_channels = set_channels

        self.layer1 = self._make_layer(_set_in_channels, _set_channels[0], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.layer2 = self._make_layer(_set_channels[0], _set_channels[1], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.layer3 = self._make_layer(_set_channels[1], _set_channels[2], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        # self.layer4 = self._make_layer(_set_channels[2], _set_channels[3])

    def _make_layer(self, inplanes, planes, kernel_size, stride, padding):

        layers = []
        layers.append(BasicConv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(Residual(STMixedConv3d(planes, planes)))

        return nn.Sequential(*layers)

    def forward(self, x):
        r"""Forward function.
            Args:
                >>> Inputs: [B C T H W]
                >>> Output: [B C T H W]
        """

        # base stage 
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)

        return out