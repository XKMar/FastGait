import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "SetPooling",
    "HorizontalMapping",
    "GeMPartMapping"
]

class SetPooling(nn.Module):
    r""" The goal of Set Pooling is to condense a set of gait information,
    formulated as z = G(V).
        Args:
            bin_num (List[int]): Number of horizontal split parts.
        Return:
            feature (torch): The global gait features
        Example:
            >>> bin_num = [16]
            >>> inputs = [torch.rand(128, 30, 256, 16, 11)] # [B C T H W]
            >>> output = SetPooling(inputs)
            >>> print(f'output.shape = {output.shape}')
            outputs.shape = torch.Size([128, 256, 16, 11]) # [B C H W]
    """
    def __init__(
            self, 
            bin_num,
        ):
        super().__init__()
        if not isinstance(bin_num, list):
            self.bin_num = [bin_num]
        else:
            self.bin_num = bin_num

    def forward(self, x):
        r"""Forward function.
            Args:
                >>> Inputs: [B C T H W]
                >>> Output: [B C H W]
        """
        return torch.max(x, 2)[0]

class HorizontalMapping(nn.Module):
    r""" The goal of Horizontal Mapping (HP) is to horizontal split the feature map into 
    {bin_num} strips.

    Args:
        bin_num (List[int]): number of horizontal split parts
    Return:
        feature (torch): the splited part features
    Example:
        >>> bin_num = [1,2,4,8,16]
        >>> inputs = [torch.rand(128, 256, 30, 16, 11)] # [B C T H W]
        >>> output = HorizontalMapping(inputs)
        >>> print(f'output.shape = {output.shape}')
        outputs.shape = torch.Size([128, 30, 256, 16]) # [B T C P]
    """
    def __init__(
            self, 
            bin_num,
        ):
        super().__init__()

        if not isinstance(bin_num, list):
            self.bin_num = [bin_num]
        else:
            self.bin_num = bin_num

    def forward(self, x):
        r"""Forward function.
            Args:
                >>> Inputs: [B C T H W]
                >>> Output: [B T C P]
        """
        # multiple backbone outputs
        if not isinstance(x, list):
            feat = [x]
        else:
            feat = x

        # horizontal split parts
        feature = list()
        for sub_feat in feat:
            size_info = sub_feat.size()[:-2] # [B C T]
            for bin in self.bin_num:
                z = sub_feat.view(*size_info, bin, -1)
                z = z.max(-1)[0] + z.mean(-1)
                feature.append(z)

        feature = torch.cat(feature,  dim=-1).contiguous() # [B C T P]
        feature = feature.permute(0, 2, 1, 3).contiguous() # [B T C P]

        return feature

class GeMPartMapping(nn.Module):
    r""" The goal of GeM Part Mapping is to horizontal split the feature into 
    {bin_num} parts and keep the temporal dimension.

    Args:
        bin_num (List[int]): number of horizontal split parts.
    Return:
        feature (torch): the splited part features
    Example:
        >>> bin_num = [16]
        >>> inputs = [torch.rand(128, 30, 256, 16, 11)] # [B C T H W]
        >>> output = GeMPartMapping(inputs)
        >>> print(f'output.shape = {output.shape}')
        outputs.shape = torch.Size([128, 30, 256, 16]) # [B T C P]
    """
    def __init__(
            self, 
            bin_num,
            norm=6.5,
            eps=1e-6
        ):
        super().__init__()
        if not isinstance(bin_num, list):
            self.bin_num = [bin_num]
        else:
            self.bin_num = bin_num
        
        self.p = nn.Parameter(torch.ones(1)*norm)
        self.eps = eps

    def forward(self, x):
        r"""Forward function.
            Args:
                >>> Inputs: [B C T H W]
                >>> Output: [B T C P]
        """
        # multiple backbone outputs
        if not isinstance(x, list):
            feat = [x]
        else:
            feat = x

        # horizontal split parts
        feature = list()
        for sub_feat in feat:
            size_info = sub_feat.size()[:-2] # [B C T]
            for bin in self.bin_num:
                z = sub_feat.view(*size_info, bin, -1)
                z = z.clamp(min=self.eps).pow(self.p)
                z = F.avg_pool3d(z, (1, 1, z.size(-1))).pow(1.0 / self.p).squeeze(-1)
                feature.append(z)

        feature = torch.cat(feature,  dim=-1).contiguous() # [B C T P]
        feature = feature.permute(0, 2, 1, 3).contiguous() # [B T C P]

        return feature

if __name__ == "__main__":
    hp = HorizontalMapping(16)
    inputs = [torch.randn(128, 30, 256, 16, 11)]
    output = hp(inputs)