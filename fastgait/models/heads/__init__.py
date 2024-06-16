# encoding: utf-8

from .base_head import BaseHead
from .gmpa_head import GMPAHead
from .gem_head import GeMHead
from .tfa_head import TFAHead

# build the head layer
__head_factory = {
    "base": BaseHead,
    "gmpa": GMPAHead,
    "gem": GeMHead,
    "tfa": TFAHead,
}

def build_head(
            name, 
            num_bins, 
            num_parts, 
            num_classes,
            in_channels, 
            out_channels, 
            *args, 
            **kwargs):
    """build the head layer.
    Args:
        name (str): The name of pooling types.
        bin_nums (List[int]): Number of horizontal split parts.
        inplanes (int): The input feature dimension.

    Example:
        >>> build_head_layer('part', [16], 1024, 256, 73)
    """
    if name not in __head_factory:
        raise KeyError("Unknown head layer:", name)
    return __head_factory[name](
                            num_bins, 
                            num_parts, 
                            num_classes,
                            in_channels, 
                            out_channels, 
                            *args, 
                            **kwargs)
