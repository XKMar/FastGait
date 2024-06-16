# encoding: utf-8
from .setnet import SetNet
from .partnet import PartNet
from .glnet import GLNet
from .mixnet import MixedNet

__all__ = ["build_bakcbone", "names"]

__factory = {
    'glnet': GLNet,
    'setnet': SetNet,
    'partnet': PartNet,
    'mixnet': MixedNet,
}

def build_backbone(
                name, 
                set_channels, 
                *args, 
                **kwargs):
    """build the backbone model.

    Args:
        name (str): The name of backbone model.
        set_channels (List[int]): the backbone model feature dimension.
    Example:
        >>> build_backbone('mixnet', [64, 128, 256])
    """
    if name not in __factory:
        raise KeyError("Unknown backbone:", name)
    return __factory[name](
                    set_channels, 
                    *args, 
                    **kwargs)
