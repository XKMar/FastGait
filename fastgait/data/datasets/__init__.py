# encoding: utf-8
from .casia_b import CASIA_B
from .oumvlp import OUMVLP
from .grew import GREW
from .gait3d import Gait3D

__all__ = ["build_dataset", "names"]

__factory = {
    "CASIA-B": CASIA_B,
    "OUMVLP": OUMVLP,
    "GREW": GREW,
    "Gait3D": Gait3D
}


def names():
    return sorted(__factory.keys())


def build_dataset(name, root, mode, *args, **kwargs):
    """Create a dataset instance.
    Args:
        name (str): The dataset name.
        root (str): The path to the dataset directory.
        mode (str): The subset for the dataset, e.g. [train | val | test]
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, mode, *args, **kwargs)

