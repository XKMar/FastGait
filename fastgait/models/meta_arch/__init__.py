from .basic import Basic

# build Architecture
__factory = {
    "basic": Basic,
}

def build_arch(name, 
            num_parts, 
            num_classes, 
            set_channels, 
            embd_feature,
            *args, 
            **kwargs):
    """build the Architecture.
    Args:
        name (str): The name of Architecture.
        num_parts (List[int]): Number of horizontal split parts.
        num_classes  (int) : Number of classes for cross-entropy loss.
        set_channels (List[int]): The backbone model feature dimension.
        embd_feature (int) The dimension of the output features.

    Example:
        >>> build_algorithm('basenet', [16], 73, [32, 64, 128], 256)
    """
    if name not in __factory:
        raise KeyError("Unknown architecture:", name)
    return __factory[name](
                        num_parts, 
                        num_classes, 
                        set_channels, 
                        embd_feature, 
                        *args, 
                        **kwargs)