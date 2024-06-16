import random
import numpy as np

import torch


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
        CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def batch_processor(data):  # list of dict
    assert isinstance(
        data, (list, dict)
    ), "the data for batch processor should be within a List or Dict"

    for sub_data in data:
        image = [torch.from_numpy(sub_data['image']).float().cuda()]
        label = [torch.from_numpy(sub_data['label']).long().cuda() ]
        views = [torch.from_numpy(sub_data['views']).long().cuda() ]
        types = [torch.from_numpy(sub_data['types']).long().cuda() ]
        index = [torch.from_numpy(sub_data['index']).long().cuda() ]

    return {"image": image,
            "label": label,
            "views": views,
            "types": types,
            "index": index}