import time
from datetime import timedelta

import numpy as np
import torch

from fastgait.utils.dist_utils import get_dist_info
from fastgait.models.utils.rank import evaluate_rank
from fastgait.models.utils.extract import extract_features

@torch.no_grad()
def test_gait(
    cfg, model, data_loader, test_data, dataset_name=None, rank=None, epoch=None, **kwargs
):

    start_time = time.monotonic()

    sep = "*******************************"
    if dataset_name is not None:
        print(f"\n{sep} Start testing {dataset_name} {sep}\n")

    if rank is None:
        rank, _, _ = get_dist_info()

    # parse ground-truth IDs and camera IDs
    label = np.array([pid for _, pid, _, _ in test_data])
    views = np.array([vid for _, _, vid, _ in test_data])
    types = np.array([tid for _, _, _, tid in test_data])

    # extract features with the given model
    features = extract_features(
        model,
        data_loader,
        test_data,
        normalize=cfg.TEST.norm_feat,
        prefix="Test: ",
        **kwargs,
    )

    if rank == 0:
        # evaluate with original distance
        evaluate_rank(cfg.TEST, features, label, views, types, dataset_name)

    end_time = time.monotonic()
    print("Testing time: ", timedelta(seconds=end_time - start_time))
    print(f"\n{sep} Finished testing {sep}\n")

    del features
    torch.cuda.empty_cache()

