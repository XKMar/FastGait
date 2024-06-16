import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from ..utils.dist_utils import get_dist_info
from .datasets import build_dataset
from .samplers import build_test_sampler, build_train_sampler
from .transformers import build_test_transformer, build_train_transformer
from .utils.dataset_wrapper import IterLoader
from .utils.collate_fn import CollateFn

__all__ = ["build_train_dataloader", "build_test_dataloader"]

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_train_dataloader(
    cfg, epoch=0, **kwargs
):
    """
    Build training data loader
    """

    rank, world_size, dist = get_dist_info()

    data_root = cfg.DATA_ROOT  # PATH, str

    dataset_names = list(cfg.TRAIN.datasets.keys())     # list of str
    dataset_modes = list(cfg.TRAIN.datasets.values())   # list of str
    for mode in dataset_modes:
        assert mode in [
            "train",
        ], "subset for training should be selected in [train]"

    # build individual datasets
    datasets = []
    for idx, (dn, dm) in enumerate(zip(dataset_names, dataset_modes)):
        # build transformer
        if cfg.DATA.TRAIN.PIPLINE is not None:
            train_transformer = build_train_transformer(cfg.DATA.TRAIN.PIPLINE)
        else:
            train_transformer = None

        # build dataset
        datasets.append(build_dataset(dn, data_root, dm, transform=train_transformer, 
                    height=cfg.DATA.height, width=cfg.DATA.width))

    # build sampler
    train_sampler = build_train_sampler(cfg, datasets, epoch=epoch)

    # build data loader
    if dist:
        batch_size = cfg.TRAIN.LOADER.samples_per_gpu
        num_workers = cfg.TRAIN.LOADER.workers_per_gpu
    else:
        batch_size = cfg.TRAIN.LOADER.samples_per_gpu * cfg.total_gpus
        num_workers = cfg.TRAIN.LOADER.workers_per_gpu * cfg.total_gpus

    # several individual data loaders
    data_loaders = []
    for dataset, sampler in zip(datasets, train_sampler):
        data_loaders.append(
            IterLoader(
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    sampler=sampler,
                    collate_fn=CollateFn(cfg.DATA.TRAIN.sample_type, cfg.DATA.TRAIN.sample_num),
                    shuffle=False,
                    pin_memory=True,
                    worker_init_fn = worker_init_fn,
                    drop_last=True,
                    **kwargs,),
                length=cfg.TRAIN.iters,))

    return data_loaders, datasets


def build_test_dataloader(cfg, one_gpu=False, **kwargs):
    """
    Build testing data loader
    """

    rank, world_size, dist = get_dist_info()

    data_root = cfg.DATA_ROOT  # PATH, str
    dataset_names = cfg.TEST.datasets  # list of str

    # build transformer
    test_transformer = build_test_transformer(cfg)

    # build individual datasets
    datasets, test_datas = [], []
    for dn in dataset_names:
        test_dataset = build_dataset(
            dn, data_root, "test", transform=test_transformer, height=cfg.DATA.height, width=cfg.DATA.width,)
        datasets.append(test_dataset)
        test_datas.append(test_dataset.data)

    # build sampler
    if not one_gpu:
        test_sampler = build_test_sampler(cfg, datasets)
    else:
        test_sampler = [None] * len(datasets)

    # build data loader
    if dist:
        batch_size = cfg.TEST.LOADER.samples_per_gpu
        num_workers = cfg.TEST.LOADER.workers_per_gpu
    else:
        batch_size = cfg.TEST.LOADER.samples_per_gpu * cfg.total_gpus
        num_workers = cfg.TEST.LOADER.workers_per_gpu * cfg.total_gpus

    # several individual data loaders
    data_loaders = []
    for dataset, sampler in zip(datasets, test_sampler):
        data_loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=sampler,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                **kwargs,))

    return data_loaders, test_datas