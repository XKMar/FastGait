from .distributed_identity_sampler import DistributedIdentitySampler
from .distributed_slice_sampler import DistributedSliceSampler


__all__ = ["build_train_sampler", "build_test_sampler"]


def build_train_sampler(cfg, datasets, epoch=0):

    num_instances = cfg.TRAIN.SAMPLER.num_instances
    shuffle = cfg.TRAIN.SAMPLER.is_shuffle

    if num_instances > 0:
        # adopt PxK sampler
        if isinstance(datasets, (tuple, list)):
            # for a list of individual datasets
            samplers = []
            for dataset in datasets:
                samplers.append(
                    DistributedIdentitySampler(
                        dataset.data, num_instances=num_instances, shuffle=shuffle, epoch=epoch
                    )
                )
            return samplers
        else:
            # for a single dataset
            return DistributedIdentitySampler(
                datasets.data, num_instances=num_instances, shuffle=shuffle, epoch=epoch
            )
    else:
        # adopt normal random sampler
        if isinstance(datasets, (tuple, list)):
            # for a list of individual datasets
            samplers = []
            for dataset in datasets:
                samplers.append(
                    DistributedSliceSampler(dataset.data, shuffle=shuffle, epoch=epoch,)
                )
            return samplers
        else:
            # for a single dataset
            return DistributedSliceSampler(datasets.data, shuffle=shuffle, epoch=epoch,)


def build_test_sampler(cfg, datasets):

    if isinstance(datasets, (tuple, list)):
        # for a list of individual datasets
        samplers = []
        for dataset in datasets:
            samplers.append(DistributedSliceSampler(dataset.data, shuffle=False,))
        return samplers

    else:
        # for a single dataset
        return DistributedSliceSampler(datasets.data, shuffle=False,)
