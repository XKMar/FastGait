import math

import torch

from .distributed_sampler import DistributedTemplateSampler

__all__ = ["DistributedSliceSampler",]


class DistributedSliceSampler(DistributedTemplateSampler):
    def __init__(self, data_sources, **kwargs):
        super(DistributedSliceSampler, self).__init__(data_sources, **kwargs)

        self._init_data()

    def _init_data(self):
        self.num_samples, self.total_size = self._init_data_single(self.data_sources)

    def _init_data_single(self, data_source):
        num_samples = int(math.ceil(len(data_source) * 1.0 / self.num_replicas))
        total_size = num_samples * self.num_replicas

        return num_samples, total_size

    def __len__(self):
        return self.num_samples

    def _generate_iter_list(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            slices = torch.randperm(len(self.data_sources), generator=self.g).tolist()
        else:
            slices = torch.arange(len(self.data_sources)).tolist()

        # add extra samples to make it evenly divisible
        slices += slices[: (self.total_size - len(slices))]
        assert len(slices) == self.total_size

        # slice
        slice = torch.LongTensor(slices).split(self.num_samples)[self.rank]
        slice = slice.tolist()
        assert len(slice) == self.num_samples

        yield from slice

    def __iter__(self):
        yield from self._generate_iter_list()
