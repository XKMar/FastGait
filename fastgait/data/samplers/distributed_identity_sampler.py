import math
import random
from collections import defaultdict

import numpy as np
import torch

from .distributed_sampler import DistributedTemplateSampler

__all__ = ["DistributedIdentitySampler"]


class DistributedIdentitySampler(DistributedTemplateSampler):
    def __init__(self, data_sources, num_instances=4, **kwargs):

        self.num_instances = num_instances
        super(DistributedIdentitySampler, self).__init__(data_sources, **kwargs)

        self._init_data()

    def _init_data(self):

        (   self.pid_index,
            self.pids,
            self.num_samples,
            self.total_size,
        ) = self._init_data_single(self.data_sources)

    def _init_data_single(self, data_source):
        # data statistics
        pid_index = defaultdict(list)

        for index, (_, pid, camid, tpid) in enumerate(data_source):
            pid_index[pid].append(index)

        pids = list(pid_index.keys())
        num_samples = int(math.ceil(len(pids) * 1.0 / self.num_replicas))
        total_size = num_samples * self.num_replicas

        return pid_index, pids, num_samples, total_size

    def __len__(self):
        # num_samples: IDs in one chunk
        # num_instance: samples for each ID
        return self.num_samples * self.num_instances

    def __iter__(self):
        yield from self._generate_iter_list()

    def _generate_iter_list(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            indices = torch.randperm(len(self.pids), generator=self.g).tolist()
        else:
            indices = torch.arange(len(self.pids)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        yield from self._sample_list(indices, self.pids, self.pid_index)

    def _sample_list(self, indices, pids, pid_index):
        # return a sampled list of indexes
        ret = []

        for kid in indices:
            p = pids[kid]
            t = pid_index[p]
            if len(t) >= self.num_instances:
                _indexes = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                _indexes = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(_indexes)
        return ret
