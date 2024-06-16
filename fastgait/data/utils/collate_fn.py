import random
import torch
import numpy as np


class CollateFn(object):
    """ Randomly selects some frames (fixed or random number) in the gait sequences.
    When sample_types is 'contiguous' and sample_num is 'None', collatefn could select
    unfixed num for network.
    Args:
         sample_type: The types that select the frames from gait sequence
         sample_num: The numbers that select the frames
    """

    def __init__(
        self, sample_type, sample_num
    ):
        self.sample_type = sample_type
        self.sample_num = sample_num

        assert sample_type in [
            'random',
            'continuous'
        ], "subset for sample_type should be selected in [random, contiguous]"

    def __call__(self, batch):
        batch_size = len(batch)

        # base information
        image = [batch[i]["image"] for i in range(batch_size)]
        label = [batch[i]["label"] for i in range(batch_size)]
        views = [batch[i]["views"] for i in range(batch_size)]
        types = [batch[i]["types"] for i in range(batch_size)]
        index = [batch[i]["index"] for i in range(batch_size)]

        # random sample the select num for contiguous 
        if self.sample_num is None:
            select_num = random.randint(20, 40)
        else:
            select_num = self.sample_num

        # random or continuous choice fixed or unfixed frames
        def select_frame(sample, nums):
            frame_set = np.arange(int(sample.shape[0]))
            if self.sample_type == 'random':
                if len(frame_set) >= nums:
                    frame_id_list = sorted(np.random.choice(frame_set, nums, replace=False))
                else:
                    frame_id_list = sorted(np.random.choice(frame_set, nums, replace=True))

                sample = sample[frame_id_list]
            
            elif self.sample_type == 'continuous':
                if len(frame_set) > nums:
                    frame_id_idx = np.random.choice(np.arange(len(frame_set)-nums))
                    sample = sample[frame_id_idx:(frame_id_idx + nums)]
                else:
                    frame_id_list = np.pad(frame_set, (0, nums-len(frame_set)), mode='maximum')
                    sample = sample[frame_id_list]
            
            return sample

        # Multiple transform in contrast learning
        sequs = [select_frame(image[i], select_num) for i in range(batch_size)]

        # generate data
        sequs = np.asarray([sequs[i] for i in range(batch_size)])
        label = np.asarray(label)
        views = np.asarray(views)
        types = np.asarray(types)
        index = np.asarray(index)

        return {
            "image": sequs,
            "label": label,
            "views": views,
            "types": types,
            "index": index}