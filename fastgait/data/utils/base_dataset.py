# Modified from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/datasets/dataset.py  # noqa
# to support unsupervised features

import copy
import numpy as np
import os.path as osp

from fastgait.data.utils.data_utils import load_seqs

class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset``.

    Args:
        data (list): contains tuples of (img_path(s), pid, camid).
        mode (str): 'train', 'val', 'test'.
        transform: transform function.
        verbose (bool): show information.
    """

    def __init__(
        self, data, mode, transform=None, verbose=True, sort=True, height=64, width=44, **kwargs,
    ):
        self.data = data
        self.transform = transform
        self.mode = mode
        self.verbose = verbose
        self.height = height
        self.width = width
        self.clip = int((height-width)/2)

        self.num_label, self.num_views, self.num_types = self.parse_data(self.data)

        if sort:
            self.data = sorted(self.data)

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        raise NotImplementedError

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (path(s), pid, camid, tpid)
        """
        self.label, self.views, self.types = [], [], []

        for _, pid, camid, tpid in data:
            self.label.append(pid)
            self.views.append(camid)
            self.types.append(tpid)

        label_set = set(self.label)
        views_set = set(self.views)
        types_set = set(self.types)
        
        return len(label_set), len(views_set), len(types_set)

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

        All other image datasets should subclass it.
        ``_get_single_item`` returns an image given index.
        It will return (``img``, ``img_path``, ``pid``, ``camid``, ``index``)
        where ``img`` has shape (channel, height, width). As a result,
        data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, data, mode, **kwargs):
        super(ImageDataset, self).__init__(data, mode, **kwargs)

        # "all_data" stores the original data list
        # "data" stores the pseudo-labeled data list
        self.all_data = copy.deepcopy(self.data)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, _index):
        seqs_path, _label, _views, _types = self.data[_index]
        
        # load data and clip both of the sides
        seqs = load_seqs(seqs_path)
        seqs = seqs[:, :, self.clip:-self.clip].astype('float32') / 255.0


        if self.transform is not None:
            seqs = self.transform(seqs)

        return {
            "image": seqs,
            "label": _label,
            "views": _views,
            "types": _types,
            "index": _index
        }

    def __add__(self, other):
        """
        work for combining query and gallery into the test data loader
        """
        return ImageDataset(
            self.data + other.data,
            self.mode + "+" + other.mode,
            transform=self.transform,
            verbose=False,
            sort=False,
            height=self.height, 
            width=self.width
        )

    def show_summary(self):
        print(
            "=> Loaded {} from {}".format(self.mode, self.__class__.__name__)
        )
        print("  ------------------------------------")
        print("  # ids | # items | # views | # types")
        print("  ------------------------------------")
        print(
            "  {:5d} | {:7d} | {:7d} | {:7d}".format(
                self.num_label, 
                len(self.data), 
                self.num_views, 
                self.num_types))
        print("  ------------------------------------")
