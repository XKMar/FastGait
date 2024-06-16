import os 
import os.path as osp 

from fastgait.data.utils.base_dataset import ImageDataset

class OUMVLP(ImageDataset):
    """
    Dataset statistics:
    For some reasons, the views or types of certain subjects was lost.
    Therefore, The corresponding relation between query and gallery was 
    not completely accurate.
    Dataset statistics:
        - identities: 10307.
            -: odd number (train: 5154) + even number (test: 5153).
        - views: 14 (000, 015, 030, 045, 060, 075, 090, 180, 195, 210, 225, 240, 255, 270).
        - types: 02 (01, 02).

    """
    dataset_dir = "OUMVLP/OUMVLP_cut_64_pkl"

    def __init__(self, root, mode, pid_num=5153, del_labels=False, **kwargs):

        self.dataset_dir = os.path.join(osp.abspath(osp.expanduser(root)), self.dataset_dir)
        self.del_labels = del_labels
        self.pid_num = pid_num
        self.dataset = 'OUMVLP'

        self.check_before_run(self.dataset_dir)

        subsets_cfgs = {
            "train": (self.dataset_dir,'train',),
            "test":  (self.dataset_dir,'test',),
            "val":   (self.dataset_dir,'test',),}

        try:
            cfgs = subsets_cfgs[mode]
        except KeyError:
            raise ValueError(
                "Invalid mode in OUMVLP. Got {}, but expected to be "
                "one of [train | test | val]".format(self.mode)
            )

        data = self._process_dir(*cfgs)

        super(OUMVLP, self).__init__(data, mode, **kwargs)

    def _process_dir(self, dir_path, mode):
        """Generate the train, query and gallery path"""

        label, types, views, paths = [], [], [], []

        # Counting the directory of seqs
        for _label in sorted(list(os.listdir(dir_path))):
            label_path = osp.join(dir_path, _label)
            for _types in sorted(list(os.listdir(label_path))):
                types_path = osp.join(label_path, _types)
                for _views in sorted(list(os.listdir(types_path))):
                    _paths = osp.join(types_path, _views)

                    label.append(_label)
                    types.append(_types)
                    views.append(_views)
                    paths.append(_paths)

        # Note: the query and gallery is different between datasets
        # In OUMVLP datasets, odd number ==> train, even number ==> test
        train, test = [], []
        label_set = sorted(list(set(label)))
        views_set = sorted(list(set(views)))
        types_set = sorted(list(set(types)))

        train_label_set = label_set[0:-1:2]  # 5153, "00001-10305" for train
        test_label_set  = label_set[1::2]    # 5153, "00002-10306" for test
        test_label_set.append(label_set[-1]) # 5154, "00002-10306, 10307" for test

        # renew the label set
        num_label_set = train_label_set + test_label_set
        self.label2pid = {pid: label for label, pid in enumerate(num_label_set)}
        self.views2vid = {vid: views for views, vid in enumerate(views_set)}
        self.types2tid = {tid: types for types, tid in enumerate(types_set)}

        for i , l in enumerate(label):
            # train list
            if l in train_label_set:
                if not self.del_labels:
                    pid = self.label2pid[label[i]]
                else:
                    pid = 0
                train.append((paths[i], pid, self.views2vid[views[i]], self.types2tid[types[i]]))
            elif l in test_label_set:
                if not self.del_labels:
                    pid = label[i]
                else:
                    pid = 0
                test.append((paths[i], pid, views[i], types[i]))
            else:
                raise ValueError(
                "Invalid label in OUMVLP. Got {}, but expected to be "
                "one of [train | val | test] label".format(l)
            )

        if mode == 'train':
            return train
        elif mode == 'test':
            return test
        else:
            raise ValueError(
                "Invalid mode in OUMVLP. Got {}, but expected to be "
                "one of [train | val | test]".format(mode)
            )

if __name__ == '__main__':
    casiab = OUMVLP('/data_1/makang/data/OUMVLP/silhouettes_cut_pkl/', 'train')