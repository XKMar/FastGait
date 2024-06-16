import os 
import os.path as osp 

from fastgait.data.utils.base_dataset import ImageDataset

class CASIA_B(ImageDataset):
    """
    Dataset statistics:
    For some reasons, the views or types of certain subjects was lost.
    Therefore, The corresponding relation between query and gallery was 
    not completely accurate.
    Dataset statistics:
        - identities: 124.
            -: 73 (train, "005" is removed) + 50 (test).
        - views: 11 (000, 018, 036, 054, 072, 090, 108, 126, 144, 162, 180).
        - types: 10 (bg-01, bg-02, cl-01, cl-02, nm-01, nm-02, nm-03, nm-04, nm-05, nm-06).

    """
    dataset_dir = "CASIA_B/CASIA_B_ocut_64_pkl"
    
    def __init__(self, root, mode, pid_num=73, del_labels=False, **kwargs):

        self.dataset_dir =  os.path.join(osp.abspath(osp.expanduser(root)), self.dataset_dir)
        self.del_labels = del_labels
        self.pid_num = pid_num
        self.dataset = 'CASIA_B'

        self.check_before_run(self.dataset_dir)

        subsets_cfgs = {
            "train": (self.dataset_dir,'train',),
            "test":  (self.dataset_dir,'test', ),
            "val":   (self.dataset_dir,'test', ),}

        try:
            cfgs = subsets_cfgs[mode]
        except KeyError:
            raise ValueError(
                "Invalid mode in CASIA-B. Got {}, but expected to be "
                "one of [train | test | val]".format(self.mode)
            )

        data = self._process_dir(*cfgs)

        super(CASIA_B, self).__init__(data, mode, **kwargs)

    def _process_dir(self, dir_path, mode):
        """Generate the train, query and gallery path"""

        label, types, views, paths = [], [], [], []

        # Counting the directory of seqs
        for _label in sorted(list(os.listdir(dir_path))):
            if _label == '005':
                continue
            label_path = osp.join(dir_path, _label)
            for _types in sorted(list(os.listdir(label_path))):
                types_path = osp.join(label_path, _types)
                for _views in sorted(list(os.listdir(types_path))):
                    _paths = osp.join(types_path, _views)
                    
                    label.append(_label)
                    types.append(_types)
                    views.append(_views)
                    paths.append(_paths)

        # Note: the query and gallery type is different between datasets
        train, test = [], []
        label_set = sorted(list(set(label)))
        views_set = sorted(list(set(views)))
        types_set = sorted(list(set(types)))

        train_label_set = label_set[:self.pid_num]
        test_label_set = label_set[self.pid_num:]

        self.label2pid = {pid: label for label, pid in enumerate(label_set)}
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
                "Invalid label in CASIA-B. Got {}, but expected to be "
                "one of [train | val | test] label".format(l))

        if mode == 'train':
            return train
        elif mode == 'test':
            return test
        else:
            raise ValueError(
                "Invalid mode in CASIA-B. Got {}, but expected to be "
                "one of [train | val | test]".format(mode)
            )