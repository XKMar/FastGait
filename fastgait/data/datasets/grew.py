import os 
import os.path as osp 

from fastgait.data.utils.base_dataset import ImageDataset

class GREW(ImageDataset):
    """
    Dataset statistics:
    For some reasons, the views or types of certain subjects was lost.
    Therefore, The corresponding relation between query and gallery was 
    not completely accurate.
    To adjust the dataset framework, we set the probe seqs as "0" in test,
    and others are gallery.

    Dataset statistics:
        - identities: 20000. (train)
        - views: uncertain.
        - types: uncertain.

    """
    dataset_dir = "GREW/GREW_cut_64_pkl"
    
    def __init__(self, root, mode, pid_num=20000, del_labels=False, **kwargs):

        self.dataset_dir = os.path.join(osp.abspath(osp.expanduser(root)), self.dataset_dir)
        self.del_labels = del_labels
        self.pid_num = pid_num
        self.dataset = 'GREW'

        self.check_before_run(self.dataset_dir)

        subsets_cfgs = {
            "train": (os.path.join(self.dataset_dir, 'train'),'train',),
            "test":  (os.path.join(self.dataset_dir, 'test'), 'test' ,),
            "val":   (os.path.join(self.dataset_dir, 'test'), 'test' ,),}

        try:
            cfgs = subsets_cfgs[mode]
        except KeyError:
            raise ValueError(
                "Invalid mode in GREW. Got {}, but expected to be "
                "one of [train | test | val]".format(self.mode)
            )

        data = self._process_dir(*cfgs)

        super(GREW, self).__init__(data, mode, **kwargs)

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

        # Note!!! : the query and gallery type is different between datasets
        dataset = []
        label_set = sorted(list(set(label)))
        views_set = sorted(list(set(views)))
        types_set = sorted(list(set(types)))
        
        self.label2pid = {pid: label for label, pid in enumerate(label_set)}
        self.views2vid = {vid: views for views, vid in enumerate(views_set)}
        self.types2tid = {tid: types for types, tid in enumerate(types_set)}

        for i , l in enumerate(label):
            if not self.del_labels:
                pid = self.label2pid[label[i]]
            else:
                pid = 0
            if mode == 'train':
                dataset.append((paths[i], pid, self.views2vid[views[i]], self.types2tid[types[i]]))
            else:
                dataset.append((paths[i], pid, views[i], types[i]))

        return dataset