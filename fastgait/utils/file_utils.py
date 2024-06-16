import errno
import json
import time
import os
import os.path as osp

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def read_json(fpath):
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def model_path(log_root, data_name, algor_name, model_name):
    """Creat the path to save model.
    Args: 
        log_root(str): the root path
        data_name(str): the dataset name
        algor_name(str): the algorithm 
        model_name(str): the backbone name
    """
    curr_date = time.strftime('%Y-%m-%d-%H-%M-%S')
    save_path = os.path.join(log_root, 'models', data_name, 
                             algor_name, model_name, curr_date)
    mkdir_if_missing(save_path)
    return save_path