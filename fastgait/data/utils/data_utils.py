import os
import os.path as osp
import glob
import pickle
import numpy as np 

from PIL import Image

def load_seqs(path):
    """Load seqs from path using ``pockle.load``.
    Args:
        path (str): path to an seqs.
    Returns:
        list(seqs)
    """
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    else:
        seqs_paths = glob.glob(osp.join(path, "*.pkl"))
        seqs = pickle.load(open(seqs_paths[0], 'rb'))
    
    if isinstance(seqs, list):
        seqs = np.asarray(seqs)
        
    return seqs

def read_image(path):
    """Reads image from path using ``PIL.Image``.
    Args:
        path (str): path to an image.
    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert("RGB")
            got_img = True
        except IOError:
            print(
                f'IOError incurred when reading "{path}". '
                f"Will redo. Don't worry. Just chill."
            )
    return img

def save_image(image_numpy, path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(path)
