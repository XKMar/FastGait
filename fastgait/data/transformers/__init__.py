import torchvision.transforms as T

from .transforms import (
                RandomPartBlur,
                RandomPadCrop,
                RandomRotate,
                RandomErasing,
                RandomHorizontalFlip,
                RandomPerspectives,
                RandomPartDilation,
                RandomPartErosion)

__all__ = ["build_train_transformer", "build_test_transformer"]

# build the algorithms
__transform_factory = {
    'is_blur': RandomPartBlur,
    'is_padcrop': RandomPadCrop,
    'is_rotate': RandomRotate,
    'is_erase': RandomErasing,
    'is_flip': RandomHorizontalFlip,
    'is_perspective': RandomPerspectives,
    'is_dilate': RandomPartDilation,
    'is_erosion': RandomPartErosion,
}

def build_train_transformer(pip):
    r"""Build the train transformers.

        Args:
            >>> input  (dict[]):   the train transformers.
            >>> output (list[]): the transformer function.
    """

    res = []
    for key in pip.keys():
        if key not in __transform_factory:
            raise KeyError("Unknown transform:", key)
        res.append(__transform_factory[key](pip[key]))

    if len(res) == 0:
        return None
    else:
        return T.Compose(res)


def build_test_transformer(cfg):

    return None

if __name__ == "__main__":

    data_dir = '/data_1/makang/data/CASIA_B/CASIA_B_ncut_64_pkl/001/bg-01/000/000.pkl'
    import os
    import cv2
    import pickle
    import yaml
    from easydict import EasyDict

    raw_data = pickle.load(open(data_dir, 'rb'))
    raw_data = raw_data.astype('float32')/255.0

    with open('/home/makang/code/OpenGait/tools/SpCL/config.yaml', "r") as f:
        try:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        except Exception:
            cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    random_erase = build_train_transformer(cfg)
    trans_data = random_erase(raw_data)

    # save the transfom data
    os.makedirs('/home/makang/code/OpenGait/records/examples/trans', exist_ok=True)

    for i in range(len(trans_data)):
        save_name = os.path.join('/home/makang/code/OpenGait/records/examples/trans', 'trans_{:0>3d}.png'.format(i))
        cv2.imwrite(save_name, (trans_data[i,:,:]*255.0).astype('uint8'))