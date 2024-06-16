import cv2
import math
import random
import numpy as np
from PIL import Image
from torchvision.transforms import RandomPerspective

class RandomPartBlur(object):
    """
    Apply the GaussianBlur Transformer on some regions of the gait sequence
    
    args:
        probability (float): random rotate rates between 0 to 1.
        top_range (tuple): the body part of silhs
        bot_range (tuple): the leg part of silhs
        per_frame (bool): each frame whether use the same way to be blured 

    """
    def __init__(self, configs):
        self.probability =  getattr(configs, 'blur_prob',   0.5)
        self.top_range = getattr(configs, 'top_range', ( 9, 20))
        self.bot_range = getattr(configs, 'bot_range', (29, 40))

    def __call__(self, seqs):
        '''
        Input:
            seqs: a sequence of silhouette frames, [s, h, w]
        Output:
            seqs: a sequence of agumented frames, [s, h, w]
        '''
        if random.uniform(0, 1) >= self.probability:
            return seqs
        else:
            top = random.randint(self.top_range[0], self.top_range[1])
            bot = random.randint(self.bot_range[0], self.bot_range[1])

            _seqs = seqs.copy()
            _seqs = _seqs[:, top:bot, ...]

            for i in range(_seqs.shape[0]):
                _blur_img = _seqs[i, :, :]
                _blur_img = cv2.GaussianBlur(_blur_img, ksize=(3, 3), sigmaX=0)
                _seqs[i, :, :] = (_blur_img > 0.2).astype(float)

            seqs[:, top:bot, ...] = _seqs
        return seqs

class RandomPartDilation(object):
    """
    Apply the RandomPartDiation Transformer on some regions of the gait sequence
    
    args:
        probability (float): random rotate rates between 0 to 1.
        top_range (tuple): the body part of silhs
        bot_range (tuple): the leg part of silhs
        per_frame (bool): each frame whether use the same way to be blured 

    """
    def __init__(self, configs):
        self.probability =  getattr(configs, 'dilate_prob',   0.5)
        self.top_range = getattr(configs, 'top_range', ( 9, 20))
        self.bot_range = getattr(configs, 'bot_range', (29, 40))

    def __call__(self, seqs):
        '''
        Input:
            seqs: a sequence of silhouette frames, [s, h, w]
        Output:
            seqs: a sequence of agumented frames, [s, h, w]
        '''
        if random.uniform(0, 1) >= self.probability:
            return seqs
        else:
            top = random.randint(self.top_range[0], self.top_range[1])
            bot = random.randint(self.bot_range[0], self.bot_range[1])

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            _seqs = seqs.copy()
            _seqs = _seqs[:, top:bot, ...]

            for i in range(_seqs.shape[0]):
                _dilate_img = _seqs[i, :, :]
                _seqs[i, :, :] = cv2.dilate(_dilate_img, kernel, iterations=2).astype(float)

            seqs[:, top:bot, ...] = _seqs
        return seqs

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be
            performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
    """

    def __init__(self, configs):

        self.probability =  getattr(configs, 'erase_prob', 0.5)
        self.sl = getattr(configs, 'erase_sl', 0.005)
        self.sh = getattr(configs, 'erase_sh', 0.02)
        self.r1 = getattr(configs, 'erase_r1', 0.3)

    def __call__(self, seqs):

        if random.uniform(0, 1) >= self.probability:
            return seqs

        for _attempt in range(5):
            area = seqs.shape[1] * seqs.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < seqs.shape[2] and h < seqs.shape[1]:
                x1 = random.randint(0, seqs.shape[1] - h)
                y1 = random.randint(0, seqs.shape[2] - w)
                seqs[:, x1:x1+h, y1:y1+w] = 0.

        return seqs

class RandomPartErosion(object):
    """
    Apply the RandomPartErosion Transformer on some regions of the gait sequence
    
    args:
        probability (float): random rotate rates between 0 to 1.
        top_range (tuple): the body part of silhs
        bot_range (tuple): the leg part of silhs
        per_frame (bool): each frame whether use the same way to be blured 

    """
    def __init__(self, configs):
        self.probability =  getattr(configs, 'erode_prob',   0.5)
        self.top_range = getattr(configs, 'top_range', ( 9, 20))
        self.bot_range = getattr(configs, 'bot_range', (29, 40))

    def __call__(self, seqs):
        '''
        Input:
            seqs: a sequence of silhouette frames, [s, h, w]
        Output:
            seqs: a sequence of agumented frames, [s, h, w]
        '''
        if random.uniform(0, 1) >= self.probability:
            return seqs
        else:
            top = random.randint(self.top_range[0], self.top_range[1])
            bot = random.randint(self.bot_range[0], self.bot_range[1])

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            _seqs = seqs.copy()
            _seqs = _seqs[:, top:bot, ...]

            for i in range(_seqs.shape[0]):
                _erode_img = _seqs[i, :, :]
                _seqs[i, :, :] = cv2.erode(_erode_img, kernel)

            seqs[:, top:bot, ...] = _seqs
        return seqs

class RandomHorizontalFlip(object):
    """Horizontally flip the given seqs randomly with a given probability
    The image can be a PIL Image or a torch Tensor,  in which case it is expected 
    to have [..., H, W] shape, where ... means an arbitraty number of leading 
    dimensions.
    
    args: 
        threhold (float): random flip rates between 0 to 1.
    """
    def __init__(self, configs):

        self.probability =  getattr(configs, 'flip_prob', 0.5)

    def __call__(self, seqs):

        if random.uniform(0, 1) >= self.probability:
            return seqs
        else:
            seqs = seqs[:, :, ::-1].copy()
            return seqs

class RandomPadCrop(object):
    """
    Random pad the imgs of input seqs
    
    args:
        pad_size (tuple): pad size of input imgs
        threhold (float): random pad rates between 0 to 1.
        per_frame (bool): each frame whether use the same way to pad 

    """

    def __init__(self, configs):

        self.probability =  getattr(configs, 'pad_prob', 0.5)
        self.pad_size =  getattr(configs, 'pad_size', (4, 0))
        self.per_frame =  getattr(configs, 'per_frame', True)

    def __call__(self, seqs):
        if not self.per_frame:
            if random.uniform(0, 1) >= self.probability:
                return seqs
            else:
                _, dh, dw = seqs.shape 
                seqs = self.pad_seqs(seqs)
                _, sh, sw = seqs.shape
                # bh, lw, th, rw = self.get_params((sh, sw), (dh, dw))
                if sh == dh and sw == dw:
                    bh, lw, th, rw = 0, 0, dh, dw
                else:
                    bh = random.randint(0, sh - dh)
                    lw = random.randint(0, sw - dw)
                    th = bh + dh
                    rw = lw + dw
                return seqs[:, bh:th, lw:rw]
        else:
            self.per_frame = False
            frame_num = seqs.shape[0]
            ret = [self.__call__(seqs[k][np.newaxis, ...]) for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, 0)

    def pad_seqs(self, seqs):
        return np.pad(seqs, ([0, 0], [self.pad_size[0], self.pad_size[0]], [self.pad_size[1], self.pad_size[1]]), mode='constant')

class RandomPerspectives(object):
    """
    Random rotate the imgs of input seqs
    
    args:
        threhold (float): random rotate rates between 0 to 1.
        degres (int, angle): the rotate angle

    """
    
    def __init__(self, configs):

        self.probability = getattr(configs, 'pers_prob', 0.5)
        self.distortion_scale =  getattr(configs, 'distortion_scale', 0.2)
        self.perspective_transformer = RandomPerspective(self.distortion_scale, self.probability)

    def __call__(self, seqs):
        if random.uniform(0, 1) >= self.probability:
            return seqs
        else:
            seqs = (seqs*255.0).astype('uint8')
            seqs = [self.perspective_transformer(Image.fromarray(seqs[i, ...], mode='L')) for i in range(seqs.shape[0])]
            seqs = np.concatenate([np.array(img)[np.newaxis, ...] for img in seqs], 0).astype('float32')/255.0
            return seqs

class RandomRotate(object):
    """
    Random rotate the imgs of input seqs
    
    args:
        threhold (float): random rotate rates between 0 to 1.
        degres (int, angle): the rotate angle

    """
    
    def __init__(self, configs):

        self.probability =  getattr(configs, 'rotate_prob', 0.5)
        self.degree =  getattr(configs, 'degree', 10)

    def __call__(self, seqs):
        if random.uniform(0, 1) >= self.probability:
            return seqs
        else:
            agl = random.uniform(-self.degree, self.degree)
            seqs = (seqs*255.0).astype('uint8')
            seqs = [Image.fromarray(seqs[i, ...], mode='L').rotate(agl) for i in range(seqs.shape[0])]
            seqs = np.concatenate([np.array(img)[np.newaxis, ...] for img in seqs], 0).astype('float32')/255.0
            return seqs