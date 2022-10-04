import random

import numpy as np
import skimage.color as sc

import torch
from torchvision import transforms


def get_patch(*args, patch_size=96, scale=1):
    ih, iw = args[0].shape[:2]
    tp = int(scale* patch_size)
    ip = patch_size
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = int(scale * ix), int(scale * iy)
    if args[0].ndim == 2:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip],
            *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]
        ]
    else:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip, :],
            *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
        ]
    return ret


def get_patch_y_side(*args, patch_size=96, scale=1):
    ih, iw = args[0].shape[:2]
    tp = int(scale * patch_size)
    ip = patch_size
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = int(scale * ix), iy
    if args[0].ndim == 2:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip],
            *[a[iy:iy + ip, tx:tx + tp] for a in args[1:]]
        ]
    else:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip, :],
            *[a[iy:iy + ip, tx:tx + tp, :] for a in args[1:]]
        ]
    return ret

def get_patch_x_side(*args, patch_size=96, scale=1):
    ih, iw = args[0].shape[-2], args[0].shape[-1]
    # print(args[0].shape)
    ip = patch_size
    # print(ih,iw,ip, args[0].shape, args[1].shape)
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - 200 + 1)
    # iy = random.randrange(0, ih - ip + 1)
    ret = [
        args[0][:,:, ix:ix + ip],
        *[a[:,:, ix:ix + ip] for a in args[1:]]
    ]
    return ret



def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)


        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img)
        tensor = torch.from_numpy(np_transpose).float()
        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        return img

    return [_augment(a) for a in args]

