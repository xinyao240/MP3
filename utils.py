""" utils.py
"""

import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time
import warnings

name_dataparallel = torch.nn.DataParallel.__name__
log10 = np.log(10)


def compute_psnr(x, label, max_diff):
    assert max_diff in [255, 1, 2]
    if max_diff == 255:
        x = x.clamp(0, 255)
    elif max_diff == 1:
        x = x.clamp(0, 1)
    elif max_diff == 2:
        x = x.clamp(-1, 1)

    mse = ((x - label) ** 2).mean()
    return 10 * torch.log(max_diff ** 2 / mse) / log10


def lr_warmup(epoch, warmup_length):
    if epoch < warmup_length:
        p = max(0.0, float(epoch)) / float(warmup_length)
        p = 1.0 - p
        return np.exp(-p * p * 5.0)
    else:
        return 1.0


def load_optimizer(optimizer, model, path, epoch=None):
    """
    return the epoch
    """
    if type(model).__name__ == name_dataparallel:
        model = model.module

    if epoch is None:
        for i in reversed(range(10000)):
            p = "{}/{}_epoch{}.pth".format(path, type(optimizer).__name__ + '_' + type(model).__name__, i)
            if os.path.exists(p):
                optimizer.load_state_dict(torch.load(p))
                return i
    else:
        p = "{}/{}_epoch{}.pth".format(path, type(optimizer).__name__ + '_' + type(model).__name__, epoch)
        if os.path.exists(p):
            optimizer.load_state_dict(torch.load(p))
            return epoch
        else:
            warnings.warn("resume optimizer not found at {}".format(p))

    warnings.warn("resume model not found ")
    return -1


def load_model(model, path, epoch=None, strict=True):
    """
    return the last epoch
    """
    if type(model).__name__ == name_dataparallel:
        model = model.module
    if epoch is None:
        for i in reversed(range(10000)):
            p = "{}/{}_epoch{}.pth".format(path, type(model).__name__, i)
            if os.path.exists(p):
                model.load_state_dict(torch.load(p), strict=strict)
                return i
    else:
        p = "{}/{}_epoch{}.pth".format(path, type(model).__name__, epoch)
        if os.path.exists(p):
            model.load_state_dict(torch.load(p), strict=strict)
            return epoch
        else:
            warnings.warn("resume model not found at {}".format(p))

    warnings.warn("resume model not found ")
    return -1


def set_requires_grad(module, b):
    for parm in module.parameters():
        parm.requires_grad = b


def adjust_dyn_range(x, drange_in, drange_out):
    if not drange_in == drange_out:
        scale = float(drange_out[1] - drange_out[0]) / float(drange_in[1] - drange_in[0])
        bias = drange_out[0] - drange_in[0] * scale
        x = x.mul(scale).add(bias)
    return x


def resize(x, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(size),
        transforms.ToTensor(),
    ])
    return transform(x)


def save_model(model, dirname, epoch):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    torch.save(model.state_dict(), '{}/{}_epoch{}.pth'.format(dirname, type(model).__name__, epoch))


def save_optimizer(optimizer, model, dirname, epoch):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    torch.save(optimizer.state_dict(),
               '{}/{}_epoch{}.pth'.format(dirname, type(optimizer).__name__ + '_' + type(model).__name__, epoch))


