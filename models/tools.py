"""
Reference: https://github.com/facebookresearch/denoiser/blob/main/denoiser/demucs.py

Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
author: adefossez
"""

import functools
import logging

import math

import torch as th
from torch.nn import functional as F


logger = logging.getLogger(__name__)


def sinc(t):
    """sinc.
    :param t: the input tensor
    """
    return th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype), th.sin(t) / t)


def kernel_upsample2(zeros=56):
    """kernel_upsample2.
    """
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    *other, time = x.shape # [32, 1, 32085]
    kernel = kernel_upsample2(zeros).to(x) # [1, 1, 112]
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time) # [32, 1, 32085]
    y = th.stack([x, out], dim=-1) # [32, 1, 32085, 2]
    return y.view(*other, -1)


def kernel_downsample2(zeros=56):
    """kernel_downsample2.
    """
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time)
    return out.view(*other, -1).mul(0.5)



def capture_init(init):
    """capture_init.
    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__



def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    is_mps = x.device.type == 'mps'
    if is_mps:
        x = x.cpu()
    z = th.stft(x,
                n_fft * (1 + pad),
                hop_length or n_fft // 4,
                window=th.hann_window(n_fft).to(x),
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect')
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    is_mps = z.device.type == 'mps'
    if is_mps:
        z = z.cpu()
    x = th.istft(z,
                 n_fft,
                 hop_length,
                 window=th.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    _, length = x.shape
    return x.view(*other, length)
