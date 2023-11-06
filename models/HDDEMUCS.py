"""
Copyright (c) 2023, JaeBinCHA7
All rights reserved.

This source code is created based on the implementation of ideas presented in the paper:
Kim, Doyeon, et al. "HD-DEMUCS: General Speech Restoration with Heterogeneous Decoders." arXiv preprint arXiv:2306.01411 (2023).
Available at: https://arxiv.org/abs/2306.01411

This source code is licensed under the MIT license found in the
LICENSE file at https://github.com/JaeBinCHA7?tab=repositories (if applicable).
"""

import math

import torch as th
from torch import nn
from torch.nn import functional as F
from .tools import downsample2, upsample2, capture_init


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class HDDEMUCS(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.
    """

    @capture_init
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 sample_rate=16_000):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        self.encoder = nn.ModuleList()
        self.decoder_map = nn.ModuleList()
        self.decoder_mask = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1
        dilation_factor = [1, 3, 5, 7, 9]
        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            # Suppression block
            decode_mask = []
            decode_mask += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                # decode_mask.append(nn.ReLU()) # Original DEMUCS
                decode_mask.append(nn.Sigmoid())  # HD-DEMUCS
            self.decoder_mask.insert(0, nn.Sequential(*decode_mask))

            # refinement block
            decode_map = []
            decode_map += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                # nn.ConvTranspose1d(hidden, chout, kernel_size, stride),  # Original DEMUCS
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride, dilation=dilation_factor[index],
                                   padding=7 * index)  # HD-DEMUCS
            ]
            if index > 0:
                decode_map.append(nn.ReLU())
            self.decoder_map.insert(0, nn.Sequential(*decode_map))

            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

        # Fusion block
        self.fb_conv1 = nn.Sequential()
        self.fb_conv1.append(nn.Conv1d(2, 2, 3, 1, padding=1))
        self.fb_conv1.append(nn.LeakyReLU())

        self.fb_conv2 = nn.Sequential()
        self.fb_conv2.append(nn.Conv1d(2, 2, 3, 1, padding=1))
        self.fb_conv2.append(nn.LeakyReLU())

        self.fb_conv3 = nn.Sequential()
        self.fb_conv3.append(nn.Conv1d(2, 2, 3, 1, padding=1))
        self.fb_conv3.append(nn.Sigmoid())

        self.weight = nn.Parameter(th.tensor(0.5, requires_grad=True))

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mean = mix.mean(dim=(1, 2), keepdim=True)
            std = mix.std(dim=(1, 2), keepdim=True)
            mix = (mix - mean) / (1e-5 + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        x_us = x
        skips_mask = []
        skips_map = []
        for encode in self.encoder:
            x = encode(x)
            skips_mask.append(x)

        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)

        x_mask = x
        for decode in self.decoder_mask:
            skip = skips_mask.pop(-1)
            x_mask = x_mask + skip
            x_mask = decode(x_mask)
            skips_map.append(x_mask)

        x_map = x
        for decode in self.decoder_map:
            x_map = decode(x_map)
            skip = skips_map.pop(0)
            x_map = x_map + skip

        d_s = x_mask * x_us
        d_r = x_map

        x_fb = th.concat((d_s, d_r), dim=1)

        x_fb = self.fb_conv1(x_fb)
        x_fb = self.fb_conv2(x_fb)
        x_fb = self.fb_conv3(x_fb)

        out = d_s * (1 - self.weight) * x_fb[:, :1, ...] + d_r * self.weight * x_fb[:, 1:, ...]

        if self.resample == 2:
            out = downsample2(out)

        elif self.resample == 4:
            out = downsample2(out)
            out = downsample2(out)

        out = out[..., :length]

        return std * out + mean