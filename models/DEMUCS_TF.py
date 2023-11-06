"""
Copyright (c) 2023, JaeBinCHA7
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file at https://github.com/JaeBinCHA7?tab=repositories (if applicable).
"""

import math
import torch
import torch as th
from torch import nn
from torch.nn import functional as F
import typing as tp
from einops import rearrange
from .tools import capture_init, spectro, ispectro


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim, batch_first=True)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen."""
    x0 = x
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            x = F.pad(x, (extra_pad_left, extra_pad_right))
    out = F.pad(x, paddings, mode, value)
    assert out.shape[-1] == length + padding_left + padding_right
    assert (out[..., padding_left: padding_left + length] == x0).all()
    return out


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


class DEMUCS_TF(nn.Module):
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
                 audio_channels=1,
                 channels=48,
                 growth=2,
                 nfft=512,
                 end_iters=0,
                 wiener_residual=False,
                 cac=True,
                 depth=5,
                 hybrid=True,
                 hybrid_old=False,
                 kernel_size=8,
                 stride=2,
                 causal=True,
                 context=1,
                 rescale=0.1,
                 segment=4 * 10,
                 glu=True,
                 max_hidden=10_000,
                 ):

        """
        Args:
            sources (list[str]): list of source names.
            audio_channels (int): input/output audio channels.
            channels (int): initial number of hidden channels.
            channels_time: if not None, use a different `channels` value for the time branch.
            growth: increase the number of hidden channels by this factor at each layer.
            nfft: number of fft bins. Note that changing this require careful computation of
                various shape parameters and will not work out of the box for hybrid models.
            wiener_iters: when using Wiener filtering, number of iterations at test time.
            end_iters: same but at train time. For a hybrid model, must be equal to `wiener_iters`.
            wiener_residual: add residual source before wiener filtering.
            cac: uses complex as channels, i.e. complex numbers are 2 channels each
                in input and output. no further processing is done before ISTFT.
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            hybrid (bool): make a hybrid time/frequency domain, otherwise frequency only.
            hybrid_old: some models trained for MDX had a padding bug. This replicates
                this bug to avoid retraining them.
            multi_freqs: list of frequency ratios for splitting frequency bands with `MultiWrap`.
            multi_freqs_depth: how many layers to wrap with `MultiWrap`. Only the outermost
                layers will be wrapped.
            freq_emb: add frequency embedding after the first frequency layer if > 0,
                the actual value controls the weight of the embedding.
            emb_scale: equivalent to scaling the embedding learning rate
            emb_smooth: initialize the embedding with a smooth one (with respect to frequencies).
            kernel_size: kernel_size for encoder and decoder layers.
            stride: stride for encoder and decoder layers.
            time_stride: stride for the final time layer, after the merge.
            context: context for 1x1 conv in the decoder.
            context_enc: context for 1x1 conv in the encoder.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            dconv_attn: adds attention layers in DConv branch starting at this layer.
            dconv_lstm: adds a LSTM layer in DConv branch starting at this layer.
            dconv_init: initial scale for the DConv branch LayerScale.
            rescale: weight recaling trick

        """
        super().__init__()
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.causal = causal
        self.depth = depth
        self.channels = channels
        self.segment = segment

        self.chin = chin
        self.chout = chout
        self.hidden = hidden

        self.nfft = nfft
        self.hop_length = nfft // 4
        self.end_iters = end_iters
        self.freq_emb = None
        self.hybrid = hybrid
        self.hybrid_old = hybrid_old

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        freqs = nfft // 2

        for index in range(depth):
            freq = freqs > 1
            pad = True
            if pad:
                self.pad = (self.kernel_size // self.stride) - 1
            else:
                self.pad = 0
            activation = nn.GLU(1) if glu else nn.ReLU()
            ch_scale = 2 if glu else 1

            if freq:
                kernel_size = [self.kernel_size, 1]
                stride = [self.stride, 1]
                pad = [self.pad, 0]

            encode = []
            encode += [
                nn.Conv2d(chin, hidden, kernel_size, stride, pad),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv2d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose2d(hidden, chout, kernel_size, stride, pad),
            ]

            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))

            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        if self.hybrid:
            # We re-pad the signal in order to keep the property
            # that the size of the output is exactly the size of the input
            # divided by the stride (here hop_length), when divisible.
            # This is achieved by padding by 1/4th of the kernel size (here nfft).
            # which is not supported by torch.stft.
            # Having all convolution operations follow this convention allow to easily
            # align the time and frequency branches later on.
            assert hl == nfft // 4
            le = int(math.ceil(x.shape[-1] / hl))
            pad = hl // 2 * 3
            if not self.hybrid_old:
                x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode='reflect')
            else:
                x = pad1d(x, (pad, pad + le * hl - x.shape[-1]))

        z = spectro(x, nfft, hl)[..., :-1, :]
        if self.hybrid:
            assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
            z = z[..., 2:2 + le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4 ** scale)
        # z = F.pad(z, (0, 0, 0, 1))
        if self.hybrid:
            z = F.pad(z, (2, 2))
            pad = hl // 2 * 3
            if not self.hybrid_old:
                le = hl * int(math.ceil(length / hl)) + 2 * pad
            else:
                le = hl * int(math.ceil(length / hl))
            x = ispectro(z, hl, length=le)
            if not self.hybrid_old:
                x = x[..., pad:pad + length]
            else:
                x = x[..., :length]
        else:
            x = ispectro(z, hl, length)
        return x

    def _magnitude(self, z):
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def forward(self, mix):
        x = mix
        length = x.shape[-1]

        z = self._spec(mix)
        z = z.unsqueeze(1)
        complex = self._magnitude(z).to(mix.device)

        real = complex[:, :1, :, :]
        imag = complex[:, 1:, :, :]

        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase = torch.atan2(imag, real)

        x = mag
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        saved = []  # skip connections, freq.
        lengths = []  # saved lengths to properly remove padding, freq branch.

        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            x = encode(x)

            saved.append(x)

        t, f = x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c t f -> b (t f) c')
        x, _ = self.lstm(x)
        x = rearrange(x, 'b (t f) c -> b c t f', t=t, f=f)

        for decode in self.decoder:
            skip = saved.pop(-1)
            x = x + skip
            x = decode(x)

        assert len(saved) == 0

        x = x * std + mean

        x_is_mps = x.device.type == "mps"
        if x_is_mps:
            x = x.cpu()
        mask_mags = x
        mask_mags = torch.tanh(mask_mags)

        real_out = mask_mags * torch.cos(phase)
        imag_out = mask_mags * torch.sin(phase)

        z = torch.complex(real_out, imag_out)
        x = self._ispec(z, length)

        x = x[..., :length]

        # back to mps device
        if x_is_mps:
            x = x.to('mps')

        return x