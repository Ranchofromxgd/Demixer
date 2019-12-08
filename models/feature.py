#!/usr/bin/env python
import sys
sys.path.append("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/")
# wujian@2019
"""
Implementation of front-end feature via PyTorch
"""

import math
import torch as th

from collections.abc import Sequence

import torch.nn.functional as F
import torch.nn as nn

EPSILON = th.finfo(th.float32).eps
MATH_PI = math.pi


def init_kernel(frame_len,
                frame_hop,
                normalize=True,
                round_pow_of_two=True,
                window="sqrt_hann"):
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    # FFT points
    N = 2**math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    # window
    W = th.hann_window(frame_len)**0.5
    # scale factor to make same magnitude after iSTFT
    if normalize:
        S = 0.5 * (N * N / frame_hop)**0.5
    else:
        S = 1
    # F x N/2+1 x 2
    K = th.rfft(th.eye(N) / S, 1)[:frame_len]
    # 2 x N/2+1 x F
    K = th.transpose(K, 0, 2) * W
    # N+2 x 1 x F
    K = th.reshape(K, (N + 2, 1, frame_len))
    return K


class STFTBase(nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """
    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 normalize=True,
                 round_pow_of_two=True):
        super(STFTBase, self).__init__()
        K = init_kernel(frame_len,
                        frame_hop,
                        normalize=normalize,
                        round_pow_of_two=round_pow_of_two,
                        window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window
        self.normalize = normalize
        self.num_bins = self.K.shape[0] // 2

    def extra_repr(self):
        return (f"window={self.window}, stride={self.stride}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}, " +
                f"normalize={self.normalize}")


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x, cplx=False):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, N x C x S or N x S
        return
            m: magnitude, N x C x F x T or N x F x T
            p: phase, N x C x F x T or N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__class__.__name__, x.dim()))
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = th.unsqueeze(x, 1)
            # N x 2F x T
            c = F.conv1d(x, self.K, stride=self.stride, padding=0)
            # N x F x T
            r, i = th.chunk(c, 2, dim=1)
        # else reshape NC x 1 x S
        else:
            N, C, S = x.shape
            x = x.view(N * C, 1, S)
            # NC x 2F x T
            c = F.conv1d(x, self.K, stride=self.stride, padding=0)
            # N x C x 2F x T
            c = c.view(N, C, -1, c.shape[-1])
            # N x C x F x T
            r, i = th.chunk(c, 2, dim=2)
        if cplx:
            return r, i
        m = (r**2 + i**2)**0.5
        p = th.atan2(i, r)
        return m, p


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, cplx=False, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        args
            m, p: N x F x T
        return
            s: N x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = th.unsqueeze(p, 0)
            m = th.unsqueeze(m, 0)
        if cplx:
            # N x 2F x T
            c = th.cat([m, p], dim=1)
        else:
            r = m * th.cos(p)
            i = m * th.sin(p)
            # N x 2F x T
            c = th.cat([r, i], dim=1)
        # N x 2F x T
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        # N x S
        s = s.squeeze(1)
        if squeeze:
            s = th.squeeze(s)
        return s

if __name__ == "__main__":
    fname = "/home/disk2/internship_anytime/liuhaohe/datasets/musdb18hq/test/test_0/mixed.wav"
    from util.wave_util import WaveHandler
    import matplotlib.pyplot as plt

    wh = WaveHandler()
    hop_length = int(44100 * 8 / 1000)
    win_length = int(44100 * 32 / 1000)
    print(hop_length,win_length)
    stft = STFT(frame_len=win_length,frame_hop=hop_length)
    istft = iSTFT(frame_len=win_length,frame_hop=hop_length)
    data = th.Tensor(wh.read_wave(fname)).unsqueeze(0)
    print(data.size())
    m,p = stft.forward(data)
    wave = istft.forward(m,p)
    diff = th.sum(th.abs(wave-data))/th.sum(data)
    print(diff)
