#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  ASLP@NPU    Ke Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
sys.path.append("..")
import torchaudio
import torch
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from config.wavenetConfig import Config
sys.path.append(os.path.dirname(sys.path[0]))

import logging

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps
MAX_EXP = np.log(np.finfo(np.float32).max - 10.0)

class logger(object):
    r""" Wrapping for logging
    Expample:
    >>> log = logger(sys.argv[0], 'debug')
    >>> log.show('This is debug level', 'debug')
    >>> log.show('This is info level', 'info')
    """

    def __init__(self, name, level):
        self.logger = logging.getLogger(name)
        self.handler = logging.StreamHandler()
        self.formatter = logging.Formatter(
            "%(asctime)s [%(name)s:%(lineno)s - %(levelname)s ] %(message)s")
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(self._init_level(level))

    def _init_level(self, level):
        levels = {
            'notset': logging.NOTSET,
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.INFO,
            'error': logging.ERROR,
            'critical': logging.CRITICAL,
        }
        return levels[level]

    def _show_level(self, level):
        levels = {
            'debug': self.logger.debug,
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical,
        }
        return levels[level]

    def show(self, msg, level):
        self._show_level(level)(msg)

log = logger(__file__, 'debug')


def wavread(path):
    #wav, sample_rate = sf.read(path, dtype='float32')
    wav, sample_rate = sf.read(path)
    return wav, sample_rate


def wavwrite(signal, sample_rate, path):
    signal = (signal * MAX_INT16).astype(np.int16)
    wavfile.write(path, sample_rate, signal)


def get_window(window_size, window_type, square_root_window=True):
    """Return the window"""
    window = {
        'hamming': torch.hamming_window(window_size),
        'hanning': torch.hann_window(window_size),
    }[window_type]
    if square_root_window:
        window = torch.sqrt(window)
    return window


def fft_point(dim):
    assert dim > 0
    num = math.log(dim, 2)
    num_point = 2**(math.ceil(num))
    return num_point


def pre_emphasis(signal, coefficient=0.97):
    """Pre-emphasis original signal
    y(n) = x(n) - a*x(n-1)
    """
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])


def de_emphasis(signal, coefficient=0.97):
    """De-emphasis original signal
    y(n) = x(n) + a*x(n-1)
    """
    length = signal.shape[0]
    for i in range(1, length):
        signal[i] = signal[i] + coefficient * signal[i - 1]
    return signal

def seperate_magnitude(magnitude,phase):
    real = torch.cos(phase) * magnitude
    imagine = torch.sin(phase) * magnitude
    expand_dim = len(list(real.size()))
    return torch.stack((real,imagine),expand_dim)

def stft(signal,
         sample_rate,
         frame_length=32,
         frame_shift=8,
         window_type="hanning",
         use_gpu = False,
         preemphasis=0.0,
         square_root_window=True):
    """Compute the Short Time Fourier Transform.

    Args:
        signal: input speech signal叠
        sample_rate: waveform data sample frequency (Hz)
        frame_length: frame length in milliseconds
        frame_shift: frame shift in milliseconds
        window_type: type of window
        square_root_window: square root window
    Return:
        fft: (n/2)+1 dim complex STFT restults
    """
    # if preemphasis != 0.0:
    #     signal = pre_emphasis(signal, preemphasis)
    # Frame lengh is greater and frame shift is smaller, because there should be a overlap between frames
    hop_length = int(sample_rate * frame_shift / 1000) # The greater sample_rate, the greater hop_length
    win_length = int(sample_rate * frame_length / 1000)
    num_point = fft_point(win_length)
    window = get_window(num_point, window_type, square_root_window)
    if(use_gpu == True):
        window = window.cuda(Config.device)
    # print(num_point,hop_length,win_length,window)#  2048 352 1411 [0.         0.00153473 0.00306946 ... 0.00306946 0.00153473 0.        ]
    # print(window.shape) # (20048,)
    # feat = librosa.stft(signal, n_fft=num_point, hop_length=hop_length,
    #                     win_length=win_length, window=window)
    feat = torch.stft(signal, n_fft=num_point, hop_length=hop_length,
                        win_length=window.shape[0], window=window)
    # if(feat.size()[0] == Config.batch_size):feat = feat.transpose(1,2)
    # else:feat = feat.transpose(0,1)
    return feat


def get_phase(signal,
              sample_rate,
              frame_length=32,
              frame_shift=8,
              window_type="hanning",
              preemphasis=0.0,
              square_root_window=True):
    """Compute phase imformation.

    Args:
        signal: input speech signal
        sample_rate: waveform data sample frequency (Hz)
        frame_length: frame length in milliseconds
        frame_shift: frame shift in milliseconds
        window_type: type of window
        square_root_window: square root window
    """
    feat = stft(signal, sample_rate, frame_length, frame_shift,
                window_type, preemphasis, square_root_window)
    phase = np.angle(feat)
    return phase


def istft(spectrum,
                    sample_rate,
                    frame_length=32,
                    frame_shift=8,
                    window_type="hanning",
                    preemphasis=0.0,
                    use_gpu = False,
                    square_root_window=True):
    """Convert frames to signal using overlap-and-add systhesis.
    Args:
        spectrum: magnitude spectrum [batchsize,x,y,2]
        signal: wave signal to supply phase information
    Return:
        wav: synthesied output waveform
    """
    # phase = get_phase(signal, sample_rate, frame_length, frame_shift,
    #                   window_type, preemphasis, square_root_window)
    # spectrum = np.transpose(spectrum)
    hop_length = int(sample_rate * frame_shift / 1000)
    win_length = int(sample_rate * frame_length / 1000)

    num_point = fft_point(win_length)
    if(use_gpu == True):
        window = get_window(num_point, window_type, square_root_window).cuda(Config.device)
    else:
        window = get_window(num_point, window_type, square_root_window)

    wav = torch_istft(spectrum, num_point,hop_length=hop_length,
                        win_length=window.shape[0], window=window)
    # if preemphasis != 0.0:
    #     wav = de_emphasis(wav, preemphasis)
    return wav

def spectrom2magnitude(spectrom,batchsize):
    if(not spectrom.size()[0] == batchsize):
        raise AssertionError("First dimension should be batchsize")
    real = spectrom[:,:,:,0]
    imagine = spectrom[:,:,:,1]
    res = torch.sqrt(real**2+imagine**2)
    return res

def test1():
    import util.wave_util as wu
    from config.wavenetConfig import Config
    from dataloader import dataloader
    h = wu.WaveHandler()
    dl = torch.utils.data.DataLoader(dataloader.WavenetDataloader(), batch_size=Config.batch_size, shuffle=False,
                                     num_workers=Config.num_workers)
    cnt = 0
    for background, vocal, song in dl:
        f_background = stft(background.float(), Config.sample_rate)
        f_vocal = stft(vocal.float(), Config.sample_rate)
        f_song = f_background+f_vocal
        phase = torchaudio.functional.angle(f_song)
        magnitude = spectrom2magnitude(f_song, Config.batch_size)
        rebuilt_f = seperate_magnitude(magnitude, phase)
        wav = istft(rebuilt_f, sample_rate=Config.sample_rate)
        h.save_wave(wav.numpy().astype(np.int16), fname="temp.wav", channels=1)
        break

def torch_istft(stft_matrix,          # type: Tensor
          n_fft,                # type: int
          hop_length=None,      # type: Optional[int]
          win_length=None,      # type: Optional[int]
          window=None,          # type: Optional[Tensor]
          center=True,          # type: bool
          pad_mode='reflect',   # type: str
          normalized=False,     # type: bool
          onesided=True,        # type: bool
          length=None           # type: Optional[int]
          ):
    # type: (...) -> Tensor

    stft_matrix_dim = stft_matrix.dim()
    assert 3 <= stft_matrix_dim <= 4, ('Incorrect stft dimension: %d' % (stft_matrix_dim))

    if stft_matrix_dim == 3:
        # add a channel dimension
        stft_matrix = stft_matrix.unsqueeze(0)

    dtype = stft_matrix.dtype
    device = stft_matrix.device
    fft_size = stft_matrix.size(1)
    assert (onesided and n_fft // 2 + 1 == fft_size) or (not onesided and n_fft == fft_size), (
        'one_sided implies that n_fft // 2 + 1 == fft_size and not one_sided implies n_fft == fft_size. ' +
        'Given values were onesided: %s, n_fft: %d, fft_size: %d' % ('True' if onesided else False, n_fft, fft_size))

    # use stft defaults for Optionals
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # There must be overlap
    assert 0 < hop_length <= win_length
    assert 0 < win_length <= n_fft

    if window is None:
        window = torch.ones(win_length, requires_grad=False, device=device, dtype=dtype)

    assert window.dim() == 1 and window.size(0) == win_length

    if win_length != n_fft:
        # center window with pad left and right zeros
        left = (n_fft - win_length) // 2
        window = torch.nn.functional.pad(window, (left, n_fft - win_length - left))
        assert window.size(0) == n_fft
    # win_length and n_fft are synonymous from here on

    stft_matrix = stft_matrix.transpose(1, 2)  # size (channel, n_frames, fft_size, 2)
    stft_matrix = torch.irfft(stft_matrix, 1, normalized,
                              onesided, signal_sizes=(n_fft,))  # size (channel, n_frames, n_fft)

    assert stft_matrix.size(2) == n_fft
    n_frames = stft_matrix.size(1)

    ytmp = stft_matrix * window.view(1, 1, n_fft)  # size (channel, n_frames, n_fft)
    # each column of a channel is a frame which needs to be overlap added at the right place
    ytmp = ytmp.transpose(1, 2)  # size (channel, n_fft, n_frames)

    eye = torch.eye(n_fft, requires_grad=False,
                    device=device, dtype=dtype).unsqueeze(1)  # size (n_fft, 1, n_fft)

    # this does overlap add where the frames of ytmp are added such that the i'th frame of
    # ytmp is added starting at i*hop_length in the output
    y = torch.nn.functional.conv_transpose1d(
        ytmp, eye, stride=hop_length, padding=0)  # size (channel, 1, expected_signal_len)

    # do the same for the window function
    window_sq = window.pow(2).view(n_fft, 1).repeat((1, n_frames)).unsqueeze(0)  # size (1, n_fft, n_frames)
    window_envelop = torch.nn.functional.conv_transpose1d(
        window_sq, eye, stride=hop_length, padding=0)  # size (1, 1, expected_signal_len)

    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    assert y.size(2) == expected_signal_len
    assert window_envelop.size(2) == expected_signal_len

    half_n_fft = n_fft // 2
    # we need to trim the front padding away if center
    start = half_n_fft if center else 0
    end = -half_n_fft if length is None else start + length

    y = y[:, :, start:end]
    window_envelop = window_envelop[:, :, start:end]

    # check NOLA non-zero overlap condition
    window_envelop_lowest = window_envelop.abs().min()
    assert window_envelop_lowest > 1e-11, ('window overlap add min: %f' % (window_envelop_lowest))

    y = (y / window_envelop).squeeze(1)  # size (channel, expected_signal_len)

    if stft_matrix_dim == 3:  # remove the channel dimension
        y = y.squeeze(0)
    return y

if __name__ == "__main__":
    pass