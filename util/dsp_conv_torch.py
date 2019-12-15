#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  ASLP@NPU    Ke Wang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import sys
sys.path.append("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/")
import torchaudio
import torch
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from config.wavenetConfig import Config
sys.path.append(os.path.dirname(sys.path[0]))
from models.feature import STFT,iSTFT
import logging

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps
MAX_EXP = np.log(np.finfo(np.float32).max - 10.0)

hop_length = int(Config.sample_rate * Config.stft_frame_shift / 1000)
# The greater sample_rate, the greater hop_length
win_length = int(Config.sample_rate * Config.stft_frame_length / 1000)

module_stft = STFT(frame_hop=hop_length,frame_len=win_length)
module_istft = iSTFT(frame_hop=hop_length,frame_len=win_length)

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

def seperate_magnitude(magnitude,phase):
    real = torch.cos(phase) * magnitude
    imagine = torch.sin(phase) * magnitude
    expand_dim = len(list(real.size()))
    return torch.stack((real,imagine),expand_dim)

def stft(signal,
         sample_rate = Config.sample_rate,
         ):
    m,p = module_stft.forward(signal)
    feat = seperate_magnitude(m,p)
    return feat

def angle(complex_tensor):
    return torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])

def istft(spectrum,sample_rate = Config.sample_rate,):
    phase = angle(spectrum)
    magnitude = spectrom2magnitude(spectrum)
    wav = module_istft.forward(m=magnitude,p=phase)
    return wav

def spectrom2magnitude(spectrom):
    real = spectrom[:,:,:,0]
    imagine = spectrom[:,:,:,1]
    res = torch.sqrt(real**2+imagine**2)
    return res

def plot2wav(a,b):
    plt.figure(figsize=(20,4))
    plt.subplot(211)
    plt.ylim(-9000,9000)
    plt.plot(a)
    plt.subplot(212)
    plt.ylim(-9000, 9000)
    plt.plot(b)
    plt.savefig("temp.png")

if __name__ == "__main__":
    pass
