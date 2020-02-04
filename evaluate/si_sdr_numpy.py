#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
EPS = 1e-7

def remove_dc(signal):
    """Normalized to zero mean"""
    mean = np.mean(signal)
    signal -= mean
    return signal


# def sdr(estimated,original):
#     noise = estimated - original
#     return 10 * np.log10(pow_np_norm(original) / pow_np_norm(noise))

def pow_norm(s1, s2):
    return np.sum(s1 * s2)

def plot3wav(a,b,c):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(30,6))
    plt.subplot(211)
    plt.plot(a,linewidth = 1)
    plt.subplot(212)
    plt.plot(b,linewidth = 1)
    # plt.subplot(313)
    plt.plot(c,linewidth = 1)
    plt.savefig("com.png")

def plot2wav(a,b):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,4))
    plt.subplot(211)
    plt.plot(a)
    plt.subplot(212)
    plt.plot(b)
    plt.savefig("temp.png")

def pow_np_norm(signal):
    """Compute 2 Norm"""
    return np.square(np.linalg.norm(signal, ord=2))

def si_sdr(estimated, original):
    # estimated = remove_dc(estimated.astype(np.float64))
    # original = remove_dc(original.astype(np.float64))
    estimated,original = estimated.astype(np.float64),original.astype(np.float64)
    target = pow_norm(estimated, original) * original / pow_np_norm(original)
    noise = estimated - target
    return 10 * np.log10(pow_np_norm(target) / pow_np_norm(noise))

def unify(source,target):
    source_max = np.max(np.abs(source))
    target_max = np.max(np.abs(target))
    source = source.astype(np.float32)/source_max
    return (source*target_max).astype(np.int16),target

def sdr(estimated, original):
    estimated, original = unify(estimated, original)
    # target = pow_norm(estimated, original) * original / pow_np_norm(original)
    estimated, original = estimated.astype(np.float64), original.astype(np.float64)
    # original = remove_dc(original)
    # estimated = remove_dc(estimated)
    noise = estimated - original
    return 10 * np.log10(pow_np_norm(original) / pow_np_norm(noise))

def permute_si_sdr(e1, e2, c1, c2):
    sdr1 = si_sdr(e1, c1) + si_sdr(e2, c2)
    sdr2 = si_sdr(e1, c2) + si_sdr(e2, c1)
    if sdr1 > sdr2:
        return sdr1 * 0.5
    else:
        return sdr2 * 0.5

def permute_si_sdr_single(e1, c1):
    sdr1 = si_sdr(e1, c1)
    return sdr1

if __name__ == "__main__":
    input1 = np.zeros(shape=(2,3000))
    input2 = np.zeros(shape=(2,3000))
    # input2 = np.random.randn((2000))*2000
    print(si_sdr(input2,input1))