#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.modules import Conv1d, Conv1dBlock, ConvTranspose1d, normalization, ConvTranspose1dPReLUBN

sys.path.append('/home/work_nfs/hhliu/workspace/github/wavenet-aslp/')
from config.wavenetConfig import Config
from models.show import show_model, show_params
from evaluate.si_sdr_torch import permute_si_sdr

class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)

class TasNet(nn.Module):
    def __init__(self, inchannels = 3,outchannels = 2,n_layers=9, channels_interval=24):
        super(TasNet, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [inchannels] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]
        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(inchannels + self.channels_interval, outchannels, kernel_size=1, stride=1),
            nn.Tanh()
        )
        # show_model(self)
        # show_params(self)

    def get_params(self, weight_decay):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params

    def forward(self, input):
        # print("input:",input.shape)
        input = input.float()
        # batch_size, channels, dim = input.shape
        tmp = []
        # input = input.reshape((batch_size, 1, dim))
        # input = input.float()
        o = input
        # print(o.shape)
        # Up Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        o = self.middle(o)
        # Down Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            
            if o.shape != tmp[self.n_layers - i - 1].shape:
                
                # print(o.shape)
                # print(tmp[self.n_layers - i - 1].shape)
                o = o[:,:,:tmp[self.n_layers - i - 1].shape[2]]
                # print(o.shape)
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)
            # print("decode:",o.shape)
        o = torch.cat([o, input], dim=1)
        # print("final:",o.shape)
        o = self.out(o)
        # print("output:",o.shape)
        return o

    def calloss(self, output, source, device):
        source = source.reshape(source.shape[0], 2, -1)
        # source = source.float()
        source = source.float()
        output = output.reshape(output.shape[0], 2, -1)
        output = output.float()
        # print(output.shape)
        # print(source.shape)
        loss = permute_si_sdr(output, source, device)
        return loss

if __name__ == "__main__":
    model = torch.load("/home/work_nfs/hhliu/workspace/github/wavenet-aslp/saved_models/model34000.pkl",map_location='cpu')
    test_path = "/home/work_nfs3/yhfu/dataset/musdb18hq/test/test_0/"
    from util.wave import WaveHandler
    import numpy as np
    import matplotlib.pyplot as plt

    wav = WaveHandler()
    vocals = wav.read_wave(test_path+"vocals.wav")
    mixed = wav.read_wave(test_path+"mixed.wav")
    maxVal = max(np.max(np.abs(vocals)), np.max(np.abs(mixed)))
    vocals,mixed = vocals/maxVal,mixed/maxVal
    song = torch.Tensor((vocals+mixed)).unsqueeze(0)
    song = song.unsqueeze(1)
    output = (30000*model.forward(song[:,:,:])).detach().numpy().astype(np.int16)
    wav.save_wave(output,"out.wav",channels=1)