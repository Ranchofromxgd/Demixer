#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from config.wavenetConfig import Config
"""
Neural network modules for WaveNet
References :
    https://arxiv.org/pdf/1609.03499.pdf
    https://github.com/ibab/tensorflow-wavenet
    https://qiita.com/MasaEguchi/items/cd5f7e9735a120f27e2a
    https://github.com/musyoku/wavenet/issues/4
"""
import math
import os
import sys

import torch
import numpy as np
sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from evaluate.si_sdr_torch import permute_si_sdr

class DilatedCausalConv1d(torch.nn.Module):
    """Dilated Causal Convolution for WaveNet"""
    def __init__(self, channels, dilation=1):
        super(DilatedCausalConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(channels, channels,
                                    kernel_size=3, stride=1,  # Fixed for WaveNet
                                    dilation=dilation,
                                    padding=dilation,  # Fixed for WaveNet dilation
                                    # bias=False  # Fixed for WaveNet but not sure
                                    )
        if(Config.use_gpu == True):
            self.conv = self.conv
    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        return output


class CausalConv1d(torch.nn.Module):
    """Causal Convolution for WaveNet"""
    def __init__(self, in_channels, out_channels):
        super(CausalConv1d, self).__init__()

        # padding=1 for same size(length) between input and output for causal convolution
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    # bias=False  # Fixed for WaveNet but not sure
                                    )
        if (Config.use_gpu == True):
            self.conv = self.conv.cuda(Config.device)

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        # remove last value for causal convolution
        # return output[:, :, :-1]
        return output


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        """
        Residual block
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :param dilation:
        """
        super(ResidualBlock, self).__init__()

        self.dilated = DilatedCausalConv1d(res_channels, dilation=dilation)
        self.gated = DilatedCausalConv1d(res_channels, dilation=dilation)
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()
        if (Config.use_gpu == True):
            self.conv_res = self.conv_res.cuda()
            self.conv_skip = self.conv_skip.cuda()
            self.gate_tanh = self.gate_tanh.cuda()
            self.gate_sigmoid = self.gate_sigmoid.cuda()

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output1 = self.dilated(x)
        output2 = self.gated(x)
        # PixelCNN gate
        gated_tanh = self.gate_tanh(output1)
        gated_sigmoid = self.gate_sigmoid(output1)
        gated = gated_tanh * gated_sigmoid

        # Residual network
        output = self.conv_res(gated)
        # input_cut = x[:, :, -output.size(2):]
        input_cut = x
        output += input_cut

        # Skip connection
        skip = self.conv_skip(gated)
        # skip = skip[:, :, -skip_size:]

        return output, skip


class ResidualStack(torch.nn.Module):
    def __init__(self, layer_size, stack_size, res_channels, skip_channels):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :return:
        """
        super(ResidualStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size

        self.res_blocks = self.stack_res_block(res_channels, skip_channels)

    @staticmethod
    def _residual_block(res_channels, skip_channels, dilation):
        block = ResidualBlock(res_channels, skip_channels, dilation)

        # if torch.cuda.device_count() > 1:
        #     block = torch.nn.DataParallel(block)
        #
        # if torch.cuda.is_available():
        #     block.cuda()

        return block

    def build_dilations(self):
        dilations = []

        # 5 = stack[layer1, layer2, layer3, layer4, layer5]
        for s in range(0, self.stack_size):
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l in range(0, self.layer_size):
                dilations.append(2 ** l)

        return dilations

    def stack_res_block(self, res_channels, skip_channels):
        """
        Prepare dilated convolution blocks by layer and stack size
        :return:
        """
        res_blocks = []
        dilations = self.build_dilations()

        for dilation in dilations:
            block = self._residual_block(res_channels, skip_channels, dilation)
            res_blocks.append(block)

        return res_blocks

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            # output is the next input
            output, skip = res_block(output, skip_size)
            skip_connections.append(skip)

        return torch.stack(skip_connections)


class DensNet(torch.nn.Module):
    def __init__(self, channels):
        """
        The last network of WaveNet
        :param channels: number of channels for input and output
        :return:
        """
        super(DensNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size = 3, padding  = 1)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size = 3, padding  = 1)
        self.relu = torch.nn.ReLU()
        # self.softmax = torch.nn.Softmax(dim=1)
        self.conv3 = torch.nn.Conv1d(channels, 1, 1)

        if (Config.use_gpu == True):
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()
            self.relu = self.relu.cuda()
            self.conv3 = self.conv3.cuda()

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.conv3(output)
        return output


class TasNet(torch.nn.Module):
    def __init__(self, layer_size=8, stack_size=2, in_channels=3, res_channels=128):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param in_channels: number of channels for input data. skip channel is same as input channel
        :param res_channels: number of residual channel for input, output
        :return:
        """
        super(TasNet, self).__init__()

        self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size)

        self.causal = CausalConv1d(in_channels, res_channels)

        self.res_stack = ResidualStack(layer_size, stack_size, res_channels, res_channels)

        self.densnet = DensNet(res_channels)

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

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        num_receptive_fields = np.sum(layers)

        return int(num_receptive_fields)

    def calc_output_size(self, x):
        output_size = int(x.size(2)) - self.receptive_fields

        return output_size

    def forward(self, x, length=64000):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        x = x.permute(0, 2, 1).float()
        # print("x",x.shape)
        # output = x.transpose(1, 2)
        output = x
        output_size = self.calc_output_size(output)
        # print(output_size)
        # print("receptive_fields",self.receptive_fields)
        output = self.causal(output)
        # print("output",output.shape)
        skip_connections = self.res_stack(output, output_size)
        # print("skip_connections",skip_connections.shape)
        output = torch.sum(skip_connections, dim=0).cuda()

        output = self.densnet(output)
        # output2 = x[:, 1, :,].unsqueeze(dim=1) - output

        # output = torch.stack([output, output2], dim = 1).squeeze(dim=2)
        return output
        # return output.transpose(1, 2).contiguous()

    def calloss(self, output, source, device):
        # print("losssource:",source.shape)
        # print("lossoutput:",output.shape)
        # batch_size = source.shape[0]
        # length = 64000
        # source = source.reshape(batch_size, 2, -1)
        # source = source.float()
        loss = permute_si_sdr(output, source, device)
        return loss





if __name__ == "__main__":
    model = TasNet(layer_size=10,stack_size=2,in_channels=256,res_channels=32)
    print(model)