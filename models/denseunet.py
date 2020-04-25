# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   denseunet.py
@Contact :   liu.8948@buckeyemail.osu.edu
@License :   (C)Copyright 2020-2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/4/1 8:01 PM   Haohe Liu      1.0         None
'''


import sys
sys.path.append("..")
from models.dense_net import _DenseBlock
import torch.nn.functional as F
import torch
import torch.nn as nn


class DenseUnet(nn.Module):
    def __init__(self, n_channels_in,
                 n_channel_out,
                 block_number = 4,
                 first_channel = 64,
                 denselayers = 4,
                 bn_size=2,
                 growth_rate=32,
                 bilinear=True,
                 dropout = 0.2,
                 ):
        '''
        :param n_channels_in: input channel of Denseunet, input shape: [batchsize, channel, Freq_bin, Time_step]
        :param n_channel_out: output channel numbers
        :param block_number: Number of DenseBlocks in down path or up path
        :param first_channel: Output channels of the first two convolution (for embedding)
        :param denselayers: number of denselayers in a denseblock
        :param bn_size:  pass
        :param growth_rate: pass
        :param bilinear: pass
        :param dropout: pass
        '''
        super(DenseUnet, self).__init__()
        self.channel_in = n_channels_in
        self.channel_out = n_channel_out
        self.block_number = block_number
        self.first_channel = first_channel
        self.denselayers = denselayers
        self.bn_size = bn_size
        self.growth_rate = growth_rate
        self.bilinear = bilinear
        self.dropout = dropout
        self.cnt = 0
        self.inc = DoubleConv(n_channels_in, self.first_channel)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        def get_ch(ch1,ch2 = None):
            if(ch1 == 0):
                return self.first_channel
            increase_rate = self.denselayers*self.growth_rate # 36
            if(ch2 is None):return self.first_channel+increase_rate*ch1
            else: return get_ch(ch1)+get_ch(ch2)

        for i in range(0,block_number):
            self.down.append(Down(get_ch(i),
                                  denselayers=self.denselayers,
                                  bn_size=self.bn_size,
                                  growth_rate=self.growth_rate,
                                  dropout=self.dropout,
                                  ))

        last_layer_ch_out = get_ch(self.block_number)

        for i in range(self.block_number):
            out_ch = get_ch(self.block_number-i) # out_ch == concact_ch
            self.up.append(Up(last_layer_ch_out,
                              out_ch,
                              denselayers=self.denselayers,
                              bn_size=self.bn_size,
                              growth_rate=self.growth_rate,
                              dropout=self.dropout,
                              ))
            last_layer_ch_out = out_ch
        self.outc = OutConv(last_layer_ch_out, n_channel_out)

    def forward(self, x):
        out_down = []
        x = self.inc(x)
        for i in range(self.block_number):
            x,x_before_pooling = self.down[i](x)
            out_down.append(x_before_pooling)
        for i in range(self.block_number):
            x = self.up[i](x,out_down[-i-1])
        x = self.outc(x)
        return x

    def get_parameter_number(self,net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        # return {'Total': total_num, 'Trainable': trainable_num}
        return total_num

    def get_param_num(self):
        num = 0
        num += self.get_parameter_number(self.inc)
        for each in self.down:
            num+=self.get_parameter_number(each)
        for each in self.up:
            num+=self.get_parameter_number(each)
        num += self.get_parameter_number(self.outc)
        return num

    def print_model(self):
        '''
        :return: print the predefined DenseUnet model
        '''
        print(self.inc)
        for cnt,each in enumerate(self.down):
            print(cnt,each)
        for cnt,each in enumerate(self.up):
            print(cnt,each)
        print(self.outc)




class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self,
                 in_channels,
                 denselayers = 4,
                 bn_size = 2,
                 growth_rate = 32,
                 dropout = 0.2,

    ):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2),
        )
        self.dense_conv = _DenseBlock(num_layers=denselayers,
                        num_input_features=in_channels,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        drop_rate=dropout)
    def forward(self, x):
        x = self.dense_conv(x)
        return self.maxpool(x),x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,
                 in_channels,
                 trans_channels,
                 bilinear=False,
                 denselayers=4,
                 bn_size=2,
                 growth_rate=32,
                 dropout=0.2,
                 ):
        '''

        :param in_channels: input channel of DenseBlock
        :param trans_channels: the output of Upsampling layer, also can be used as concatenated feature's channel
        :param bilinear: upsample or convTranspose
        :param denselayers: number of layers in a denseblock
        :param bn_size: pass
        :param growth_rate: pass
        :param dropout: pass
        '''
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = _DenseBlock(num_layers=denselayers,
                        num_input_features=in_channels,
                        concact_input_features=trans_channels,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        drop_rate=dropout)
        self.conv1_1 = nn.Conv2d(in_channels+trans_channels+denselayers*growth_rate,trans_channels,kernel_size=(1,1))

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        return self.conv1_1(self.conv(x1,x2))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.outConv = nn.Sequential(
            # nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.outConv(x)

if __name__ == "__main__":
    model = DenseUnet(n_channels_in=8, n_channel_out=8, growth_rate=32, denselayers=3, bn_size=2, block_number=5)
    # data = torch.randn(size=(1,2,300,200))
    # print(model.forward(data).size())
    # model.print_model()
    print(model)
    print(model.get_param_num())
