import os
import sys
import torch
import torch.nn as nn
import numpy as np
import scipy
from util.matutil import load_mat2numpy
from config.mainConfig import Config

class PQMF(nn.Module):
    def __init__(self, N, M, file_path=Config.project_root+"filters/"):
        super().__init__()
        self.N = N #nsubband
        self.M = M #nfilter
        try:
            assert (N,M) in [(8,64),(4,64),(2,64)]
        except:
            print("Warning:",N,"subband and ",M," filter is not supported")

        self.name = str(N)+"_"+str(M)+".mat"
        self.ana_conv_filter = nn.Conv1d(1, out_channels=N, kernel_size=M, stride=N, bias=False)
        data = load_mat2numpy(file_path+"f_"+self.name)
        data = data['f'].astype(np.float32)/N
        data=np.flipud(data.T).T
        data=np.reshape(data, (N, 1, M)).copy()
        dict_new = self.ana_conv_filter.state_dict().copy()
        dict_new['weight'] = torch.from_numpy(data)
        self.ana_pad = nn.ConstantPad1d((M-N,0), 0)
        self.ana_conv_filter.load_state_dict(dict_new)

        self.syn_pad = nn.ConstantPad1d((0, M//N-1), 0)
        self.syn_conv_filter = nn.Conv1d(N, out_channels=N, kernel_size=M//N, stride=1, bias=False)
        gk= load_mat2numpy(file_path+"h_"+self.name)
        gk = gk['h'].astype(np.float32)
        gk = np.transpose(np.reshape(gk, (N, M//N, N)), (1, 0, 2))*N
        gk = np.transpose(gk[::-1, :, :], (2, 1, 0)).copy()
        dict_new = self.syn_conv_filter.state_dict().copy()
        dict_new['weight'] = torch.from_numpy(gk)
        self.syn_conv_filter.load_state_dict(dict_new)

        for param in self.parameters():
            param.requires_grad = False

    def analysis(self, inputs):
        return self.ana_conv_filter(self.ana_pad(inputs))
    def synthesis(self, inputs):
        return self.syn_conv_filter(self.syn_pad(inputs))
        #return self.syn_conv_filter(inputs)
    def forward(self, inputs):
        return self.ana_conv_filter(self.ana_pad(inputs))

# def get_parameter_number(net):
#     total_num = sum(p.numel() for p in net.parameters())
#     trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}
    # return total_num
#
# if __name__ == "__main__":
#     a=PQMF(8, 64)
#     from util.wave_util import WaveHandler
#     wh = WaveHandler()
#     wavfile='/Users/liuhaohe/College/Experiments/music separation/outputs/test/test_0/combined.wav'
#     x = wh.read_wave(wavfile,channel=2)
#     x_bak = x.copy()
#     #x = np.load('data/train/audio/010000.npy')
#     x=torch.from_numpy(x.astype(np.float32) / 32768)
#     x=np.reshape(x, (1, 1, -1))
#     # x = torch.from_numpy(x)
#     b=a.analysis(x)
#     c=a.synthesis(b)
#     print(x.shape, c.shape)
#     b=(b * 32768).numpy().astype(np.int16)
#     # b=np.reshape(np.transpose(b, (0, 2, 1)), (-1, 1)).astype(np.int16)
#     b[0,0,:].tofile(a.name+'b1.pcm')
#     b[0,1,:].tofile(a.name+'b2.pcm')
#
#     c = np.reshape(np.transpose(c.numpy()*32768, (0, 2, 1)), (-1,1)).astype(np.int16)
#     c.tofile(a.name+'here2.pcm')
#     x = np.reshape(x_bak,(-1,1))
#     minleng = min(x.shape[0],c.shape[0])
#     diff = x[:minleng]-c[:minleng]
#     print(c[:100]/x[:100])
#     print(np.sum(x[:minleng]-c[:minleng])/c.shape[0])

