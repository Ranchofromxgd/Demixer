import sys,os
sys.path.append("..")
from config.wavenetConfig import Config
save_root = Config.datahub_root+"pure_vocal_mp3/"
from util.wave_util import WaveHandler
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import webrtcvad


def plotWav(a,name = ""):
    plt.figure(figsize=(10,4))
    plt.plot(a,linewidth = 0.5)
    plt.savefig(name+"_.png")

class VocalFilter():
    def __init__(self):
        self.vad = webrtcvad.Vad()
        self.wh = WaveHandler()
        self.kernal = np.ones(44100*1)/4410*5
        self.threshold = 20
        # self.kernal = np.append(np.linspace(0,1,44100*1.5),np.linspace(1,0,44100*1.5))
    def normalize(self,frames):
        return frames/np.max(frames)

    def variance(self,frames):
        return np.var(frames)

    def flattern(self,arr,smooth = 44100*2):
        for i in range(arr.shape[0]):
            arr[i] = np.sum(arr[i:i+smooth])/smooth

    def conv(self,arr,ker):
        return scipy.signal.convolve(arr, ker)

    def calculate_variance(self,fpath,name = ""):
        self.frames = self.wh.read_wave(fpath)
        self.frames = self.frames[self.frames>=0.0]
        self.frames = self.conv(self.frames,self.kernal)
        # frames = self.flattern(frames)
        length = self.wh.get_framesLength(fpath)
        zero_count = np.sum(~(self.frames>self.threshold))
        if(name != ""):plotWav(self.frames,name)
        self.frames = self.frames[44100*100:44100*100+10000]
        return zero_count/length

    def filter_music(self,pth):
        if(pth[-1] != '/'):
            raise ValueError("Error: Path should end with /")
        dict = {}
        for each in os.listdir(pth):
            fpath = pth+each
            ratio = self.calculate_variance(fpath)
            dict[each] = ratio
            print(ratio,each)

    def myVad(self):
        self.vad.set_mode(0)
        sample_rate = 16000
        frame_duration = 10
        frame = b'\x10\x20' * int(sample_rate * frame_duration / 1000)
        print('Contains speech: %s' % (self.vad.is_speech(frame, sample_rate)))


vf = VocalFilter()
# res = vf.calculate_variance("/home/disk2/internship_anytime/liuhaohe/datasets/musdb18hq/test/test_0/vocals.wav",name = "vocal")
# res2 = vf.calculate_variance("/home/disk2/internship_anytime/liuhaohe/datasets/musdb18hq/test/test_0/combined.wav",name = "bac")
# print(res,res2)
# vf.filter_music("/home/disk2/internship_anytime/liuhaohe/datasets/pure_vocal_wav/纯男声清唱 安排一下？/")
vf.myVad()