import sys
sys.path.append("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/")
import os
import json
from config.wavenetConfig import Config
from util import wave_util
from util.dsp import stft,istft,spectrom2magnitude,seperate_magnitude
from util.wave_util import save_pickle,load_pickle,write_json
import torch
import numpy as np
from evaluate.si_sdr_numpy import si_sdr,sdr
import time
import matplotlib.pyplot as plt
name = "phase_spleeter_l7_l8_lr8e-05_bs4_fl1.5_ss4000_85lnu5emptyEvery50"
model_name = "model78000"

def plot2wav(a,b):
    plt.figure(figsize=(20,4))
    plt.subplot(211)
    plt.plot(a)
    plt.subplot(212)
    plt.plot(b)
    plt.savefig("temp.png")

class SpleeterUtil:
    def __init__(self,model_pth = "",map_location = Config.device):
        if(type(model_pth) == type("")):
            print("Load model from " + model_pth)
            self.model = torch.load(model_pth, map_location=map_location)
            print("done")
        else:
            self.model = model_pth
        self.model.eval()
        self.test_pth = Config.musdb_test_pth
        self.wh = wave_util.WaveHandler()

    def evaluate(self,save_wav = True):
        performance = {}
        pth = os.listdir(self.test_pth)
        pth.sort()
        for each in pth:
            print("evaluating: ",each,end="  ")
            performance[each] = {}
            mixed_fpath = self.test_pth+each+"/mixed.wav"
            vocal_fpath = self.test_pth+each+"/vocals.wav"
            mixed, vocal, origin_mixed, origin_vocals = self.split(mixed_fpath,vocal_fpath,use_gpu=True,save=False)
            mixed_min_length = min(mixed.shape[0],origin_mixed.shape[0])
            vocal_min_length = min(vocal.shape[0],origin_vocals.shape[0])

            sdr_mixed = sdr(mixed[:mixed_min_length],origin_mixed[:mixed_min_length])
            sdr_vocal = sdr(vocal[:vocal_min_length],origin_vocals[:vocal_min_length])

            if (save_wav == True):
                if (not os.path.exists(Config.project_root + "outputs/" + name + model_name + "/")):
                    print("mkdir: " + Config.project_root + "outputs/" + name + model_name + "/")
                    os.mkdir(Config.project_root + "outputs/" + name + model_name + "/")
                self.wh.save_wave((mixed[:mixed_min_length]).astype(np.int16),
                                  Config.project_root + "outputs/" + name + model_name + "/mixed_" + each + ".wav",
                                  channels=1)
                self.wh.save_wave((origin_mixed[:mixed_min_length]).astype(np.int16),
                                  Config.project_root + "outputs/" + name + model_name + "/origin_mixed_" + each + ".wav",
                                  channels=1)
                self.wh.save_wave((vocal[:vocal_min_length]).astype(np.int16),
                                  Config.project_root + "outputs/" + name + model_name + "/vocal_" + each + ".wav",
                                  channels=1)
                self.wh.save_wave((origin_vocals[:vocal_min_length]).astype(np.int16),
                                  Config.project_root + "outputs/" + name + model_name + "/origin_vocals_" + each + ".wav",
                                  channels=1)
            performance[each]["sdr_mixed"] = sdr_mixed
            performance[each]["sdr_vocal"] = sdr_vocal
            print("mixed:",sdr_mixed,"vocal:",sdr_vocal)
            # print("hit rate: ",self.wh.hit_times/self.wh.read_times)
        performance["ALL"] = {}
        performance["ALL"]["sdr_mixed"] = 0
        performance["ALL"]["sdr_vocal"] = 0
        performance["ALL"]["sdr_mixed"] = [performance[each]["sdr_mixed"] for each in performance.keys()]
        performance["ALL"]["sdr_vocal"] = [performance[each]["sdr_vocal"] for each in performance.keys()]
        performance["ALL"]["sdr_mixed"] = sum(performance["ALL"]["sdr_mixed"] )/(len(performance["ALL"]["sdr_mixed"])-1)
        performance["ALL"]["sdr_vocal"] = sum(performance["ALL"]["sdr_vocal"] )/(len(performance["ALL"]["sdr_vocal"])-1)
        write_json(performance, Config.project_root+"evaluate/result_" + name + model_name + ".json")

    def split(self,
              mixed_fpath,
              vocal_fpath = None,
              require_merge = True,
              use_gpu = False,
              save = True
              ):
        '''
        :param mixed_fpath: required, if vocal_fpath is None, then mixed_path should be file with both background and vocal
        :param vocal_fpath: optional, if provided, mixed_fpath will be deemed as pure background and vocal_fpath is pure vocal
        :param require_merge: if True, vocal_fpath and mixed_fpath should both be provided
        :param use_gpu: whether use gpu or not
        :param model_pth: the pre-trained model to be used
        :return: None
        '''
        mixed = np.empty([])
        vocal = np.empty([])
        with torch.no_grad():
            start = time.time()
            # for portion_start in np.linspace(0,0.985,198):
            # for portion_start in np.linspace(0,0.98,50):
            for portion_start in np.linspace(0,0.95,20):
            # for portion_start in np.linspace(0,0.9,10):
                portion_end = portion_start + 0.05
                input_t_mixed = self.wh.read_wave(fname=mixed_fpath,
                                                  sample_rate=Config.sample_rate,
                                                  portion_end=portion_end,
                                                  portion_start=portion_start)
                if(require_merge == True):
                    origin_mixed = self.wh.read_wave(fname=mixed_fpath, sample_rate=Config.sample_rate)
                    origin_vocal = self.wh.read_wave(fname=vocal_fpath, sample_rate=Config.sample_rate)

                    input_mixed = self.wh.read_wave(fname=mixed_fpath
                                               ,sample_rate=Config.sample_rate
                                               ,portion_end = portion_end,
                                                    portion_start=portion_start)
                    input_vocals = self.wh.read_wave(fname=vocal_fpath
                                                ,sample_rate=Config.sample_rate
                                                ,portion_end = portion_end,
                                                     portion_start=portion_start)
                    input_f_mixed = stft(torch.Tensor(input_mixed).float(),Config.sample_rate).unsqueeze(0)
                    input_f_vocals = stft(torch.Tensor(input_vocals).float(),Config.sample_rate).unsqueeze(0)
                    input_f = (input_f_vocals+input_f_mixed)
                else:
                    origin_mixed = self.wh.read_wave(fname=mixed_fpath, sample_rate=Config.sample_rate)
                    input_mixed = self.wh.read_wave(fname=mixed_fpath,
                                               sample_rate=Config.sample_rate,
                                               portion_end=portion_end,portion_start=portion_start)
                    input_f = stft(torch.Tensor(input_mixed).float(), Config.sample_rate).unsqueeze(0)
                out = []
                for i in range(self.model.channels):
                    input_f = input_f.cuda(Config.device)
                    data = self.model.forward(i, input_f)*input_f
                    out.append(data.squeeze(0))
                for count, each in enumerate(out):
                    if(count == 0):
                        construct_mixed = istft(each,sample_rate=Config.sample_rate,use_gpu=use_gpu).cpu().numpy()
                    else:
                        construct_vocal = istft(each,sample_rate=Config.sample_rate,use_gpu=use_gpu).cpu().numpy()

                # Compensate the lost samples points from istft
                pad_length = input_t_mixed.shape[0]- construct_mixed.shape[0]
                construct_mixed = np.pad(construct_mixed,(0,pad_length),'constant',constant_values=(0,0))
                construct_vocal = np.pad(construct_vocal,(0,pad_length),'constant',constant_values=(0,0))

                mixed = np.append(mixed,construct_mixed)
                vocal = np.append(vocal,construct_vocal)

            end = time.time()
            print('time cost',end-start,'s')
        if(save == True):
            self.wh.save_wave((mixed).astype(np.int16),Config.project_root+"outputs/"+mixed_fpath+"_mixed.wav",channels=1)
            self.wh.save_wave((vocal).astype(np.int16),Config.project_root+"outputs/"+mixed_fpath+"_vocal.wav",channels=1)
            print("Split work finish!")
        else:
            if(not vocal_fpath == None): return mixed,vocal,origin_mixed,origin_vocal
            else: return mixed,vocal,origin_mixed

if __name__ == "__main__":
    # split("/home/work_nfs3/yhfu/dataset/musdb18hq/test/test_0/mixed.wav",
    #         "/home/work_nfs3/yhfu/dataset/musdb18hq/test/test_0/vocals.wav",
    #         model_pth = "/home/work_nfs/hhliu/workspace/github/wavenet-aslp/saved_models/phase_spleeter_l4_l5_lr0003_bs4_fl1.5_ss8000_85lnu5emptyEvery50/model18000.pkl",
    #         use_gpu=True)
    path = "/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/saved_models/"+name+"/"+model_name+".pkl"
    su = SpleeterUtil(model_pth = path)
    # su.split("../xuemaojiao.wav",
    #         require_merge=False,
    #         use_gpu=True)
    # su.evaluate():q
    su.split(mixed_fpath="../welcome_to_beijing.wav",save=True,require_merge=False,use_gpu=True)




