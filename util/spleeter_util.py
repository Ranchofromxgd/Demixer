import sys
sys.path.append("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/")
import os
import json
from config.wavenetConfig import Config
from util import wave_util
from util.dsp_torch import stft,istft,spectrom2magnitude,seperate_magnitude
from util.wave_util import save_pickle,load_pickle,write_json
import torch
import numpy as np
from evaluate.si_sdr_numpy import si_sdr,sdr
import time
import decimal
import matplotlib.pyplot as plt
name = "2019_12_14_phase_spleeter_l2_l3_lr001_bs4_fl1.5_ss8000_85lnu5emptyEvery50"
model_name = "model42000"

def plot2wav(a,b):
    plt.figure(figsize=(20,4))
    plt.subplot(211)
    plt.plot(a)
    plt.subplot(212)
    plt.plot(b)
    plt.savefig("temp.png")



class SpleeterUtil:
    def __init__(self,model_pth = "",map_location = Config.device,test_pth = Config.musdb_test_pth):
        if(type(model_pth) == type("")):
            print("Load model from " + model_pth)
            self.model = torch.load(model_pth, map_location=map_location)
            print("done")
        else:
            self.model = model_pth
        self.test_pth = test_pth
        self.wh = wave_util.WaveHandler()
        self.start = []
        self.end = []
        for each in np.linspace(0,95,20):
            self.start.append(each/100)
            self.end.append((each+5)/100)

    def evaluate(self,save_wav = True,save_json = True):
        performance = {}
        pth = os.listdir(self.test_pth)
        pth.sort()
        for each in pth:
            print("evaluating: ",each,end="  ")
            performance[each] = {}
            background_fpath = self.test_pth+each+"/"+Config.background_fname
            vocal_fpath = self.test_pth+each+"/"+Config.vocal_fname
            background, vocal, origin_background, origin_vocals = self.split(background_fpath,vocal_fpath,use_gpu=True,save=False)
            background_min_length = min(background.shape[0],origin_background.shape[0])
            vocal_min_length = min(vocal.shape[0],origin_vocals.shape[0])

            sdr_background = sdr(background[:background_min_length],origin_background[:background_min_length])
            sdr_vocal = sdr(vocal[:vocal_min_length],origin_vocals[:vocal_min_length])

            if (save_wav == True):
                if (not os.path.exists(Config.project_root + "outputs/" + name + model_name + "/")):
                    print("mkdir: " + Config.project_root + "outputs/" + name + model_name + "/")
                    os.mkdir(Config.project_root + "outputs/" + name + model_name + "/")
                self.wh.save_wave((background[:background_min_length]).astype(np.int16),
                                  Config.project_root + "outputs/" + name + model_name + "/background_" + each + ".wav",
                                  channels=1)
                # self.wh.save_wave((origin_background[:background_min_length]).astype(np.int16),
                #                   Config.project_root + "outputs/" + name + model_name + "/origin_background_" + each + ".wav",
                #                   channels=1)
                self.wh.save_wave((vocal[:vocal_min_length]).astype(np.int16),
                                  Config.project_root + "outputs/" + name + model_name + "/vocal_" + each + ".wav",
                                  channels=1)
                # self.wh.save_wave((origin_vocals[:vocal_min_length]).astype(np.int16),
                #                   Config.project_root + "outputs/" + name + model_name + "/origin_vocals_" + each + ".wav",
                #                   channels=1)
            performance[each]["sdr_background"] = sdr_background
            performance[each]["sdr_vocal"] = sdr_vocal
            print("background:",sdr_background,"vocal:",sdr_vocal)
            # print("hit rate: ",self.wh.hit_times/self.wh.read_times)
        performance["ALL"] = {}
        performance["ALL"]["sdr_background"] = 0
        performance["ALL"]["sdr_vocal"] = 0
        performance["ALL"]["sdr_background"] = [performance[each]["sdr_background"] for each in performance.keys()]
        performance["ALL"]["sdr_vocal"] = [performance[each]["sdr_vocal"] for each in performance.keys()]
        performance["ALL"]["sdr_background"] = sum(performance["ALL"]["sdr_background"] )/(len(performance["ALL"]["sdr_background"])-1)
        performance["ALL"]["sdr_vocal"] = sum(performance["ALL"]["sdr_vocal"] )/(len(performance["ALL"]["sdr_vocal"])-1)
        if(save_json == True):
            write_json(performance, Config.project_root+"evaluate/result_" + name + model_name + ".json")
        print("Result:")
        print("Evaluation on "+str(len(pth))+" songs")
        print("sdr_background: "+str(performance["ALL"]["sdr_background"]))
        print("sdr_vocal: "+str(performance["ALL"]["sdr_vocal"]))
        return performance["ALL"]["sdr_background"],performance["ALL"]["sdr_vocal"]

    def split(self,
              background_fpath,
              vocal_fpath = None,
              require_merge = True,
              use_gpu = False,
              save = True
              ):
        '''
        :param background_fpath: required, if vocal_fpath is None, then background_path should be file with both background and vocal
        :param vocal_fpath: optional, if provided, background_fpath will be deemed as pure background and vocal_fpath is pure vocal
        :param require_merge: if True, vocal_fpath and background_fpath should both be provided
        :param use_gpu: whether use gpu or not
        :param model_pth: the pre-trained model to be used
        :return: None
        '''
        background = np.empty([])
        vocal = np.empty([])
        with torch.no_grad():
            start = time.time()
            # for portion_start in np.linspace(0,0.985,198):
            # for portion_start in np.linspace(0,0.98,50):
            # for portion_start in np.linspace(0,0.9,10):
            for i in range(len(self.start)):
                portion_start,portion_end = self.start[i],self.end[i]
                input_t_background = self.wh.read_wave(fname=background_fpath,
                                                  sample_rate=Config.sample_rate,
                                                  portion_end=portion_end,
                                                  portion_start=portion_start)
                if(require_merge == True):
                    origin_background = self.wh.read_wave(fname=background_fpath, sample_rate=Config.sample_rate)
                    origin_vocal = self.wh.read_wave(fname=vocal_fpath, sample_rate=Config.sample_rate)

                    input_background = self.wh.read_wave(fname=background_fpath
                                               ,sample_rate=Config.sample_rate
                                               ,portion_end = portion_end,
                                                    portion_start=portion_start)
                    input_vocals = self.wh.read_wave(fname=vocal_fpath
                                                ,sample_rate=Config.sample_rate
                                                ,portion_end = portion_end,
                                                     portion_start=portion_start)
                    # To avoid taking one sample point into calculation twice
                    if(i != 0):
                        input_background,input_vocals = input_background[1:],input_vocals[1:]
                    input_f_background = stft(torch.Tensor(input_background).float(),Config.sample_rate).unsqueeze(0)
                    input_f_vocals = stft(torch.Tensor(input_vocals).float(),Config.sample_rate).unsqueeze(0)
                    input_f = (input_f_vocals+input_f_background)
                else:
                    origin_background = self.wh.read_wave(fname=background_fpath, sample_rate=Config.sample_rate)
                    input_background = self.wh.read_wave(fname=background_fpath,
                                               sample_rate=Config.sample_rate,
                                               portion_end=portion_end,portion_start=portion_start)
                    if (i != 0):
                        input_background = input_background[1:]
                    input_f = stft(torch.Tensor(input_background).float(), Config.sample_rate).unsqueeze(0)
                out = []
                for i in range(self.model.channels):
                    input_f = input_f.cuda(Config.device)
                    data = self.model.forward(i, input_f)*input_f
                    out.append(data.squeeze(0))
                for count, each in enumerate(out):
                    if(count == 0):
                        construct_background = istft(each,sample_rate=Config.sample_rate,use_gpu=use_gpu).cpu().numpy()
                    else:
                        construct_vocal = istft(each,sample_rate=Config.sample_rate,use_gpu=use_gpu).cpu().numpy()

                # Compensate the lost samples points from istft
                pad_length = input_t_background.shape[0]- construct_background.shape[0]
                construct_background = np.pad(construct_background,(0,pad_length),'constant',constant_values=(0,0))
                construct_vocal = np.pad(construct_vocal,(0,pad_length),'constant',constant_values=(0,0))
                background = np.append(background,construct_background)
                vocal = np.append(vocal,construct_vocal)

            end = time.time()
            print('time cost',end-start,'s')
        if (save == True):
            print(background.shape)
            self.wh.save_wave((background).astype(np.int16),Config.project_root+"outputs/"+background_fpath+"_background.wav",channels=1)
            self.wh.save_wave((vocal).astype(np.int16),Config.project_root+"outputs/"+background_fpath+"_vocal.wav",channels=1)
            print("Split work finish!")
        if(not vocal_fpath == None): return background,vocal,origin_background,origin_vocal
        else: return background,vocal,origin_background

if __name__ == "__main__":
    vocal = "/home/disk2/internship_anytime/liuhaohe/datasets/musdb18hq/test/test_5/vocals.wav"
    background = "/home/disk2/internship_anytime/liuhaohe/datasets/musdb18hq/test/test_5/background.wav"
    # split("/home/work_nfs3/yhfu/dataset/musdb18hq/test/test_0/background.wav",
    #         "/home/work_nfs3/yhfu/dataset/musdb18hq/test/test_0/vocals.wav",
    #         model_pth = "/home/work_nfs/hhliu/workspace/github/wavenet-aslp/saved_models/phase_spleeter_l4_l5_lr0003_bs4_fl1.5_ss8000_85lnu5emptyEvery50/model18000.pkl",
    #         use_gpu=True)
    path = "/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/saved_models/"+name+"/"+model_name+".pkl"
    su = SpleeterUtil(model_pth = path)
    # su.evaluate()
    # background,vocal,origin_background,origin_vocal = su.split(background_fpath=background,vocal_fpath=vocal,use_gpu=True,save=True)
    # background_min_length = min(background.shape[0], origin_background.shape[0])
    # vocal_min_length = min(vocal.shape[0], origin_vocal.shape[0])

    # sdr_background = sdr(background[:background_min_length], origin_background[:background_min_length])
    # sdr_vocal = sdr(vocal[:vocal_min_length], origin_vocal[:vocal_min_length])
    # print(sdr_background,sdr_vocal)

    su.split(background_fpath="../xuemaojiao.wav",save=True,require_merge=False,use_gpu=True)




