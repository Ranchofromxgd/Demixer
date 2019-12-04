import sys
sys.path.append("/home/work_nfs/hhliu/workspace/github/wavenet-aslp")
import os
import json
from config.wavenetConfig import Config
from util import wave_util
from util.dsp import stft,istft,spectrom2magnitude,seperate_magnitude
from util.wave_util import save_pickle,load_pickle,write_json
import torch
import numpy as np
from evaluate.si_sdr_numpy import si_sdr
import time

class SpleeterUtil:
    def __init__(self,model_pth = "",map_location = Config.device):
        print("Load model from " + model_pth)
        self.model = torch.load(model_pth, map_location=map_location)
        self.model.eval()
        print("done")
        self.test_pth = Config.musdb_test_pth
        self.wh = wave_util.WaveHandler()

    def evaluate(self):
        performance = {}
        for each in os.listdir(self.test_pth):
            print("evaluating: ",each,end="  ")
            performance[each] = {}
            mixed_fpath = self.test_pth+each+"/mixed.wav"
            vocal_fpath = self.test_pth+each+"/vocals.wav"
            mixed, vocal, origin_mixed, origin_vocals = self.split(mixed_fpath,vocal_fpath,use_gpu=True,save=False)

            # self.wh.save_wave((mixed).astype(np.int16), "./outputs/mixed_"+each+".wav", channels=1)
            # self.wh.save_wave((vocal).astype(np.int16), "./outputs/vocal_"+each+".wav", channels=1)
            # self.wh.save_wave((origin_mixed).astype(np.int16), "./outputs/origin_mixed_"+each+".wav", channels=1)
            # self.wh.save_wave((origin_vocals).astype(np.int16), "./outputs/origin_vocals_"+each+".wav", channels=1)

            mixed_min_length = min(mixed.shape[0],origin_mixed.shape[0])
            vocal_min_length = min(vocal.shape[0],origin_vocals.shape[0])

            sdr_mixed = si_sdr(mixed[:mixed_min_length],origin_mixed[:mixed_min_length])
            sdr_vocal = si_sdr(vocal[:vocal_min_length],origin_vocals[:vocal_min_length])

            performance[each]["sdr_mixed"] = sdr_mixed
            performance[each]["sdr_vocal"] = sdr_vocal

            print(sdr_mixed,sdr_vocal)
        write_json(performance,"../evaluate/result.json")

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
                if(require_merge == True):
                    origin_mixed = self.wh.read_wave(fname=mixed_fpath, sample_rate=Config.sample_rate)
                    origin_vocal = self.wh.read_wave(fname=vocal_fpath, sample_rate=Config.sample_rate)
                    input_mixed = self.wh.read_wave(fname=mixed_fpath
                                               ,sample_rate=Config.sample_rate
                                               ,portion_end = portion_end,portion_start=portion_start)
                    input_vocals = self.wh.read_wave(fname=vocal_fpath
                                                ,sample_rate=Config.sample_rate
                                                ,portion_end = portion_end,portion_start=portion_start)
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
                mixed = np.append(mixed,construct_mixed)
                vocal = np.append(vocal,construct_vocal)

            end = time.time()
            print('time cost',end-start,'s')
        if(save == True):
            self.wh.save_wave((mixed).astype(np.int16),"../outputs/mixed.wav",channels=1)
            self.wh.save_wave((vocal).astype(np.int16),"../outputs/vocal.wav",channels=1)
            print("Split work finish!")
        else:
            if(not vocal_fpath == None): return mixed,vocal,origin_mixed,origin_vocal
            else: return mixed,vocal,origin_mixed

def pow_np_norm(signal):
    """Compute 2 Norm"""
    return np.square(np.linalg.norm(signal, ord=2))

def pow_norm(s1, s2):
    return np.sum(s1 * s2)

def si_sdr(estimated, original):
    #estimated = remove_dc(estimated)
    #original = remove_dc(original)
    target = pow_norm(estimated, original) * original / pow_np_norm(original)
    noise = estimated - target
    return 10 * np.log10(pow_np_norm(target) / pow_np_norm(noise))

if __name__ == "__main__":
    # split("/home/work_nfs3/yhfu/dataset/musdb18hq/test/test_0/mixed.wav",
    #         "/home/work_nfs3/yhfu/dataset/musdb18hq/test/test_0/vocals.wav",
    #         model_pth = "/home/work_nfs/hhliu/workspace/github/wavenet-aslp/saved_models/phase_spleeter_l4_l5_lr0003_bs4_fl1.5_ss8000_85lnu5emptyEvery50/model18000.pkl",
    #         use_gpu=True)
    su = SpleeterUtil(model_pth = "/home/work_nfs/hhliu/workspace/github/wavenet-aslp/saved_models/phase_spleeter_l4_l5_lr0003_bs4_fl1.5_ss8000_85lnu5emptyEvery50/model18000.pkl")
    # su.split("../xuemaojiao.wav",
    #         require_merge=False,
    #         use_gpu=True)
    su.evaluate()




    # wh = wave_util.WaveHandler()
    # input = wh.read_wave("/home/work_nfs/hhliu/workspace/github/wavenet-aslp/util/outputs/input_vocals_test_0.wav")
    # out = wh.read_wave("/home/work_nfs/hhliu/workspace/github/wavenet-aslp/util/outputs/vocal_test_0.wav",channel=1)
    # print(si_sdr(out,input))



