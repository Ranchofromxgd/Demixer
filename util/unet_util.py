import sys

sys.path.append("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/")
import os
import json
from config.wavenetConfig import Config
from util import wave_util
from util.dsp_torch import stft, istft, spectrom2magnitude, seperate_magnitude
from util.wave_util import save_pickle, load_pickle, write_json
import torch
import numpy as np
from evaluate.si_sdr_numpy import si_sdr, sdr
import time
import matplotlib.pyplot as plt

name = "unet_phase_only_musdb_l8_lr001_bs8_fl1.5_ss8000_85lnu5emptyEvery50"
model_name = "model18000"


def plot2wav(a, b):
    plt.figure(figsize=(20, 4))
    plt.subplot(211)
    plt.plot(a)
    plt.subplot(212)
    plt.plot(b)
    plt.savefig("temp.png")


class UnetSpleeter:
    def __init__(self, model_pth="", map_location=Config.device):
        if (type(model_pth) == type("")):
            print("Load model from " + model_pth)
            self.model = torch.load(model_pth, map_location=map_location)
            print("done")
        else:
            self.model = model_pth
        self.model.eval()
        self.test_pth = Config.musdb_test_pth
        self.wh = wave_util.WaveHandler()

    def evaluate(self, save_wav=True):
        performance = {}
        pth = os.listdir(self.test_pth)
        pth.sort()
        for each in pth:
            print("evaluating: ", each, end="  ")
            performance[each] = {}
            mixed_fpath = self.test_pth + each + "/mixed.wav"
            vocal_fpath = self.test_pth + each + "/vocals.wav"
            background, vocal, background, origin_vocals = self.split(background_fpath, vocal_fpath, use_gpu=True, save=False)
            background_min_length = min(background.shape[0], origin_background.shape[0])
            vocal_min_length = min(vocal.shape[0], origin_vocals.shape[0])

            sdr_background = sdr(background[:background_min_length], origin_background[:background_min_length])
            sdr_vocal = sdr(vocal[:vocal_min_length], origin_vocals[:vocal_min_length])

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
            print("background:", sdr_background, "vocal:", sdr_vocal)
            # print("hit rate: ",self.wh.hit_times/self.wh.read_times)
        performance["ALL"] = {}
        performance["ALL"]["sdr_background"] = 0
        performance["ALL"]["sdr_vocal"] = 0
        performance["ALL"]["sdr_background"] = [performance[each]["sdr_background"] for each in performance.keys()]
        performance["ALL"]["sdr_vocal"] = [performance[each]["sdr_vocal"] for each in performance.keys()]
        performance["ALL"]["sdr_background"] = sum(performance["ALL"]["sdr_background"]) / (
                    len(performance["ALL"]["sdr_background"]) - 1)
        performance["ALL"]["sdr_vocal"] = sum(performance["ALL"]["sdr_vocal"]) / (
                    len(performance["ALL"]["sdr_vocal"]) - 1)
        write_json(performance, Config.project_root + "evaluate/result_" + name + model_name + ".json")

    def split(self,
              background_fpath,
              vocal_fpath=None,
              use_gpu=False,
              save=True
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
        if(not vocal_fpath == None):
            origin_background = self.wh.read_wave(fname=background_fpath, sample_rate=Config.sample_rate)
            origin_vocal = self.wh.read_wave(fname=vocal_fpath, sample_rate=Config.sample_rate)
        else:
            origin_background = self.wh.read_wave(fname=background_fpath, sample_rate=Config.sample_rate)
        with torch.no_grad():
            start = time.time()
            # for portion_start in np.linspace(0,0.985,198):
            # for portion_start in np.linspace(0,0.98,50):
            for portion_start in np.linspace(0, 0.95, 20):
                # for portion_start in np.linspace(0,0.9,10):
                portion_end = portion_start + 0.05
                # The wave segment used to do separation
                input_t_background = self.wh.read_wave(fname=background_fpath,
                                                  sample_rate=Config.sample_rate,
                                                  portion_end=portion_end,
                                                  portion_start=portion_start)
                if (not vocal_fpath == None):
                    origin_song = origin_background+origin_vocal
                    input_background = self.wh.read_wave(fname=background_fpath
                                                    , sample_rate=Config.sample_rate
                                                    , portion_end=portion_end,
                                                    portion_start=portion_start)
                    input_vocals = self.wh.read_wave(fname=vocal_fpath
                                                     , sample_rate=Config.sample_rate
                                                     , portion_end=portion_end,
                                                     portion_start=portion_start)
                    input_f_background = stft(torch.Tensor(input_background).float(), Config.sample_rate).unsqueeze(0)
                    input_f_vocals = stft(torch.Tensor(input_vocals).float(), Config.sample_rate).unsqueeze(0)
                    input_f = (input_f_vocals + input_f_background)
                else:
                    origin_song = origin_background
                    input_background = self.wh.read_wave(fname=background_fpath,
                                                    sample_rate=Config.sample_rate,
                                                    portion_end=portion_end, portion_start=portion_start)
                    input_f = stft(torch.Tensor(input_background).float(), Config.sample_rate).unsqueeze(0)

                input_f = input_f.cuda(Config.device)
                sigmoid = torch.nn.Sigmoid()
                data = sigmoid(self.model.forward(input_f)) * input_f
                data = data.squeeze(0)
                construct_vocal = istft(data, sample_rate=Config.sample_rate, use_gpu=use_gpu).cpu().numpy()

                # Compensate the lost samples points from istft
                pad_length = input_t_background.shape[0] - construct_vocal.shape[0]
                construct_vocal = np.pad(construct_vocal, (0, pad_length), 'constant', constant_values=(0, 0))

                # Stack
                vocal = np.append(vocal, construct_vocal)
            min_length = min(origin_song.shape[0],vocal.shape[0])
            background = origin_song[:min_length]-vocal[:min_length]
            end = time.time()
            print('time cost', end - start, 's')
        if (save == True):
            self.wh.save_wave((background).astype(np.int16), Config.project_root + "outputs/" + background_fpath + "_background.wav",
                              channels=1)
            self.wh.save_wave((vocal).astype(np.int16), Config.project_root + "outputs/" + background_fpath + "_vocal.wav",
                              channels=1)
            print("Split work finish!")
        else:
            if (not vocal_fpath == None):
                return background, vocal, origin_background, origin_vocal
            else:
                return background, vocal, origin_background


if __name__ == "__main__":
    path = "/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/saved_models/" + name + "/" + model_name + ".pkl"
    su = UnetSpleeter(model_pth=path)
    # su.split("../xuemaojiao.wav",
    #         require_merge=False,
    #         use_gpu=True)
    su.evaluate()
    # su.split(background_fpath="../welcome_to_beijing.wav",save=True,require_merge=False,use_gpu=True)




