import time
import decimal
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import sys
sys.path.append("..")
from evaluate.sdr import sdr_evaluate
from config.wavenetConfig import Config
from util import wave_util
from util.dsp_torch import stft, istft
from util.wave_util import  write_json
from evaluate.si_sdr_numpy import si_sdr
from util.vad import VoiceActivityDetector
from scipy import signal

vocal_b, vocal_a = signal.butter(8, 2*11000/44100, 'lowpass')
back_b, back_a = signal.butter(8, 2*17000/44100, 'lowpass')


def plot2wav(a, b, fname):
    plt.figure(figsize=(20, 4))
    plt.subplot(211)
    plt.plot(a)
    plt.subplot(212)
    plt.plot(b)
    plt.savefig(fname)


def plot3wav(a, b, c):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(30, 6))
    plt.subplot(311)
    plt.plot(a, linewidth=1)
    plt.subplot(312)
    plt.plot(b, linewidth=1)
    plt.subplot(313)
    plt.plot(c, linewidth=1)
    plt.savefig("commmmm1.png")


class SpleeterUtil:
    def __init__(self, model_pth="", map_location=Config.device, test_pth=Config.musdb_test_pth):
        if (type(model_pth) == type("")):
            self.model_name = Config.load_model_path.split("/")[-1]
            self.start_point = Config.start_point
            print("Load model from " + model_pth)
            self.model = torch.load(model_pth, map_location=map_location)
            print("done")
        else:
            self.model_name = Config.trail_name
            self.model = model_pth
            self.start_point = self.model.cnt
        self.test_pth = test_pth
        self.wh = wave_util.WaveHandler()

        # We use this segment to avoid STFT's loss of sample points
        self.tiny_segment = 10

        # This is for better precision in float!
        self.start = []
        self.end = []
        self.realend = []
        # for each in np.linspace(0, 980, 50):
        #     self.start.append(each / 1000)
        #     self.end.append((each + 20 + self.tiny_segment) / 1000)
        #     self.realend.append((each + 20) / 1000)
        # for each in np.linspace(0, 900, 10):
        #     self.start.append(each / 1000)
        #     self.end.append((each + 100 + self.tiny_segment) / 1000)
        #     self.realend.append((each + 100) / 1000)
        for each in np.linspace(0,950,20):
            self.start.append(each/1000)
            self.end.append((each+50+self.tiny_segment)/1000)
            self.realend.append((each+50)/1000)

    def activation(self,x):
        return 1. / (1 + torch.exp(-15 * (x - 0.35)))
        # x[x > 0.1] = 1
        # x[x < 0.1] = 0
        # return x

    def is_restrained(self,x):
        return torch.abs(x) < 0.45

    def posterior_handling(self,mask, smooth_length=4, freq_bin_portion=0.25,min_gap = 1400):
        mask = mask.squeeze(0)
        freq_bin = mask.shape[0]
        mask_bak = mask.clone()
        mask = mask[:int(freq_bin * freq_bin_portion), :, :]
        mask = torch.sum(torch.sum(mask, 2), 0)
        mask /= torch.max(torch.abs(mask))
        for i in range(mask.shape[0]):
            mask[i] = torch.sum(mask[i - int(smooth_length / 2):i + int(smooth_length / 2)]) / smooth_length
        mask = self.activation(mask)
        vstart, vend = None, None
        mstart, mend = None, None
        is_vocal = True
        not_music = False
        for i in range(mask.shape[0]):
            if (self.is_restrained(mask[i]) and is_vocal == True):
                is_vocal = False
                vstart = i
                continue
            elif (not self.is_restrained(mask[i]) and is_vocal == False):
                is_vocal = True
                vend = i
                if (abs(vend - vstart) < min_gap):
                    mask[vstart:vend] = torch.ones(vend - vstart)
        for i in range(mask.shape[0]):
            if (not self.is_restrained(mask[i]) and not_music == False):
                not_music = True
                mstart = i
                continue
            elif (self.is_restrained(mask[i]) and not_music == True):
                not_music = False
                mend = i
                if (abs(mend - mstart) < min_gap):
                    # try:
                    #     mask[mstart+800:mend] = torch.zeros(mend - (mstart+800))
                    # except:
                    #     continue
                    pass
        for i in range(mask.shape[0]):
            mask_bak[:, i, :] *= mask[i]
        return mask_bak.unsqueeze(0),mask

    def evaluate(self, save_wav=True, save_json=True):
        performance = {}
        dir_pth = self.test_pth
        pth = os.listdir(dir_pth) # + os.listdir(self.test_pth)
        pth.sort()
        for each in pth: # TODO
            print("evaluating: ", each, end="  ")
            performance[each] = {}
            background_fpath = dir_pth + each + "/" + Config.background_fname
            vocal_fpath = dir_pth + each + "/" + Config.vocal_fname
            background, vocal, origin_background, origin_vocals = self.split(background_fpath,
                                                                             vocal_fpath,
                                                                             save=False,
                                                                             save_path=Config.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point) + "/",
                                                                             fname = each
                                                                             )
            background_min_length = min(background.shape[0], origin_background.shape[0])
            vocal_min_length = min(vocal.shape[0], origin_vocals.shape[0])
            sdr_background = si_sdr(background,origin_background)
            sdr_vocal = si_sdr(vocal,origin_vocals)
            if (save_wav == True):
                if (not os.path.exists(
                        Config.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point) + "/")):
                    print("mkdir: " + Config.project_root + "outputs/musdb_test/" + self.model_name + str(
                        self.start_point) + "/")
                    os.mkdir(
                        Config.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point) + "/")
                background = signal.filtfilt(back_b, back_a, background)
                self.wh.save_wave((background[:background_min_length]).astype(np.int16),
                                  Config.project_root + "outputs/musdb_test/" + self.model_name + str(
                                      self.start_point) + "/background_" + each + ".wav",
                                  channels=1)
                vocal = signal.filtfilt(vocal_b, vocal_a, vocal)
                self.wh.save_wave((vocal[:vocal_min_length]).astype(np.int16),
                                  Config.project_root + "outputs/musdb_test/" + self.model_name + str(
                                      self.start_point) + "/vocal_" + each + ".wav",
                                  channels=1)
            performance[each]["sdr_background"] = sdr_background
            performance[each]["sdr_vocal"] = sdr_vocal
            print("background:", sdr_background, "vocal:", sdr_vocal)
        performance["ALL"] = {}
        performance["ALL"]["sdr_background"] = 0
        performance["ALL"]["sdr_vocal"] = 0
        performance["ALL"]["sdr_background"] = [performance[each]["sdr_background"] for each in performance.keys()]
        performance["ALL"]["sdr_vocal"] = [performance[each]["sdr_vocal"] for each in performance.keys()]
        performance["ALL"]["sdr_background"] = sum(performance["ALL"]["sdr_background"]) / (
                    len(performance["ALL"]["sdr_background"]) - 1)
        performance["ALL"]["sdr_vocal"] = sum(performance["ALL"]["sdr_vocal"]) / (
                    len(performance["ALL"]["sdr_vocal"]) - 1)
        if (save_json == True):
            write_json(performance, Config.project_root + "outputs/musdb_test/" + self.model_name + str(
                self.start_point) + "/result_" + self.model_name + str(self.start_point) + ".json")
        print("Result:")
        print("Evaluation on " + str(len(pth)) + " songs")
        print("sdr_background: " + str(performance["ALL"]["sdr_background"]))
        print("sdr_vocal: " + str(performance["ALL"]["sdr_vocal"]))
        return performance["ALL"]["sdr_background"], performance["ALL"]["sdr_vocal"]

    def split(self,
              background_fpath,
              vocal_fpath=None,
              save=True,
              fname="temp",
              save_path=Config.project_root,
              scale=1.0
              ):
        if (save_path[-1] != '/'):
            raise ValueError("Error: path should end with /")
        background = np.empty(0)
        vocal = np.empty(0)  # Different from empty([])
        if (vocal_fpath != None):
            origin_background = self.wh.read_wave(fname=background_fpath, sample_rate=Config.sample_rate)
            origin_vocal = self.wh.read_wave(fname=vocal_fpath, sample_rate=Config.sample_rate)
        else:
            origin_background = self.wh.read_wave(fname=background_fpath, sample_rate=Config.sample_rate)
        start = time.time()
        if(not os.path.exists(save_path + "background_"+fname+".wav") or not os.path.exists(save_path + "vocal_"+fname+".wav")):
            print("Not found: ",save_path + "background_"+fname+".wav",save_path + "vocal_"+fname+".wav")
            with torch.no_grad():
                # mask_all = [None]*self.model.channels
                switch_all = None
                for i in range(len(self.start)):
                    portion_start, portion_end, real_end = self.start[i], self.end[i], self.realend[i]
                    input_t_background = self.wh.read_wave(fname=background_fpath,
                                                           sample_rate=Config.sample_rate,
                                                           portion_end=real_end,
                                                           portion_start=portion_start)

                    if (vocal_fpath != None):
                        input_background = self.wh.read_wave(fname=background_fpath
                                                             , sample_rate=Config.sample_rate
                                                             , portion_end=portion_end,
                                                             portion_start=portion_start)
                        input_vocals = self.wh.read_wave(fname=vocal_fpath
                                                         , sample_rate=Config.sample_rate
                                                         , portion_end=portion_end,
                                                         portion_start=portion_start)
                        # if (i != 0):
                        #     input_background,input_vocals = input_background[1:],input_vocals[1:]
                        # Construct Spectrom of Song
                        input_f_background = stft(torch.Tensor(input_background).float().cuda(Config.device),
                                                  Config.sample_rate, use_gpu=True).unsqueeze(0)
                        input_f_vocals = stft(torch.Tensor(input_vocals).float().cuda(Config.device), Config.sample_rate,
                                              use_gpu=True).unsqueeze(0)
                        input_f = (input_f_vocals + input_f_background)
                    else:
                        input_background = self.wh.read_wave(fname=background_fpath,
                                                             sample_rate=Config.sample_rate,
                                                             portion_end=portion_end, portion_start=portion_start)
                        # if (i != 0):
                        #     input_background = input_background[1:]
                        input_f = stft(torch.Tensor(input_background).float().cuda(Config.device), Config.sample_rate,
                                       use_gpu=True).unsqueeze(0)
                    out = []
                    # Forward and mask
                    self.model.eval()
                    for ch in range(self.model.channels):

                        mask = self.model.forward(ch, input_f)

                        # If vocal, add posterior handling
                        if(ch == 1):
                            mask,switch = self.posterior_handling(mask)
                        #     # Visualize switch
                        #     switch = switch.cpu().detach().numpy()
                        #     if (switch_all is None):
                        #         switch_all = switch
                        #     else:
                        #         switch_all = np.append(switch_all, switch)

                        if (Config.OUTPUT_MASK):
                            data = mask * input_f
                        else:
                            data = mask
                        out.append(data.squeeze(0) * scale)
                    self.model.train()

                    # Inverse STFT
                    for count, each in enumerate(out):
                        if (count == 0):
                            construct_background = istft(each, sample_rate=Config.sample_rate, use_gpu=True).cpu().numpy()
                        else:
                            construct_vocal = istft(each, sample_rate=Config.sample_rate, use_gpu=True).cpu().numpy()

                    # The last segment's stft loss could not be compensatedn with a little margin
                    if (i == (len(self.start) - 1)):
                        pad_length = input_t_background.shape[0] - construct_background.shape[0]
                        construct_background = np.pad(construct_background, (0, pad_length), 'constant',
                                                      constant_values=(0, 0))
                        construct_vocal = np.pad(construct_vocal, (0, pad_length), 'constant', constant_values=(0, 0))
                    else:
                        real_length = input_t_background.shape[0]
                        construct_background = construct_background[:real_length]
                        construct_vocal = construct_vocal[:real_length]
                    background = np.append(background, construct_background)
                    vocal = np.append(vocal, construct_vocal)
                    smooth = 6
                    scope = 20
                    if(not abs(background.shape[0]-real_length)<10):
                        for i in range(-scope,scope):
                            probe = -real_length+i
                            background[probe] = sum(background[probe-int(smooth/2):probe+int(smooth/2)])/smooth
                            vocal[probe] = sum(vocal[probe-int(smooth/2):probe+int(smooth/2)])/smooth
                # vad = VoiceActivityDetector(rate=Config.sample_rate,data=vocal,channels=Config.channels)
                # vocal = vad.select_speech_regions()
        else:
            background = self.wh.read_wave(save_path + "background_"+fname+".wav",channel=1)
            vocal = self.wh.read_wave(save_path +  "vocal_"+fname+".wav",channel=1)
        end = time.time()
        print('time cost', end - start, 's')
        # plt.figure(figsize=(20,4))
        # plt.plot(switch_all)
        # plt.savefig(save_path + "vocal_"+fname+".png")
        # save_pickle(mask_all, "mask_curve.pkl")
        if (save == True):
            background = signal.filtfilt(back_b, back_a, background)
            self.wh.save_wave((background).astype(np.int16), save_path + "background_"+fname+".wav", channels=1)
            vocal = signal.filtfilt(vocal_b, vocal_a, vocal)  # data为要过滤的信号
            self.wh.save_wave((vocal).astype(np.int16), save_path +  "vocal_"+fname+".wav", channels=1)
            print("Split work finish!")
        if (not vocal_fpath == None):
            return background, vocal, origin_background, origin_vocal
        else:
            return background, vocal, origin_background

    def Split_listener(self,
                       pth=Config.project_root + "evaluate/raw_wave/",
                       fname=None):
        output_path = Config.project_root + "outputs/listener/" + self.model_name + str(self.start_point) + "/"
        if (not os.path.exists(output_path)):
            os.mkdir(output_path)
        for each in os.listdir(pth):
            if (each.split('.')[-1] != 'wav'):
                continue
            if (fname is not None and fname != each):
                continue
            print(each)
            file = each.split('.')[-2]
            self.split(background_fpath=pth + each,
                       save=True,
                       save_path=output_path,
                       fname=file,
                       scale=0.5)


if __name__ == "__main__":
    from evaluate.sdr import sdr_evaluate
    path = Config.load_model_path +"/model" + str(Config.start_point) + ".pkl"
    su = SpleeterUtil(model_pth=path)
    # su.evaluate(save_wav=True)
    su.Split_listener()
    # background,vocal,origin_background,origin_vocal = su.split(background_fpath=background,vocal_fpath=vocal,use_gpu=True,save=True)
    # background_min_length = min(background.shape[0], origin_background.shape[0])
    # vocal_min_length = min(vocal.shape[0], origin_vocal.shape[0])

    # sdr_background = sdr(background[:background_min_length], origin_background[:background_min_length])
    # sdr_vocal = sdr(vocal[:vocal_min_length], origin_vocal[:vocal_min_length])
    # print(sdr_background,sdr_vocal)
