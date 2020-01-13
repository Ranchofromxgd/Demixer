import time
import decimal
import matplotlib.pyplot as plt
import os
import json
import torch
import numpy as np
import sys
sys.path.append("..")

from config.wavenetConfig import Config
from util import wave_util
from util.dsp_torch import stft,istft,spectrom2magnitude,seperate_magnitude
from util.wave_util import save_pickle,load_pickle,write_json
from evaluate.si_sdr_numpy import si_sdr,sdr

# model162000
# name="2019_12_30_phase_spleeter_l1_l2_l3_lr0001_bs2_fl1.5_ss120000_8lnu6mu0.5sig0.1low0.2hig0.5"
# model_name = "model192000"
# name="2019_12_25_phase_spleeter_l1_l2_l3_lr0005_bs4_fl1.5_ss6000_8lnu5"
# model_name = "model99000"
# name = "2020_1_8_phase_spleeter_l1_l2_l3_lr0001_bs1_fl2.5_ss60000_8lnu6mu0.5sig0.2low0hig0.5"
# model_name = "model468000"
name = "2020_1_13_DenseUnet_spleeter_l1_l2_l3_lr0001_bs1_fl2_ss60000_8lnu3mu0.5sig0.2low0.49hig0.5"
model_name = "model12000"

def plot2wav(a,b,fname):
    plt.figure(figsize=(20,4))
    plt.subplot(211)
    plt.plot(a)
    plt.subplot(212)
    plt.plot(b)
    plt.savefig(fname)
def plot3wav(a,b,c):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(30,6))
    plt.subplot(311)
    plt.plot(a,linewidth = 1)
    plt.subplot(312)
    plt.plot(b,linewidth = 1)
    plt.subplot(313)
    plt.plot(c,linewidth = 1)
    plt.savefig("commmmm1.png")


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

        # We use this segment to avoid STFT's loss of sample points
        self.tiny_segment = 1

        #This is for better precision in float!
        self.start = []
        self.end = []
        self.realend = []
        for each in np.linspace(0,980,50):
            self.start.append(each/1000)
            self.end.append((each+20+self.tiny_segment)/1000)
            self.realend.append((each+20)/1000)
        # for each in np.linspace(0,950,20):
        #     self.start.append(each/1000)
        #     self.end.append((each+50+self.tiny_segment)/1000)
        #     self.realend.append((each+50)/1000)

    def activation(self,x):
        return 1. / (1 + torch.exp(-100*(x-0.1)))

    def posterior_handling(self,mask, smooth_length=2, freq_bin_portion=0.25):
        mask = mask.squeeze(0)
        freq_bin = mask.shape[0]
        mask_bak = mask.clone()
        mask = mask[:int(freq_bin * freq_bin_portion), :, :]
        mask = torch.sum(torch.sum(mask, 2), 0)
        mask /= torch.max(torch.abs(mask))
        for i in range(mask.shape[0]):
            mask[i] = torch.sum(mask[i-int(smooth_length/2):i + int(smooth_length/2)]) / smooth_length
        mask = self.activation(mask)
        for i in range(mask.shape[0]):
            mask_bak[:, i, :] *= mask[i]
        return mask_bak.unsqueeze(0)

    def evaluate(self,save_wav = True,save_json = True):
        performance = {}
        pth = os.listdir(self.test_pth)#+os.listdir(Config.musdb_train_pth)
        pth.sort()
        for each in pth:
            print("evaluating: ",each,end="  ")
            performance[each] = {}
            background_fpath = self.test_pth+each+"/"+Config.background_fname
            vocal_fpath = self.test_pth+each+"/"+Config.vocal_fname
            background, vocal, origin_background, origin_vocals = self.split(background_fpath,
                                                                             vocal_fpath,
                                                                             use_gpu=True,
                                                                             save=False
                                                                             )
            background_min_length = min(background.shape[0],origin_background.shape[0])
            vocal_min_length = min(vocal.shape[0],origin_vocals.shape[0])

            sdr_background = si_sdr(background[:background_min_length],origin_background[:background_min_length])
            sdr_vocal = si_sdr(vocal[:vocal_min_length],origin_vocals[:vocal_min_length])

            if (save_wav == True):
                if (not os.path.exists(Config.project_root + "outputs/musdb_test/" + Config.trail_name + str(self.model.cnt) + "/")):
                    print("mkdir: " + Config.project_root + "outputs/musdb_test/" + Config.trail_name + str(self.model.cnt) + "/")
                    os.mkdir(Config.project_root + "outputs/musdb_test/" + Config.trail_name + str(self.model.cnt) + "/")
                self.wh.save_wave((background[:background_min_length]).astype(np.int16),
                                  Config.project_root + "outputs/musdb_test/" + Config.trail_name + str(self.model.cnt) + "/background_" + each + ".wav",
                                  channels=1)
                # self.wh.save_wave((origin_background[:background_min_length]).astype(np.int16),
                #                   Config.project_root + "outputs/" + Config.trail_name + str(self.model.cnt) + "/origin_background_" + each + ".wav",
                #                   channels=1)
                self.wh.save_wave((vocal[:vocal_min_length]).astype(np.int16),
                                  Config.project_root + "outputs/musdb_test/" + Config.trail_name + str(self.model.cnt) + "/vocal_" + each + ".wav",
                                  channels=1)
                # self.wh.save_wave((origin_vocals[:vocal_min_length]).astype(np.int16),
                #                   Config.project_root + "outputs/" + Config.trail_name + str(self.model.cnt) + "/origin_vocals_" + each + ".wav",
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
            write_json(performance, Config.project_root+"outputs/musdb_test/"+Config.trail_name + str(self.model.cnt) + "/result_" + Config.trail_name + str(self.model.cnt) + ".json")
        print("Result:")
        print("Evaluation on "+str(len(pth))+" songs")
        print("sdr_background: "+str(performance["ALL"]["sdr_background"]))
        print("sdr_vocal: "+str(performance["ALL"]["sdr_vocal"]))
        return performance["ALL"]["sdr_background"],performance["ALL"]["sdr_vocal"]

    def split(self,
              background_fpath,
              vocal_fpath = None,
              use_gpu = False,
              save = True,
              fname = "temp",
              save_path=Config.project_root,
              scale = 1.0
              ):
        '''
        :param background_fpath: required, if vocal_fpath is None, then background_path should be file with both background and vocal
        :param vocal_fpath: optional, if provided, background_fpath will be deemed as pure background and vocal_fpath is pure vocal
        :param require_merge: if True, vocal_fpath and background_fpath should both be provided
        :param use_gpu: whether use gpu or not
        :param model_pth: the pre-trained model to be used
        :return: None
        '''
        if(save_path[-1]!='/'):
            raise ValueError("Error: path should end with /")
        background = np.empty(0)
        vocal = np.empty(0) # Different from empty([])
        if (vocal_fpath != None):
            origin_background = self.wh.read_wave(fname=background_fpath, sample_rate=Config.sample_rate)
            origin_vocal = self.wh.read_wave(fname=vocal_fpath, sample_rate=Config.sample_rate)
        else:
            origin_background = self.wh.read_wave(fname=background_fpath, sample_rate=Config.sample_rate)
        with torch.no_grad():
            start = time.time()
            # mask_all = [None]*self.model.channels
            for i in range(len(self.start)):
                portion_start,portion_end,real_end= self.start[i],self.end[i],self.realend[i]
                input_t_background = self.wh.read_wave(fname=background_fpath,
                                                  sample_rate=Config.sample_rate,
                                                  portion_end=real_end,
                                                  portion_start=portion_start)

                if(vocal_fpath != None):
                    input_background = self.wh.read_wave(fname=background_fpath
                                               ,sample_rate=Config.sample_rate
                                               ,portion_end = portion_end,
                                                    portion_start=portion_start)
                    input_vocals = self.wh.read_wave(fname=vocal_fpath
                                                ,sample_rate=Config.sample_rate
                                                ,portion_end = portion_end,
                                                     portion_start=portion_start)
                    # if (i != 0):
                    #     input_background,input_vocals = input_background[1:],input_vocals[1:]
                    # Construct Spectrom of Song
                    input_f_background = stft(torch.Tensor(input_background).float().cuda(Config.device),Config.sample_rate,use_gpu=True).unsqueeze(0)
                    input_f_vocals = stft(torch.Tensor(input_vocals).float().cuda(Config.device),Config.sample_rate,use_gpu=True).unsqueeze(0)
                    input_f = (input_f_vocals+input_f_background)
                else:
                    input_background = self.wh.read_wave(fname=background_fpath,
                                               sample_rate=Config.sample_rate,
                                               portion_end=portion_end,portion_start=portion_start)
                    # if (i != 0):
                    #     input_background = input_background[1:]
                    input_f = stft(torch.Tensor(input_background).float().cuda(Config.device), Config.sample_rate,use_gpu=True).unsqueeze(0)
                out = []
                # Forward and mask
                for ch in range(self.model.channels):
                    mask = self.model.forward(ch, input_f)
                    # if(ch == 1):
                    #     mask = self.posterior_handling(mask)
                    if(Config.OUTPUT_MASK):data = mask  * input_f
                    else:data = mask
                    out.append(data.squeeze(0)*scale)
                    # mask = mask.cpu().numpy()[0,:,:,:]
                    # if (mask_all[ch] is None):
                    #     # mask_all[ch] = mask
                    #     mask_all[ch] = []
                    # else:
                    #     # mask_all[ch] = np.concatenate((mask_all[ch], mask), axis=1)
                    #     mask_all.append(np.sum(mask))
                # Inverse STFT
                for count, each in enumerate(out):
                    if(count == 0):
                        construct_background = istft(each,sample_rate=Config.sample_rate,use_gpu=use_gpu).cpu().numpy()
                    else:
                        construct_vocal = istft(each,sample_rate=Config.sample_rate,use_gpu=use_gpu).cpu().numpy()

                # The last segment's stft loss could not be compensatedn with a little margin
                if(i == (len(self.start)-1)):
                    pad_length = input_t_background.shape[0]- construct_background.shape[0]
                    construct_background = np.pad(construct_background,(0,pad_length),'constant',constant_values=(0,0))
                    construct_vocal = np.pad(construct_vocal,(0,pad_length),'constant',constant_values=(0,0))
                else:
                    real_length = input_t_background.shape[0]
                    construct_background = construct_background[:real_length]
                    construct_vocal = construct_vocal[:real_length]
                background = np.append(background,construct_background)
                vocal = np.append(vocal,construct_vocal)
            end = time.time()
            print('time cost',end-start,'s')
            # save_pickle(mask_all, "mask_curve.pkl")
        if (save == True):
            self.wh.save_wave((background).astype(np.int16),save_path+fname+"_background.wav",channels=1)
            self.wh.save_wave((vocal).astype(np.int16),save_path+fname+"_vocal.wav",channels=1)
            print("Split work finish!")
        if(not vocal_fpath == None): return background,vocal,origin_background,origin_vocal
        else: return background,vocal,origin_background

    def Split_listener(self,
                       pth = Config.project_root+"evaluate/raw_wave/",
                       fname = None):
        output_path = Config.project_root+"outputs/listener/"+Config.trail_name+str(self.model.cnt)+"/"
        if(not os.path.exists(output_path)):
            os.mkdir(output_path)
        for each in os.listdir(pth):
            if(each.split('.')[-1] != 'wav'):
                continue
            if(fname is not None and fname != each):
                continue
            print(each)
            file = each.split('.')[-2]
            self.split(background_fpath=pth+each,
                     save=True,
                     save_path=output_path,
                     fname = file,
                     use_gpu=True,
                     scale=0.5)

if __name__ == "__main__":
    # split("/home/work_nfs3/yhfu/dataset/musdb18hq/test/test_0/background.wav",
    #         "/home/work_nfs3/yhfu/dataset/musdb18hq/test/test_0/vocals.wav",
    #         model_pth = "/home/work_nfs/hhliu/workspace/github/wavenet-aslp/saved_models/phase_spleeter_l4_l5_lr0003_bs4_fl1.5_ss8000_85lnu5emptyEvery50/model18000.pkl",
    #         use_gpu=True)
    path = Config.project_root+"saved_models/"+name+"/"+model_name+".pkl"
    su = SpleeterUtil(model_pth = path)
    su.evaluate(save_wav=True)
    # su.Split_listener(fname="西原健一郎_-_Serendipity_vocal.wav")
    # background,vocal,origin_background,origin_vocal = su.split(background_fpath=background,vocal_fpath=vocal,use_gpu=True,save=True)
    # background_min_length = min(background.shape[0], origin_background.shape[0])
    # vocal_min_length = min(vocal.shape[0], origin_vocal.shape[0])

    # sdr_background = sdr(background[:background_min_length], origin_background[:background_min_length])
    # sdr_vocal = sdr(vocal[:vocal_min_length], origin_vocal[:vocal_min_length])
    # print(sdr_background,sdr_vocal)




