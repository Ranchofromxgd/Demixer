import time
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import sys
sys.path.append("..")
from config.mainConfig import Config
from util import wave_util
from util.dsp_torch import stft, istft
from util.wave_util import  write_json,save_pickle
from evaluate.si_sdr_numpy import si_sdr,sdr
from scipy import signal
from evaluate.museval.mus_eval import bss_eval

vocal_b, vocal_a = signal.butter(8, 2*11000/44100, 'lowpass')
back_b, back_a = signal.butter(8, 2*17000/44100, 'lowpass')


def plot2wav(a, b, fname):
    plt.figure(figsize=(20, 4))
    plt.subplot(211)
    plt.plot(a)
    plt.subplot(212)
    plt.plot(b)
    plt.savefig(fname)
    plt.close("all")

def draw_f(f_domain_data,fname = None):
    plt.figure(figsize=(30,5))
    im = plt.imshow(f_domain_data)
    plt.colorbar(im)
    if(fname is not None):
        plt.savefig(fname)
    else:
        plt.show()
    plt.close("all")

def visualize_mask(res,pth):
    res = np.flipud(res[:,:,0])
    draw_f(res,pth)

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
    plt.close("all")


class SpleeterUtil:
    def __init__(self, load_model_pth="",start_point = 0, map_location=Config.device, test_pth=Config.musdb_test_pth):
        if (type(load_model_pth) == type("")):
            if(start_point == 0):
                print("Warning: Start point is 0")
            self.model_name = load_model_pth.split("/")[-1]
            self.start_point = start_point
            model_pth = load_model_pth +"/model" + str(start_point) + ".pkl"
            print("Load model from " + model_pth)
            self.model = torch.load(model_pth, map_location=map_location)
            print("done")
        else:
            self.model_name = Config.trail_name
            self.model = load_model_pth
            self.start_point = self.model.cnt
        self.test_pth = test_pth
        self.wh = wave_util.WaveHandler()
        self.map_location = map_location
        # We use this segment to avoid STFT's loss of sample points
        self.tiny_segment = 10

        # This is for better precision in float!
        self.start = []
        self.end = []
        self.realend = []
        for each in np.linspace(0, 980, 50):
            self.start.append(each / 1000)
            self.end.append((each + 20 + self.tiny_segment) / 1000)
            self.realend.append((each + 20) / 1000)
        # for each in np.linspace(0, 900, 10):
        #     self.start.append(each / 1000)
        #     self.end.append((each + 100 + self.tiny_segment) / 1000)
        #     self.realend.append((each + 100) / 1000)
        # for each in np.linspace(0,950,20):
        #     self.start.append(each/1000)
        #     self.end.append((each+50+self.tiny_segment)/1000)
        #     self.realend.append((each+50)/1000)


    def unify_length(self,seq1,seq2):
        min_len = min(seq1.shape[0],seq2.shape[0])
        return seq1[:min_len],seq2[:min_len]



    def evaluate(self, save_wav=True, save_json=True,firstN = None):
        eval = {}
        eval["ALL"] = {}
        dir_pth = self.test_pth
        pth = os.listdir(dir_pth)
        pth.sort()
        bac_keys = ["sdr_bac", "sisdr_bac", "mus_sdr_bac", "mus_isr_bac", "mus_sir_bac", "mus_sar_bac"]
        voc_keys = ["sdr_voc", "sisdr_voc", "mus_sdr_voc", "mus_isr_voc", "mus_sir_voc", "mus_sar_voc"]

        def fm(num):
            return format(num,".2f")

        def reshape_source(src):
            return np.reshape(src,newshape=(1,src.shape[0],1))

        def stack_two_source(vocal,background):
            return np.concatenate((reshape_source(vocal),reshape_source(background)),axis=0)

        def __get_key_average(key):
            util_list = [eval[each][key] for each in eval.keys()]
            return sum(util_list) / (len(util_list) - 1)

        def roc_val(item,key:list,value:list):
            for each in zip(key,value):
                eval[item][each[0]] = each[1]
                print(each[0],fm(each[1]),end="  ")
            print("")

        def cal_avg_val(keys:list):
            for each in keys:
                eval["ALL"][each] = 0
                eval["ALL"][each] = __get_key_average(each)
                print(each,fm(eval["ALL"][each]))

        for cnt,each in enumerate(pth): # TODO
            if(firstN is not None and cnt == firstN):
                break
            print("evaluating: ", each)
            eval[each] = {}
            background_fpath = dir_pth + each + "/" + "background.wav"
            vocal_fpath = dir_pth + each + "/" + "vocals.wav"
            bac, voc, origin_bac, origin_voc = self.split(background_fpath,
                                                                             vocal_fpath,
                                                                             save=False,
                                                                             save_path=Config.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point) + "/",
                                                                             fname = each
                                                                             )
            if (save_wav == True):
                print("saving...")
                if (not os.path.exists(
                        Config.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point) + "/")):
                    print("mkdir: " + Config.project_root + "outputs/musdb_test/" + self.model_name + str(
                        self.start_point) + "/")
                    os.mkdir(
                        Config.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point) + "/")
                # bac = signal.filtfilt(back_b, back_a, bac)
                self.wh.save_wave((bac).astype(np.int16),
                                  Config.project_root + "outputs/musdb_test/" + self.model_name + str(
                                      self.start_point) + "/background_" + each + ".wav",
                                  channels=1)
                # voc = signal.filtfilt(vocal_b, vocal_a, voc)
                self.wh.save_wave((voc).astype(np.int16),
                                  Config.project_root + "outputs/musdb_test/" + self.model_name + str(
                                      self.start_point) + "/vocal_" + each + ".wav",
                                  channels=1)

            bac, origin_bac = self.unify_length(bac, origin_bac)
            voc, origin_voc = self.unify_length(voc, origin_voc)
            sdr_bac = sdr(bac, origin_bac)
            sdr_voc = sdr(voc, origin_voc)
            sisdr_bac = si_sdr(bac, origin_bac)
            sisdr_voc = si_sdr(voc, origin_voc)
            estimate = stack_two_source(bac, voc)
            target = stack_two_source(origin_bac, origin_voc)
            mus_sdr, mus_isr, mus_sir, mus_sar, _ = bss_eval(target, estimate)

            bac_values = [sdr_bac,sisdr_bac,mus_sdr[0],mus_isr[0],mus_sir[0],mus_sar[0]]
            voc_values = [sdr_voc,sisdr_voc,mus_sdr[1],mus_isr[1],mus_sir[1],mus_sar[1]]
            roc_val(each,bac_keys,bac_values)
            roc_val(each,voc_keys,voc_values)

        print("Result:")
        print("Evaluation on " + str(len(pth)) + " songs")
        cal_avg_val(bac_keys)
        cal_avg_val(voc_keys)

        if (save_json == True):
            if(not os.path.exists(Config.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point))):
                os.mkdir(Config.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point))
            write_json(eval, Config.project_root + "outputs/musdb_test/" + self.model_name + str(
                self.start_point) + "/result_" + self.model_name + str(self.start_point) + ".json")

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
        if(not os.path.exists(save_path)):
            os.mkdir(save_path)
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
                # mask_all = np.empty([1025,0,2])
                # mask_processed_all = np.empty([1025,0,2])
                # switch_all = None
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
                        input_f_background = stft(torch.Tensor(input_background).float(),
                                                  Config.sample_rate, use_gpu=False).unsqueeze(0)
                        input_f_vocals = stft(torch.Tensor(input_vocals).float(), Config.sample_rate,
                                              use_gpu=False).unsqueeze(0)
                        input_f = (input_f_vocals + input_f_background)
                    else:
                        input_background = self.wh.read_wave(fname=background_fpath,
                                                             sample_rate=Config.sample_rate,
                                                             portion_end=portion_end, portion_start=portion_start)
                        # if (i != 0):
                        #     input_background = input_background[1:]
                        input_f = stft(torch.Tensor(input_background).float(), Config.sample_rate,
                                       use_gpu=False).unsqueeze(0)
                    out = []
                    # Forward and mask
                    self.model.eval()
                    for ch in range(self.model.channels):
                        mask = self.model(ch, input_f) # TODO
                        # if(ch == 1):
                        #     mask_all = np.append(mask_all,orig_mask.squeeze(0).cpu().detach().numpy(),axis=1)
                        #     mask_processed_all = np.append(mask_processed_all,mask.squeeze(0).cpu().detach().numpy(),axis=1)
                        data = mask * input_f
                        out.append(data.squeeze(0) * scale)
                    self.model.train()

                    # Inverse STFT
                    for count, each in enumerate(out):
                        if (count == 0):
                            construct_background = istft(each, sample_rate=Config.sample_rate, use_gpu=False).numpy()
                        else:
                            construct_vocal = istft(each, sample_rate=Config.sample_rate, use_gpu=False).numpy()

                    # The last segment's stft loss could not be compensatedn with a little margin
                    if (i == (len(self.start) - 1)):
                        pass
                        # pad_length = input_t_background.shape[0] - construct_background.shape[0]
                        # construct_background = np.pad(construct_background, (0, pad_length), 'constant',
                        #                               constant_values=(0, 0))
                        # construct_vocal = np.pad(construct_vocal, (0, pad_length), 'constant', constant_values=(0, 0))
                    else:
                        real_length = input_t_background.shape[0]
                        construct_background = construct_background[:real_length]
                        construct_vocal = construct_vocal[:real_length]
                    background = np.append(background, construct_background)
                    vocal = np.append(vocal, construct_vocal)
                    # smooth = 6
                    # scope = 20
                    # if(not abs(background.shape[0]-real_length)<10):
                    #     for i in range(-scope,scope):
                    #         probe = -real_length+i
                    #         background[probe] = sum(background[probe-int(smooth/2):probe+int(smooth/2)])/smooth
                    #         vocal[probe] = sum(vocal[probe-int(smooth/2):probe+int(smooth/2)])/smooth
                # vad = VoiceActivityDetector(rate=Config.sample_rate,data=vocal,channels=Config.channels)
                # vocal = vad.select_speech_regions()
            # save_pickle(mask_all, save_path + "mask_vocal_" + fname + ".pkl")
            # visualize_mask(mask_all,save_path + "mask_vocal_" + fname + ".png")
            # visualize_mask(mask_processed_all,save_path + "mask_processed_vocal_" + fname + ".png")
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
            # background = signal.filtfilt(back_b, back_a, background)
            self.wh.save_wave((background).astype(np.int16), save_path + "background_"+fname+".wav", channels=1)
            # vocal = signal.filtfilt(vocal_b, vocal_a, vocal)  # data为要过滤的信号
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

def save_params(model,name = ""):
    params = []
    for each in model.parameters():
        params.append(each)
    save_pickle(params, name)

if __name__ == "__main__":

    load_model_path = [None]
    start_point = [None]
    # load_model_path[0] = Config.project_root+"saved_models/2_2020_3_22__unet_spleeter_spleeter_sf0_l1_l2_l3_lr0001_bs2-48_fl3_ss32000_85lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0.5split_bandFalse"
    # start_point[0] = 160000
    # load_model_path[0] = Config.project_root+"saved_models/2_2020_2_6__unet_spleeter_spleeter_sf0_l1_l2_l3_l7_l8_lr0001_bs1_fl3_ss16000_85lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32"
    # start_point[0] = 88000
    # load_model_path[0] = Config.project_root+"saved_models/1_2020_3_15__unet_spleeter_spleeter_sf48000_l1_l2_l3_lr0008_bs4_fl3_ss16000_9lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32"
    # start_point[0] = 320000
    load_model_path[0] = Config.project_root+"saved_models/3_2020_3_6__unet_spleeter_spleeter_sf72000_l1_l2_l3_lr7e-05_bs2_fl3_ss32000_85lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32"
    start_point[0] = 224000
    for each in zip(load_model_path,start_point):
        print(each)
        su = SpleeterUtil(load_model_pth=each[0],start_point=each[1],map_location="cpu")
        su.evaluate(save_wav=True,save_json=True)
        # su.Split_listener()
        # su.Split_listener()
    # background,vocal,origin_background,origin_vocal = su.split(background_fpath=background,vocal_fpath=vocal,use_gpu=True,save=True)
    # background_min_length = min(background.shape[0], origin_background.shape[0])
    # vocal_min_length = min(vocal.shape[0], origin_vocal.shape[0])

    # sdr_background = sdr(background[:background_min_length], origin_background[:background_min_length])
    # sdr_vocal = sdr(vocal[:vocal_min_length], origin_vocal[:vocal_min_length])
    # print(sdr_background,sdr_vocal)
