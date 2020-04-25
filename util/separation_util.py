import time
import os
import torch
import numpy as np
import sys
import logging
sys.path.append("..")
from config.mainConfig import Config
from util import wave_util
from util.wave_util import  write_json,save_pickle
from scipy import signal
from models.subband_util import before_forward_f,after_forward_f
from util.wave_util import load_json
from evaluate.museval.evaluate_track import eval_mus_track

vocal_b, vocal_a = signal.butter(8, 2*11000/44100, 'lowpass')
back_b, back_a = signal.butter(8, 2*17000/44100, 'lowpass')

class SeparationUtil:
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
        self.start = []
        self.end = []
        self.realend = []
        if(not Config.split_band):
            self.tiny_segment = 2
            for each in np.linspace(0, 995, 200):
                self.start.append(each / 1000)
                self.end.append((each + 5 + self.tiny_segment) / 1000)
                self.realend.append((each + 5) / 1000)
        else:
            self.tiny_segment = 5
            for each in np.linspace(0, 950, 20):
                self.start.append(each / 1000)
                self.end.append((each + 50 + self.tiny_segment) / 1000)
                self.realend.append((each + 50) / 1000)

    def evaluate(self, save_wav=True,
                 save_json=True,
                 split_band = True,
                 test_mode = False):
        '''
        Do evaluation on musdb dataset
        Args:
            save_wav: Whether to save the hole separation results
            save_json: Whether to save the metrics evaluation result (json file)
            split_band: Whether to use subband input
            test_mode: test the hole system
        Returns:
            None
        '''
        def __fm(num):
            return format(num,".2f")

        def __unify_length(seq1, seq2):
            min_len = min(seq1.shape[0], seq2.shape[0])
            return seq1[:min_len], seq2[:min_len]

        def __reshape_source(src):
            return np.reshape(src,newshape=(1,src.shape[0],1))

        def __stack_two_source(vocal,background):
            return np.concatenate((__reshape_source(vocal),__reshape_source(background)),axis=0)

        def __get_aproperate_keys():
            keys = []
            for each in list(res.keys()):
                if ("ALL" not in each):
                    keys.append(each)
            return keys

        def __get_key_average(key,keys):
            util_list = [res[each][key] for each in keys]
            return np.mean(util_list)  # sum(util_list) / (len(util_list) - 1)

        def __get_key_median(key,keys):
            util_list = [res[each][key] for each in keys]
            return np.median(util_list)

        def __get_key_std(key,keys):
            util_list = [res[each][key] for each in keys]
            return np.std(util_list)

        def __roc_val(item, key: list, value: list):
            for each in zip(key, value):
                res[item][each[0]] = each[1]

        def __cal_avg_val(keys: list):
            proper_keys = __get_aproperate_keys()
            for each in keys:
                res["ALL_median"][each] = 0
                res["ALL_mean"][each] = 0
                res["ALL_std"][each] = 0
                res["ALL_median"][each] = __get_key_median(each,proper_keys)
                res["ALL_mean"][each] = __get_key_average(each,proper_keys)
                res["ALL_std"][each] = __get_key_std(each,proper_keys)
                print(each,":")
                print( __fm(res["ALL_median"][each]),",", __fm(res["ALL_mean"][each]),",",__fm(res["ALL_std"][each]))

        def unify_energy(source, target):
            def abs_max(arr):
                return np.max(np.abs(arr))
            ratio = abs_max(target) / abs_max(source)
            return source * ratio, target

        json_file_alias = Config.project_root + "outputs/musdb_test/" + self.model_name + str(
            self.start_point) + "/result_" + self.model_name + str(self.start_point) + ".json"
        bac_keys = ["mus_sdr_bac", "mus_isr_bac", "mus_sir_bac", "mus_sar_bac"]
        voc_keys = ["mus_sdr_voc", "mus_isr_voc", "mus_sir_voc", "mus_sar_voc"]
        save_pth = Config.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point)
        if (os.path.exists(json_file_alias+"@")): # todo here we just do not want this program to find these json file
            res = load_json(json_file_alias)
            # print("Find:",res)
            res["ALL_median"] = {}
            res["ALL_mean"] = {}
            res["ALL_std"] = {}
        else:
            res = {}
            res["ALL_median"] = {}
            res["ALL_mean"] = {}
            res["ALL_std"] = {}
            dir_pth = self.test_pth
            pth = os.listdir(dir_pth)
            pth.sort()
            for cnt,each in enumerate(pth):
                print("evaluating: ", each)
                res[each] = {}
                background_fpath = dir_pth + each + "/" + "background.wav"
                vocal_fpath = dir_pth + each + "/" + "vocals.wav"
                try:
                    bac, voc, origin_bac, origin_voc = self.split(background_fpath,
                                                                  vocal_fpath,
                                                                  split_band=split_band,
                                                                  save=save_wav,
                                                                  save_path= save_pth + "/",
                                                                  fname = each,
                                                                  test_mode = test_mode
                                                                  )
                    if(not test_mode):
                        bac, origin_bac = __unify_length(bac,origin_bac)
                        voc,origin_voc = __unify_length(voc,origin_voc)
                        bac, origin_bac = unify_energy(bac,origin_bac)
                        voc,origin_voc = unify_energy(voc,origin_voc)

                        eval_targets = ['vocals', 'accompaniment']
                        origin, estimate = {}, {}
                        origin[eval_targets[0]], origin[eval_targets[1]] = origin_voc, origin_bac
                        estimate[eval_targets[0]], estimate[eval_targets[1]] = voc, bac
                        data = eval_mus_track(origin, estimate,
                                              output_dir=save_pth,
                                              track_name=each)
                        print(data)
                        museval_res = data.get_result()
                        bac_values = [museval_res['accompaniment']['SDR'], museval_res['accompaniment']['ISR'],
                                      museval_res['accompaniment']['SIR'], museval_res['accompaniment']['SAR']]
                        voc_values = [museval_res['vocals']['SDR'], museval_res['vocals']['ISR'],
                                      museval_res['vocals']['SIR'], museval_res['vocals']['SAR']]
                        __roc_val(each, bac_keys, bac_values)
                        __roc_val(each, voc_keys, voc_values)

                except Exception as e:
                    print("ERROR: splitting error...")
                    logging.exception(e)

        if(not test_mode):
            print("Result:")
            print("Median,", "Mean,", "Std")
            __cal_avg_val(bac_keys)
            __cal_avg_val(voc_keys)

        if (save_json == True):
            if (not os.path.exists(Config.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point))):
                os.mkdir(Config.project_root + "outputs/musdb_test/" + self.model_name + str(self.start_point))
            write_json(res, Config.project_root + "outputs/musdb_test/" + self.model_name + str(
                self.start_point) + "/result_" + self.model_name + str(self.start_point) + ".json")

    def split(self,
              background_fpath,
              vocal_fpath=None,
              split_band = True,
              save=True,
              fname="temp",
              save_path=Config.project_root,
              scale=1.0,
              test_mode = False
              ):
        '''
        This function has two mode:
            mode1: read vocal and background, combine them, and then do separation job
            mode2: read mixture, then do separation job
        mode1 need you to specify both the background and vocal path
        Mode2 need you to sepecify only the background path and leave vocal as None
        Args:
            background_fpath: The path to background wav
            vocal_fpath: The path to vocal wav
            split_band: boolean, whether do band splitting or not
            save: boolean, whether save the splitting result
            fname: an alias for you separation result
            save_path: saving directory, only take effect when save is True
            scale: the separation result will time this factor
            test_mode: test the system
        Returns:
            mode1: background, vocal, origin_background, origin_vocal
            mode2: background, vocal
        '''
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
                        input_background = torch.Tensor(input_background).float().cuda(self.map_location)
                        input_vocals = torch.Tensor(input_vocals).float().cuda(self.map_location)
                        input_f_background,input_f_vocals = before_forward_f(input_background.unsqueeze(0),input_vocals.unsqueeze(0),
                                                              use_subband = split_band,
                                                                subband_num=Config.subband,
                                                              use_gpu = Config.use_gpu,
                                                              sample_rate=Config.sample_rate)
                        input_f = (input_f_vocals + input_f_background)
                    else:
                        input_song = self.wh.read_wave(fname=background_fpath,
                                                             sample_rate=Config.sample_rate,
                                                             portion_end=portion_end, portion_start=portion_start)
                        input_song = torch.Tensor(input_song).float().cuda(self.map_location)
                        input_f = before_forward_f(input_song.unsqueeze(0),
                                                   use_subband=split_band,
                                                   subband_num=Config.subband,
                                                   use_gpu=Config.use_gpu,
                                                   sample_rate=Config.sample_rate)
                    # Forward and mask
                    self.model.eval()
                    for ch in range(self.model.channels):
                        mask = self.model(ch, input_f)
                        data = mask * input_f * scale
                        if (ch == 0):
                            construct_background = after_forward_f(data,
                                                                       use_gpu=True,
                                                                       sample_rate=Config.sample_rate,
                                                                       subband=split_band,
                                                                   subband_num=Config.subband).cpu().numpy()
                        else:
                            construct_vocal = after_forward_f(data,
                                                                   use_gpu=True,
                                                                   sample_rate=Config.sample_rate,
                                                                   subband=split_band,
                                                              subband_num=Config.subband).cpu().numpy()
                    self.model.train()
                    if(test_mode):break
                    real_length = input_t_background.shape[0]
                    construct_background = construct_background.reshape(-1)[:real_length]
                    construct_vocal = construct_vocal.reshape(-1)[:real_length]
                    background = np.append(background, construct_background)
                    vocal = np.append(vocal, construct_vocal)
            if (save == True):
                if (not os.path.exists(save_path)):
                    print("Creat path", save_path)
                # background = signal.filtfilt(back_b, back_a, background)
                self.wh.save_wave((background).astype(np.int16), save_path + "background_" + fname + ".wav",
                                  channels=1)
                # vocal = signal.filtfilt(vocal_b, vocal_a, vocal)  # data为要过滤的信号
                self.wh.save_wave((vocal).astype(np.int16), save_path + "vocal_" + fname + ".wav", channels=1)
                print("Split work finish!")
        else:
            background = self.wh.read_wave(save_path + "background_"+fname+".wav",channel=1)
            vocal = self.wh.read_wave(save_path +  "vocal_"+fname+".wav",channel=1)

        end = time.time()
        print('time cost', end - start, 's')
        if (not vocal_fpath == None):
            return background, vocal, origin_background, origin_vocal
        else:
            return background, vocal

    def Split_listener(self,
                       pth=Config.project_root + "evaluate/raw_wave/",
                       split_band = True,
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
                       split_band = split_band,
                       fname=file,
                       scale=0.5)

def save_params(model,name = ""):
    params = []
    for each in model.parameters():
        params.append(each)
    save_pickle(params, name)


if __name__ == "__main__":
    from util.EvaluationHelper import EvaluationHelper
    test = {
        # "1":{
        #     "path": "1_2020_4_12__unet_spleeter_spleeter_sf0_l1_l2_l3_lr001_bs8-1_fl3_ss32000_9lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0split_bandTrue_8",
        #     "start_points": [96000],
        #     "subband": 8,
        # },
        # "2": {
        #     "path": "1_2020_4_12__unet_spleeter_spleeter_sf0_l1_l2_l3_lr001_bs2-1_fl3_ss32000_9lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0split_bandTrue_2",
        #     "start_points": [352000],
        #     "subband": 2,
        # }
        "7": {
            "path": "1_2020_4_18__unet_spleeter_spleeter_sf32000_l1_l2_l3_lr001_bs2-1_fl3_ss64000_9lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0split_bandFalse_4",
            "start_points": [288000, 224000],  # [192000, 160000, 128000, 96000, 64000],
            "subband": 1,
        }
    }

    eh = EvaluationHelper()
    for each in list(test.keys()):
        path,start_points,subband = test[each]["path"],test[each]["start_points"],test[each]["subband"]
        eh.batch_evaluation_subband(Config.project_root+"saved_models/"+path,start_points,
                                    save_wav=True,
                                    save_json=True,
                                    test_mode=False,
                                    split_musdb=True,
                                    split_listener=False,
                                    subband = subband)



