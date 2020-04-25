
import wave
import numpy as np
import scipy.signal as signal
# import util.dsp_conv_torch as dsp
import pickle
import json
import time
# import librosa


class WavObj:
    def __init__(self,fname,content):
        self.fname =fname
        self.content = content

# class FixLengthDictionary:
#     def __init__(self,maxlength = 10):
#         self.content = {}
#         self.maxlength = maxlength
#
#     def put(self,wavobj):
#         if(len(list(self.content.keys())) >= self.maxlength):
#             self.content.pop(list(self.content.keys())[0])
#         self.content[wavobj.fname] = wavobj.content
#
#     def get(self,fname):
#         if(fname in list(self.content.keys())):
#             return self.content[fname]
#         return np.empty([])

class WaveHandler:
    def __init__(self):
        # self.wav_cache = FixLengthDictionary(maxlength=20)
        self.read_times = 0
        self.hit_times = 0
    def save_wave(self, frames, fname, bit_width=2, channels=1, sample_rate=44100):
        f = wave.open(fname, "wb")
        f.setnchannels(channels)
        f.setsampwidth(bit_width)
        f.setframerate(sample_rate)
        f.writeframes(frames.tostring())
        f.close()

    # Only get the first channel
    def read_wave(self, fname,
                  channel=2,
                  convert_to_f_domain = False,
                  sample_rate = 44100,
                  portion_start = 0,
                  portion_end = 1,
                  needRaw = False): # Whether you want raw bytes
        if(portion_end > 1 and portion_end < 1.1):
            portion_end = 1
        f = wave.open(fname)
        params = f.getparams()
        if(portion_end <= 1):
            raw = f.readframes(params[3])
            if (needRaw == True): return raw
            frames = np.fromstring(raw, dtype=np.short)
            if(frames.shape[0] % 2 == 1):frames = np.append(frames,0)
            # Convert to mono
            frames.shape = -1, channel
            start, end = int(frames.shape[0] * portion_start), int(frames.shape[0] * portion_end)
            frames = frames[start:end, 0]
            # Resample
            # if(params[2] != sample_rate):
            #     frames = librosa.resample(frames.astype(np.float), params[2], sample_rate).astype(np.int16)
        else:
            f.setpos(portion_start)
            raw = f.readframes(int(portion_end-portion_start))
            if(needRaw == True):return raw
            frames = np.fromstring(raw, dtype=np.short)
            if (frames.shape[0] % 2 == 1): frames = np.append(frames, 0)
            frames.shape = -1, channel
            frames = frames[:,0]
            # if (params[2] != sample_rate):
            #     frames = librosa.resample(frames.astype(np.float), params[2], sample_rate).astype(np.int16)

        # if(convert_to_f_domain == True):
        #     frames = dsp.stft(frames.astype(np.float32),sample_rate = sample_rate)
        return frames

    def get_channels_sampwidth_and_sample_rate(self,fname):
        f = wave.open(fname)
        params = f.getparams()
        return (params[0],params[1],params[2]) == (2,2,44100),(params[0],params[1],params[2])

    def get_channels(self,fname):
        f = wave.open(fname)
        params = f.getparams()
        return params[0]

    def get_sample_rate(self,fname):
        f = wave.open(fname)
        params = f.getparams()
        return params[2]

    def get_duration(self,fname):
        f = wave.open(fname)
        params = f.getparams()
        return params[3]/params[2]

    def get_framesLength(self,fname):
        f = wave.open(fname)
        params = f.getparams()
        return params[3]

    def restore_wave(self,zxx):
        _,w = signal.istft(zxx)
        return w

def save_pickle(obj,fname):
    print("Save pickle at "+fname)
    with open(fname,'wb') as f:
        pickle.dump(obj,f)

def load_pickle(fname):
    print("Load pickle at "+fname)
    with open(fname,'rb') as f:
        res = pickle.load(f)
    return res

def write_json(dict,fname):
    print("Save json file at"+fname)
    json_str = json.dumps(dict)
    with open(fname, 'w') as json_file:
        json_file.write(json_str)

def load_json(fname):
    with open(fname,'r') as f:
        data = json.load(f)
        return data
# 15412577 15544877

if __name__ == "__main__":
    wh = WaveHandler()
    # print(wh.get_framesLength("/home/work_nfs/hhliu/workspace/datasets/musdb18hq_splited/train/Alexander Ross - Goodbye Bolero/drums.wav"))
    res = wh.read_wave(#"/home/work_nfs/hhliu/workspace/datasets/musdb18hq_splited/train/Music Delta - Grunge/other.wav",
         "/home/work_nfs/hhliu/workspace/datasets/musdb18hq_splited/train/Alexander Ross - Goodbye Bolero/drums.wav",
                       portion_start=13412577,
                       portion_end=14544877,
                       channel=2
                        )
    print(res.shape)





