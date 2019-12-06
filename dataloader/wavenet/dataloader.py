import sys
sys.path.append("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/")
from torch.utils.data import Dataset
from config.wavenetConfig import Config
import util.wave_util as wave_util
import os
import numpy as np
from util.wave_util import save_pickle,load_pickle

class WavenetDataloader(Dataset):
    def __init__(self,sample_length = Config.sample_rate*Config.frame_length,empty_every_n = 10000000):
        self.music_folders = self.readList(Config.musdb_train_mixed)
        self.vocal_folders = self.readList(Config.musdb_train_vocal)*5+self.readList(Config.song_vocal_pth)
        self.sample_length = int(sample_length)
        self.cnt = 0
        self.empty_every_n = empty_every_n
        self.wh = wave_util.WaveHandler()


    def __getitem__(self, item):
        np.random.seed(os.getpid()+self.cnt)
        # Select mixed(background only) and vocal file randomly
        self.cnt += 1
        music_fname = self.music_folders[np.random.randint(0,len(self.music_folders))]
        vocal_fname = self.vocal_folders[np.random.randint(0,len(self.vocal_folders))]
        mixed_frames = self.wh.read_wave(music_fname)
        # if (self.cnt % self.empty_every_n == 0):
        #     mixed_start = np.random.randint(0, mixed_frames.shape[0] - self.sample_length)
        #     mixed_crop = mixed_frames[mixed_start: mixed_start + self.sample_length]
        #     return mixed_crop,np.zeros(mixed_crop.shape).astype(np.int16),mixed_crop
        vocal_frames = self.wh.read_wave(vocal_fname)
        # Randomly crop
        mixed_start = np.random.randint(0,mixed_frames.shape[0]-self.sample_length)
        vocal_start = np.random.randint(0,vocal_frames.shape[0]-self.sample_length)

        mixed_crop = mixed_frames[mixed_start: mixed_start+self.sample_length]
        vocal_crop = vocal_frames[vocal_start: vocal_start+self.sample_length]
        # maxVal = max(np.max(np.abs(vocal_crop)),np.max(np.abs(mixed_crop)))
        # mixed_crop,vocal_crop = mixed_crop/maxVal,vocal_crop/maxVal
        return mixed_crop,vocal_crop,mixed_crop+vocal_crop #,music_fname.split('/')[-2]+music_fname.split('/')[-1],vocal_fname.split('/')[-2]+vocal_fname.split('/')[-1]

    def __len__(self):
        # Actually infinit due to the random dynamic sampling
        return int(36000/Config.frame_length)

    def normalize(self,data):
        max_val = np.max(np.abs(data))
        return data/(max_val+1e-7)

    def quantize_data(self,data, classes):
        mu_x = self.mu_law_encoding(self.normalize(data), classes)
        bins = np.linspace(-1, 1, classes)
        quantized = np.digitize(mu_x, bins) - 1
        return quantized

    def mu_law_encoding(self,data, mu):
        mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
        return mu_x

    def mu_law_expansion(self,data, mu):
        s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
        return s

    def readList(self,fname):
        result = []
        with open(fname,"r") as f:
            for each in f.readlines():
                each = each.strip('\n')
                result.append(each)
        return result

def zero():
    return 0

if __name__ == "__main__":
    # hit = load_pickle("hit.pkl")
    # print(len(hit))
    # smooth = 10
    # smoothed = []
    # for each in range(len(hit)):
    #     smoothed.append(sum(hit[each:each+smooth])/smooth)
    # print(smoothed)
    import torch
    # import scipy.signal as signal
    # import util.wave_util as wu
    s = WavenetDataloader()
    dl = torch.utils.data.DataLoader(s, batch_size=Config.batch_size, shuffle=False, num_workers=1)
    cnt =0
    for d in dl:
        print(d)
    #     if(count % 100 == 0):
    #         print(count)
    #     if(count == 200):
    #         break
    #     print(each)
    #     f_music,f_vocal = each[-1]
    #     for name in f_music:
    #         music[name] += 1
    #         cnt += 1
    # for each in music.keys():
    #     music[each]/=cnt
    # print(music)
    # print(len(list(music.keys())))
    # print(max(music.values()),min(music.values()))