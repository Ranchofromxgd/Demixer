import sys
sys.path.append("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/")
from torch.utils.data import Dataset
from config.wavenetConfig import Config
import util.wave_util as wave_util
import os
import time
import numpy as np
from util.wave_util import save_pickle,load_pickle

class WavenetDataloader(Dataset):
    def __init__(self,sample_length = Config.sample_rate*Config.frame_length,num_worker = Config.num_workers,empty_every_n = 10000000):
        # self.music_folders = self.readList(Config.musdb_train_background)
        self.music_folders = []
        for each in Config.background_data:
            self.music_folders += self.readList(each)
        self.vocal_folders = []
        for each in Config.vocal_data:
            self.vocal_folders += self.readList(each)
        self.sample_length = int(sample_length)
        self.cnt = 0
        self.num_worker = num_worker
        self.empty_every_n = empty_every_n
        self.wh = wave_util.WaveHandler()


    def __getitem__(self, item):
        np.random.seed(os.getpid()+self.cnt)
        # Select background(background only) and vocal file randomly
        self.cnt += self.num_worker
        random_music = np.random.randint(0,len(self.music_folders))
        random_vocal = np.random.randint(0,len(self.vocal_folders))
        music_fname = self.music_folders[random_music]
        vocal_fname = self.vocal_folders[random_vocal]
        background_frames = self.wh.read_wave(music_fname)
        # if (self.cnt % self.empty_every_n == 0):
        #     background_start = np.random.randint(0, background_frames.shape[0] - self.sample_length)
        #     background_crop = background_frames[background_start: background_start + self.sample_length]
        #     return background_crop,np.zeros(background_crop.shape).astype(np.int16),background_crop
        vocal_frames = self.wh.read_wave(vocal_fname)

        # Randomly crop
        background_start = np.random.randint(0,background_frames.shape[0]-self.sample_length)
        vocal_start = np.random.randint(0,vocal_frames.shape[0]-self.sample_length)
        background_crop = background_frames[background_start: background_start+self.sample_length]
        vocal_crop = vocal_frames[vocal_start: vocal_start+self.sample_length]

        vocal_crop = vocal_crop.astype(np.float32)
        max_background = np.max(np.abs(background_crop))
        max_vocal = np.max(np.abs(vocal_crop))

        # To avoid magnify the blank vocal
        if(not max_vocal == 0 and (max_background/max_vocal)<50):
            vocal_crop /= max_vocal
            background_crop, vocal_crop = background_crop, (vocal_crop * max_background).astype(np.int16)
            background_crop, vocal_crop = background_crop/2, vocal_crop/2

        background_crop, vocal_crop = background_crop.astype(np.int16), vocal_crop.astype(np.int16)
        return background_crop, vocal_crop,background_crop+vocal_crop#  ,(music_fname.split('/')[-2]+music_fname.split('/')[-1],vocal_fname.split('/')[-2]+vocal_fname.split('/')[-1])

    def __len__(self):
        # Actually infinit due to the random dynamic sampling
        return int(36000/Config.frame_length)

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
    import torch
    from util.wave_util import WaveHandler
    wh = WaveHandler()
    s = WavenetDataloader()
    dl = torch.utils.data.DataLoader(s, batch_size=Config.batch_size, shuffle=False, num_workers=1)
    cnt = 0
    for each in dl:
        print(each[-1])
        cnt += 1
