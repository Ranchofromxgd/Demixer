import sys
sys.path.append("..")
from util.wave_util import WaveHandler
from config.wavenetConfig import Config
from torch.utils.data import Dataset
# These part should below 'import util'
import os
import time
import torch
import numpy as np
from util.dsp_torch import stft

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(Config.device)
        self.preload()

    def preload(self):
        try:
            self.next_background, self.next_vocal,self.next_song = next(self.loader)
        except StopIteration:
            self.next_background = None
            self.next_vocal = None
            self.next_song= None
            return
        with torch.cuda.stream(self.stream):
            self.next_background = self.next_background.cuda(Config.device,non_blocking=True)
            self.next_vocal = self.next_vocal.cuda(Config.device,non_blocking=True)
            self.next_song = self.next_song.cuda(Config.device,non_blocking=True)

    def next(self):
        torch.cuda.current_stream(Config.device).wait_stream(self.stream)
        background = self.next_background
        vocal = self.next_vocal
        song = self.next_song
        if background is not None:
            background.record_stream(torch.cuda.current_stream(Config.device))
        if vocal is not None:
            vocal.record_stream(torch.cuda.current_stream(Config.device))
        if song is not None:
            song.record_stream(torch.cuda.current_stream(Config.device))
        self.preload()
        return background, vocal,song

class WavenetDataloader(Dataset):
    def __init__(self,sample_length = Config.sample_rate*Config.frame_length,
                 num_worker = Config.num_workers,
                 sampleNo=20000,
                 mu = Config.mu,
                 sigma = Config.sigma,
                 alpha_low = Config.alpha_low,
                 alpha_high = Config.alpha_high # If alpha_high get a value greater than 0.5, it would have probability to overflow
    ):
        # self.music_folders = self.readList(Config.musdb_train_background)
        self.music_folders = []
        for each in Config.background_data:
            self.music_folders += self.readList(each)
        self.vocal_folders = []
        for each in Config.vocal_data:
            self.vocal_folders += self.readList(each)
        self.sample_length = int(sample_length)
        self.cnt = 0
        self.data_counter = 0
        self.sampleNo = sampleNo
        self.num_worker = num_worker
        self.wh = WaveHandler()
        # This alpha is to balance the energy between vocal and background
        # Also, this alpha is used to simulate different energy leval between vocal and background
        np.random.seed(0)
        self.normal_distribution = np.random.normal(mu, sigma, sampleNo)
        self.normal_distribution = self.normal_distribution[self.normal_distribution > alpha_low]
        self.normal_distribution = self.normal_distribution[self.normal_distribution < alpha_high]
        self.sampleNo = self.normal_distribution.shape[0]

    def __getitem__(self, item):
        self.data_counter += 1
        np.random.seed(os.getpid()+self.cnt)
        # Select background(background only) and vocal file randomly
        self.cnt += self.num_worker
        random_music = np.random.randint(0,len(self.music_folders))
        random_vocal = np.random.randint(0,len(self.vocal_folders))
        music_fname = self.music_folders[random_music]
        vocal_fname = self.vocal_folders[random_vocal]
        music_length = self.wh.get_framesLength(music_fname)
        vocal_length = self.wh.get_framesLength(vocal_fname)

        background_start = np.random.randint(0, music_length - self.sample_length)
        vocal_start = np.random.randint(0, vocal_length - self.sample_length)
        background_crop = self.wh.read_wave(music_fname,
                                            portion_start=background_start,
                                            portion_end=background_start+self.sample_length)

        vocal_crop = self.wh.read_wave(vocal_fname,
                                       portion_start=vocal_start,
                                       portion_end=vocal_start+self.sample_length)

        # Randomly crop
        vocal_crop = vocal_crop.astype(np.float32)
        try:
            max_background = np.max(np.abs(background_crop))
        except:
            print(background_crop)
            exit(0)
        max_vocal = np.max(np.abs(vocal_crop))

        # To avoid magnify the blank vocal

        if(not max_vocal == 0 and (max_background/max_vocal)<50):
            vocal_crop /= max_vocal
            background_crop, vocal_crop = background_crop, (vocal_crop * max_background).astype(np.int16)
            alpha_vocal = self.normal_distribution[self.data_counter % self.sampleNo]
            alpha_background = self.normal_distribution[-(self.data_counter % self.sampleNo)]
            background_crop, vocal_crop = background_crop*alpha_background, vocal_crop*alpha_vocal

        background_crop, vocal_crop = background_crop.astype(np.int16), vocal_crop.astype(np.int16)
        b,v,s = torch.Tensor(background_crop), torch.Tensor(vocal_crop), torch.Tensor(background_crop + vocal_crop)
        if(not Config.time_domain_loss):
            b,v,s = stft(b.float(),Config.sample_rate),stft(v.float(),Config.sample_rate),stft(s.float(),Config.sample_rate)
        return b,v,s#  ,(music_fname.split('/')[-2]+music_fname.split('/')[-1],vocal_fname.split('/')[-2]+vocal_fname.split('/')[-1])

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

    wh = WaveHandler()
    s = WavenetDataloader()
    dl = torch.utils.data.DataLoader(s, batch_size=Config.batch_size, shuffle=False, num_workers=1)
    # start = 0
    # for each in dl:
    #     end = time.time()
    #     print(start - end)
    #     start = time.time()

    pref = data_prefetcher(dl)
    b,v,s = pref.next()
    iteration = 0

    while s is not None:
        start = time.time()
        iteration += 1
        b,v,s = pref.next()
        # wh.save_wave(s.cpu().numpy().astype(np.int16),"tempOutput/"+str(iteration)+".wav")
        end = time.time()
        print(start-end,iteration)
