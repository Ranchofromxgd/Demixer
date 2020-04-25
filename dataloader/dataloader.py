import sys
sys.path.append("..")
from util.wave_util import WaveHandler
from config.mainConfig import Config
from torch.utils.data import Dataset
# These part should below 'import util'
import os
import time
import torch
import numpy as np
from util.wave_util import load_json
from util.dsp_torch import stft

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(Config.device)
        self.preload()

    def preload(self):
        try:
            self.next_background, self.next_vocal,self.next_song,self.next_name = next(self.loader)
        except StopIteration:
            self.next_background = None
            self.next_vocal = None
            self.next_song= None
            self.next_name= None
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
        name = self.next_name
        if background is not None:
            background.record_stream(torch.cuda.current_stream(Config.device))
        if vocal is not None:
            vocal.record_stream(torch.cuda.current_stream(Config.device))
        if song is not None:
            song.record_stream(torch.cuda.current_stream(Config.device))
        self.preload()
        if(not Config.MODE_CLEAN_DATA):return background, vocal,song
        else:return background, vocal,song,name

class WavenetDataloader(Dataset):
    def __init__(self,
                 frame_length = Config.frame_length,
                 sample_rate = Config.sample_rate,
                 num_worker = Config.num_workers,
                 sampleNo=20000,
                 mu = Config.mu,
                 empty_every_n = 50,
                 sigma = Config.sigma,
                 alpha_low = Config.alpha_low,
                 alpha_high = Config.alpha_high # If alpha_high get a value greater than 0.5, it would have probability to overflow
    ):
        np.random.seed(1)
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        # self.music_folders = self.readList(Config.musdb_train_background)
        self.music_folders = []
        for each in Config.background_data:
            self.music_folders += self.readList(each)
        self.vocal_folders = []
        for each in Config.vocal_data:
            self.vocal_folders += self.readList(each)
        # prev_data_size = len(self.vocal_folders)
        # if(Config.exclude_list != ""):
        #     for each in self.readList(Config.exclude_list):
        #         self.vocal_folders.remove(each)
        # print(prev_data_size-len(self.vocal_folders)," songs were removed from vocal datasets")
        self.sample_length = int(self.sample_rate*self.frame_length)
        self.cnt = 0
        self.data_counter = 0
        self.empty_every_n = empty_every_n
        self.sampleNo = sampleNo
        self.num_worker = num_worker
        self.wh = WaveHandler()
        # This alpha is to balance the energy between vocal and background
        # Also, this alpha is used to simulate different energy leval between vocal and background
        self.normal_distribution = np.random.normal(mu, sigma, sampleNo)
        self.normal_distribution = self.normal_distribution[self.normal_distribution > alpha_low]
        self.normal_distribution = self.normal_distribution[self.normal_distribution < alpha_high]
        self.sampleNo = self.normal_distribution.shape[0]

    def update_empty_n(self):
        # dataloader_dict = load_json(Config.project_root+"config/json/Dataloader.json")
        # self.empty_every_n = dataloader_dict["empty_every_n"]
        pass

    def __getitem__(self, item):
        self.data_counter += 1
        # np.random.seed(os.getpid()+self.cnt)
        # Select background(background only) and vocal file randomly
        self.cnt += self.num_worker
        while(True):
            random_music = np.random.randint(0,len(self.music_folders))
            random_vocal = np.random.randint(0,len(self.vocal_folders))
            music_fname = self.music_folders[random_music]
            vocal_fname = self.vocal_folders[random_vocal]
            music_length = self.wh.get_duration(music_fname)
            vocal_length = self.wh.get_duration(vocal_fname)
            if((music_length - self.frame_length) <= 0 or (vocal_length - self.frame_length) <= 0):continue
            else:
                music_sr,vocal_sr = self.wh.get_sample_rate(music_fname),self.wh.get_sample_rate(vocal_fname)
                music_length,vocal_length = music_length * music_sr , vocal_length * vocal_sr
                break
        background_start = np.random.randint(0, music_length - self.sample_length)
        # print(background_start,background_start + self.frame_length * music_sr,self.wh.get_channels(music_fname))
        background_crop = self.wh.read_wave(music_fname,
                                            portion_start=background_start,
                                            portion_end=background_start + self.frame_length * music_sr,
                                            channel=self.wh.get_channels(music_fname),
                                            sample_rate=self.sample_rate)
        # if (self.cnt % self.empty_every_n == 0):
        #     return background_crop,np.zeros(background_crop.shape).astype(np.int16),background_crop,(music_fname,"_empty_")

        vocal_start = np.random.randint(0, vocal_length - self.frame_length*vocal_sr)
        vocal_crop = self.wh.read_wave(vocal_fname,
                                       portion_start=vocal_start,
                                       portion_end=vocal_start+self.frame_length*vocal_sr,
                                       channel = self.wh.get_channels(vocal_fname),
                                       sample_rate=self.sample_rate).astype(np.float32)
        max_background = np.max(np.abs(background_crop))
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
        # if(not Config.time_domain_loss):
        #     b,v,s = stft(b.float(),Config.sample_rate),stft(v.float(),Config.sample_rate),stft(s.float(),Config.sample_rate)

        return b,v,s ,(music_fname,vocal_fname)

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
    import psutil

    wh = WaveHandler()
    s = WavenetDataloader()
    dl = torch.utils.data.DataLoader(s, batch_size=Config.batch_size, shuffle=False, num_workers=1)
    start = 0
    cnt = 0
    for each in dl:
        end = time.time()
        # print(each[-1])
        print("here")
        start = time.time()

    # pref = data_prefetcher(dl)
    # b,v,s,n = pref.next()
    # iteration = 0
    #
    # while s is not None:
    #     start = time.time()
    #     iteration += 1
    #     b,v,s,n = pref.next()
    #     print(n)
    #     # wh.save_wave(s.cpu().numpy().astype(np.int16),"tempOutput/"+str(iteration)+".wav")
    #     end = time.time()
    #     print(start-end,iteration)
