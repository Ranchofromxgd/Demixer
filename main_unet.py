from models.spleeter import Spleeter
from dataloader.wavenet import dataloader
from config.wavenetConfig import Config
from torch.utils.data import DataLoader
from util.dsp import stft,istft,spectrom2magnitude,seperate_magnitude
from util.wave_util import WaveHandler
from evaluate.si_sdr_torch import si_sdr
from util.wave_util import save_pickle,load_pickle
from models.unet_model import UNet
import torch
import os
import numpy as np
import torchaudio
from util.spleeter_util import SpleeterUtil

if(not Config.trail_name[:4] == "unet"):
    raise ValueError("trail name should start with: unet")

if(not os.path.exists("./saved_models/"+Config.trail_name)):
    os.mkdir("./saved_models/"+Config.trail_name+"/")
    print("MakeDir: "+"./saved_models/"+Config.trail_name)
sigmoid = torch.nn.Sigmoid()
loss = torch.nn.L1Loss()
loss_cache = []
background_sisdr = []
vocal_sisdr = []
wh = WaveHandler()
model = UNet(n_channels = 2,n_classes = 2).cuda(Config.device)
dl = torch.utils.data.DataLoader(dataloader.WavenetDataloader(empty_every_n=Config.empty_every_n), batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.step_size, gamma=Config.gamma)

def train( # Frequency domain
            target_mixed,
            target_vocal,
            target_song,
             # Time domain
            target_t_mixed,
            target_t_vocal,
            target_t_song
            ):
    # Put data on GPU
    if (not Config.device == 'cpu'):
        target_song = target_song.cuda(Config.device)
        target_vocal = target_vocal.cuda(Config.device)
        target_t_vocal = target_t_vocal.cuda(Config.device)
    # For each channels
        output_vocal = sigmoid(model.forward(target_song)) * target_song
        # All tracks is done
            # Reconstruct to Time domain
        if('l5' in Config.loss_component or 'l8' in Config.loss_component):
            output_t_vocal = istft(output_vocal
                                   ,sample_rate=Config.sample_rate
                                   ,use_gpu=True)
            min_length_vocal = min(output_t_vocal.size()[1], target_t_vocal.size()[1])
        if('l3' in Config.loss_component):lossVal = loss(output_vocal,target_vocal)
        if('l5' in Config.loss_component):
            val = -si_sdr(output_t_vocal.float()[:,:min_length_vocal],target_t_vocal.float()[:,:min_length_vocal])
            lossVal += val
            vocal_sisdr.append(float(val))
        if('l8' in Config.loss_component):
            val = loss(output_t_vocal.float()[:,:min_length_vocal],target_t_vocal.float()[:,:min_length_vocal])
            lossVal += val
        # Backward
        loss_cache.append(float(lossVal))
        optimizer.zero_grad()
        lossVal.backward()
        optimizer.step()

if(not Config.start_point == 0):
    model = torch.load("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/saved_models/"+Config.trail_name+"/model"+str(Config.start_point)+".pkl",
                               map_location=Config.device)
    print("Start from ",model.cnt)

for epoch in range(Config.epoches):
    print("EPOCH: ",epoch)
    # su = SpleeterUtil(model_pth = model)
    # su.evaluate()
    for mixed,vocal,song in dl:
        # [4, 1025, 188, 2]
        f_mixed, f_vocal, f_song = stft(mixed.float(),sample_rate=Config.sample_rate),stft(vocal.float(),sample_rate=Config.sample_rate),stft(song.float(),sample_rate=Config.sample_rate)
        train(target_mixed=f_mixed,
              target_vocal=f_vocal,
              target_song=f_song,
              target_t_mixed=mixed,
              target_t_vocal=vocal,
              target_t_song=song)
        scheduler.step()
        every_n = 100
        if(model.cnt % every_n == 0 and not model.cnt == 0 ):
            print("Loss",(sum(loss_cache[-every_n:])/every_n),
                  "SI-SDR:",
                  "bak",(sum(background_sisdr[-every_n:])/every_n),
                  "voc",(sum(vocal_sisdr[-every_n:])/every_n))
            if (model.cnt % 2000 == 0 and not model.cnt == 0):
                print("Save model")
                torch.save(model,"./saved_models/"+Config.trail_name+"/model" + str(model.cnt) + ".pkl")
        model.cnt += 1
