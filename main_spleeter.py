from models.spleeter import Spleeter
from dataloader.wavenet import dataloader
from config.wavenetConfig import Config
from torch.utils.data import DataLoader
from util.dsp import stft,istft,spectrom2magnitude,seperate_magnitude
from util.wave_util import WaveHandler
from evaluate.si_sdr_torch import si_sdr
from util.wave_util import save_pickle,load_pickle
import torch
import os
import numpy as np
import torchaudio
from util.spleeter_util import SpleeterUtil

if(not os.path.exists("./saved_models/"+Config.trail_name)):
    os.mkdir("./saved_models/"+Config.trail_name+"/")
    print("MakeDir: "+"./saved_models/"+Config.trail_name)
loss = torch.nn.L1Loss()
loss_cache = []
background_sisdr = []
vocal_sisdr = []
wh = WaveHandler()
model = Spleeter(channels=2,unet_inchannels=2,unet_outchannels=2).cuda(Config.device)
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
        target_mixed = target_mixed.cuda(Config.device)
        target_vocal = target_vocal.cuda(Config.device)
        target_t_mixed = target_t_mixed.cuda(Config.device)
        target_t_vocal = target_t_vocal.cuda(Config.device)
        target_t_song = target_t_song.cuda(Config.device)
    output_track = []
    # For each channels
    for track_i in range(Config.channels):
        out = model.forward(track_i,target_song)* target_song
        output_track.append(out)
        # All tracks is done
        if(track_i == Config.channels-1):
            # Preprocessing
            if (type(output_track) is list):
                output_track_sum = sum(output_track)
            output_mixed = output_track[0]
            output_vocal = output_track[1]
            # Reconstruct to Time domain
            if('l4' in Config.loss_component
                    or 'l5' in Config.loss_component
                    or 'l6' in Config.loss_component
                    or 'l7' in Config.loss_component
                    or 'l8' in Config.loss_component):
                output_t_mixed = istft(output_mixed
                                       ,sample_rate=Config.sample_rate
                                       ,use_gpu=True)
                output_t_vocal = istft(output_vocal
                                       ,sample_rate=Config.sample_rate
                                       ,use_gpu=True)
                min_length_mixed = min(output_t_mixed.size()[1], target_t_mixed.size()[1])
                min_length_vocal = min(output_t_vocal.size()[1], target_t_vocal.size()[1])
                min_length_vocal_mixed = min(min_length_mixed, min_length_vocal)
            # Loss function: Energy conservation & mixed spectrom & vocal spectrom & mixed wave & vocal wave
            if('l1' in Config.loss_component):lossVal = loss(output_track_sum, target_song)
            if('l2' in Config.loss_component):lossVal = loss(output_mixed,target_mixed)
            if('l3' in Config.loss_component):lossVal += loss(output_vocal,target_vocal)
            if('l4' in Config.loss_component):
                val = -si_sdr(output_t_mixed.float()[:,:min_length_mixed],target_t_mixed.float()[:,:min_length_mixed])
                lossVal = val
                background_sisdr.append(float(val))
            if('l5' in Config.loss_component):
                val = -si_sdr(output_t_vocal.float()[:,:min_length_vocal],target_t_vocal.float()[:,:min_length_vocal])
                lossVal += val
                vocal_sisdr.append(float(val))
            if('l6' in Config.loss_component):
                lossVal += loss(output_t_mixed.float()[:,:min_length_vocal_mixed]+output_t_vocal.float()[:,:min_length_vocal_mixed],target_t_song.float()[:,:min_length_vocal_mixed])
            if('l7' in Config.loss_component):
                val = loss(output_t_mixed.float()[:,:min_length_mixed],target_t_mixed.float()[:,:min_length_mixed])
                lossVal = val
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
        every_n = 10
        if(model.cnt % every_n == 0):
            print("Loss",(sum(loss_cache[-every_n:])/every_n),
                  "SI-SDR:",
                  "bak",(sum(background_sisdr[-every_n:])/every_n),
                  "voc",(sum(vocal_sisdr[-every_n:])/every_n))
            if (model.cnt % 2000 == 0 and not model.cnt == 0):
                print("Save model")
                torch.save(model,"./saved_models/"+Config.trail_name+"/model" + str(model.cnt) + ".pkl")
        model.cnt += 1
