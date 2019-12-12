from models.spleeter import Spleeter
from dataloader import dataloader
from config.wavenetConfig import Config
from torch.utils.data import DataLoader
from util.dsp_torch import stft,istft
from util.wave_util import WaveHandler
from evaluate.si_sdr_torch import si_sdr
import torch
import os
from util.spleeter_util import SpleeterUtil

if(not os.path.exists("./saved_models/"+Config.trail_name)):
    os.mkdir("./saved_models/"+Config.trail_name+"/")
    print("MakeDir: "+"./saved_models/"+Config.trail_name)

loss = torch.nn.L1Loss()

# Cache for data
loss_cache = []
sdr_background_cache, sdr_vocal_cache = [],[]

wh = WaveHandler()
model = Spleeter(channels=2,unet_inchannels=2,unet_outchannels=2).cuda(Config.device)
dl = torch.utils.data.DataLoader(
    dataloader.WavenetDataloader(empty_every_n=Config.empty_every_n),
    batch_size=Config.batch_size,
    shuffle=False,
    num_workers=Config.num_workers)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.step_size, gamma=Config.gamma)

def save_and_evaluation():
    print("Save model")
    torch.save(model, "./saved_models/" + Config.trail_name + "/model" + str(model.cnt) + ".pkl")
    print("Start evaluation...")
    model.eval()
    su = SpleeterUtil(model_pth=model)
    sdr_background, sdr_vocal = su.evaluate(save_wav=False, save_json=False)
    sdr_background_cache.append(sdr_background)
    sdr_vocal_cache.append(sdr_vocal)
    print("REFERENCE FOR EARILY STOPPING:")
    print("sdr_vocal", sdr_vocal_cache)
    print("sdr_background", sdr_background_cache)
    model.train()

def train( # Frequency domain
            target_background,
            target_vocal,
            target_song,
             # Time domain
            target_t_background,
            target_t_vocal,
            target_t_song
            ):
    # Put data on GPU
    if (not Config.device == 'cpu'):
        target_song = target_song.cuda(Config.device)
        target_background = target_background.cuda(Config.device)
        target_vocal = target_vocal.cuda(Config.device)
        target_t_background = target_t_background.cuda(Config.device)
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
            output_background = output_track[0]
            output_vocal = output_track[1]
            # Reconstruct to Time domain
            if('l4' in Config.loss_component
                    or 'l5' in Config.loss_component
                    or 'l6' in Config.loss_component
                    or 'l7' in Config.loss_component
                    or 'l8' in Config.loss_component):
                output_t_background = istft(output_background
                                       ,sample_rate=Config.sample_rate
                                       ,use_gpu=True)
                output_t_vocal = istft(output_vocal
                                       ,sample_rate=Config.sample_rate
                                       ,use_gpu=True)
                min_length_background = min(output_t_background.size()[1], target_t_background.size()[1])
                min_length_vocal = min(output_t_vocal.size()[1], target_t_vocal.size()[1])
                min_length_vocal_background = min(min_length_background, min_length_vocal)
            # Loss function: Energy conservation & mixed spectrom & vocal spectrom & background wave & vocal wave
            if('l1' in Config.loss_component):lossVal = loss(output_track_sum, target_song)
            if('l2' in Config.loss_component):lossVal = loss(output_background,target_background)
            if('l3' in Config.loss_component):lossVal += loss(output_vocal,target_vocal)
            if('l4' in Config.loss_component):
                val = -si_sdr(output_t_background.float()[:,:min_length_background],target_t_background.float()[:,:min_length_background])
                lossVal = val
            if('l5' in Config.loss_component):
                val = -si_sdr(output_t_vocal.float()[:,:min_length_vocal],target_t_vocal.float()[:,:min_length_vocal])
                lossVal += val
            if('l6' in Config.loss_component):
                lossVal += loss(output_t_background.float()[:,:min_length_vocal_background]+output_t_vocal.float()[:,:min_length_vocal_background],target_t_song.float()[:,:min_length_vocal_background])
            if('l7' in Config.loss_component):
                val = loss(output_t_background.float()[:,:min_length_background],target_t_background.float()[:,:min_length_background])
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
    for background,vocal,song in dl:
        # [4, 1025, 188, 2]
        f_background, f_vocal, f_song = stft(background.float(),sample_rate=Config.sample_rate),stft(vocal.float(),sample_rate=Config.sample_rate),stft(song.float(),sample_rate=Config.sample_rate)
        train(target_background=f_background,
              target_vocal=f_vocal, 
              target_song=f_song,
              target_t_background=background,
              target_t_vocal=vocal,
              target_t_song=song)
        scheduler.step()
        every_n = 10
        if(model.cnt % every_n == 0):
            print("Loss",(sum(loss_cache[-every_n:])/every_n))
            if (model.cnt % 3000 == 0 and not model.cnt == 0):
                save_and_evaluation()
        model.cnt += 1


