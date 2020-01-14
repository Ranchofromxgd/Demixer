from models.spleeter import Spleeter
from dataloader import dataloader
from config.wavenetConfig import Config
from torch.utils.data import DataLoader
from util.dsp_torch import stft,istft
from util.wave_util import WaveHandler
from evaluate.si_sdr_torch import si_sdr
import torch
import os
import time
from util.spleeter_util import SpleeterUtil


def write_list(l,fname):
    with open(fname,'w') as f:
        for each in l:
            f.write(each+"\n")

def L1loss(estimate,target):
    if(estimate.size()!=target.size()):
        raise ValueError("Error: estimate and the target tensor shape should be asame")
    return torch.sum(torch.abs(target-estimate))/estimate.nelement()

if (not os.path.exists(Config.project_root + "saved_models/" + Config.trail_name)):
    os.mkdir(Config.project_root + "saved_models/" + Config.trail_name + "/")
    print("MakeDir: " + Config.project_root + "saved_models/" + Config.trail_name)

print("Vocal Data:")
for each in Config.vocal_data:
    print(each)
print("Background Data")
for each in Config.background_data:
    print(each)

write_list(Config.background_data, Config.project_root + "saved_models/" + Config.trail_name + "/" + "data_background.txt")
write_list(Config.vocal_data, Config.project_root + "saved_models/" + Config.trail_name + "/" + "data_vocal.txt")

loss = torch.nn.L1Loss()
# loss = torch.nn.MSELoss()

# Cache for data
wav_loss_cache = []
freq_loss_cache = []
wav_cons_loss_cache = []
freq_cons_loss_cache = []

sdr_background_cache, sdr_vocal_cache = [],[]

wh = WaveHandler()
model = Spleeter(channels=2,unet_inchannels=2,unet_outchannels=2).cuda(Config.device)
# print(model)
print(Config.model_name)
dl = torch.utils.data.DataLoader(
    dataloader.WavenetDataloader(),
    batch_size=Config.batch_size,
    shuffle=False,
    num_workers=Config.num_workers)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
optimizer.zero_grad()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.step_size, gamma=Config.gamma)

def save_and_evaluation():
    print("Save model")
    torch.save(model, "./saved_models/" + Config.trail_name + "/model" + str(model.cnt) + ".pkl")
    print("Start evaluation...")
    model.eval()
    su = SpleeterUtil(model_pth=model)
    sdr_background, sdr_vocal = su.evaluate(save_wav=True, save_json=False)
    su.Split_listener()
    sdr_background_cache.append(sdr_background)
    sdr_vocal_cache.append(sdr_vocal)
    print("REFERENCE FOR EARILY STOPPING:")
    print("sdr_vocal", sdr_vocal_cache)
    print("sdr_background", sdr_background_cache)
    model.train()
    del su

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
    output_track = []
    # For each channels
    for track_i in range(Config.channels):
        if(Config.OUTPUT_MASK): out = model.forward(track_i,target_song)* target_song
        else: out = model.forward(track_i,target_song)
        output_track.append(out)
        # All tracks is done
        if(track_i == Config.channels-1):
            wav_loss = 0
            freq_loss = 0
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
            if('l1' in Config.loss_component):
                lossVal = loss(output_track_sum, target_song)/Config.accumulation_step
                freq_cons_loss_cache.append(float(lossVal)*Config.accumulation_step)
            if('l2' in Config.loss_component):
                temp2 = loss(output_background,target_background)/Config.accumulation_step
                lossVal += temp2
                freq_loss += float(temp2)*Config.accumulation_step
            if('l3' in Config.loss_component):
                temp3 = loss(output_vocal,target_vocal)/Config.accumulation_step
                lossVal += temp3
                freq_loss_cache.append(freq_loss+float(temp3)*Config.accumulation_step)
            # if('l4' in Config.loss_component):
            #     val = -si_sdr(output_t_background.float()[:,:min_length_background],target_t_background.float()[:,:min_length_background])
            #     lossVal = val
            # if('l5' in Config.loss_component):
            #     val = -si_sdr(output_t_vocal.float()[:,:min_length_vocal],target_t_vocal.float()[:,:min_length_vocal])
            #     lossVal += val
            if('l6' in Config.loss_component):
                temp6 = L1loss(output_t_background.float()[:,:min_length_vocal_background]+output_t_vocal.float()[:,:min_length_vocal_background],target_t_song.float()[:,:min_length_vocal_background])
                lossVal += temp6
                wav_cons_loss_cache.append(float(temp6))
            if('l7' in Config.loss_component):
                val7 = L1loss(output_t_background.float()[:,:min_length_background],target_t_background.float()[:,:min_length_background])
                lossVal += val7
                wav_loss += float(val7)
            if('l8' in Config.loss_component):
                val8 = L1loss(output_t_vocal.float()[:,:min_length_vocal],target_t_vocal.float()[:,:min_length_vocal])
                lossVal += val8
                wav_loss_cache.append(wav_loss+float(val8))
            # Backward
            lossVal.backward()
            if(model.cnt%Config.accumulation_step == 0 and model.cnt != Config.start_point):
                # Optimize
                optimizer.step()
                optimizer.zero_grad()

if(not Config.start_point == 0):
    model = torch.load(Config.project_root+"saved_models/"+Config.trail_name+"/model"+str(Config.start_point)+".pkl",
                               map_location=Config.device)
    print("Start from ",model.cnt)

every_n = 100
t0 = time.time()
for epoch in range(Config.epoches):
    print("EPOCH: ", epoch)
    start = time.time()
    pref = dataloader.data_prefetcher(dl)
    background, vocal, song = pref.next()
    while(background is not None):
        if (model.cnt % every_n == 0 and model.cnt != Config.start_point):#and model.cnt != Config.start_point
            t1 = time.time()
            print(str(model.cnt)+" - Raw wav L1loss", (sum(wav_loss_cache[-every_n:]) / every_n),
                  "\tRaw wav conserv-loss", (sum(wav_cons_loss_cache[-every_n:]) / every_n),
                  "\tFreq L1loss", (sum(freq_loss_cache[-every_n:]) / every_n),
                  "\tFreq conserv-loss", (sum(freq_cons_loss_cache[-every_n:]) / every_n),
                  "\tlr:",optimizer.param_groups[0]['lr'],
                  "\tspeed:",(every_n*Config.frame_length*Config.batch_size)/(t1-t0))
            wav_loss_cache = []
            freq_loss_cache = []
            wav_cons_loss_cache = []
            freq_cons_loss_cache = []
            if (model.cnt % 12000 == 0):
                save_and_evaluation()
            t0 = time.time()
        # f_background, f_vocal, f_song = stft(background.float(),sample_rate=Config.sample_rate),stft(vocal.float(),sample_rate=Config.sample_rate),stft(song.float(),sample_rate=Config.sample_rate)
        f_background, f_vocal, f_song = background, vocal, song
        train(target_background=f_background,
              target_vocal=f_vocal, 
              target_song=f_song,
              target_t_background=background,
              target_t_vocal=vocal,
              target_t_song=song)
        scheduler.step()
        background, vocal, song = pref.next()
        model.cnt += 1
    end = time.time()
    print("Epoch "+str(epoch)+" finish, total time: "+str(end-start))

