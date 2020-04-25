import sys
sys.path.append("..")
from models.spleeter import Spleeter,Mmasker
from dataloader import dataloader
from torch.utils.data import DataLoader
from util.wave_util import WaveHandler,load_json
from models.subband_util import before_forward_f,after_forward_f
import torch
import time
from util.separation_util import SeparationUtil,save_params
from dataloader.construct_dataset import get_total_time_in_txt
import logging
import os
from config.mainConfig import Config

vocal_data = list(set(Config.vocal_data))
back_data = list(set(Config.background_data))

t_vocal,t_back = 0,0
number_vocal,number_back = 0,0
for each in vocal_data:
    print(each)
    vocal,num = get_total_time_in_txt(each)
    t_vocal+= vocal
    number_vocal += num

for each in back_data:
    print(each)
    vocal,num = get_total_time_in_txt(each)
    t_back += vocal
    number_back += num

print(t_vocal,t_back)
print(number_vocal,number_back)

if (not os.path.exists(Config.project_root + "saved_models/" + Config.trail_name)):
    os.mkdir(Config.project_root + "saved_models/" + Config.trail_name + "/")
    print("MakeDir: " + Config.project_root + "saved_models/" + Config.trail_name)

print("Vocal Data:")
for each in Config.vocal_data:
    print(each)
print("Background Data")
for each in Config.background_data:
    print(each)

def write_list(l,fname):
    with open(fname,'w') as f:
        for each in l:
            f.write(each+"\n")


write_list(Config.background_data, Config.project_root + "saved_models/" + Config.trail_name + "/" + "data_background.txt")
write_list(Config.vocal_data, Config.project_root + "saved_models/" + Config.trail_name + "/" + "data_vocal.txt")

# Cache for data
freq_bac_loss_cache = []
freq_voc_loss_cache = []
freq_cons_loss_cache = []

best_sdr_vocal,best_sdr_background = Config.best_sdr_vocal, Config.best_sdr_background

# exclude_dict = load_json("config/json/ExcludeData.json")
# exclude_start_point,vocal_sisdr_min,vocal_sisdr_max,background_sisdr_min,background_sisdr_max = exclude_dict["start_exclude_point"],exclude_dict["vocal_sisdr"][0],exclude_dict["vocal_sisdr"][1],exclude_dict["background_sisdr"][0],exclude_dict["background_sisdr"][1]

wh = WaveHandler()
loss = torch.nn.L1Loss()

if(not Config.start_point == 0):
    model = torch.load(Config.load_model_path+"/model"+str(Config.start_point)+".pkl",
                               map_location=Config.device)
else:
    if(Config.split_band):
        model = Spleeter(channels=2, unet_inchannels=2*Config.subband, unet_outchannels=2*Config.subband).cuda(Config.device)
    else:
        model = Spleeter(channels=2, unet_inchannels=2, unet_outchannels=2).cuda(Config.device)

print("Model structure: \n",model)
print("Start from ",model.cnt,Config.model_name)

dl = torch.utils.data.DataLoader(
    dataloader.WavenetDataloader(),
    batch_size=Config.batch_size,
    shuffle=True,
    num_workers=Config.num_workers)

optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
optimizer.zero_grad()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.step_size, gamma=Config.gamma)

def save_and_evaluation(save_wav=False):
    print("Save model")
    torch.save(model, "./saved_models/" + Config.trail_name + "/model" + str(model.cnt) + ".pkl")
    su = SeparationUtil(load_model_pth=model)
    print("Start evaluation...")
    try:
        su.evaluate(save_wav=save_wav, save_json=True,split_band=Config.split_band)
        if(save_wav == True):su.Split_listener()
    except Exception as e:
        print("Error occured while evaluating...")
        logging.exception(e)
    del su

def train( # Time Domain
            target_background,
            target_vocal,
            target_song,
            ):
    gt_bac, gt_voc, gt_song = before_forward_f(target_background, target_vocal, target_song,
                                               use_subband=Config.split_band,
                                               use_gpu=True,
                                               subband_num=Config.subband,
                                               sample_rate=Config.sample_rate)
    if('l1' in Config.loss_component):
        output_track = []
        for track_i in range(Config.channels):
            mask = model(track_i,gt_song)
            out = mask*gt_song
            output_track.append(out)
            # All tracks is done
            if(track_i == Config.channels-1):
                # Preprocessing
                output_track_sum = sum(output_track)
                output_background = output_track[0]
                output_vocal = output_track[1]
                # Loss function: Energy conservation & mixed spectrom & vocal spectrom & background wave & vocal wave
                if('l1' in Config.loss_component):
                    lossVal = loss(output_track_sum, gt_song)/Config.accumulation_step
                    freq_cons_loss_cache.append(float(lossVal)*Config.accumulation_step)
                if('l2' in Config.loss_component):
                    temp2 = loss(output_background,gt_bac)/Config.accumulation_step
                    lossVal += temp2
                    freq_bac_loss_cache.append(float(temp2)*Config.accumulation_step)
                if('l3' in Config.loss_component):
                    temp3 = loss(output_vocal,gt_voc)/Config.accumulation_step
                    lossVal += temp3
                    freq_voc_loss_cache.append(float(temp3)*Config.accumulation_step)
        # Backward
        lossVal.backward()
        if(model.cnt%Config.accumulation_step == 0 and model.cnt != Config.start_point):
            # Optimize
            optimizer.step()
            optimizer.zero_grad()
        return True
    else:
        #An momory efficient version
        for track_i in range(Config.channels):
            mask = model(track_i,gt_song)
            out = mask*gt_song
            if(track_i == 1):
                lossVal = loss(out,gt_voc)
                freq_voc_loss = float(lossVal)
            else:
                lossVal = loss(out,gt_bac)
                freq_bac_loss = float(lossVal)
            # Backward
            lossVal.backward()
            # Optimize
            optimizer.step()
            optimizer.zero_grad()
        freq_bac_loss_cache.append(freq_bac_loss)
        freq_voc_loss_cache.append(freq_voc_loss)
        return True

every_n = 10

t0 = time.time()
for epoch in range(Config.epoches):
    print("EPOCH: ", epoch)
    start = time.time()
    pref = dataloader.data_prefetcher(dl)
    background, vocal, song,name = pref.next()
    while(background is not None):
        if (model.cnt % every_n == 0 and model.cnt != Config.start_point):#and model.cnt != Config.start_point
            t1 = time.time()
            print(str(model.cnt)+
                  " Freq L1loss voc", format((sum(freq_voc_loss_cache[-every_n:]) / every_n), '.7f'),
                  " Freq L1loss bac", format((sum(freq_bac_loss_cache[-every_n:]) / every_n), '.7f'),
                  " Freq conserv-loss", format((sum(freq_cons_loss_cache[-every_n:]) / every_n), '.7f'),
                  " lr:",optimizer.param_groups[0]['lr'],
                  " speed:",format((Config.frame_length*Config.batch_size)/(t1-t0), '.2f'))
            freq_voc_loss_cache = []
            freq_bac_loss_cache = []
            freq_cons_loss_cache = []
        if (model.cnt % Config.validation_interval == 0 and model.cnt != Config.start_point):
            if(model.cnt % (Config.validation_interval*4) == 0):
                save_and_evaluation(save_wav=True)
            else:
                save_and_evaluation(save_wav=False)
            # max_tolerate_validation = 8
            # if(len(validation_result_cache)>max_tolerate_validation and sum(validation_result_cache[-max_tolerate_validation:]) == 0):
            #     print("Continuelly appeared bad result on validation set! Not improved in ",max_tolerate_validation," tries")
            #     raise RuntimeError("Train process terminate with exit code 0")
        t0 = time.time()
        status = train(
              target_background=background,
              target_vocal=vocal,
              target_song=song)
        # If the training data is not clean
        if(not status):
            background, vocal, song,name = pref.next()
            continue
        background, vocal, song, name = pref.next()
        if(model.cnt>100):
            scheduler.step()
        model.cnt += 1
    end = time.time()
    print("Epoch "+str(epoch)+" finish, total time: "+str(end-start))
