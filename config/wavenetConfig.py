import torch
import os


import datetime


class Config:
    test_path = "/home/work_nfs3/yhfu/dataset/musdb18hq/test/"
    train_path = "/home/work_nfs3/yhfu/dataset/musdb18hq/train/"
    background_fname = "background.wav"
    vocal_fname = "vocals.wav"
    epoches = 25
    use_gpu = True
    learning_rate = 0.0003
    step_size = 120000
    gamma = 0.5
    sample_rate = 44100
    batch_size = 2
    num_workers = batch_size
    frame_length = 1.5
    empty_every_n = 50
    project_root = "/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/"
    datahub_root = "/home/disk2/internship_anytime/liuhaohe/datasets/"

    # trail_name = 'phase_spleeter_musdb'
    cur = datetime.datetime.now()
    trail_name = str(cur.year)+"_"+str(cur.month)+"_"+str(cur.day)+"_"+'phase_spleeter_'
    # Dataset
    #musdb18hq
    musdb_test_pth = datahub_root+"musdb18hq/test/"
    musdb_train_pth = datahub_root+"musdb18hq/train/"

    musdb_train_vocal =  datahub_root + "datahub/musdb_train_vocal.txt"
    musdb_train_background = datahub_root + "datahub/musdb_train_backtrack.txt"
    musdb_test_vocal = datahub_root +"datahub/musdb_test_vocal.txt"
    musdb_test_background = datahub_root +"datahub/musdb_test_backtrack.txt"

    # musdb: 100
    vocal_data = [
        datahub_root + "datahub/song_vocal_data_44_1.txt", # 1440
        datahub_root + "datahub/k_pop.txt", # 44
    ]
    background_data = [
        datahub_root + "datahub/Eminem歌曲纯伴奏单.txt",
        datahub_root + "datahub/超舒服的说唱伴奏（Rap Beat）.txt",
        # datahub_root + "datahub/抖腿 | 刷题必听电音(无人声).txt",
        # datahub_root + "datahub/pure_music_7.txt",
        # datahub_root + "datahub/pure_music_9.txt",
        # datahub_root + "datahub/纯伴奏byLHH.txt",
        datahub_root + "datahub/纯伴奏byLHH.txt",
    ]
    background_data += [datahub_root + "datahub/Avril Lavigne Instrumental Version.txt"]*2
    background_data += [datahub_root + "datahub/Eminem歌曲纯伴奏单.txt"]
    background_data += [datahub_root + "datahub/超舒服的说唱伴奏（Rap Beat）.txt"]
    # background_data += [datahub_root + "datahub/抖腿 | 刷题必听电音(无人声).txt"]*3

    device = torch.device("cuda:0" if use_gpu else "cpu")

    # config for stft and istft
    stft_frame_shift = 8
    stft_frame_length = 32

    # Reload pre-trained model
    start_point = 0 
    # model
    layer_numbers_unet = 6
    # Loss function
    '''
    l1: Frequency domain energy conservation l1 loss
    l2: Frequency domain l1 loss on background
    l3: Frequency domain l1 loss on vocal
    l4: Time domain sdr loss on background
    l5: Time domain sdr loss on vocal
    l6: Time domain energy conservation l1 loss
    l7: Time domain l1 loss on background
    l8: Time domain l1 loss on vocal
    '''
    loss_component = [
                        'l1',
                      'l2',
                      'l3',
                      # 'l4',
                      #  'l5',
                      # 'l6',
                      #  'l7',
                      #  'l8',
                      ]
    channels = 2

    for each in loss_component:
        trail_name += each+"_"
    trail_name.strip("_")
    trail_name+="lr"+str(learning_rate).split(".")[-1]+"_"\
                +"bs"+str(batch_size)+"_"\
                +"fl"+str(frame_length)+"_"\
                +"ss"+str(step_size)+"_"+str(gamma).split(".")[-1]\
                +"lnu"+str(layer_numbers_unet)\
                +"emptyEvery"+str(empty_every_n)

    temp = []



if __name__ == "__main__":
    print(Config.trail_name)
