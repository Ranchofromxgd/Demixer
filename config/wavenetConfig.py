import torch
import os


import datetime
class Config:
    model_name =  "DenseUnet" # "Unet" #"DenseUnet"
    if(model_name == "DenseUnet"):
        dense_block = 3
        dense_bn = 4
        dense_layers = 4
        dense_growth_rate = 10
        drop_rate = 0.2
        model_name_alias = model_name+"_"+str(dense_block)+\
                      "_"+str(dense_bn)+\
                      "_"+str(dense_layers)+\
                      "_"+str(dense_growth_rate)+\
                      "_"+str(drop_rate)+"_"

    OUTPUT_MASK = True
    test_path = "/home/work_nfs3/yhfu/dataset/musdb18hq/test/"
    train_path = "/home/work_nfs3/yhfu/dataset/musdb18hq/train/"
    background_fname = "background.wav"
    vocal_fname = "vocals.wav"
    epoches = 200
    use_gpu = True
    learning_rate = 0.0003
    accumulation_step = 5
    step_size = 60000
    gamma = 0.8
    sample_rate = 44100
    batch_size = 1
    num_workers = batch_size
    frame_length = 2

    project_root = "/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/"
    datahub_root = "/home/disk2/internship_anytime/liuhaohe/datasets/"

    # trail_name = 'phase_spleeter_musdb'
    cur = datetime.datetime.now()
    if(OUTPUT_MASK == False):trail_name = str(cur.year)+"_"+str(cur.month)+"_"+str(cur.day)+"_"+ model_name_alias+'_NM_spleeter_'
    else: trail_name = str(cur.year)+"_"+str(cur.month)+"_"+str(cur.day)+"_"+ model_name_alias+'_spleeter_'
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
        musdb_train_vocal,
        datahub_root + "datahub/song_vocal_data_44_1.txt", # 1440
        datahub_root + "datahub/k_pop.txt", # 44
        datahub_root + "datahub/干声（纯人声系）.txt",
        datahub_root + "datahub/干声素材！Acapella！音乐制作人工具包.txt",
        datahub_root + "datahub/[Rap清唱]感受最真实的声音.txt",
        datahub_root + "datahub/贺国丰（清唱陕北民歌）.txt",
    ]

    vocal_data += [datahub_root + "datahub/song_vocal_data_44_1.txt"]*3
    vocal_data += [musdb_train_vocal]*3

    background_data = [
        musdb_train_background,
        datahub_root + "datahub/Eminem歌曲纯伴奏单.txt",
        datahub_root + "datahub/超舒服的说唱伴奏（Rap Beat）.txt",
        datahub_root + "datahub/抖腿 | 刷题必听电音(无人声).txt",
        datahub_root + "datahub/pure_music_7.txt",
        datahub_root + "datahub/pure_music_1.txt",
        datahub_root + "datahub/pure_music_9.txt",
        datahub_root + "datahub/pure_music_8.txt",
        datahub_root + "datahub/Artpop(Intrumental).txt",
        datahub_root + "datahub/纯伴奏byLHH.txt",
        datahub_root + "datahub/纯伴奏byLHH.txt",
        datahub_root + "datahub/抖腿 | 刷题必听电音(无人声).txt",
        datahub_root + "datahub/Avril Lavigne Instrumental Version.txt",
    ]
    background_data += [datahub_root + "datahub/Avril Lavigne Instrumental Version.txt"]*2
    background_data += [datahub_root + "datahub/Eminem歌曲纯伴奏单.txt"]
    background_data += [datahub_root + "datahub/超舒服的说唱伴奏（Rap Beat）.txt"]
    background_data += [datahub_root + "datahub/抖腿 | 刷题必听电音(无人声).txt"]*3

    mu = 0.5
    sigma = 0.2
    alpha_low = 0.49
    alpha_high = 0.5

    device = torch.device("cuda:0" if use_gpu else "cpu")

    # config for stft and istft
    stft_frame_shift = 8
    stft_frame_length = 32

    # Reload pre-trained model
    start_point = 0
    # model
    layer_numbers_unet = 5
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
                +"mu"+str(mu)+"sig"+str(sigma)+"low"+str(alpha_low)+"hig"+str(alpha_high)
    temp = []



if __name__ == "__main__":
    print(Config.trail_name)
