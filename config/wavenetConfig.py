import torch
import os

class Config:
    test_path = "/home/work_nfs3/yhfu/dataset/musdb18hq/test/"
    train_path = "/home/work_nfs3/yhfu/dataset/musdb18hq/train/"
    mix_fname = "mixed.wav"
    vocal_fname = "vocals.wav"
    epoches = 25
    use_gpu = True
    num_workers = 16
    learning_rate = 0.001
    step_size = 8000
    gamma = 0.85
    sample_rate = 44100
    batch_size = 4
    frame_length = 1.5
    empty_every_n = 50
    project_root = "/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/"
    datahub_root = "/home/disk2/internship_anytime/liuhaohe/datasets/datahub/"

    trail_name = 'phase_spleeter_musdb'
    netease_included = [7]
    # Dataset
    #musdb18hq
    musdb_test_pth = "/home/disk2/internship_anytime/liuhaohe/datasets/musdb18hq/test/"
    # musdb: 100
    musdb_train_vocal =  datahub_root + "musdb_train_vocal.txt"
    musdb_train_background = datahub_root + "musdb_train_backtrack.txt"
    musdb_test_vocal = datahub_root +"musdb_test_vocal.txt"
    musdb_test_background = datahub_root +"musdb_test_backtrack.txt"
    # song_vocal: 1440
    song_vocal_pth = datahub_root + "song_vocal_data_44_1.txt"
    # kpop: 44
    kpop_vocal_pth = datahub_root + "k_pop.txt"
    # Netease music
    netease_background_pth = []
    for i in netease_included:
        netease_background_pth.append(datahub_root+"pure_music_"+str(i)+".txt")
    if(not len(netease_included) == 0):
        trail_name+="_netease"
        for each in netease_included:
           trail_name+= str(each)
        trail_name += "_"

    # # mir1k
    # mir1k_music = "/home/work_nfs3/yhfu/workspace/multichannel_separation/speech_noise_rir_list/mir1k_music.txt"
    # mir1k_vocal = "/home/work_nfs3/yhfu/workspace/multichannel_separation/speech_noise_rir_list/mir1k_vocal.txt"
    # visiable_device = os.environ["CUDA_VISIBLE_DEVICES"]
    # dev_num = os.environ["CUDA_VISIBLE_DEVICES"]
    # print("device",dev_num)
    device = torch.device("cuda:1" if use_gpu else "cpu")

    # config for stft and istft
    stft_frame_shift = 8
    stft_frame_length = 32

    # Reload pre-trained model
    start_point = 0
    # model
    layer_numbers_unet = 5
    # Loss function
    '''
    l1: frequency domain energy conservation l1 loss
    l2: frequency domain l1 loss on background
    l3: frequency domain l1 loss on vocal
    l4: time domain sisdr loss on background
    l5: time domain sisdr loss on vocal
    l6: time domain energy conservation l1 loss
    l7: time domain l1 loss on background
    l8: time domain l1 loss on vocal
    '''
    loss_component = [
                        # 'l1',
                      # 'l2',
                      # 'l3',
                      # 'l4',
                      #  'l5',
                      # 'l6',
                        'l7',
                         'l8',
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
