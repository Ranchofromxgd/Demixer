import torch
import os


import datetime
class Config:
    # Project configurations
    project_root = "/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/"
    datahub_root = "/home/disk2/internship_anytime/liuhaohe/datasets/"
    # Model configurations
    model_name =  "Demixer" # "Unet" #"Demixer"
    if(model_name == "Demixer"):
        block = "DNN"
        dense_block = 2
        dense_bn = 4
        dense_layers = 4
        dense_growth_rate = 12
        drop_rate = 0.2
        model_name_alias = model_name+"_"+block+"_"+str(dense_block)+\
                      "_"+str(dense_bn)+\
                      "_"+str(dense_layers)+\
                      "_"+str(dense_growth_rate)+\
                      "_"+str(drop_rate)+"_"
    else:
        model_name_alias = "_unet_spleeter_"
    # Reload pre-trained model
    load_model_path = project_root+"saved_models/1_2020_1_18_Demixer_DNN_2_4_4_12_0.2_spleeter_sf0_l1_l2_l3_lr0003_bs1_fl3_ss60000_8lnu5mu0.5sig0.2low0.3hig0.5"
    start_point  = 468000
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
                       'l7',
                       'l8',
                      ]
    if ('l4' in loss_component
            or 'l5' in loss_component
            or 'l6' in loss_component
            or 'l7' in loss_component
            or 'l8' in loss_component):
        time_domain_loss = True
    else:
        time_domain_loss = False
    channels = 2
    OUTPUT_MASK = True
    background_fname = "background.wav"
    vocal_fname = "vocals.wav"
    epoches = 200
    use_gpu = True
    learning_rate = 0.0005
    accumulation_step = 100
    step_size = 120000
    gamma = 0.8
    sample_rate = 44100
    batch_size = 1
    num_workers = batch_size
    frame_length = 3
    device_str = "cuda:1"
    device = torch.device( device_str if use_gpu else "cpu")

    # Dataset
    ## musdb18hq
    musdb_test_pth = datahub_root+"musdb18hq/test/"
    musdb_train_pth = datahub_root+"musdb18hq/train/"
    musdb_train_vocal =  datahub_root + "datahub/musdb_train_vocal.txt"
    musdb_train_background = datahub_root + "datahub/musdb_train_backtrack.txt"
    musdb_test_vocal = datahub_root +"datahub/musdb_test_vocal.txt"
    musdb_test_background = datahub_root +"datahub/musdb_test_backtrack.txt"

    ## Config data path
    ### vocal data
    vocal_data = [
        musdb_train_vocal,
        musdb_train_vocal,
        musdb_train_vocal,
        musdb_train_vocal,
        musdb_train_vocal,
        musdb_train_vocal,
        musdb_train_vocal,
        musdb_train_vocal,
        musdb_train_vocal,
        datahub_root + "datahub/song_vocal_data_44_1.txt", # 1440
        datahub_root + "datahub/song_vocal_data_44_1.txt", # 1440
        datahub_root + "datahub/song_vocal_data_44_1.txt", # 1440
        datahub_root + "datahub/song_vocal_data_44_1.txt", # 1440
        datahub_root + "datahub/song_vocal_data_44_1.txt", # 1440
        datahub_root + "datahub/song_vocal_data_44_1.txt", # 1440
        datahub_root + "datahub/k_pop.txt", # 44
        datahub_root + "datahub/k_pop.txt", # 44
        datahub_root + "datahub/k_pop.txt", # 44
        datahub_root + "datahub/k_pop.txt", # 44
        datahub_root + "datahub/k_pop.txt", # 44
        datahub_root + "datahub/k_pop.txt", # 44
        datahub_root + "datahub/k_pop.txt", # 44
        datahub_root + "datahub/干声（纯人声系）.txt",
        datahub_root + "datahub/干声素材！Acapella！音乐制作人工具包.txt",
        datahub_root + "datahub/[Rap清唱]感受最真实的声音.txt",
        datahub_root + "datahub/贺国丰（清唱陕北民歌）.txt",
        datahub_root + "datahub/【Acapella Remix干声素材】.txt",
        datahub_root + "datahub/Pentatonix歌单-纯人声.txt",
        datahub_root + "datahub/Remix干声素材.txt",
        datahub_root + "datahub/Remix混音 | 干声素材 | Acapella.txt",
        datahub_root + "datahub/新垣结衣清唱收集.txt",
        datahub_root + "datahub/说唱干声素材(Acapella).txt",
    ]

    ### background data
    background_data = [
        musdb_train_background,
        musdb_train_background,
        musdb_train_background,
        musdb_train_background,
        musdb_train_background,
        musdb_train_background,
        musdb_train_background,
        musdb_train_background,
        musdb_train_background,
        musdb_train_background,
        datahub_root + "datahub/Eminem歌曲纯伴奏单.txt",
        datahub_root + "datahub/超舒服的说唱伴奏（Rap Beat）.txt",
        datahub_root + "datahub/抖腿 | 刷题必听电音(无人声).txt",
        datahub_root + "datahub/pure_music_7.txt",
        datahub_root + "datahub/Artpop(Intrumental).txt",
        datahub_root + "datahub/纯伴奏byLHH.txt",
        datahub_root + "datahub/纯伴奏byLHH.txt",
        datahub_root + "datahub/纯伴奏byLHH.txt",
        datahub_root + "datahub/抖腿 | 刷题必听电音(无人声).txt",
        datahub_root + "datahub/奥斯卡电影交响乐.txt",
        datahub_root + "datahub/柴可夫斯基钢琴独奏作品全集.txt",
        datahub_root + "datahub/吹奏乐1000首.txt",
        datahub_root + "datahub/纯音乐|鼓点节奏 •器乐 •电音 |BGM.txt",
        datahub_root + "datahub/纯音乐·西洋乐器.txt",
        datahub_root + "datahub/单簧管独奏曲收藏.txt",
        datahub_root + "datahub/德国著名黑管演奏家雨果.斯特拉瑟.txt",
        datahub_root + "datahub/『电音』动感纯音乐 节奏感强（建议随机.txt",
        datahub_root + "datahub/东方钢琴独奏~黑白键下的梦与历史~.txt",
        datahub_root + "datahub/俄罗斯手风琴独奏.txt",
        datahub_root + "datahub/风华国乐-传统器乐&amp;国风电子.txt",
        datahub_root + "datahub/各种乐器独奏协奏.txt",
        datahub_root + "datahub/黑白琴键的独奏.txt",
        datahub_root + "datahub/吉他独奏（流行·影视金曲集选）.txt",
        datahub_root + "datahub/交响乐 电影背景音乐.txt",
        datahub_root + "datahub/【爵士-黑管-单簧管】.txt",
        datahub_root + "datahub/猫和老鼠背景音乐交响乐.txt",
        datahub_root + "datahub/纳卡里亚科夫 小号独奏.txt",
        datahub_root + "datahub/牛逼到炸裂的电吉他独奏神曲.txt",
        datahub_root + "datahub/琵琶独奏.txt",
        datahub_root + "datahub/器乐爵士 ‖ 午夜小酒馆里的醉人音调.txt",
        datahub_root + "datahub/西非:非洲鼓(打击乐).txt",
        datahub_root + "datahub/气势恢宏----史诗级交响乐【大片即视感】.txt",
        datahub_root + "datahub/西洋器乐曲—大号.txt",
        datahub_root + "datahub/弦中呐喊 纯器乐精选.txt",
        datahub_root + "datahub/旋律型吉他独奏（solo）.txt",
        datahub_root + "datahub/一入电音深似海 | 纯音乐电音.txt",
        datahub_root + "datahub/一入电音深似海 | 纯音乐电音.txt",
    ]+[datahub_root + "datahub/牛逼到炸裂的电吉他独奏神曲.txt"]*15\
    +[datahub_root + "datahub/单簧管独奏曲收藏.txt"]*10\
    +[datahub_root + "datahub/德国著名黑管演奏家雨果.斯特拉瑟.txt"]*5\
    +[datahub_root + "datahub/西非:非洲鼓(打击乐).txt"]*20

    # Config for energy variance
    mu = 0.5
    sigma = 0.2
    alpha_low = 0.3
    alpha_high = 0.5

    # Config for stft and istft
    stft_frame_shift = 8
    stft_frame_length = 32

    # Build trail name
    cur = datetime.datetime.now()
    if(OUTPUT_MASK == False):trail_name = str(cur.year)+"_"+str(cur.month)+"_"+str(cur.day)+"_"+ model_name_alias+'_NM_spleeter_'
    else: trail_name = str(cur.year)+"_"+str(cur.month)+"_"+str(cur.day)+"_"+ model_name_alias+'spleeter_'+"sf"+str(start_point)+"_"
    counter = 1
    for each in os.listdir(project_root+"saved_models"):
        t = str(cur.year)+"_"+str(cur.month)+"_"+str(cur.day)
        if(t in each):
            for dirName in os.listdir(project_root+"saved_models/"+each):
                if("model" in dirName):
                    counter+= 1
                    break
    trail_name = str(counter)+"_"+trail_name
    for each in loss_component:
        trail_name += each+"_"
    trail_name.strip("_")
    trail_name+="lr"+str(learning_rate).split(".")[-1]+"_"\
                +"bs"+str(batch_size)+"_"\
                +"fl"+str(frame_length)+"_"\
                +"ss"+str(step_size)+"_"+str(gamma).split(".")[-1]\
                +"lnu"+str(layer_numbers_unet)\
                +"mu"+str(mu)+"sig"+str(sigma)+"low"+str(alpha_low)+"hig"+str(alpha_high)

if __name__ == "__main__":
    print(Config.trail_name)
