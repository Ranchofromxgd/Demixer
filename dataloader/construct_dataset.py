import sys
sys.path.append("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/")
import os
import matplotlib.pyplot as plt
import numpy as np
from util.wave_util import WaveHandler

def intership_server():
    test_path = "/home/disk2/internship_anytime/liuhaohe/datasets/musdb18hq/test/"
    train_path = "/home/disk2/internship_anytime/liuhaohe/datasets/musdb18hq/train/"

    test_files = os.listdir(test_path)
    train_files = os.listdir(train_path)

    fname1 = "musdb_test_mixed.txt"
    fname2 = "musdb_test_vocal.txt"
    fname3 = "musdb_train_mixed.txt"
    fname4 = "musdb_train_vocal.txt"


    with open(fname1,"w") as f:
        for each in os.listdir(test_path):
            f.write(test_path+each+"/mixed.wav\n")

    with open(fname2,"w") as f:
        for each in os.listdir(test_path):
            f.write(test_path+each+"/vocals.wav\n")

    with open(fname3,"w") as f:
        for each in os.listdir(train_path):
            f.write(train_path+each+"/mixed.wav\n")

    with open(fname4,"w") as f:
        for each in os.listdir(train_path):
            f.write(train_path+each+"/vocals.wav\n")

def analysis_cache():
    fname = "/home/work_nfs/hhliu/workspace/github/wavenet-aslp/dataloader/wavenet/temp"
    with open(fname,'r') as f:
        line = f.readline().strip().split()
    for i in range(len(line)):
        line[i] = int(line[i])
    smooth = 10
    smoothed = []
    for start in range(len(line[smooth:])):
        smoothed.append(sum(line[start:start+smooth])/smooth)
    plt.figure(figsize=(10,5))
    smoothed = smoothed[:3000]
    l1, = plt.plot(smoothed,linewidth = 1)
    l2, = plt.plot(np.zeros((len(smoothed),)),linewidth = 1)
    plt.legend([l1,l2],["cached","without cache"],loc = 'upper right')
    plt.title("Hit rate increases during training(150 songs' cache)")
    plt.savefig("hit_rate.png")

def write_list(l,fname):
    with open(fname,'w') as f:
        for each in l:
            f.write(each+"\n")

def Song_data():
    dir = "/home/disk2/internship_anytime/liuhaohe/datasets/seg_song_data/"
    res = []
    for cnt,fname in enumerate(os.listdir(dir)):
        res.append(dir+fname)
    write_list(res,"/home/disk2/internship_anytime/liuhaohe/datasets/song_vocal_data_44_1.txt")


def seg_data():
    wh = WaveHandler()
    dir = "/home/work_nfs/hhliu/datasets/song/441_song_data/"
    seg_dir = "/home/work_nfs/hhliu/datasets/song/seg_song_data/"
    for cnt,fname in enumerate(os.listdir(dir)):
        print("Doing segmentation on ",fname+"...")
        unseg_f = dir+fname
        data = wh.read_wave(unseg_f, channel=2)
        length = data.shape[0]
        for start in np.linspace(0,0.95,20):
            seg_data = data[int(start*length):int((start+0.05)*length)]
            wh.save_wave(seg_data,seg_dir+fname.split('.')[-2]+"_"+str('%.2f' % start)+".wav",channels=2)


def pure_music():
    path = "/home/work_nfs/hhliu/datasets/pure_music/"
    pth_name = [path+each for each in os.listdir(path)]
    log = []
    for cnt,each in enumerate(pth_name):
        log.append(each+" -> "+path+"pure_music_"+str(cnt))
        os.rename(each,path+"pure_music_"+str(cnt))
    write_list(log,path+"readme")

def plot2wav(a,b):
    plt.figure(figsize=(20,4))
    plt.subplot(211)
    plt.plot(a)
    plt.subplot(212)
    plt.plot(b)
    plt.savefig("temp.png")

def varify_sisdr():
    from evaluate.si_sdr_torch import si_sdr
    wh = WaveHandler()
    voice_output_dir = "/home/work_nfs/hhliu/workspace/github/wavenet-aslp/util/outputs/vocal_test_0.wav"
    voice_input_dir = "/home/work_nfs3/yhfu/dataset/musdb18hq/test/test_0/vocals.wav"
    output = wh.read_wave(voice_output_dir,channel=1)
    input = wh.read_wave(voice_input_dir)
    length = min(output.shape[0],input.shape[0])
    input,output = input[:length],output[:length]
    # plot2wav(input,output)
    print(si_sdr(input,output))

Song_data()