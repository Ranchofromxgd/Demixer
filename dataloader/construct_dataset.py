import sys
sys.path.append("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/")
import os
import matplotlib.pyplot as plt
import numpy as np
from util.wave_util import WaveHandler
from pydub import AudioSegment

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

def list_and_save_folder(path,save_path):
    if(not path[-1] == '/'):
        raise ValueError("Error: Path should end with / ")
    files = os.listdir(path)
    log = []
    for cnt,each in enumerate(files):
        log.append(path+each)
    write_list(log,save_path)

def plot2wav(a,b):
    plt.figure(figsize=(20,4))
    plt.subplot(211)
    plt.plot(a)
    plt.subplot(212)
    plt.plot(b)
    plt.savefig("temp.png")

# delete_unproper_training_data("/home/disk2/internship_anytime/liuhaohe/datasets/pure_music_wav/pure_music_7/")
def delete_unproper_training_data(path):
    if (not path[-1] == '/'):
        raise ValueError("Error: path should end with /")
    wh = WaveHandler()
    files = os.listdir(path)
    for cnt,each in enumerate(files):
        if(cnt % 20 == 0):
            print("finished:",cnt)
        file_pth = path+each
        judge = wh.get_channels_sampwidth_and_sample_rate(file_pth)
        if(not judge[0]):
            print(judge[1],each)

def plot3wav(a,b,c):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(30,6))
    plt.subplot(311)
    plt.plot(a,linewidth = 1)
    plt.subplot(312)
    plt.plot(b,linewidth = 1)
    plt.subplot(313)
    plt.plot(c,linewidth = 1)
    plt.savefig("com.png")

def trans_mp3_folder_to_wav(root_path,save_folder):
    if(not save_folder[-1] == '/' or not root_path[-1] == '/'):
        raise ValueError("Error: path should end with /")
    if(not os.path.exists(root_path)):
        raise ValueError(root_path+" does not exist!")
    if(not os.path.exists(save_folder)):
        os.mkdir(save_folder)
    for cnt,each in enumerate(os.listdir(root_path)):
        if(each.split('.')[-1] == 'mp3'):
            try:
                trans_mp3_to_wav(root_path+each,save_folder)
            except:
                print("Failed to transfer: "+each)
        if(cnt % 20 == 0):
            print(cnt,"files finished")

def trans_mp3_to_wav(filepath:str,savepath:str):
    if(not savepath[-1] == '/'):
        raise ValueError("Error: savepath should end with /")
    name = filepath.split('.')[-2].split('/')[-1]
    song = AudioSegment.from_mp3(filepath)
    song.export(savepath+name+".wav", format="wav")

def get_total_time_in_folder(path):
    if(not path[-1] == '/'):
        raise ValueError("Error: path should end with /")
    wh = WaveHandler()
    total_time = 0
    for cnt,file in enumerate(os.listdir(path)):
        if(cnt % 20 == 0):
            print(cnt,"finished")
        total_time += wh.get_duration(path+file)
    print("total: ")
    print(total_time,"s")
    print(total_time/60,"min")
    print(total_time/3600,"h")

def readList(fname):
    result = []
    with open(fname,"r") as f:
        for each in f.readlines():
            each = each.strip('\n')
            result.append(each)
    return result

def get_total_time_in_txt(txtpath):
    wh = WaveHandler()
    cnt = 0
    files = readList(txtpath)
    total_time = 0
    for cnt,file in enumerate(files):
        if(cnt % 100 == 0):
            print(cnt,"finished")
        try:
            total_time += wh.get_duration(file)
            cnt += 1
        except:
            print("error:",file)
    print("total: ")
    print(total_time,"s")
    print(total_time/60,"min")
    print(total_time/3600,"h")
    return total_time/3600,cnt

# delete_unproper_training_data("/home/disk2/internship_anytime/liuhaohe/datasets/pure_vocal/k_pop/")
# list_and_save_folder("/home/disk2/internship_anytime/liuhaohe/datasets/pure_vocal/k_pop/",fname="k_pop.txt")
# from config.wavenetConfig import Config
totalsongs =0
totalduration = 0
from config.wavenetConfig import Config
for each in Config.netease_background_pth:
    duration,num = get_total_time_in_txt(each)
    totalsongs+=num
    totalduration += duration
print(totalsongs,totalduration)
