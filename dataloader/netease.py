#!/usr/bin/python3
# -*- coding: utf-8 -*-
import requests
import threading
import os
import re
from bs4 import BeautifulSoup
import sys
sys.path.append("..")
from config.wavenetConfig import Config
save_root = Config.datahub_root+"pure_music_mp3/"
pure_vocal_to_do = [
    # 2019-12-13
    # "https://music.163.com/#/playlist?id=2313554742",
    # "https://music.163.com/#/playlist?id=2021261354",
    # "https://music.163.com/#/playlist?id=140935570",
    # "https://music.163.com/#/playlist?id=2286848",
    # "https://music.163.com/#/playlist?id=530191151",
    # "https://music.163.com/#/playlist?id=158725481",
    # "https://music.163.com/#/playlist?id=33275157",
    # "https://music.163.com/#/playlist?id=2521740264",
    # "https://music.163.com/#/playlist?id=2181589504",
    # "https://music.163.com/#/playlist?id=98038128",
    # "https://music.163.com/#/playlist?id=608623860",
    # "https://music.163.com/#/playlist?id=85487211",
    # "https://music.163.com/#/playlist?id=430660988",
    "https://music.163.com/#/playlist?id=158725481",
    "https://music.163.com/#/playlist?id=771535658",
    "https://music.163.com/#/playlist?id=161474530",
    "https://music.163.com/#/playlist?id=2422694017"
]


background_to_do = [
    # "https://music.163.com/#/playlist?id=527938618", # clean
    # "https://music.163.com/#/playlist?id=2715138104", # Hot
    # "https://music.163.com/#/playlist?id=465992653", # clean
    # "https://music.163.com/#/playlist?id=2787072920", # not clean
    # "https://music.163.com/#/playlist?id=2279310366", # foke song
    # "https://music.163.com/#/playlist?id=2997764076", #lady gaga
    # "https://music.163.com/#/playlist?id=96446995",
    # "https://music.163.com/#/playlist?id=3074099553"
    "https://music.163.com/#/playlist?id=383075247", #交响乐 电影背景音乐 841
    "https://music.163.com/#/playlist?id=460326234", # 风华国乐-传统器乐&国风电子
    "https://music.163.com/#/playlist?id=90401069", # 器乐爵士 ‖ 午夜小酒馆里的醉人音调
    "https://music.163.com/#/playlist?id=39694263", # 纯音乐|鼓点节奏 •器乐 •电音 |BGM
    "https://music.163.com/#/playlist?id=2491655154",# 西洋器乐曲—大号 39
    "https://music.163.com/#/playlist?id=7163705", # 弦中呐喊 纯器乐精选 36
    "https://music.163.com/#/playlist?id=116214280", # 【器乐】架子鼓 25
    "https://music.163.com/#/playlist?id=2036778465" # 架子鼓 对耳朵的享受 124
    "https://music.163.com/#/playlist?id=431256642" , # 贝斯独奏 不一样的感觉
    "https://music.163.com/#/playlist?id=523210850",# 牛逼到炸裂的电吉他独奏神曲
    "https://music.163.com/#/playlist?id=2497426288", # 吉他独奏（流行·影视金曲集选）
    "https://music.163.com/#/playlist?id=108012586", # 莫扎特钢琴独奏作品
    "https://music.163.com/#/playlist?id=387359964", # 东方钢琴独奏~黑白键下的梦与历史~
    "https://music.163.com/#/playlist?id=2698239642", # 吉他独奏（纯音乐）
    "https://music.163.com/#/playlist?id=1986062420", # 单簧管独奏曲收藏
    "https://music.163.com/#/playlist?id=422785916", # 纳卡里亚科夫 小号独奏
    "https://music.163.com/#/playlist?id=691124760", # 旋律型吉他独奏（solo）
    "https://music.163.com/#/playlist?id=2718633718", # 琵琶独奏
    "https://music.163.com/#/playlist?id=2300144261", # 俄罗斯手风琴独奏
    "https://music.163.com/#/playlist?id=517402175", # 柴可夫斯基钢琴独奏作品全集
    "https://music.163.com/#/playlist?id=61311676", # 黑白琴键的独奏
    "https://music.163.com/#/playlist?id=602664262", # 李斯特钢琴独奏曲60卷合集之（25-46卷）
    "https://music.163.com/#/playlist?id=34280073", # 各种乐器独奏协奏
    "https://music.163.com/#/playlist?id=2385707999", # 纯音乐/电音可下载(2020/1/10)
    "https://music.163.com/#/playlist?id=2491616279", # 猫和老鼠背景音乐交响乐
]



class NeteaseDownloader(threading.Thread):
    musicData = []
    user_agent = 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.82 Safari/537.36'

    def __init__(self, threadID, name, counter):
        # 多线程
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.url_text = ""
        self.list_name = ""

        if not os.path.exists(save_root):
            raise ValueError(save_root+" does not exist!")
        if(not save_root[-1] == '/'):
            raise ValueError("Error: Path should end with /")

    def __del__(self):
        pass

    def run(self):
        print("Start downloading...")
        self.get(self.musicData)

    def download(self,url):
        self.url_text = url
        self.musicData = []
        self.musicData = self.getMusicData(self.url_text.replace("#/", ""))
        print(self.musicData)
        if len(self.musicData) > 1:
            self.start()

    def get(self, values):
        print(len(values))
        downNum = 0
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        for x in values:
            x['name'] = re.sub(rstr, "_", x['name'])
            if not os.path.exists(save_root + self.list_name+"/"+ x['name'] + '.mp3'):
                print('***** ' + x['name'] + '.mp3 ***** Downloading...\n')

                url = 'http://music.163.com/song/media/outer/url?id=' + x['id'] + '.mp3'
                try:
                    # urllib.request.urlretrieve(url,'./music/' + x['name'] + '.mp3')
                    self.saveFile(url, save_root +self.list_name+"/"+ x['name'] + '.mp3')
                    downNum = downNum + 1
                except:
                    x = x - 1
                    print(u'Download wrong~\n')
        print('Download complete ' + str(downNum) + ' files !\n')

    def getMusicData(self, url):
        headers = {'User-Agent': self.user_agent}
        webData = requests.get(url, headers=headers).text
        soup = BeautifulSoup(webData, 'lxml')
        find_list = soup.find('ul', class_="f-hide").find_all('a')
        self.list_name = soup.find_all(name='h2',attrs={"class":"f-ff2 f-brk"})
        self.list_name = str(self.list_name).split('<')[-2].split('>')[-1]
        if(not os.path.exists(save_root+self.list_name)):
            os.mkdir(save_root+self.list_name)
        tempArr = []
        for a in find_list:
            music_id = a['href'].replace('/song?id=', '')
            music_name = a.text
            tempArr.append({'id': music_id, 'name': music_name})
        return tempArr

    def saveFile(self, url, path):
        headers = {'User-Agent': self.user_agent,
                   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                   'Upgrade-Insecure-Requests': '1'}
        response = requests.get(url, headers=headers)
        with open(path, 'wb') as f:
            f.write(response.content)
            f.flush()

if __name__ == '__main__':
    for each in background_to_do:
        frame = NeteaseDownloader(1, "Thread-1", 1)
        frame.download(each)