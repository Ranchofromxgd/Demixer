#!/usr/bin/python3
# -*- coding: utf-8 -*-
import requests
import threading
import os
import re
from bs4 import BeautifulSoup

save_root = "/home/disk2/internship_anytime/liuhaohe/datasets/pure_vocal_mp3/"
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
    "https://music.163.com/#/playlist?id=85487211"
]

background_to_do = [
    # "https://music.163.com/#/playlist?id=527938618", # clean
    # "https://music.163.com/#/playlist?id=2715138104", # Hot
    # "https://music.163.com/#/playlist?id=465992653", # clean
    # "https://music.163.com/#/playlist?id=2787072920", # not clean
    # "https://music.163.com/#/playlist?id=2279310366", # foke song
    "https://music.163.com/#/playlist?id=2997764076", #lady gaga
    "https://music.163.com/#/playlist?id=96446995",
    "https://music.163.com/#/playlist?id=3074099553"
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
    for each in pure_vocal_to_do:
        frame = NeteaseDownloader(1, "Thread-1", 1)
        frame.download(each)