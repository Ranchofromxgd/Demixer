#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   EvaluationHelper.py    
@Contact :   liu.8948@buckeyemail.osu.edu
@License :   (C)Copyright 2020-2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/4/13 7:42 PM   Haohe Liu      1.0         None
'''
import os
import sys
import time
sys.path.append("..")
from config.mainConfig import Config
from util.separation_util import SeparationUtil
from config.global_tool import GlobalTool
import logging

class EvaluationHelper:
    def __init__(self,maplocation):
        self.maplocation = maplocation
    def evaluate(self,path,start_point,
                 save_wav = True,
                 save_json = True,
                 subband = 1,
                 test_mode = False,
                 split_musdb = True,
                 split_listener = True):
        if(subband != 1):
            split_band = True
            Config.split_band = True
            Config.subband = subband
            GlobalTool.refresh_subband(subband)
        else:
            Config.split_band = False
            split_band = False
        print(Config.subband)
        su = SeparationUtil(load_model_pth=path, start_point=start_point,map_location=self.maplocation)
        if(split_musdb):su.evaluate(save_wav=save_wav, save_json=save_json, split_band=split_band,test_mode=test_mode)
        if(split_listener):su.Split_listener(split_band=split_band)
        del su

    def batch_evaluation_subband(self,path:str,start_points:list,
                                save_wav = True,
                                save_json = True,
                                split_musdb = True,
                                split_listener = True,
                                test_mode = False,
                                subband = 4):
        for start in start_points:
            self.evaluate(path,start,
                          save_wav = save_wav,
                          save_json=save_json,
                          test_mode=test_mode,
                          split_musdb=split_musdb,
                          split_listener = split_listener,
                          subband = subband)


if __name__ == "__main__":
    print("start")
    test = {
        # "1":{
        #     "path": "1_2020_4_12__unet_spleeter_spleeter_sf0_l1_l2_l3_lr001_bs8-1_fl3_ss32000_9lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0split_bandTrue_8",
        #     "start_points": [144000,136000,128000,120000,112000,104000],
        #     "subband": 8,
        # },
        # "2": {
        #     "path": "1_2020_4_12__unet_spleeter_spleeter_sf0_l1_l2_l3_lr001_bs2-1_fl3_ss32000_9lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0split_bandTrue_2",
        #     "start_points": [176000,160000,144000,128000,112000,96000,80000],
        #     "subband": 2,
        # },
        # "3": {
        #     "path": "1_2020_4_19__unet_spleeter_spleeter_sf136000_l1_l2_l3_lr0006_bs4-1_fl3_ss64000_9lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0split_bandTrue_4",
        #     "start_points":  [256000, 224000, 192000, 352000, 320000, 288000],
        #     "subband": 4,
        # },
        # "4": {
        #     "path": "1_2020_4_13__unet_spleeter_spleeter_sf0_l1_l2_l3_lr001_bs4-1_fl3_ss32000_9lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0split_bandTrue_4",
        #     "start_points": [136000, 128000, 120000, 112000, 104000, 96000],
        #     "subband": 4,
        # },
        # "5":{
        #     "path": "1_2020_4_18__unet_spleeter_spleeter_sf112000_l1_l2_l3_lr001_bs4-1_fl3_ss64000_9lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0split_bandTrue_4",
        #     "start_points": [576000, 544000, 512000, 480000, 448000, 416000,384000,352000,320000,288000,256000],
        #     "subband": 4,
        # },
        # "6": {
        #     "path": "1_2020_2_10__unet_spleeter_spleeter_sf0_l1_l2_l3_l7_l8_lr0001_bs1_fl3_ss16000_85lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32",
        #     "start_points": [280000],
        #     "subband": 1,
        # },
        "7": {
            "path": "1_2020_4_18__unet_spleeter_spleeter_sf32000_l1_l2_l3_lr001_bs2-1_fl3_ss64000_9lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0split_bandFalse_4",
            "start_points": [288000,224000] ,#[192000, 160000, 128000, 96000, 64000],
            "subband": 1,
        }
    }


    def TimeStampToTime(timestamp):
        timeStruct = time.localtime(timestamp)
        return time.strftime('%Y-%m-%d %H:%M:%S', timeStruct)
    # test = {
    #     "1":{
    #         "path": "1_2020_2_10__unet_spleeter_spleeter_sf0_l1_l2_l3_l7_l8_lr0001_bs1_fl3_ss16000_85lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32",
    #         "start_points": [280000],
    #         "subband": 1,
    #     },
    #     "2": {
    #         "path": "1_2020_4_18__unet_spleeter_spleeter_sf32000_l1_l2_l3_lr001_bs2-1_fl3_ss64000_9lnu5mu0.5sig0.2low0.3hig0.5fshift8flength32drop0split_bandFalse_4",
    #         "start_points": [192000, 160000, 128000, 96000, 64000],
    #         "subband": 1,
    #     },
    # }

    print(test)

    for each in list(test.keys()):
        path,start_points,subband = test[each]["path"],test[each]["start_points"],test[each]["subband"]
        for start in start_points:
            if(not os.path.exists(Config.project_root+"saved_models/"+path+"/model"+str(start)+".pkl")):
                raise ValueError(start_points,path,"none exist")
    print("Found all models success")

    eh = EvaluationHelper(maplocation=Config.device)
    for each in list(test.keys()):
        path,start_points,subband = test[each]["path"],test[each]["start_points"],test[each]["subband"]
        try:
            eh.batch_evaluation_subband(Config.project_root+"saved_models/"+path,start_points,
                                    save_wav=True,
                                    save_json=True,
                                    test_mode=False,
                                    split_musdb=True,
                                    split_listener=False,
                                    subband = subband)
        except Exception as e:
            print("error...")
            logging.exception(e)
            continue