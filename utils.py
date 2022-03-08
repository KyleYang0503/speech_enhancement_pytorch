# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 01:59:03 2022

@author: Kyle
"""
import collections
import os
import random
import shutil
fileDir = "./origin/train/clean/"    #源圖片資料夾路徑
fileDir_mix = "./origin/train/mix/"    #源圖片資料夾路徑
tarDir = './origin/dev/clean/'    #移動到新的資料夾路徑
tarDir_mix = './origin/dev/mix/'

def moveFile(fileDir):
    pathDir = os.listdir(fileDir)    #取圖片的原始路徑
    filenumber=len(pathDir)
    rate=0.1    #自定義抽取圖片的比例，比方說100張抽10張，那就是0.1
    picknumber=int(filenumber*rate) #按照rate比例從資料夾中取一定數量圖片
    sample = random.sample(pathDir, picknumber)  #隨機選取picknumber數量的樣本圖片
    print (sample)
    for name in sample:
        shutil.move(fileDir+name, tarDir+name)
        shutil.move(fileDir_mix+name, tarDir_mix+name)
    return

def file_rename(path):
    f = os.listdir(path)
    n = 0
    i = 0
    for i in f:
        oldname = f[n]
        numbers = ''.join([x for x in oldname if x.isdigit()])
        newname = 'vocal_%s'%numbers + '.flac'
        os.rename(path+oldname, path+newname)
        print(oldname, '======>', newname)
        n += 1

def count_rename(path):
    f = os.listdir(path)
    n = 0
    i = 0
    count = []
    for i in f:
        oldname = f[n]
        count.append(oldname.split('_')[-1])
        n += 1
    print(collections.Counter(count))
    
def move(path,name,number):
    f = os.listdir(path)
    n = 0
    count = 0
    for i in f:
        oldname = f[n]
        n=n+1
        numbers = ''.join([x for x in oldname if x.isdigit()])
        if oldname.split('_')[-1] == name and count < number:
            print(oldname)
            shutil.move(fileDir+'vocal_%s.flac'%numbers, tarDir+'vocal_%s.flac'%numbers)
            shutil.move(fileDir_mix+oldname, tarDir_mix+oldname)
            count =count+1
def move_by_type(mode,name,mix_path,tar_mix,clean_path=None,tar_clean=None):
    f = os.listdir(mix_path)
    n = 0
    i = 0
    count = 0
    for i in f:
            oldname = f[n]
            name_ = oldname.split('_')[-1]
            number = ''.join([x for x in oldname if x.isdigit()])
            n += 1
            if (name_ == name):
                if mode == 'train':
                    shutil.move(clean_path+'/vocal_%s.flac'%number, tar_clean+'/vocal_%s.flac'%number)
                shutil.move(mix_path +'/' +oldname, tar_mix +'/' +oldname)
                count +=1
    print("number of moved item:%d"%count)            

def create():
    path = "./dataset_apart"
    f = os.listdir(path)
    for x in f:
        os.makedirs("./Model/%s"%x)
def create_dev():
    from random import sample
    f = ['blower','cleaner','fan','grinding','horn','idling','jackhammer','market','music',
         'playing','rainy','shot','silence','siren','street_music','traffic','train','truck']
    for x in f:
        y = os.listdir("./dataset_apart/%s/train/mix"%x)
        sample_ = sample(y,int(len(y)*0.1))
        print(x)
        for z in sample_:
            shutil.copy("./dataset_apart/%s/train/mix/%s"%(x,z),"./dataset_apart/%s/dev/mix/%s"%(x,z))
            shutil.copy("./dataset_apart/%s/train/clean/%s"%(x,z),"./dataset_apart/%s/dev/clean/%s"%(x,z))
