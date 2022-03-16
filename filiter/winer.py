# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 22:13:24 2022

@author: Kyle
"""


import librosa
from librosa.core.spectrum import amplitude_to_db
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import lfilter, firwin
import os
def add_nosie(clean,noise,snr):
    if not len(clean) == len(noise):
        print("")
        return False
 
    p_clean = np.sum(np.abs(clean)**2)
    
    p_noise = np.sum(np.abs(noise)**2)

    scale =  np.sqrt( (p_clean/p_noise) * np.power(10,-snr/10) )

    noisy = clean + scale * noise
    
    return noisy




def gen_color_noise(N,order_filter,fs,f_L,f_H):
    
    noise = np.random.randn(N)
    m_firwin = firwin(order_filter, [2*f_L/fs, 2*f_H/fs], pass_zero="bandpass")
    color_noise = lfilter(m_firwin, 1.0, noise)
    return color_noise


def train_wiener_filter(cleans,noises,para):
    n_fft = para["n_fft"]
    hop_length = para["hop_length"]
    win_length = para["win_length"]
    alpha = para["alpha"]
    beta = para["beta"]
    Pxxs = []
    Pnns =[]
    for clean,noise in zip(cleans,noises):
        S_clean = librosa.stft(clean,n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        S_noise = librosa.stft(noise,n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        Pxx = np.mean((np.abs(S_clean))**2,axis=1,keepdims=True) # Dx1
        Pnn = np.mean((np.abs(S_noise))**2,axis=1,keepdims=True)
        Pxxs.append(Pxx)
        Pnns.append(Pnn)
    
    train_Pxx = np.mean(np.concatenate(Pxxs,axis=1),axis=1,keepdims=True)
    train_Pnn = np.mean(np.concatenate(Pnns,axis=1),axis=1,keepdims=True)
    
    H = (train_Pxx/(train_Pxx+alpha*train_Pnn))**beta
    
    return H    
if __name__ == "__main__":
    f = ['air_condition','bark','blower','cleaner','drilling','fan','grinding','horn',
         'idling','jackhammer','market','music','playing','rainy','shot','siren',
         'street_music','traffic','train','truck']

    for _type in f:
        path ="./dataset_apart/%s/train/"%_type
        cleans =[]
        noises= []
        for x in os.listdir(path+"clean"):
            clean,sr = librosa.load(path+"clean/"+x,sr=None)
            mix,sr = librosa.load('./Test/apart/winer_data/'+x,sr=None)
            cleans.append(clean)
            noises.append(mix)

        para_wiener = {}
        para_wiener["n_fft"] = 256
        para_wiener["hop_length"] = 128
        para_wiener["win_length"] = 256
        para_wiener["alpha"] = 1
        para_wiener["beta"] =3.3
        print(_type)

        H= train_wiener_filter(cleans,noises,para_wiener)
        
        test_path = "./Test/apart/%s/"%_type
        for x in os.listdir(test_path):

            numbers = ''.join([y for y in x if y.isdigit()])
            noise_file = "./Test/apart/esti/vocal_%s.flac"%numbers
            test_noisy,fs = librosa.load(noise_file,sr=None)

            S_test_noisy = librosa.stft(test_noisy,
                                        n_fft=para_wiener["n_fft"], 
                                        hop_length=para_wiener["hop_length"], 
                                        win_length=para_wiener["win_length"])
            S_test_enhec = S_test_noisy*H
            test_enhenc = librosa.istft(S_test_enhec, 
                                        hop_length=para_wiener["hop_length"], 
                                        win_length=para_wiener["win_length"])
            
            sf.write("./Test/apart/winer/vocal_%s.flac"%numbers,test_enhenc,fs)

    