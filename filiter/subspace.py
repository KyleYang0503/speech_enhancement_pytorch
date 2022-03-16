# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 23:51:30 2022

@author: Kyle
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 12:21:24 2022

@author: Kyle
"""

import librosa
from librosa.util import frame
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz



def sine_taper(L, N):
    tapers= np.zeros( [N, L]);
    index = np.array([i+1  for i in range(N)])
    for i in range(L):
        tapers[:,i] = np.sqrt(2/(N+1))* np.sin(np.pi * (i+1)*index/(N+1))
    
    return tapers


def estimate_R(x,p,W):
    
    N,L = np.shape(W)
    x_rep = np.tile(x,[L,1]) #  L x N
    x_rep = x_rep.T          # N x L
    

    x_w= W* x_rep
    

    R1= np.dot(x_w, x_w.T)
   
    r = np.zeros(p)
    for i in range(p):
        r[i] = np.sum(np.diag(R1,k=i))
    
    R_est = toeplitz(r)
    return R_est


def frame2singal(frames):
    N,d = np.shape(frames)
    
    half_frame = int(d/2)
    overlap = np.zeros(half_frame)
    len_singal = d +(N-1)*(half_frame)
    start =0
    singal = np.zeros(len_singal)
    for i in range(N):
        temp = frames[i]
        singal[start:start+half_frame] = temp[:half_frame] + overlap
        overlap = temp[half_frame:]
        start = start +half_frame
    singal[start:] = overlap
    return singal


L=16 
vad_thre= 1.2  
mu_vad= 0.98   

mu_max=5 
mu_toplus= 1 

mu_tominus= mu_max;  
mu_slope= (mu_tominus- mu_toplus )/ 25;
mu0= mu_toplus+ 20* mu_slope;

import os

path ="./Test/apart/winer/"
for x in os.listdir(path):

    noisy_wav_file = path + x
    noisy_speech,fs = librosa.load(noisy_wav_file,sr=None)
    
    
    subframe_dur= 4  
    len_subframe= int(np.floor( fs* subframe_dur/ 1000))
    
    P= len_subframe  
    
    frame_dur= 32  
    len_frame = int(np.floor(frame_dur* fs/ 1000)) 
    
    len_step_frame= int(len_frame/ 2)    
    
    window_frame = np.hamming(len_frame)
    window_subframe= np.hamming(len_subframe) 
    

    noise_dur = 120 
    N_noise=int(np.floor( noise_dur* fs/ 1000))
    noise= noisy_speech[:N_noise]
    

    tapers= sine_taper( L, N_noise)

    Rn = estimate_R(noise,P,tapers)
    iRn = np.linalg.inv(Rn)
    
    
    

    noisy_frames = frame(noisy_speech, len_frame, len_step_frame,axis = 0 )

    N_frame = noisy_frames.shape[0]
    

    tapers_noisy = sine_taper( L, len_frame)
    

    enh_frames = np.zeros(np.shape(noisy_frames))
    

    for n in range(N_frame):
      

        noisy = noisy_frames[n]
    

        Ry = estimate_R(noisy,P,tapers_noisy)
       

        vad_ratio= Ry[0,0]/ Rn[0,0]
        if vad_ratio<= vad_thre: 
            Rn= mu_vad* Rn+ (1- mu_vad)* Ry
            iRn= np.linalg.inv( Rn)
    

        In= np.eye(P)   
        iRnRx= np.dot(iRn, Ry)- In
        

        d, V = np.linalg.eig(iRnRx)  
        iV= np.linalg.inv(V)
        

        d[d<0]=0
        dRx = d
        

        SNR  = np.sum(dRx)/P
        SNR_db = 10 * np.log10( SNR+ 1e-10)
        if SNR_db >= 20:
            mu = mu_toplus    
        elif SNR_db< 20  and SNR_db>= -5 :
            mu = mu0- SNR_db * mu_slope
        else:
            mu = mu_tominus
          

        gain_vals= dRx/( dRx+mu) 
        G= np.diag( gain_vals)
        

        H = np.dot(np.dot(iV.T,G),V.T)
       

        sub_frames = frame(noisy, len_subframe, int(len_subframe/2),axis = 0)  # N * d

        enh_sub_frames = np.dot(sub_frames,H.T) # N x d

        enh_sub_frames = enh_sub_frames * window_subframe

        enh_signal = frame2singal(enh_sub_frames)
        enh_frames[n] = enh_signal
    

    enh_frames = enh_frames*window_frame
    enh_wav = frame2singal(enh_frames)
    
    
    sf.write("./Test/apart/subspace/%s"%x,enh_wav,fs)
        
''' 
plt.subplot(2,1,1)
plt.specgram(noisy_speech,NFFT=256,Fs=fs)
plt.xlabel("noisy specgram")
plt.subplot(2,1,2)
plt.specgram(enh_wav,NFFT=256,Fs=fs)
plt.xlabel("enh specgram")   
plt.show()
'''