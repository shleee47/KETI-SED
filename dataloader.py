import json
import sys
sys.path.append('..')
from torch.utils.data.dataset import Dataset
from pathlib import Path
import pickle
import random
import pdb
import torch
import numpy as np
import argparse
import os
import sys
# import h5py
import librosa
import numpy as np
# import pandas as pd
import scipy.io as sio
from scipy import signal
from tqdm import tqdm
from SED.utils.specaug import SpecAugment
import warnings
warnings.filterwarnings("ignore")

def Audio_Collate(batch):
    #pdb.set_trace() 
    data, class_num = list(zip(*batch))
    class_num = np.concatenate(class_num)
    #data = np.concatenate(data)
    
    data_len = [len(x) for x in data]
    if len(data_len) == 0:
        return -1

    max_len = max(data_len)
    wrong_indices = []
    
    for i, a_ in enumerate(class_num):
        if a_ == -1:
            wrong_indices.append(i)

    B = len(data)
    #pdb.set_trace()
    inputs = torch.zeros(B-len(wrong_indices), max_len)
    labels = torch.zeros(B-len(wrong_indices), 2)
    j = 0

    '''zero pad'''    
    for i in range(B):
        if i in wrong_indices:
            continue
        inputs[j, :data[i].shape[0]] = torch.from_numpy(data[i])
        labels[j, class_num[i]] = 1.0
        j += 1

#    '''replica'''
#    for i in range(B):
#        if i in wrong_indices:
#            continue
#
#        inputs[j, :, :data[i].size(1),:] = data[i]
#        labels[j, class_num[i]] = 1.0
#        num_pad = max_len - data[i].size(1) # To be padded
#        idx_ = data[i].size(1)
#        while num_pad > 0:
#            if num_pad > data[i].size(1):
#                inputs[j, :, idx_:idx_+data[i].size(1),:] = data[i]
#                idx_ += data[i].size(1)
#                num_pad -= data[i].size(1)
#            else:
#                inputs[j, :, idx_:idx_+num_pad,:] = data[i][:,:num_pad,:]
#                num_pad = 0
#        j += 1
    #pdb.set_trace()
    data = (inputs, labels, data_len)
    return data

class Audio_Reader(Dataset):
    def __init__(self, yes_datalist, no_datalist):
        super(Audio_Reader, self).__init__()
        self.yes_datalist = yes_datalist
        self.no_datalist = no_datalist
        self.datalist = np.concatenate([self.yes_datalist,self.no_datalist],axis=None)
        self.classlist = ['Pig','Not-Pig']
        #self.classlist = ['Speech','Not-Speech']
        #self.classlist = ['Crying','Not-Crying']
        #self.classlist = ['Screaming','Not-Screaming']
        self.nfft = 512
        self.hopsize = self.nfft // 4
        self.window = 'hann'
        self.spec_aug = SpecAugment()

    def shuffle_dataset(self):
       #pdb.set_trace()
       np.random.shuffle(self.no_datalist)
       self.datalist = np.concatenate([self.yes_datalist,self.no_datalist[:len(self.yes_datalist)*3]],axis=None)
       #print(self.no_datalist[:5])

    def __len__(self):
        #print(len(self.datalist))
        return len(self.datalist)

    def LogMelExtractor(self, sig):
        def logmel(sig):
            D = np.abs(librosa.stft(y=sig,
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2        
            #pdb.set_trace() 
            SS = librosa.feature.melspectrogram(S=D,sr=16000)
            S = librosa.amplitude_to_db(SS)
            #S[:10, :] = 0.0
            return S

        def transform(audio):
            feature_logmel = logmel(audio)
            return feature_logmel
        
        return transform(sig)

    def __getitem__(self, idx):
        
        #pdb.set_trace()
        audio_path = self.datalist[idx]
        if audio_path.split('/')[-2] == self.classlist[0]:
            class_num = 0
        else:
            class_num = 1
            
        try:
            audio, _ = librosa.load(audio_path, sr=32000, dtype=np.float32)
        except:
            #pdb.set_trace()
            return np.zeros((320000,),dtype=float),np.array([-1])
            
        audio_len = len(audio)
        if audio_len >= 320000:
            audio = audio[:320000]
        else:
            #pdb.set_trace()
            dum = np.zeros(320000-audio_len)
            audio = np.concatenate([audio,dum])

        return audio, np.array([class_num])
        
        
class Test_Reader(Dataset):
    def __init__(self, datalist):
        super(Test_Reader, self).__init__()
        self.datalist = datalist
        #self.classlist = ['Speech','Not-Speech']
        #self.classlist = ['Screaming','Not-Screaming']
        #self.classlist = ['Crying','Not-Crying']
        #self.classlist = ['Explosion','Not-Explosion']
        self.classlist = ['Pig','Not-Pig']
        self.nfft = 512
        self.hopsize = self.nfft // 4
        self.window = 'hann'
        self.nan_cnt = 0

    def __len__(self):
        #print('Number of Data for Inference: '.format(len(self.datalist)))
        return len(self.datalist)

    def LogMelExtractor(self, sig):
        def logmel(sig):
            D = np.abs(librosa.stft(y=sig,
                                    n_fft=self.nfft,
                                    hop_length=self.hopsize,
                                    center=True,
                                    window=self.window,
                                    pad_mode='reflect'))**2        
            #pdb.set_trace() 
            SS = librosa.feature.melspectrogram(S=D,sr=16000)
            S = librosa.amplitude_to_db(SS)
            #S[:10, :] = 0.0
            return S

        def transform(audio):
            feature_logmel = logmel(audio)
            return feature_logmel
        
        return transform(sig)

    def __getitem__(self, idx):
        #pdb.set_trace()
        audio_path = self.datalist[idx]
        class_name = audio_path.split('/')[-2] 
        if class_name == self.classlist[0]:
            class_num = 0
        else:
            class_num = 1
        try:
            audio, _ = librosa.load(audio_path, sr=32000, dtype=np.float32)
        except:
            #pdb.set_trace()
            self.nan_cnt +=1
            return np.zeros((320000,),dtype=float), np.array([-1])
        
        '''Exception'''
        if audio.sum() == 0.0:
            self.nan_cnt +=1
            return np.zeros((320000,),dtype=float), np.array([-1])

        else:
            #feature = self.LogMelExtractor(audio)
            #return torch.FloatTensor(feature).transpose(0,1), np.array([class_num])
            #return audio, np.array([class_num])
            return audio, np.array([class_num]),audio_path.split('/')[-1]
