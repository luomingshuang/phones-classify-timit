#coding:utf-8
import numpy as np
import glob
import os
import re
import random
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import options as opt
from pydub import AudioSegment

#the length of mfcc features max:
class MyDataset(Dataset):
    
    def __init__(self, data_path):
        self.data_path = data_path
        #self.audio_pad = audio_pad
        #self.txts = glob.glob(os.path.join(audio_path, '*', '*', '*', '*.TXT'))
        self.samples = []
        self.phones = []
        self.sequences = []
        with open('./label_phones.txt') as myfile:
            self.phones = myfile.read().splitlines()

        f = open(self.data_path, 'r')
        for line in f.readlines():
            line = line.strip().split(' ')
            target_index = self.phones.index(line[3])
            self.sequences = MyDataset.load_feature(line[4], line[1], line[2])
            for idx in range(len(self.sequences)):
                self.samples.append((self.sequences[idx], target_index))
                #print((self.sequences[idx], target_index))
        

    def __getitem__(self, idx):
        #(audio_path, start, end, label) = self.samples[idx]
        #audio = self._load_audio(audio_path, start, end)
        #audio = self._padding(audio, self.audio_pad)
        sample, label = self.samples[idx]
        label = torch.from_numpy(np.array(int(label)))
        #print(audio.shape)
        #print(torch.FloatTensor(sample), label)
        return {'audio':torch.FloatTensor(sample), 'target':label}

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, f, start, end):
        #sound = AudioSegment.from_file(f)
        #print(sound)
        fs, sound = wav.read(f)
        sound = sound[int(float(start)*fs):int(float(end)*fs)]
        mfcc_features = mfcc(sound, samplerate=16000)
    
        return mfcc_features

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    @staticmethod
    def load_feature(f, start, end):
        fs, sound = wav.read(f)
        sound = sound[int(float(start)*fs):int(float(end)*fs)]
        mfcc_features = mfcc(sound, samplerate=16000)

        return mfcc_features
'''
train_data = MyDataset(opt.train_path, opt.audio_pad)
data = sorted(train_data, key=lambda x:len(x['audio']), reverse=True)
print(data[0], data[0]['audio'].size())
##### (353, 13)) ####
'''