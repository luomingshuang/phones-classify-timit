#coding:utf-8
import os
from python_speech_features import mfcc
from pydub import AudioSegment

utt_F = ['A', 'B', 'C']
au_ph = 'files/out_ths30_train.ali'
#au_ph = 'out1.ali'
#au_ph = 'files/out_ths30_dev.ali'
au_ph = 'files/out_ths30_tst.ali'
phones = []
uttID = []
data = []
path = []
f = open(au_ph, 'r')
#f_data = open('files/data_train.txt', 'w')
#f_data = open('files/dataset_dev.txt', 'w')
f_data = open('files/dataset_tst.txt', 'w')
#wavscp = 'files/wav_ths30_train.scp'
#wavscp = 'files/wav_ths30_dev.scp'
wavscp = 'files/wav_ths30_tst.scp'
f_train = open(wavscp, 'r')
for line in f_train.readlines():
    line = line.strip().split(' ')
    path.append((line[0], line[1]))
print(path)
n = 0
for line in f.readlines():  
    #print(line, path[n])
    line = line.strip()
    First = line[0]
    if First in ('A','B','C','D'):
        uttID.append(line.strip())
    if First not in ('A','B','C','D','.'):
        #print(line)
        line_phone = line.strip().split(' ')
        #print(line_phone)
        #if line_phone[2] not in phones:
        #    phones.append(line_phone[2])
        #print(uttID[0], path[n][0], path[n][1])
        print(uttID, path[n])
        if uttID[0] == path[n][0]:
            data.append((uttID[0], line_phone[0], line_phone[1], line_phone[2], path[n][1]))
            #print(data)
        #print(line_phone)
    if First == '.':
        uttID = []
        n += 1

#print(phones, len(phones))
'''
label = 'label_phones.txt'
#print(data, len(data))
with open(label, 'w') as f:
    for p in phones:
        f.write(p+'\n')
'''
print(data)
for f in data:
    f_data.writelines(f[0]+' '+f[1]+' '+f[2]+' '+f[3]+' '+f[4])
    f_data.write('\n')
f_data.close()

####n_phones=188 in train data####

####测试所有训练片段中最长的音频的mfcc 序列长是多少####


