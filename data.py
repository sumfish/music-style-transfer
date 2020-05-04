"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import os
import random
import torch.utils.data as data
from torchvision import transforms


def load_npy(data_path):
    data=np.load(data_path)
    return data

'''
class AudioAutoVClDataset(data.Dataset):
    def __init__(self,
                 root,
                 label,
                 fix_length,
                 transform=None):
        self.transform = transform
        self.fix_length = fix_length
        self.data=load_npy(os.path.join(root,'data.npy'))
        self.label=load_npy(os.path.join(label,'label.npy'))

    def __getitem__(self,index):
        # source
        con_audio = self.data[index]
        con_label = self.label[index]

        # target
        rand = random.randint(0,len(self.data)-1)
        sty_audio = self.data[rand]
        sty_class = self.label[rand][1]

        

    def __len__(self):
        return len(self.data)
'''
class AudioParallelDataset(data.Dataset):
    def __init__(self,
                 root,
                 label,
                 fix_length,
                 transform=None):
        self.transform = transform
        self.fix_length = fix_length
        self.data=load_npy(os.path.join(root,'data.npy'))
        self.label=load_npy(os.path.join(label,'label.npy'))

    def __getitem__(self,index):
        # source
        con_audio = self.data[index]
        con_label = self.label[index]

        # target
        rand = random.randint(0,len(self.data)-1)
        sty_audio = self.data[rand]
        sty_class = self.label[rand][1]

        # parallel
        para_label = [con_label[0], sty_class, con_label[2]]
        try:
            parallel_index = self.label.tolist().index(para_label)
        except ValueError:  # if some song is too long
            con_audio = self.data[index-10]
            con_label = self.label[index-10]
            para_label = [con_label[0], sty_class, con_label[2]]
            parallel_index = self.label.tolist().index(para_label)

        parallel_audio = self.data[parallel_index]

        if self.fix_length:
            j = random.randint(0, con_audio.shape[1] - 128)
            con_audio = con_audio[:, j:j+128]
            sty_audio = sty_audio[:, j:j+128]
            parallel_audio = parallel_audio[:, j:j+128]

        if self.transform is not None:
            con_audio=self.transform(con_audio) #is already float data
            sty_audio=self.transform(sty_audio)
            parallel_audio=self.transform(parallel_audio)

        #category=int(self.label[index][1]) # song, instrument, segment   '01'->1

        return con_audio, sty_audio, parallel_audio

    def __len__(self):
        return len(self.data)

class AudioDataset(data.Dataset):
    def __init__(self,
                 root,
                 label,
                 fix_length,
                 transform=None):
        self.transform = transform
        self.fix_length = fix_length
        self.data=load_npy(os.path.join(root,'data.npy'))
        self.label=load_npy(os.path.join(label,'label.npy'))

    def __getitem__(self,index):
        audio=self.data[index]
        #print(audio.shape) #(128,Time) 
        
        if self.fix_length:
            j = random.randint(0, audio.shape[1] - 128)
            audio = audio[:, j:j+128]

        if self.transform is not None:
            audio=self.transform(audio) #is already float data

        category=int(self.label[index][1]) # song, instrument, segment   '01'->1

        return audio, category

    def __len__(self):
        return len(self.data)

'''
transform_list = [transforms.ToTensor()]
transform = transforms.Compose(transform_list)

test, label=AudioDataset('./datasets/npy_txt/hung/2s_4class_128(no_normalize)/test','./datasets/npy_txt/hung/2s_4class_128(no_normalize)/test', True, transform).__getitem__(0)
print(test.shape)

c, s, t=AudioParallelDataset('./datasets/npy_txt/hung/2s_4class_512_no_normalize/train','./datasets/npy_txt/hung/2s_4class_512_no_normalize/train', True, transform).__getitem__(16)
print(c.shape)
'''