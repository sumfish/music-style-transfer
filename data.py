"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import os
import torch.utils.data as data


def load_npy(data_path):
    data=np.load(data_path)
    return data

class AudioDataset(data.Dataset):
    def __init__(self,
                 root,
                 label,
                 transform=None):
        self.transform = transform
        self.data=load_npy(os.path.join(root,'data.npy'))
        self.label=load_npy(os.path.join(label,'label.npy'))

    def __getitem__(self,index):
        audio=self.data[index]
        #print(audio.shape) #(128,Time) 
        
        if self.transform is not None:
            audio=self.transform(audio) #is already float data

        category=int(self.label[index][1]) # song, instrument, segment   '01'->1

        return audio, category

    def __len__(self):
        return len(self.data)

'''
test, label=AudioDataset('./datasets/npy_txt/hung/2s_4class_128(no_normalize)/test','./datasets/npy_txt/hung/2s_4class_128(no_normalize)/test').__getitem__(0)
print(test)
print(label)
print(test.shape)
'''