import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
#from model import AE
#from utils import *
from trainer import Trainer

from functools import reduce
import json
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from scipy.io.wavfile import write
import random
from preprocess.tacotron.utils import melspectrogram2wav
from preprocess.tacotron.utils import get_spectrograms
import librosa 
import librosa.display
import matplotlib.pyplot as plt
from utils import mkdir

class Inferencer(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)
        # args store other information
        self.args = args
        print(self.args)

        # init the model with config
        self.build_model()

        # load model
        self.load_model()

        with open(self.args.attr, 'rb') as f:
            self.attr = pickle.load(f)

    def load_model(self):
        print(f'Load model from {self.args.model}')
        last_model_name = self.get_model_list(self.args.model, "gen")
        print(last_model_name)
        self.model.load_ckpt(last_model_name)
        self.model.eval()
        '''
        self.model.gen.load_state_dict(torch.load(last_model_name))
        print(last_model_name)
        last_model_name = self.get_model_list(self.args.model, "dis")
        self.model.dis.load_state_dict(torch.load(last_model_name))
        '''
        return

    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = Trainer(self.config)
        self.model.cuda()
        print(self.model)
        return

    def utt_make_frames(self, x): #data to model
        frame_size = 1
        #print(x.size(0)) #T 時間長度
        remains = x.size(0) % frame_size 
        if remains != 0:
            x = F.pad(x, (0, remains))
        #######################################################
        #print(out.shape) #torch[1, n_mel, T]
        out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
        return out
    
    def get_model_list(self, dirname, key):
        if os.path.exists(dirname) is False:
            return None
        gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                    os.path.isfile(os.path.join(dirname, f)) and
                    key in f and ".pt" in f]
        if gen_models is None:
            return None
        gen_models.sort()
        last_model_name = gen_models[-1]
        return last_model_name

    #def translate():

    def translate_a_song(self, x, x_cond):
        content_spec = self.utt_make_frames(x)
        style_spec = self.utt_make_frames(x_cond)
        content_split=torch.split(content_spec,173,dim=2)
        style_split=torch.split(style_spec,87,dim=2)

        #content
        c_data=[]
        for i in range(len(content_split)):
            c_data.append(content_split[i])

        # style
        for i in range(len(style_split)-1):
            with torch.no_grad():
                scode=style_split[i].unsqueeze(1)
                temp=self.model.model.gen.enc_class_model(scode)
            if i==0:
                class_code=temp
                one_code=temp
            else:
                class_code+=temp
        final_class_code=class_code/(len(style_split)-1)

        
        #####################################
        # Assembling generated Spectrogram chunks into final Spectrogram
        con=np.array([])
        flag=False
        with torch.no_grad():
            for i in range((len(c_data))):
                ccode=c_data[i].unsqueeze(1)
                dec = self.model.model.translate_simple(ccode, final_class_code)
                #dec = self.model.model.translate_simple(ccode, one_code)
                dec = dec.squeeze(0).squeeze(0)
                dec = dec.detach().cpu().numpy()
                if not flag:
                    con=dec
                    flag=True
                else:
                    con = np.concatenate((con,dec),axis=1)
        #dec = self.denormalize(dec)
        ### transpose
        dec=con.transpose((1,0))
        wav_data = melspectrogram2wav(dec)
        return wav_data, con

    def denormalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = (x - m) / s
        return ret

    def write_wav_to_file(self, wav_data, output_path):
        write(output_path, rate=self.args.sample_rate, data=wav_data)
        return

    def turn(self, x):
        dec=x
        wav_data = melspectrogram2wav(dec)
        return wav_data, dec

    def write_images(self, ori_mel, trans_mel,outputpath):

        librosa.display.specshow(trans_mel, hop_length=256)
        plt.colorbar()
        plt.savefig(os.path.join(outputpath,'trans.png'), dpi='figure', bbox_inches='tight')
        #plt.colorbar(format='%+2.0f dB')
        plt.clf()

        librosa.display.specshow(ori_mel, hop_length=256)
        plt.colorbar()
        plt.savefig(os.path.join(outputpath,'source.png'), dpi='figure', bbox_inches='tight')
        plt.colorbar()
        plt.clf()

    def inference_from_path(self):
        #Loading sound file
        src_audio, sr = librosa.load(self.args.source, sr=self.args.sample_rate)
        tar_audio, sr = librosa.load(self.args.target, sr=self.args.sample_rate)

        src_mel, _ = get_spectrograms(src_audio)
        tar_mel, _ = get_spectrograms(tar_audio)
        
        ######## 可以注意一下normalize維度
        #src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        #tar_mel = torch.from_numpy(self.normalize(tar_mel)).cuda()
        src_mel = torch.from_numpy(src_mel).cuda()
        tar_mel = torch.from_numpy(tar_mel).cuda()

        conv_wav, conv_mel = self.translate_a_song(src_mel, tar_mel)

        ####### output
        #############################################################
        outputpath=os.path.join(self.args.outpath,self.args.source_dir)
        mkdir(outputpath)
        self.write_images(src_mel.detach().cpu().numpy().transpose((1,0)),conv_mel,outputpath)
        self.write_wav_to_file(conv_wav, os.path.join(outputpath, self.args.output))
        
        ####### 測試直接從原本資料轉回來
        conv_wav_ori, _ = self.turn(src_mel.detach().cpu().numpy())
        self.write_wav_to_file(conv_wav_ori,os.path.join(outputpath,'ori.wav'))
        return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-attr', '-a', 
                        type=str,
                        default='test_audio/attr.pkl',
                        help='(normalize data)attr file path')  ####need renew
    parser.add_argument('-config', '-c', 
                        type=str,
                        default='configs/funit_music.yaml',
                        help='config file path')
    parser.add_argument('-model', '-m', 
                        type=str,
                        default='outputs/funit_music_no_D_0423/checkpoints',
                        help='model path')
    parser.add_argument('-source', '-s', 
                        type=str,
                        default='test_audio/01_01.mp3',
                        help='source wav path')
    parser.add_argument('-target', '-t', 
                        type=str,
                        default='test_audio/00_00.mp3',
                        help='target wav path')
    parser.add_argument('-source_dir','-d',
                        type=str,
                        default='01_01_to_00_00',
                        help='source wav name')
    parser.add_argument('-outpath','-op',
                        type=str,
                        default='transfered_output',
                        help='output wav path')
    parser.add_argument('-output', '-o', 
                        type=str,
                        default='output.wav',
                        help='output wav file')
    parser.add_argument('-sample_rate', '-sr', help='sample rate', default=22050, type=int)
    args = parser.parse_args()
    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)
    inferencer = Inferencer(config=config, args=args)
    inferencer.inference_from_path()


'''
-c: the path of config file.
-m: the path of model checkpoint.
-a: the attribute file for normalization ad denormalization.
-s: the path of source file (.wav).
-t: the path of target file (.wav).
-o: the path of output converted file (.wav).
'''