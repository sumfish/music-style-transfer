"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import yaml
import time
import librosa.display
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils

from data import AudioDataset, AudioParallelDataset, AudioAutoVCDataset

def mkdir(directory):
    if not os.path.exists(directory):
        print('making dir:{0}'.format(directory))
        os.makedirs(directory)
    else:
        print('already exist: {0}'.format(directory))


def loader_from_list(
        root,
        label,
        fix_length,
        batch_size,
        mode,
        num_workers=4,
        shuffle=True,
        return_paths=False,
        drop_last=True):
    transform_list = [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    if mode =='no_D_xt_xa_recon':
        dataset = AudioParallelDataset(root,
                                label,
                                fix_length,
                                transform)        
    elif mode == 'no_D_AUTOVC':
        dataset = AudioAutoVCDataset(root,
                                label,
                                fix_length,
                                transform)  
    else:
        dataset = AudioDataset(root,
                                label,
                                fix_length,
                                transform)
    loader = DataLoader(dataset,
                        batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=num_workers)
    return loader


def get_evaluation_loaders(conf, shuffle_content=False):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    fix_length = conf['fix_length']

    content_loader = loader_from_list(
            root=conf['data_train'],
            label=conf['label_train'],
            fix_length=fix_length,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle_content,
            return_paths=True,
            drop_last=False)

    class_loader = loader_from_list(
            root=conf['data_test'],
            label=conf['label_test'],
            fix_length=fix_length,
            batch_size=batch_size * conf['k_shot'],
            num_workers=1,
            shuffle=False,
            return_paths=True,
            drop_last=False)
    return content_loader, class_loader

def get_special_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers'] ############
    fix_length = conf['fix_length']
    mode = conf['loss_mode']

    train_loader = loader_from_list(
            root=conf['data_train'],
            label=conf['label_train'],
            fix_length=fix_length,
            batch_size=batch_size,
            mode=mode,
            num_workers=num_workers)

    test_loader = loader_from_list(
            root=conf['data_test'],
            label=conf['label_test'],
            fix_length=fix_length,
            batch_size=batch_size,
            mode=mode,
            num_workers=num_workers)

    return (train_loader, test_loader)

def get_train_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers'] ############
    fix_length = conf['fix_length']
    mode = conf['loss_mode']

    train_content_loader = loader_from_list(
            root=conf['data_train'],
            label=conf['label_train'],
            fix_length=fix_length,
            batch_size=batch_size,
            mode=mode,
            num_workers=num_workers)
    train_class_loader = loader_from_list(
            root=conf['data_train'],
            label=conf['label_train'],
            fix_length=fix_length,
            batch_size=batch_size,
            mode=mode,
            num_workers=num_workers)
    test_content_loader = loader_from_list(
            root=conf['data_test'],
            label=conf['label_test'],
            fix_length=fix_length,
            batch_size=batch_size,
            mode=mode,
            num_workers=0)
    test_class_loader = loader_from_list(
            root=conf['data_test'],
            label=conf['label_test'],
            fix_length=fix_length,
            batch_size=batch_size,
            mode=mode,
            num_workers=0)

    return (train_content_loader, train_class_loader, test_content_loader,
            test_class_loader)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def make_result_folders(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def __write_images(im_outs, dis_img_n, file_name):
    im_outs = [images.view(-1, 128, 128) for images in im_outs]
    image_tensor = torch.cat([images[:dis_img_n] for images in im_outs], 0)
    ####spec   
    plt.figure(1,figsize=(5*4,4*4))
    for i in range(4):
        for j in range(5):#4
            index= (i*5)+j+1
            plt.subplot(4, 5, index)
            
            librosa.display.specshow(image_tensor[index-1].cpu().numpy(), hop_length=256)
            plt.set_cmap("magma")
    plt.savefig(file_name, dpi='figure', bbox_inches='tight')
    plt.clf()
    '''
    image_grid = vutils.make_grid(image_tensor.data,
                                  nrow=dis_img_n, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)
    '''

def write_1images(image_outputs, image_directory, postfix):
    display_image_num = image_outputs[0].size(0)  #batchsize4  #output[0] -> [4,1,128,173] #img_out len=6
    __write_images(image_outputs, display_image_num,
                   '%s/gen_%s.jpg' % (image_directory, postfix))


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr))]
    for m in members:
        # getattr 拿某屬性的值
        # tag --> section/plot tensorboard會按照section來分組顯示
        # ex: accuracy/ vali or test
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)  #tag, value, iteration


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
