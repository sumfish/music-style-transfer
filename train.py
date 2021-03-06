"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import os
import sys
import argparse
import shutil

from tensorboardX import SummaryWriter

from utils import get_config, get_train_loaders, get_special_loaders, make_result_folders
from utils import write_loss, write_1images, Timer
from trainer import Trainer

import torch.backends.cudnn as cudnn
# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='configs/funit_music.yaml',
                    help='configuration file for training and testing')
parser.add_argument('--output_path',
                    type=str,
                    default='.',
                    help="outputs path")
parser.add_argument('--multigpus',
                    action="store_true")
parser.add_argument('--batch_size',
                    type=int,
                    default=0)
parser.add_argument('--test_batch_size',
                    type=int,
                    default=4)
parser.add_argument("--resume",
                    action="store_true")
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
# Override the batch size if specified.4
if opts.batch_size != 0:
    config['batch_size'] = opts.batch_size

trainer = Trainer(config)
trainer.cuda()
if opts.multigpus: ####use multiple gpu
    ngpus = torch.cuda.device_count()
    config['gpus'] = ngpus
    print("Number of GPUs: %d" % ngpus)
    trainer.model = torch.nn.DataParallel(
        trainer.model, device_ids=range(ngpus))
else:
    config['gpus'] = 1

if config['loss_mode']=='no_D_xt_xa_recon' or config['loss_mode']=='no_D_AUTOVC':
    print('get_special_dataset:{}'.format(config['loss_mode']))
    loaders = get_special_loaders(config)
    train_loader = loaders[0]
    test_loader = loaders[1]
else:
    loaders = get_train_loaders(config)  ###################
    train_content_loader = loaders[0]
    train_class_loader = loaders[1]
    test_content_loader = loaders[2]
    test_class_loader = loaders[3]

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]+'_new_0520_Auto_blank'
#model_name = os.path.splitext(os.path.basename(opts.config))[0]+'_no_D_0503_xsc_recon_fine_tune_lr_wei_adjust'
print('model name:{}'.format(model_name))
# 建立實體資料的存放
train_writer = SummaryWriter(
    os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = make_result_folders(output_directory)
# 複製某檔案到路徑之下
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

iterations = trainer.resume(checkpoint_directory,
                            hp=config,
                            multigpus=opts.multigpus) if opts.resume else 0 #opts.resume

print('LOSS MODE:{}'.format(config['loss_mode']))

content_train=True
while True:
    if config['loss_mode']=='no_D_xt_xa_recon' or config['loss_mode']=='no_D_AUTOVC':
        for it, (co_data, cl_data, trans) in enumerate(train_loader):  #####class=style

            if(config['exchange_train_mode']):
                g_loss = trainer.gen_update(co_data, cl_data, config,
                                    opts.multigpus, trans, content_train)
                if((iterations+1)%config['exchange_fre']==0):
                    content_train=not(content_train)
                    trainer.setup_exchange_mode(content_train,config)

            else:   
                g_loss = trainer.gen_update(co_data, cl_data, config,
                                    opts.multigpus, trans)
            torch.cuda.synchronize() #####g.d together
            print('G recon_x loss: %.4f' % (g_loss))

            if ((iterations + 1) % config['image_save_iter'] == 0):          
                key_str = '%08d' % (iterations + 1)
                with torch.no_grad():  ###########################
                    for t, (val_co_data, val_cl_data, trans) in enumerate(train_loader):
                        if t >= opts.test_batch_size:
                            break
                        val_image_outputs = trainer.test(val_co_data, val_cl_data,
                                                        opts.multigpus, config)
                        write_1images(val_image_outputs, image_directory,
                                    'train_%s_%02d' % (key_str, t))
                    for t, (test_co_data, test_cl_data, trans) in enumerate(test_loader):
                        if t >= opts.test_batch_size:
                            break
                        test_image_outputs = trainer.test(test_co_data,
                                                        test_cl_data,
                                                        opts.multigpus, config)
                        write_1images(test_image_outputs, image_directory,
                                    'test_%s_%02d' % (key_str, t))
                                    
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations, opts.multigpus)
                print('Saved model at iteration %d' % (iterations + 1))

            iterations += 1
            if iterations >= max_iter:
                print("Finish Training")
                sys.exit(0)

    #### other loss mode
    else:
        for it, (co_data, cl_data) in enumerate(
                zip(train_content_loader, train_class_loader)):  #####class=style
            
            #with Timer("Elapsed time in update: %f"):
            if(config['loss_mode']=='D'):          
                if(it%5==0):
                    d_acc = trainer.dis_update(co_data, cl_data, config)

            g_loss = trainer.gen_update(co_data, cl_data, config,
                                        opts.multigpus)
            torch.cuda.synchronize() #####g.d together
            #print('D acc: %.4f\t G acc: %.4f' % (d_acc, g_acc))
            print('G loss: %.4f' % (g_loss))

            ######################################## below unseen 
            if ((iterations + 1) % config['image_save_iter'] == 0):          
                key_str = '%08d' % (iterations + 1)
                with torch.no_grad():  ###########################
                    for t, (val_co_data, val_cl_data) in enumerate(
                            zip(train_content_loader, train_class_loader)):
                        if t >= opts.test_batch_size:
                            break
                        val_image_outputs = trainer.test(val_co_data, val_cl_data,
                                                        opts.multigpus, config)
                        write_1images(val_image_outputs, image_directory,
                                    'train_%s_%02d' % (key_str, t))
                    for t, (test_co_data, test_cl_data) in enumerate(
                                zip(test_content_loader, test_class_loader)):
                        if t >= opts.test_batch_size:
                            break
                        test_image_outputs = trainer.test(test_co_data,
                                                        test_cl_data,
                                                        opts.multigpus, config)
                        write_1images(test_image_outputs, image_directory,
                                    'test_%s_%02d' % (key_str, t))

            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations, opts.multigpus)
                print('Saved model at iteration %d' % (iterations + 1))

            iterations += 1
            if iterations >= max_iter:
                print("Finish Training")
                sys.exit(0)
