"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy
import os
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler

from funit_model import FUNITModel

############## ????????
def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


class Trainer(nn.Module):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.model = FUNITModel(cfg)
        lr_gen = cfg['lr_gen']
        lr_dis = cfg['lr_dis']
        dis_params = list(self.model.dis.parameters())
        gen_params = list(self.model.gen.parameters())
        style_en_params = list(map(id, self.model.gen.enc_class_model.parameters()))
        cont_en_params = list(map(id, self.model.gen.enc_content.parameters()))

        ########## load pre-train
        if cfg['pre_train']:
            print('----------Load pre-train model!!!!----------')
            style_en_pt=torch.load(cfg['style_pt'])
            content_en_pt=torch.load(cfg['content_pt'])
            self.model.gen.enc_class_model.load_state_dict(style_en_pt['model'])
            self.model.gen.enc_content.load_state_dict(content_en_pt['encoder'])
        
            ########## freeze encoder parameter
            print('----------Freeze the encoder parameter!!!!----------')
            for p in self.model.gen.enc_content.parameters():
                p.requires_grad= False
            for p in self.model.gen.enc_class_model.parameters():
                p.requires_grad= False

            self.gen_opt = torch.optim.RMSprop(
                [p for p in gen_params if p.requires_grad],
                lr=lr_dis, weight_decay=cfg['weight_decay'])
        
        else:
            #################### parameter
            base_params = filter(lambda p : id(p) not in style_en_params+cont_en_params, self.model.gen.parameters())
            #print(list(base_params))

            self.gen_opt = torch.optim.RMSprop([
            {'params': base_params},
            {'params': self.model.gen.enc_content.parameters(),'lr':(lr_gen/10)},
            {'params': self.model.gen.enc_class_model.parameters(),'lr':(lr_gen/10)}],
            lr=lr_gen, weight_decay=cfg['weight_decay'])
        
        self.dis_opt = torch.optim.RMSprop(
            [p for p in dis_params if p.requires_grad],
            lr=lr_gen, weight_decay=cfg['weight_decay'])
    
        
        ##################### not understand
        self.dis_scheduler = get_scheduler(self.dis_opt, cfg) 
        self.gen_scheduler = get_scheduler(self.gen_opt, cfg) 
        self.apply(weights_init(cfg['init'])) 
 

    def gen_update(self, co_data, cl_data, hp, multigpus, trans=None):
        self.gen_opt.zero_grad()
        
        if(hp['loss_mode']=='D'):
            total, ad, xr, cr, sr, ac = self.model(co_data, cl_data, hp, 'gen_update')
            self.loss_gen_total = torch.mean(total)
            self.loss_gen_recon_x = torch.mean(xr)
            self.loss_gen_recon_c = torch.mean(cr)
            self.loss_gen_recon_s = torch.mean(sr)
            self.loss_gen_adv = torch.mean(ad)
            self.accuracy_gen_adv = torch.mean(ac)

        elif(hp['loss_mode']=='no_D_sc_recon'):
            total, xr, cr, sr = self.model(co_data, cl_data, hp, 'gen_update')
            self.loss_gen_total = torch.mean(total)
            self.loss_gen_recon_x = torch.mean(xr)
            self.loss_gen_recon_c = torch.mean(cr)
            self.loss_gen_recon_s = torch.mean(sr)

        elif(hp['loss_mode']=='no_D_xt_xa_recon'):
            total, xr, xtr = self.model(co_data, cl_data, hp, 'gen_update', trans)
            self.loss_gen_total = torch.mean(total)
            self.loss_gen_recon_x = torch.mean(xr)
            self.loss_gen_recon_trans = torch.mean(xtr)

        elif(hp['loss_mode']=='only_self_recon'):
            xr = self.model(co_data, cl_data, hp, 'gen_update')
            self.loss_gen_recon_x = torch.mean(xr)

        self.gen_opt.step()
        #return self.accuracy_gen_adv.item()
        return self.loss_gen_recon_x.item()

    def dis_update(self, co_data, cl_data, hp):
        self.dis_opt.zero_grad()
        al, lfa, lre, reg, acc = self.model(co_data, cl_data, hp, 'dis_update')
        self.loss_dis_total = torch.mean(al)
        self.loss_dis_fake_adv = torch.mean(lfa)
        self.loss_dis_real_adv = torch.mean(lre)
        self.loss_dis_reg = torch.mean(reg)
        self.accuracy_dis_adv = torch.mean(acc)
        self.dis_opt.step()
        return self.accuracy_dis_adv.item()

    def test(self, co_data, cl_data, multigpus, config):
        this_model = self.model.module if multigpus else self.model
        return this_model.test(co_data, cl_data, config)

    def resume(self, checkpoint_dir, hp, multigpus):
        this_model = self.model.module if multigpus else self.model

        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        ###################
        ###################
        this_model.gen.load_state_dict(state_dict['gen'])
        ########## load dec only ///encoder was pre-loaded
        #this_model.gen.dec.load_state_dict(state_dict['gen.dec'])
        iterations = int(last_model_name[-11:-3])

        '''
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        this_model.dis.load_state_dict(state_dict['dis'])
        
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        '''
        self.dis_scheduler = get_scheduler(self.dis_opt, hp, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hp, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations, multigpus):
        this_model = self.model.module if multigpus else self.model
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        #dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        #######################
        #### gen_dec --> only for decoder
        #### so you can load encoder/decoder seperately
        #######################
        torch.save({'gen': this_model.gen.state_dict(),
                    'gen.dec':this_model.gen.dec.state_dict()}, gen_name) 
        #torch.save({'dis': this_model.dis.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)

    def load_ckpt(self, ckpt_name):
        state_dict = torch.load(ckpt_name)
        self.model.gen.load_state_dict(state_dict['gen'])

    def translate(self, co_data, cl_data):
        return self.model.translate(co_data, cl_data)

    def translate_k_shot(self, co_data, cl_data, k, mode):
        return self.model.translate_k_shot(co_data, cl_data, k, mode)

    def forward(self, *inputs):
        print('Forward function not implemented.')
        pass


def get_model_list(dirname, key):
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


def get_scheduler(optimizer, hp, it=-1):
    if 'lr_policy' not in hp or hp['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hp['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hp['step_size'],
                                        gamma=hp['gamma'], last_epoch=it)
    else:
        return NotImplementedError('%s not implemented', hp['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun
