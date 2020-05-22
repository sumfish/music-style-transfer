"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy

import torch
import torch.nn as nn

from networks import FewShotGen, GPPatchMcResDis


def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


class FUNITModel(nn.Module):
    def __init__(self, hp):
        super(FUNITModel, self).__init__()
        self.gen = FewShotGen(hp['gen'])
        self.dis = GPPatchMcResDis(hp['dis'])

    def forward(self, co_data, cl_data, hp, mode, trans=None, train_mode=None):
        if(hp['loss_mode']=='no_D_xt_xa_recon'):
            xa = co_data.cuda()
            xb = cl_data.cuda()
            trans = trans.cuda()
        elif(hp['loss_mode']=='no_D_AUTOVC'):
            xa = co_data.cuda()
            ### target in style method below
        else:
            xa = co_data[0].cuda()
            la = co_data[1].cuda() #class label
            xb = cl_data[0].cuda()
            lb = cl_data[1].cuda() #class label


        if mode == 'gen_update':
            # content method
            c_xa = self.gen.enc_content(xa)

            # style method
            if(hp['loss_mode']=='no_D_AUTOVC'):
                for count in range(len(cl_data)):
                    scode=cl_data[count][:,:,:,:87].cuda()
                    temp=self.gen.enc_class_model(scode)
                    if count==0:
                        class_code=temp
                    else:
                        class_code+=temp
                s_xb=class_code/(len(cl_data))
            else:
                s_xa = self.gen.enc_class_model(xa[:,:,:,:87])
                s_xb = self.gen.enc_class_model(xb[:,:,:,:87])  #[B,64]

            if(hp['loss_mode']!='no_D_AUTOVC'):
                xr = self.gen.decode(c_xa, s_xa)  # reconstruction
            ### every mode needs
            xt = self.gen.decode(c_xa, s_xb)  # translation

            
            if(hp['loss_mode']=='D'):
                # gan loss
                l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(xt, lb)
                l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(xr, la)
                
                # feature matching loss
                _, xb_gan_feat = self.dis(xb, lb)
                _, xa_gan_feat = self.dis(xa, la)
                l_c_rec = recon_criterion(xr_gan_feat.mean(3).mean(2),
                                        xa_gan_feat.mean(3).mean(2))
                #r1=xr_gan_feat.mean(3).mean(2)
                #r2=xr_gan_feat.mean(3)   #[B,1024(channel)]
                l_s_rec = recon_criterion(xt_gan_feat.mean(3).mean(2),
                                        xb_gan_feat.mean(3).mean(2))
                
                # x_a recons
                l_x_rec = recon_criterion(xr, xa)

                l_adv = 0.5 * (l_adv_t + l_adv_r)
                acc = 0.5 * (gacc_t + gacc_r)
                l_total = (hp['gan_w'] * l_adv + hp['r_w'] * l_x_rec + hp[
                    'fm_w'] * (l_c_rec + l_s_rec))
                l_total.backward()

                return l_total, l_adv, l_x_rec, l_c_rec, l_s_rec, acc
            
            elif(hp['loss_mode']=='no_D_sc_recon'):
                # c_a recons
                c_xt = self.gen.enc_content(xt)
                l_ca_rec = recon_criterion(c_xt,c_xa)

                # s_b recons
                s_xt = self.gen.enc_class_model(xt)
                l_sb_rec = recon_criterion(s_xt,s_xb) 

                # x_a recons
                l_x_rec = recon_criterion(xr, xa)
                
                if(hp['not_use_style_loss']):
                #if (hp['freeze_whom']=='only_s'):
                    l_total = hp['r_w'] * l_x_rec + hp['r_w']* l_ca_rec
                    l_total.backward()
                else:
                    ## weight setting from x_rec=1 c_rec=1
                    # https://github.com/auspicious3000/autovc
                    l_total = hp['r_w'] * l_x_rec + hp['r_w']* (l_ca_rec+l_sb_rec)*0.5 
                    l_total.backward()

                return l_total, l_x_rec, l_ca_rec, l_sb_rec

            elif(hp['loss_mode']=='no_D_xt_xa_recon'):
                # x_a recons
                l_x_rec = recon_criterion(xr, xa)
                l_xtr_rec = recon_criterion(xt, trans)

                l_total= l_x_rec+l_xtr_rec
                l_total.backward()

                return l_total, l_x_rec, l_xtr_rec

            elif(hp['loss_mode']=='no_D_AUTOVC'):
                # x_a recons
                l_x_rec = recon_criterion(xt, xa)  #trans is recons
                
                # c_a recons
                c_xt = self.gen.enc_content(xt)
                l_c_rec = recon_criterion(c_xa, c_xt)

                # s_b recons
                s_xt = self.gen.enc_class_model(xt)
                l_s_rec = recon_criterion(s_xb, s_xt)

                if(hp['not_use_style_loss']):
                    l_total= l_x_rec+l_c_rec
                    l_total.backward()

                    return l_total, l_x_rec, l_c_rec, l_s_rec
                else:
                    if(hp['exchange_train_mode']):
                        if train_mode: #True->train content
                            l_total= l_x_rec+ l_c_rec
                            l_total.backward()
                        else:
                            l_total= l_x_rec+ l_s_rec
                            l_total.backward()

                    else:    
                        l_total= l_x_rec+ 0.5*(l_c_rec+l_s_rec)
                        l_total.backward()
                    
                    return l_total, l_x_rec, l_c_rec, l_s_rec

            elif(hp['loss_mode']=='only_self_recon'):
                # x_a recons
                l_x_rec = recon_criterion(xr, xa)
                l_x_rec.backward()
                
                return l_x_rec

        elif mode == 'dis_update':
            xb.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(xb, lb)
            l_real = hp['gan_w'] * l_real_pre
            l_real.backward(retain_graph=True)

            ################################################
            ### important gradient penalty loss
            ################################################
            l_reg_pre = self.dis.calc_grad2(resp_r, xb)
            l_reg = 10 * l_reg_pre
            l_reg.backward()
            with torch.no_grad():
                c_xa = self.gen.enc_content(xa)
                s_xb = self.gen.enc_class_model(xb[:,:,:,:87])
                xt = self.gen.decode(c_xa, s_xb)
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach(),
                                                                  lb)
            l_fake = hp['gan_w'] * l_fake_p
            l_fake.backward()
            l_total = l_fake + l_real + l_reg
            acc = 0.5 * (acc_f + acc_r)
            return l_total, l_fake_p, l_real_pre, l_reg_pre, acc
        else:
            assert 0, 'Not support operation'

    def test(self, co_data, cl_data, config):
        self.eval()
        self.gen.eval()
        if (config['loss_mode']=='no_D_xt_xa_recon'):
            xa = co_data.cuda()
            xb = cl_data.cuda()
        elif(config['loss_mode']=='no_D_AUTOVC'):
            xa = co_data.cuda()
            xb = cl_data[0].cuda()
        else:
            xa = co_data[0].cuda()
            xb = cl_data[0].cuda()
        
        ##################
        c_xa_current = self.gen.enc_content(xa)
        if(config['loss_mode']=='no_D_AUTOVC'):
            for count in range(len(cl_data)):
                scode=cl_data[count][:,:,:,:87].cuda()
                temp=self.gen.enc_class_model(scode)
                if count==0:
                    class_code=temp
                else:
                    class_code+=temp
            s_xb_current=class_code/(len(cl_data))
            xr_current = self.gen.decode(c_xa_current, s_xb_current)

        else:
            s_xa_current = self.gen.enc_class_model(xa[:,:,:,:87])
            s_xb_current = self.gen.enc_class_model(xb[:,:,:,:87])
            xr_current = self.gen.decode(c_xa_current, s_xa_current)
        
        xt_current = self.gen.decode(c_xa_current, s_xb_current)
        self.train()
        return xa, xr_current, xt_current, xb

    def translate_k_shot(self, co_data, cl_data, k):
        self.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa_current = self.gen.enc_content(xa)
        if k == 1:
            c_xa_current = self.gen.enc_content(xa)
            s_xb_current = self.gen.enc_class_model(xb[:,:,:,:87])########################
            xt_current = self.gen.decode(c_xa_current, s_xb_current)
        else:
            s_xb_current_before = self.gen.enc_class_model(xb[:,:,:,:87])##########################
            s_xb_current_after = s_xb_current_before.squeeze(-1).permute(1,
                                                                         2,
                                                                         0)
            s_xb_current_pool = torch.nn.functional.avg_pool1d(
                s_xb_current_after, k)
            s_xb_current = s_xb_current_pool.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen.decode(c_xa_current, s_xb_current)
        return xt_current

    def compute_k_style(self, style_batch, k):
        self.eval()
        style_batch = style_batch.cuda()
        s_xb_before = self.gen.enc_class_model(style_batch[:,:,:,:87])########################
        s_xb_after = s_xb_before.squeeze(-1).permute(1, 2, 0)
        s_xb_pool = torch.nn.functional.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool.permute(2, 0, 1).unsqueeze(-1)
        return s_xb

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.cuda()
        s_xb_current = class_code.cuda()
        c_xa_current = self.gen.enc_content(xa)
        xt_current = self.gen.decode(c_xa_current, s_xb_current)
        return xt_current

    def translate_a_shot(self, content_spec, style_spec):
        self.eval()
        xa = content_spec.cuda()
        xb = style_spec.cuda()
        c_xa_current = self.gen.enc_content(xa)
        s_xb_current = self.gen.enc_class_model(xb[:,:,:,:87])
        xt_current = self.gen.decode(c_xa_current, s_xb_current)
        return xt_current
