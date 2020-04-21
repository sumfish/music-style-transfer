"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F

from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock


class GPPatchMcResDis(nn.Module):
    def __init__(self, hp):
        super(GPPatchMcResDis, self).__init__()
        assert hp['n_res_blks'] % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = hp['n_res_blks'] // 2
        nf = hp['nf']
        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, hp['num_classes'], 1, 1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x, y):
        # y: data class label
        assert(x.size(0) == y.size(0))
        #x=x.view(-1,1,x.size(1),x.size(2))
        # get the last layer output of D
        # eg.[6,1024,16,22]
        feat = self.cnn_f(x) 
        # patchGAN output: 幾個分類就有幾張feature map, training set how many class
        # eg.[6,4,16,22]
        out = self.cnn_c(feat) 
        # index == ex.[0,1,2,3,4,5] torch.size[batchsize=6]
        index = torch.LongTensor(range(out.size(0))).cuda()  
        # return label種類的feature map
        # eg.[6,16,22] 
        out = out[index, y, :, :] 
        return out, feat

    def calc_dis_fake_loss(self, input_fake, input_label):
        resp_fake, gan_feat = self.forward(input_fake, input_label)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean()
        correct_count = (resp_fake < 0).sum()
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss(self, input_real, input_label):
        resp_real, gan_feat = self.forward(input_real, input_label)
        total_count = torch.tensor(np.prod(resp_real.size()),
                                   dtype=torch.float).cuda()
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label):
        ############### true ################
        # gan_feat --> former last layer of D
        # resp_faks --> label class D matrix feature map [B(6),H(16),W(22)]
        resp_fake, gan_feat = self.forward(input_fake, input_fake_label)
        # np.prod 所有乘積
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        ############## ad loss
        loss = -resp_fake.mean()
        ##############
        correct_count = (resp_fake >= 0).sum()
        #### type as: turn correct_count type to as same as loss
        accuracy = correct_count.type_as(loss) / total_count
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()/batch_size
        return reg


class FewShotGen(nn.Module):
    def __init__(self, hp):
        super(FewShotGen, self).__init__()
        ch = hp['ch']
        mel_d = hp['mel_d']
        latent_dim = hp['latent_dim']
        num_classes = hp['num_classes']
        kernel = hp['kernel']
        self.content_dim= hp['content_dim']
        self.style_dim = hp['latent_dim']
        self.enc_class_model = ClassModelEncoder(latent_dim,
                                                 num_classes,
                                                 ch)

        self.enc_content = ContentEncoder(mel_d,
                                            kernel)

        self.dec = Decoder(mel_d,
                            kernel)


    def forward(self, one_image, model_set):
        # reconstruct an image
        content, model_codes = self.encode(one_image, model_set)
        
        print('model_codesss shape{}'.format(model_codes.shape))
        
        model_code = torch.mean(model_codes, dim=0).unsqueeze(0)
        images_trans = self.decode(content, model_code)
        return images_trans

    '''
    def encode(self, one_image, model_set):
        # extract content code from the input image
        content = self.enc_content(one_image)
        # extract class code from the images in the model set
        ####### set ####### 
        class_codes = self.enc_class_model(model_set)
        class_code = torch.mean(class_codes, dim=0).unsqueeze(0)    
        
        print('class_codesss shape{}'.format(class_codes.shape))
        print('class_code shape{}'.format(class_code.shape))
        
        return content, class_code
    '''

    def decode(self, content, model_code):
        # decode content and style codes to an image
        # concatenate
        model_code_flat=model_code.view(-1)  #1d
        #print(model_code)
        temp_style=model_code_flat.repeat(self.content_dim,1).t()
        #print(temp_style)
        temp_style=temp_style.view(-1,self.style_dim,self.content_dim)
        #print(temp_style)
        content=content.view(-1,1,self.content_dim)
        #print(content)
        combined=torch.cat((content,temp_style),dim=1)
        #print(combined.shape)
        images = self.dec(combined)
        return images

class res_block2d(nn.Module):
    def __init__(self, inp, out, kernel):
        super(res_block2d, self).__init__()
        if kernel==3:
            last_kernel=1
        else: 
            last_kernel=5
        self.bn1 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, out, (kernel,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, (kernel,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(out)
        self.up = nn.Conv2d(inp, out, (last_kernel,1), padding=(0,0))
    
    def forward(self, x):
        out = self.conv1(self.bn1(x)) #before is a cnn layer
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        out += self.up(x) ##########################
        return out

class ClassModelEncoder(nn.Module):
    def __init__(self, latent_dim, num_classes, ch):
        super(ClassModelEncoder, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(1, ch, (5,1), padding=(1,0))
        self.fc1 = nn.Linear(ch*3, latent_dim)
        self.fc2 = nn.Linear(latent_dim, num_classes)
        self.lin_drop = nn.Dropout(p=0.5)
        
        self.head = nn.Sequential(
            res_block2d(ch, ch*2, 5),
            nn.Dropout(p=0.25),
            nn.MaxPool2d((3,1),(3,1)), #(42,T)
            
            res_block2d(ch*2, ch*3, 3),
            nn.BatchNorm2d(ch*3),
            nn.ReLU(inplace=True),
        )
        

    def forward(self, _input):
        x = _input
        #print('original:{}'.format(x.shape))
        x = self.conv(x)
        x = self.head(x)
        #print('level 1(after res):{}'.format(x.shape))
        x = self.avgpool(x)
        #print('level 2:{}'.format(x.shape))
        x = torch.flatten(x, 1)
        #print('level 3:{}'.format(x.shape))
        last_layer = self.lin_drop(F.relu(self.fc1(x))) ########################
        #print('level 4:{}'.format(x.shape))
        out = F.softmax(self.fc2(last_layer), dim=0) ####classifier
        #print('level 5:{}'.format(last_layer.shape))
        return last_layer
        
class res_block1d(nn.Module):
    def __init__(self, inp, out, kernel, stride_list):
        super(res_block1d, self).__init__()
        self.conv1 = nn.Conv1d(inp, inp, kernel_size=kernel, stride=stride_list[0], padding=kernel//2)
        self.bn2 = nn.BatchNorm1d(inp)
        self.conv2 = nn.Conv1d(inp, out, kernel_size=kernel, stride=stride_list[1], padding=kernel//2)
        self.bn3 = nn.BatchNorm1d(out)
        self.add_conv = nn.Conv1d(inp, out, kernel_size=kernel, stride=stride_list[1], padding=kernel//2)
        if inp!=out:
            print('downsample')
            downsample = True
        else:
            downsample = False
        self.downsample = downsample
        #self.up = nn.Conv1d(inp, out, kernel_size=kernel)
    
    def forward(self, x):
        '''
        block x:torch.Size([10, 128, 173])
        f(x):torch.Size([10, 128, 173])
        f(x):torch.Size([10, 128, 87])
        block x:torch.Size([10, 128, 87])
        '''
        #print('in')
        #print('block x:{}'.format(x.shape))  #shape(N,128,173)
        ori = x
        out = self.conv1(x) 
        out = F.relu(self.bn2(out))
        #print('f(x):{}'.format(out.shape))
        out = self.conv2(out)
        out = F.relu(self.bn3(out))
        #print('f(x):{}'.format(out.shape))
        if self.downsample:
            out = out + self.bn3(self.add_conv(ori))
            out = F.relu(out)
            #print('f(x):{}'.format(self.add_conv(ori).shape))
        else:
            out = out + F.avg_pool1d(ori, kernel_size=2, ceil_mode=True)
        #print('block x:{}'.format(out.shape))
        return out 

class ContentEncoder(nn.Module):   #new ,1d ,no connected layer
    def __init__(self, mel_d, kernel):
        super(ContentEncoder, self).__init__()
        self.c_in=mel_d ###128=mel dimension
        self.c_out=1
        self.c_m=64
        self.kernel=kernel
        self.stride=[1,2]
        self.conv1 = nn.Conv1d(self.c_in, self.c_in, kernel_size=1) 
        self.norm_layer = nn.BatchNorm1d(self.c_in)
        self.act = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.25) 
        #self.conv_last = res_block1d(self.c_m, self.c_out, self.kernel, self.stride)     

        self.head = nn.Sequential(
            res_block1d(self.c_in, self.c_in, self.kernel, self.stride),
            res_block1d(self.c_in, self.c_in, self.kernel, self.stride),
            res_block1d(self.c_in, self.c_m, self.kernel, self.stride),
            res_block1d(self.c_m, self.c_m, self.kernel, self.stride),
            res_block1d(self.c_m, self.c_out, self.kernel, self.stride),
        )
    def forward(self, _input):
        x = _input
        #print('original:{}'.format(x.shape))
        x = x.view(-1, x.size(2), x.size(3))
        #print('after view:{}'.format(x.shape)) 
        
        #### conv bank??????

        #### dimension up
        x = self.conv1(x)
        x = self.norm_layer(x)
        x = self.act(x)
        x = self.drop_out(x)

        #### residual
        x = self.head(x)
        #print('level 1(after res):{}'.format(x.shape))
        #x = self.conv_last(embed)
        x = x.view(-1, x.size(2))
        #print('level 1(after res):{}'.format(x.shape))
        #input()
        return x

class res_block_up(nn.Module):
    def __init__(self, inp, out, kernel, stride_list):
        super(res_block_up, self).__init__()
        self.conv1 = nn.Conv1d(inp, inp, kernel_size=kernel, stride=stride_list[0], padding=kernel//2)
        self.bn2 = nn.BatchNorm1d(inp)
        self.conv2 = nn.Conv1d(inp, out, kernel_size=kernel, stride=stride_list[0], padding=kernel//2)
        self.bn3 = nn.BatchNorm1d(out)
        self.add_conv = nn.Conv1d(inp, out, kernel_size=kernel, stride=stride_list[0], padding=kernel//2)
        if inp!=out:
            print('upsmaple')
            upsample = True
        else:
            upsample = False
        self.upsample = upsample
        #self.up = nn.Conv1d(inp, out, kernel_size=kernel)
    
    def forward(self, x):
        ori = x
        out = self.conv1(x) 
        out = F.relu(self.bn2(out))
        #print('f(x):{}'.format(out.shape))
        out = self.conv2(out)
        out = F.relu(self.bn3(out))
        #print('f(x):{}'.format(out.shape))
        '''
        if self.upsample:
            out = out + pixel_shuffle_1d(out, scale_factor=2) #####################
            out = F.relu(out)
            #print('f(x):{}'.format(self.add_conv(ori).shape))
        else:
            out = out + upsample(ori, scale_factor=2) 
        '''
        #print('block x:{}'.format(out.shape))
        out = out+ori
        return out 

class Decoder(nn.Module):
    def __init__(self, mel_d, kernel):
        super(Decoder, self).__init__()
        self.c_out=mel_d ###128=mel dimension
        self.c_in=65 #128+1
        self.c_m=64
        self.kernel=kernel
        self.stride=[1]
        self.conv_first = nn.Conv1d(self.c_in, self.c_m, kernel_size=1) 
        self.conv_mid = nn.Conv1d(self.c_m, self.c_m, kernel_size=2) 
        self.conv_last = nn.Conv1d(self.c_m, self.c_out, kernel_size=4) 
        self.res1=res_block_up(self.c_m, self.c_m, self.kernel, self.stride)
        
        self.head = nn.Sequential(
            #PrintLayer(),
            res_block_up(self.c_m, self.c_m, self.kernel, self.stride),
            Upsample(scale_factor=2),
            #PrintLayer(),
            res_block_up(self.c_m, self.c_m, self.kernel, self.stride),
            Upsample(scale_factor=2),
            #PrintLayer(),
            res_block_up(self.c_m, self.c_m, self.kernel, self.stride),
            Upsample(scale_factor=2),
            #PrintLayer(),
            res_block_up(self.c_m, self.c_m, self.kernel, self.stride),
            Upsample(scale_factor=2),
            #PrintLayer(),
        )

    def forward(self, x):
        x = self.conv_first(x)
        x = self.res1(x)
        x = upsample(x,scale_factor=2)
        x = self.conv_mid(x)
        x = self.head(x)
        x = self.conv_last(x)
        x = x.view(-1,1,x.size(1),x.size(2))
        #print('decoder:{}'.format(x.shape))
        return x
'''
class Decoder(nn.Module):
    def __init__(self, ups, n_res, dim, out_dim, res_norm, activ, pad_type):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        self.model += [PrintLayer()]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)
'''
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor=scale_factor

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x_up

def pixel_shuffle_1d(inp, scale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= scale_factor
    out_width = in_width * scale_factor
    inp_view = inp.contiguous().view(batch_size, channels, scale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out

def upsample(x, scale_factor=2):
    x_up = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x_up