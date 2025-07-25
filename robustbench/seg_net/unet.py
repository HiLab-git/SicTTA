from robustbench.losses import DiceLoss
import cv2
from scipy import ndimage
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import os.path as osp
import SimpleITK as sitk
from torch.optim import lr_scheduler
from torch.nn import init


# from utils.convert import *


def get_scheduler(optimizer):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
    return scheduler
def denorm(img):
    return (img+1.0)/2
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def get_largest_component(image):
    """
    get the largest component from 2D or 3D binary image
    image: nd array
    """
    dim = len(image.shape)
    # print(dim,image.shape)
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return  output
def showseg(output):
    output = F.softmax(output,dim=1)
    output = torch.argmax(output,dim=1)
    output = output.reshape(1,1,256,320)       
    output = output.to(torch.float64)

    return output

class UNetConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self,in_channels, out_channels, dropout_p):
        """
        dropout_p: probability to be zeroed
        """
        super(UNetConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)
 
class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode, dropout_p):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTransposed2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode=='upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        self.conv_block = UNetConvBlock(in_chans, out_chans, dropout_p)

    def centre_crop(self, layer, target_size):
        _,_,layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.centre_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out

class Encoder(nn.Module):
    def __init__(self,
        in_chns,
        n_classes,
        ft_chns,
        dropout_p
        ):
        super().__init__()
        self.in_chns   = in_chns
        self.ft_chns   = ft_chns
        self.n_class   = n_classes
        self.dropout   = dropout_p
        self.down_path = nn.ModuleList()
        self.down_path.append(UNetConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[0]))
        self.down_path.append(UNetConvBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[0]))

    def forward(self, x):
        blocks=[]
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x ,2)
        return blocks, x

class Decoder(nn.Module):
    def __init__(self, 
        in_chns,
        n_classes,
        ft_chns,
        dropout_p,
        up_mode):
        super().__init__()
        self.in_chns   = in_chns
        self.ft_chns   = ft_chns
        self.n_class   = n_classes
        self.dropout   = dropout_p
        self.up_path = nn.ModuleList()
        self.up_path.append(UNetUpBlock(self.ft_chns[4], self.ft_chns[3], up_mode, self.dropout[1]))
        self.up_path.append(UNetUpBlock(self.ft_chns[3], self.ft_chns[2], up_mode, self.dropout[0]))
        self.up_path.append(UNetUpBlock(self.ft_chns[2], self.ft_chns[1], up_mode, self.dropout[0]))
        self.up_path.append(UNetUpBlock(self.ft_chns[1], self.ft_chns[0], up_mode, self.dropout[0]))
        self.last = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1)

    def forward(self, x, blocks):
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i -1])
        return self.last(x),x
    def get_output_last(self,x):
        return self.last(x)
    def get_output_med(self,x):
        return self.last(x)

class UNet(nn.Module):
    def __init__(
        self, params):
        super(UNet, self).__init__()
        lr = params['lr']
        self.in_chns = params['in_chns']
        n_classes = params['n_classes']
        ft_chns = params['ft_chns']
        dropout_p = params['dropout_p']
        up_mode = params['up_mode']
 
        self.enc = Encoder(self.in_chns,n_classes,ft_chns,dropout_p)
        self.dec1 = Decoder(self.in_chns,n_classes,ft_chns,dropout_p,up_mode)
        # setting the optimzer
        opt = 'adam'
        if opt == 'adam':
            self.enc_opt = torch.optim.Adam(self.enc.parameters(),lr=lr,betas=(0.9,0.999))
            self.dec1_opt = torch.optim.Adam(self.dec1.parameters(),lr=lr,betas=(0.9,0.999))

        # setting the using of loss  
        self.segloss = DiceLoss(n_classes).to(torch.device('cuda'))
        
        self.enc_opt_sch = get_scheduler(self.enc_opt)
        self.dec_1_opt_sch = get_scheduler(self.dec1_opt)

        # if True:
        #     newBN=AdaBN
        #     self.newBN = newBN
        #     warm_n=5
        #     self.enc = replace_bn_with_adabn(self.enc, newBN, warm_n=warm_n)
        #     self.dec1 = replace_bn_with_adabn(self.dec1, newBN, warm_n=warm_n)
    def initialize(self):
        init_weights(self.enc)
        init_weights(self.dec1)
    def change_BN_status(self, new_sample=True):
        for nm, m in self.named_modules():
            if isinstance(m, self.newBN):
                m.new_sample = new_sample

    def update_lr(self):
        self.enc_opt_sch.step()
        self.dec_1_opt_sch.step()  
    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        self.blocks1, self.latent_A1 = self.enc(x)
        self.output,self.last_x = self.dec1(self.latent_A1, self.blocks1)
        if (len(x_shape) == 5):
            new_shape = [N, D] + list(self.output.shape)[1:]
            self.output = torch.reshape(self.output, new_shape)
            self.output = torch.transpose(self.output, 1, 2)
        return self.output
    def get_feature(self,x, loc = 'med'):
        if loc == 'last':
            self.blocks1, self.latent_A1 = self.enc(x)
            _,self.last_x = self.dec1(self.latent_A1, self.blocks1)
            return self.last_x
        elif loc == 'med':
            self.blocks1, self.latent_A1 = self.enc(x)
            return self.latent_A1
        
    def get_output(self,x,loc = 'med'):
        if loc == 'last':
            return self.dec1.get_output_last(x)
        elif loc == 'med':
            output,_ = self.dec1(x, self.blocks1)
            return output
    
    
    def train_source(self,imagesa,labelsa):
        self.enc_opt.zero_grad()
        self.dec1_opt.zero_grad()
        self.forward(imagesa)
        seg_loss = self.segloss(self.output,labelsa,softmax = True ,one_hot = True)
        seg_loss.backward()
        self.enc_opt.step()
        self.dec1_opt.step()
        print(seg_loss.item())
