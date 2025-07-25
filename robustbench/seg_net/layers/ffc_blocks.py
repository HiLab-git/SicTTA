import torch.nn as nn
from layers.ffc import *


def normalize(x, norm_type):
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(x)
    elif norm_type == 'instancenorm':
        return nn.InstanceNorm2d(x)
    else:
        return nn.BatchNorm2d(x) #temp


def ffconv_lrelu(in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
    return nn.Sequential(
        FFC(in_channels, out_channels, stride, groups, enable_lfu),
        nn.LeakyReLU(0.2, inplace=True)
    )


def ffconv_bn_lrelu(in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
    return nn.Sequential(
        FFC(in_channels, out_channels, stride, groups, enable_lfu),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )


def ffconv_in_lrelu(in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
    return nn.Sequential(
        FFC(in_channels, out_channels, stride, groups, enable_lfu),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )


def ffconv_bn_relu(in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
    return nn.Sequential(
        FFC(in_channels, out_channels, stride, groups, enable_lfu),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def ffconv_relu(in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
    return nn.Sequential(
        FFC(in_channels, out_channels, stride, groups, enable_lfu),
        nn.ReLU(inplace=True)
    )


def ffconv_no_activ(in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
    return FFC(in_channels, out_channels, stride, groups, enable_lfu)


def upffconv(in_channels, out_channels, norm='batchnorm'):
    return nn.Sequential(
        FFC(in_channels, out_channels, 1, 1, False),
        normalize(out_channels, norm)
    )


def ffconv_block_unet(in_channels, out_channels, stride=1, groups=1, enable_lfu=False, norm='batchnorm'):
    return nn.Sequential(
        FFC(in_channels, out_channels, stride, groups, enable_lfu),
        normalize(out_channels, norm),
        nn.LeakyReLU(inplace=True),
        FFC(out_channels, out_channels, stride, groups, enable_lfu),
        normalize(out_channels, norm),
        nn.LeakyReLU(inplace=True),
    )


def ffconv_preactivation_relu(in_channels, out_channels, stride=1, groups=1, enable_lfu=False, norm='batchnorm'):
    return nn.Sequential(
        nn.ReLU(inplace=False),
        FFC(in_channels, out_channels, stride, groups, enable_lfu),
        normalize(out_channels, norm)
    )


class LastffConv(nn.Module):
    def __init__(self, ndf, norm):
        super(LastffConv, self).__init__()
        """
        Args:
            ndf: constant number from channels
        """
        self.ndf = ndf
        self.norm = norm
        self.ffconv = ffconv_preactivation_relu(self.ndf, self.ndf * 2, 1, 1, False, self.norm)

    def forward(self, x):
        out = self.ffconv(x)

        return out
