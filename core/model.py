"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os



from core.spectral_normalization import SpectralNorm
from core.utils import sobel

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pg_modules.discriminator import ProjectedDiscriminator
from torch.nn.utils import spectral_norm

from core.wing import FAN

class DenseBlock(nn.Module):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, dim_in,drop_rate):
        super(DenseBlock, self).__init__()
        self.dim_in  = dim_in
        self.drop_rate = drop_rate
        self.norm =nn.InstanceNorm2d(dim_in,affine=True)
        self.norm2 =nn.InstanceNorm2d(int(dim_in+dim_in/2),affine=True)
        self.relu =nn.LeakyReLU(0.2)
        self.conv1 =nn.Conv2d(dim_in, dim_in,kernel_size=1, stride=1, bias=False)
        self.conv2 =nn.Conv2d(dim_in, int(dim_in/2),kernel_size=3, stride=1, padding=1,bias=False)
        self.conv3 = nn.Conv2d(int(dim_in+dim_in/2), int(dim_in+dim_in/2), kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(int(dim_in+dim_in/2),int(dim_in/2), kernel_size=3, stride=1, padding=1, bias=False)
    def _denselayer1(self,x):
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out
    def _denselayer2(self,x):
        out = self.norm2(x)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv4(out)
        return out
    def forward(self, x):
        # print(x.size())
        new_features = self._denselayer1(x)
        # print(new_features.size())
        new_features  = torch.cat([x, new_features], 1)
        # print(new_features.size())
        new_features2 = self._denselayer2(new_features)
        new_features2 = torch.cat([new_features, new_features2], 1)
        # print(new_features2.size())
        if self.drop_rate > 0:
            new_features2 = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return  new_features2

class Transition(nn.Module):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, downsample=False,upsample=False):
        super(Transition, self).__init__()
        self.downsample = downsample
        self.upsample = upsample
    def forward(self, x):
        out = x
        if self.downsample:
            self.pool = nn.AvgPool2d(2, stride=2)
            out = self.pool(x)
        if self.upsample:
            out = F.interpolate(x, scale_factor=2, mode='nearest')
        return out

# class Generator(nn.Module):
#     def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
#         super().__init__()
#         dim_in = 2**14 // img_size
#         self.img_size = img_size
#         self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
#         self.encode = nn.ModuleList()
#         self.decode = nn.ModuleList()
#         self.to_rgb = nn.Sequential(
#             # SwitchNormalization(dim_in),
#             nn.InstanceNorm2d(dim_in, affine=True),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(dim_in, 3, 1, 1, 0))


#         # down/up-sampling blocks
#         self.skip_connections = [None, None, None]  # 仅保存前三层
#         repeat_num = int(np.log2(img_size)) - 4
#         if w_hpf > 0:
#             repeat_num += 1
#         for _ in range(repeat_num):
#             dim_out = min(dim_in*2, max_conv_dim)
#             if dim_in != max_conv_dim:
#                 self.encode.append(DenseBlock(dim_in, 0))
#                 self.encode.append(Transition(downsample=True))
#             self.decode.insert(
#                 0, AdainResBlk(dim_out, dim_in, style_dim,
#                                w_hpf=w_hpf, upsample=True))
#             dim_in = dim_out
#         # bottleneck blocks
#         self.encode.append(Transition(downsample=True))
#         for _ in range(2):
#             self.encode.append(
#                 ResBlk(dim_out, dim_out, normalize=True))
#             self.decode.insert(
#                 0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))


#         if w_hpf > 0:
#             device = torch.device(
#                 'cuda' if torch.cuda.is_available() else 'cpu')
#             self.hpf = HighPass(w_hpf, device)

#     def forward(self, x, s, masks=None):
#         x = self.from_rgb(x)
#         for i, block in enumerate(self.encode):
#             x = block(x)
#             if i ==1 or i==3 or i==5:
#                 self.skip_connections.append(x)
#         self.skip_connections.reverse()
#         # print(x.size())
#         for i, block in enumerate(self.decode):
#             if  i >= 3:
#                 x = x + self.skip_connections[i-3]
#             x = block(x, s)
#         return self.to_rgb(x)


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, feature=False):
        super().__init__()
        #
        self.feature = feature
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            # self.norm1 = SwitchNormalization(3)
            # self.norm2 = SwitchNormalization(3)
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        if self.feature:
            return x
        return x / math.sqrt(2)  # unit variance

class SEResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, feature=False):
        super().__init__()
        #
        self.feature = feature
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            # self.norm1 = SwitchNormalization(3)
            # self.norm2 = SwitchNormalization(3)
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

        if dim_out == 64:
            self.globalAvgPool = nn.AvgPool2d(256, stride=1)
        elif dim_out == 128:
            self.globalAvgPool = nn.AvgPool2d(128, stride=1)
        elif dim_out == 256:
            self.globalAvgPool = nn.AvgPool2d(64, stride=1)
        elif dim_out == 512:
            self.globalAvgPool = nn.AvgPool2d(16, stride=1)
        self.fc1 = nn.Linear(in_features=dim_out, out_features=round(dim_out / 16))
        self.fc2 = nn.Linear(in_features=round(dim_out / 16), out_features=dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)

        original_out = x
        x = self.globalAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = x * original_out

        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        if self.feature:
            return x
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        # self.norm = SwitchNormalization(num_features)
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta
    
# class AdainResBlk(nn.Module):
#     def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
#                  actv=nn.LeakyReLU(0.2), upsample=False):
#         super().__init__()
#         self.w_hpf = w_hpf
#         self.actv = actv
#         self.upsample = upsample
#         self.learned_sc = dim_in != dim_out
#         self._build_weights(dim_in, dim_out, style_dim)

#     def _build_weights(self, dim_in, dim_out, style_dim=64):
#         self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
#         self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
#         self.norm1 = AdaIN(style_dim, dim_in)
#         self.norm2 = AdaIN(style_dim, dim_out)
#         if self.upsample:
#             self.upsample_conv1 = nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1)
#             self.upsample_conv2 = nn.ConvTranspose2d(dim_out, dim_out, 4, 2, 1)
#         if self.learned_sc:
#             self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

#     def _shortcut(self, x):
#         if self.upsample:
#             x = self.upsample_conv1(x)
#         if self.learned_sc:
#             x = self.conv1x1(x)
#         return x

#     def _residual(self, x, s):
#         x = self.norm1(x, s)
#         x = self.actv(x)
#         if self.upsample:
#             x = self.upsample_conv1(x)
#         x = self.conv1(x)
#         x = self.norm2(x, s)
#         x = self.actv(x)
#         x = self.conv2(x)
#         return x

#     def forward(self, x, s):
#         out = self._residual(x, s)
#         if self.w_hpf == 0:
#             out = (out + self._shortcut(x)) / math.sqrt(2)
#         return out

class SEAdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

        if dim_out == 64:
            self.globalAvgPool = nn.AvgPool2d(256, stride=1)
        elif dim_out == 128:
            self.globalAvgPool = nn.AvgPool2d(128, stride=1)
        elif dim_out == 256:
            self.globalAvgPool = nn.AvgPool2d(64, stride=1)
        elif dim_out == 512:
            self.globalAvgPool = nn.AvgPool2d(16, stride=1)
        self.fc1 = nn.Linear(in_features=dim_out, out_features=round(dim_out / 16))
        self.fc2 = nn.Linear(in_features=round(dim_out / 16), out_features=dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)

        original_out = x
        x = self.globalAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = x * original_out

        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
    
# def conv2d(*args, **kwargs):
#     return spectral_norm(nn.Conv2d(*args, **kwargs))

# class Swish(nn.Module):
#     def forward(self, feat):
#         return feat * torch.sigmoid(feat)
# #  SEB d4   
# class SEBlock(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super().__init__()
#         self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
#                                     conv2d(ch_in, ch_out, 4, 4, 0, bias=False), Swish(),
#                                     conv2d(ch_out, ch_out, 3, 1, 1, bias=False), nn.Sigmoid() )

#     def forward(self, feat_small, feat_big ):
#         # print("small",feat_small.shape)
#         # print("big",feat_big.shape)
#         # print("after_big",self.main(feat_small).shape)
#         return feat_big * self.main(feat_small)
    
# class SEBlock(nn.Module):
#     def __init__(self, ch_in, ch_out, style_dim=64):
#         super(SEBlock, self).__init__()
#         self.norm1 = AdaIN(style_dim, ch_in)
#         self.norm2 = AdaIN(style_dim, ch_out)
#         self.pool = nn.AdaptiveAvgPool2d(4)
#         self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=4, padding=0, bias=False)
#         self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
#         self.act1 = nn.LeakyReLU(0.2)
#         self.act2 = Swish()
#         self.act3 = nn.LeakyReLU(0.2)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, feat_small, feat_big, s):
#         # print(feat_small.shape)
#         x = self.norm1(feat_small, s)
#         # print(x.shape)
#         x = self.act1(x)
#         x = self.pool(x)
#         x = self.conv1(x)
#         x = self.act2(x)
#         # x = self.norm2(x, s)
#         # x = self.act3(x)
#         x = self.conv2(x)
#         x = self.sigmoid(x)
#         return feat_big * x
    
    
class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

# class Generator(nn.Module):
#     def __init__(self, img_size=256, style_dim=64, max_conv_dim=256, w_hpf=0):
#         super().__init__()
#         dim_in = 2**14 // img_size
#         self.img_size = img_size
#         self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
#         self.encode = nn.ModuleList()
#         self.decode = nn.ModuleList()
#         self.to_rgb = nn.Sequential(
#             # SwitchNormalization(dim_in),
#             nn.InstanceNorm2d(dim_in, affine=True),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(dim_in, 3, 1, 1, 0))


#         # down/up-sampling blocks
#         repeat_num = int(np.log2(img_size)) - 6

#         for _ in range(repeat_num):
#             dim_out = min(dim_in*2, max_conv_dim)
#             self.encode.append(
#                 ResBlk(dim_in, dim_out, normalize=True, downsample=True))
#             self.decode.insert(
#                 0, AdainResBlk(dim_out, dim_in, style_dim,
#                                w_hpf=w_hpf, upsample=True))  # stack-like
#             dim_in = dim_out
#         # bottleneck blocks
#         for _ in range(2):
#             self.encode.append(
#                 ResBlk(dim_out, dim_out, normalize=True))
#             self.decode.insert(
#                 0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))


#     def forward(self, x, s, masks=None):
#         x = self.from_rgb(x)
#         s = s.unsqueeze(1).to(x.dtype)
#         for block in self.encode:
#             x = block(x)
#         for i,block in enumerate(self.decode):
#             x = block(x, s)
#             print(i,x.shape)
#         return self.to_rgb(x)


class InteractionModule(nn.Module):
    def __init__(self, dim_out, style_dim):
        super(InteractionModule, self).__init__()
        self.conv1 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        self.adain1 = AdaIN(style_dim, dim_out)
        # self.in1 = nn.InstanceNorm2d(dim_out)
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        self.adain2 = AdaIN(style_dim, dim_out)
        # self.in2 = nn.InstanceNorm2d(dim_out)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x, s):
        x = self.conv1(x)
        x = self.adain1(x,s)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.adain2(x,s)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        return x

class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=256, w_hpf=0):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.redecode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            # SwitchNormalization(dim_in),
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))
        # self.to_rgbre = nn.Sequential(
        #     # SwitchNormalization(dim_in),
        #     nn.InstanceNorm2d(dim_in, affine=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(dim_in, 3, 1, 1, 0))


        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 6

        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))
            # self.redecode.insert(
            #     0, AdainResBlk(dim_out, dim_in, style_dim,
            #                    w_hpf=w_hpf, upsample=True))    
            dim_in = dim_out
        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))
            self.redecode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        self.interaction_module = InteractionModule(dim_out, style_dim)
        self.final_conv = nn.Conv2d(dim_out*2, dim_out, 3, 1, 1)

    def forward(self, x, s,l, masks=None):
        x = self.from_rgb(x)
        s = s.unsqueeze(1).to(x.dtype)
        l = l.unsqueeze(1).to(x.dtype)

        for block in self.encode:
            x = block(x)

        xs_layers = []
        xl_layers = []

        xl = x.clone()
        xs = x.clone()

        for i, block in enumerate(self.redecode):
        
            if i<2:
                xl_layers.append(xl)

            xl = block(xl, l)

        for i, block in enumerate(self.decode):
            if i <2:  # 记录前1, 2层的输出

                input_xs = xs - xl_layers[i]
                input_xs = self.interaction_module(input_xs,s)
                xs = torch.cat([input_xs,xs], dim=1)
                xs =self.final_conv(xs)
            xs = block(xs, s)
            xl = block(xl, l)

        return self.to_rgb(xs), self.to_rgb(xl)



# class Generator(nn.Module):
#     def __init__(self, img_size=256, style_dim=64, max_conv_dim=256, w_hpf=0):
#         super().__init__()
#         dim_in = 2**14 // img_size
#         self.img_size = img_size
#         self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
#         self.encode = nn.ModuleList()
#         self.decode = nn.ModuleList()
#         self.to_rgb = nn.Sequential(
#             # SwitchNormalization(dim_in),
#             nn.InstanceNorm2d(dim_in, affine=True),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(dim_in, 3, 1, 1, 0))


#         # down/up-sampling blocks
#         repeat_num = int(np.log2(img_size)) - 6

#         for _ in range(repeat_num):
#             dim_out = min(dim_in*2, max_conv_dim)
#             self.encode.append(
#                 ResBlk(dim_in, dim_out, normalize=True, downsample=True))
#             self.decode.insert(
#                 0, AdainResBlk(dim_out, dim_in, style_dim,
#                                w_hpf=w_hpf, upsample=True))
#             dim_in = dim_out

#         # bottleneck blocks
#         for _ in range(2):
#             self.encode.append(
#                 ResBlk(dim_out, dim_out, normalize=True))
#             self.decode.insert(
#                 0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))



#     def forward(self, x, s, masks=None):
#         x = self.from_rgb(x)
#         s = s.unsqueeze(1).to(x.dtype)
#         for block in self.encode:
#             x = block(x)
#         for block in self.decode:
#             x = block(x, s)
#         return self.to_rgb(x)


##PG
# Discriminator = ProjectedDiscriminator()
# Discriminator.feature_network.requires_grad_(False)

class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            # blocks += [ResBlk(dim_in, dim_out, downsample=True,sn=True)]
            blocks += [FirstResBlockDiscriminator(dim_in, dim_out, stride=2)]
            # blocks += [ResBlockDiscriminator(dim_out, dim_out, stride=2)]
            # blocks += [ResBlockDiscriminator(dim_out, dim_out)]
            # blocks += [ResBlockDiscriminator(dim_out, dim_out)]
            # blocks += [nn.ReLU()]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        # print('d_shape:',out.shape)
        # print('d_out:',out)
        return out
    

def build_model(args):
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf))
    # mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains))
    # style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains))
    # discriminator = nn.DataParallel(Discriminator())
    generator_ema = copy.deepcopy(generator)
    # Sidiscriminator = nn.DataParallel(SiameseDiscriminator(args.img_size))
    # mapping_network_ema = copy.deepcopy(mapping_network)
    # style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 # mapping_network=mapping_network,
                #  style_encoder=style_encoder,
                 discriminator=discriminator)
                #  Sidiscriminator=Sidiscriminator
    nets_ema = Munch(generator=generator_ema)
                     # mapping_network=mapping_network_ema,
                    #  style_encoder=style_encoder_ema

    if args.w_hpf > 0:
        fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
        fan.get_heatmap = fan.module.get_heatmap
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema

