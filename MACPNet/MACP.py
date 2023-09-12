#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/27 23:54
# @Author  : Denxun
# @FileName: MACP.py
# @Software: PyCharm
import timm
from PIL import Image
from timm.models.resnet import resnet34, resnet50
from torch.nn import init
from torchvision.models import Swin_V2_B_Weights, ConvNeXt_Base_Weights, Swin_V2_S_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import numpy as np
from skimage.util import img_as_ubyte
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms
import tqdm
import random
from torchvision import models
#from torchviz import make_dot

# import skimage
seed =123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# data = torch.ones(2, 1, 512, 512)
#from torchcontrib.optim import SWA
import torchvision

class decodeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decodeConv, self).__init__()
        self.quadruple_conv1 = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d( out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, 1 * out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(1* out_channels),
        )

    def forward(self, x):
        x = x
        x1 = self.quadruple_conv1(x)
        return x1

# class maxvit(nn.Module):
#     def __init__(self,imgsize):
#         super().__init__()
#         model = timm.models.create_model('maxvit_large_tf_512', pretrained=True)
#         stem = model.stem
#         stages1 = model.stages[0].blocks
#         stages2 = model.stages[1].blocks
#         stages3 = model.stages[2].blocks
#         stages4 = model.stages[3].blocks
#         maxbone = nn.Sequential()
#         maxbone.add_module('0', stem)
#         maxbone.add_module('1', stages1[0:2])
#         maxbone.add_module('2', stages2[0:2])
#         maxbone.add_module('3', stages3[0:2])
#         maxbone.add_module('4', stages4[0:2])
#         if imgsize==512:
#             self.dowm = maxbone[0]
#         elif imgsize==256:
#             self.dowm=maxbone[1]
#         elif imgsize==128:
#             self.dowm=maxbone[2]
#         elif imgsize==64:
#             self.dowm = maxbone[3]
#         elif imgsize==32:
#             self.dowm = maxbone[4]
#     def forward(self,x):
#         return self.dowm(x)
import torch
from torch import nn

class PatchMerge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.unfold = nn.Unfold(kernel_size=2, stride=2)
        self.fold = nn.Fold(output_size=(1, 1), kernel_size=(2, 2))

        self.proj = nn.Conv2d(4 * in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.size()
        assert h % 2 == 0 and w % 2 == 0, "Height and Width of input tensor must be even."

        x = self.unfold(x)
        x = x.view(b, self.in_channels, 4, h // 2, w // 2)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = x.view(b, 4 * self.in_channels, h // 2, w // 2)
        x = self.proj(x)

        return x
class maxvit(nn.Module):
    def __init__(self,imgsize):
        super().__init__()
        model = timm.models.create_model('maxvit_large_tf_512', pretrained=True)
        #print(model)
        stem = model.stem
        stages1 = model.stages[0].blocks
        stages1[0].conv=nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
                                      nn.BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))
        stages1[1].conv =nn.Identity()
        stages2 = model.stages[1].blocks
        stages2[0].conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1),padding="same"),
                                        nn.BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True,
                                                       track_running_stats=True),nn.GELU(),
                                        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),)
        stages2[1].conv = nn.Identity()
        stages3 = model.stages[2].blocks
        stages3[0].conv = nn.Sequential(
                                        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1),padding="same"),
                                        nn.BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True,
                                                       track_running_stats=True), nn.GELU(),
                                        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),)
        stages3[1].conv = nn.Identity()
        stages4 = model.stages[3].blocks
        stages4[0].conv = nn.Sequential(
                                        nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1),padding="same"),
                                        nn.BatchNorm2d(1024, eps=0.001, momentum=0.1, affine=True,
                                                       track_running_stats=True),
                                        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)), nn.GELU(),)
        stages4[1].conv = nn.Identity()
        maxbone = nn.Sequential()
        maxbone.add_module('0', stem)
        maxbone.add_module('1', stages1[0:1])
        maxbone.add_module('2', stages2[0:1])
        maxbone.add_module('3', stages3[0:1])
        maxbone.add_module('4', stages4[0:1])
        if imgsize==512:
            self.dowm = maxbone[0]
        elif imgsize==256:
            self.dowm=maxbone[1]
        elif imgsize==128:
            self.dowm=maxbone[2]
        elif imgsize==64:
            self.dowm = maxbone[3]
        elif imgsize==32:
            self.dowm = maxbone[4]
    def forward(self,x):
        return self.dowm(x)
class MACP_block(nn.Module):
    def __init__(self,inc,outc,imgsize):
        super().__init__()
        self.max_vit_block=maxvit(imgsize)
        self.conv1=nn.Sequential(nn.Conv2d(2*inc,outc,kernel_size=(1,1),stride=(1,1),padding="same"),nn.BatchNorm2d(outc))
        self.conv3=nn.Sequential(nn.Conv2d(inc,outc,kernel_size=(3,3),stride=(1,1),padding="same"),nn.BatchNorm2d(outc))
        self.conv7 = nn.Sequential(nn.Conv2d(inc, inc, kernel_size=(1,1),stride=(1,1), padding="same"),nn.BatchNorm2d(inc))
        self.conv = nn.Sequential(nn.Conv2d(2*outc, outc, kernel_size=(1,1),stride=(1,1), padding="same"),nn.BatchNorm2d(outc))
        self.LN=nn.LayerNorm(outc)
        self.act=nn.GELU()
        self.maxpool=nn.MaxPool2d(2)
    def forward(self,x):
        x_max=self.max_vit_block(x)
        xconv7 = self.conv7(x)
        xconv3=self.conv3(xconv7)
        #xconv_3_7=self.conv1(torch.cat([xconv3,xconv7],dim=1))
        xconv_3_1=self.maxpool(xconv3)
        xcombine=self.conv(torch.cat([xconv_3_1,x_max],dim=1))
        #xcombine=self.LN(xcombine.permute(0,2,3,1))
        out=self.act(xcombine)
        return x_max+out*x_max
class up_feature(nn.Module):
    def __init__(self,inc,outc,scale,channel_last=True,out_channel_last=True):
        super(up_feature,self).__init__()
        self.channel_last=channel_last
        self.outchannel_last=out_channel_last
        self.quadruple_conv1 = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=1))
        self.layer_norm = nn.LayerNorm(outc)
        self.act = nn.GELU()
        #self.upsam = nn.ConvTranspose2d(inc,outc,kernel_size=scale,stride=scale)
        self.upsam =nn.Sequential(nn.Upsample(scale_factor=scale,mode="bilinear"),nn.Conv2d(inc,outc,kernel_size=1,stride=1),nn.GELU()) #nn.ConvTranspose2d(inc, outc, kernel_size=scale, stride=scale)
    def forward(self,x):
        if self.channel_last==True:
            x = x.permute(0, 3, 1, 2)
        else:pass
        x = self.upsam(x)
        # x=self.quadruple_conv1(x).permute(0,2,3,1)
        # x=self.layer_norm(x).permute(0,3,1,2)
        # x=self.act(x)
        if self.outchannel_last==True:
            x=x.permute(0,2,3,1)
        else:pass
        return x


class MVCnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=3,stride=1,kernel_size=3,padding="same"),nn.BatchNorm2d(3))
        self.down256=maxvit(imgsize=512)
        self.MVC256=MACP_block(inc=128, outc=128, imgsize=256)
        self.MVC128 = MACP_block(inc=128, outc=256, imgsize=128)
        self.MVC64 = MACP_block(inc=256, outc=512, imgsize=64)
        self.MVC32 = MACP_block(inc=512, outc=1024, imgsize=32)
        self.up_feature_16_32 = up_feature(1024, 512, 2, channel_last=False, out_channel_last=False)
        self.up_feature_32_64 = up_feature(512, 256, 2,channel_last=False,out_channel_last=False)
        self.decode32=decodeConv(512*2,512)#512,32,32->512,32,32
        #self.up_feature_32_512 = up_feature(512, 64, 16,channel_last=False,out_channel_last=False)  # 512,32,32->128,512,512
        self.up_feature_64_128 = up_feature(256, 128, 2,channel_last=False,out_channel_last=False)
        self.decode64 = decodeConv(256*2, 256)
        #self.up_feature_64_512 = up_feature(256, 64,8,channel_last=False,out_channel_last=False)  # 256,64,64->256,64,64
        self.decode128 = decodeConv(128*2, 128)
        self.decode256 = decodeConv(128*2, 128)
        self.up_feature_128_256 = up_feature(128, 128,2,channel_last=False,out_channel_last=False)
        self.up_feature_256_512 = up_feature(128, 64, 2, channel_last=False, out_channel_last=False)
        #self.up_feature_256_512_1 = nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear"),nn.Conv2d(96,1,kernel_size=1,stride=1),nn.GELU())
        self.out=nn.Sequential(nn.Conv2d(64,64,1,1),nn.BatchNorm2d(64),
                               nn.GELU(),nn.Conv2d(64,1,1))

    def forward(self,x):
        x=self.rgb(x)
        xdown256=self.down256(x)
        xdown128=self.MVC256(xdown256)
        xdown64 = self.MVC128(xdown128)
        xdown32 = self.MVC64(xdown64)
        xdown16=self.MVC32(xdown32)
        x_16up32=self.up_feature_16_32(xdown16)+xdown32
        x_16up32=torch.cat([xdown32,x_16up32],dim=1)
        x32 = self.decode32(x_16up32)
        x_32up64=self.up_feature_32_64(x32)+xdown64
        x_32up64=torch.cat([xdown64,x_32up64],dim=1)
        x64 = self.decode64(x_32up64)
        x_64up128=self.up_feature_64_128(x64)+xdown128
        x_64up128=torch.cat([xdown128,x_64up128],dim=1)
        x128=self.decode128(x_64up128)
        x_128up256=self.up_feature_128_256(x128)+xdown256
        x_128up256=torch.cat([xdown256,x_128up256],dim=1)
        x256 = self.decode256(x_128up256)
        x_512=self.up_feature_256_512(x256)
        # x_256_512=self.up_feature_256_512(x256)
        out = self.out(x_512)
        return out

# if __name__=="__main__":
#     net=MVCnet()
#     input_tensor = torch.randn(1, 1, 512, 512)  # Replace with appropriate input size
#     output = net(input_tensor)
#     onnx_path = "MVCnet.onnx"
#     torch.onnx.export(net, input_tensor, onnx_path, verbose=True)