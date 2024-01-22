# coding=utf-8
# @FileName:FSIM.py
# @Time:2023/12/2
# @Author: gyy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FreBlock(nn.Module):
    def __init__(self):
        super(FreBlock, self).__init__()

    def forward(self, x):
        x = x + 1e-8
        mag = torch.abs(x)
        pha = torch.angle(x)

        return mag, pha

class FSIM(nn.Module):
    def __init__(self, n_feats=64):
        super().__init__()
        # i_feats =n_feats*2
        self.projection=nn.Conv2d(1920,n_feats,1,1,0)
        self.Conv1 = nn.Sequential(
            nn.Conv2d(n_feats, 2 * n_feats, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0))
        self.Conv1_1 = nn.Sequential(
            nn.Conv2d(n_feats, 2 * n_feats, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0))

        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.FF = FreBlock()
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        self.refine = nn.Conv2d(64, 1920, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, H, W = x.shape
        a = 0.35
        #fft
        mix = self.projection(x)
        mix_mag, mix_pha = self.FF(mix)
        #卷积偏置
        mix_mag = self.Conv1(mix_mag)
        mix_pha = self.Conv1_1(mix_pha)
        #放复数域 pha_only
        constant = mix_mag.mean()  #振幅设成常量
        pha_only = constant * np.e ** (1j * mix_pha)
        # irfft
        x_out_main = torch.abs(torch.fft.irfft2(pha_only, s=(H, W), norm='backward')) + 1e-8
        out1=self.Conv2(a * x_out_main + (1 - a) * mix)
        out2=self.refine(out1)
        return out2



