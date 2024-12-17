# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Shengjiang Kong,School of mathematics and statistics, Xidian University.
Email: sjkongxd@gmail.com.
Date: 2020/12/9 17:47
"""

import numpy as np
import torch
import torch.nn as nn
import models.commonblocks as B
"""
# ----------------------------------------------------------------------------
# Following code come from https://github.com/donggong1/learn-optimizer-rgdn
# ----------------------------------------------------------------------------
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        n_feature_2 = 64
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, n_feature_2, 5, 1, 2, bias=False),
            # nn.BatchNorm2d(n_feature_2),
            nn.ReLU(True),
            nn.Conv2d(n_feature_2, n_feature_2, 5, 1, 2, bias=False),
            # nn.BatchNorm2d(n_feature_2),
            nn.ReLU(True),
            nn.Conv2d(n_feature_2, n_feature_2, 5, 1, 2, bias=False),
            # nn.BatchNorm2d(n_feature_2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_feature_2, n_feature_2, 5, 1, 2, bias=False),
            # nn.BatchNorm2d(n_feature_2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_feature_2, n_feature_2, 5, 1, 2, bias=False),
            # nn.BatchNorm2d(n_feature_2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_feature_2, n_feature_2, 5, 1, 2, bias=False),
            # nn.BatchNorm2d(n_feature_2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_feature_2, in_channels, 5, 1, 2, bias=False),
        )

    def forward(self, input_):
        output = self.main(input_)
        return output


"""
# ----------------------------------------------------------------------------
# Following code come from https://github.com/cszn/DPIR
# ----------------------------------------------------------------------------
"""


class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R'):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')
        # downsample
        downsample_block = B.downsample_strideconv
        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        upsample_block = B.upsample_convtranspose

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
        h, w = x0.size()[-2:]
        padding_bot = int(np.ceil(h / 8) * 8 - h)
        padding_right = int(np.ceil(w / 8) * 8 - w)
        x0 = nn.ReplicationPad2d((0, padding_right, 0, padding_bot))(x0)
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)
        x = x[..., :h, :w]
        return x


class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


if __name__ == '__main__':

    device = torch.device('cuda')
    x = torch.randn(1, 3, 128, 128).cuda()
    net = UNetRes(in_nc=3, out_nc=3)
    net.to(device)
    out = net(x)
    print(out.shape)
