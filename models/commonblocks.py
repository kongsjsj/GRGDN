# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Shengjiang Kong,School of mathematics and statistics, Xidian University.
Email: sjkongxd@gmail.com.
Date: 2020/12/9 17:45
Define some useful basic block.
some codes from https://github.com/cszn/KAIR and https://gitlab.mpi-klsb.mpg.de/jdong/dwdn
"""

import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as f
from pytorch_wavelets import DWT, IDWT


# -----------------------------------------------------
# useful functions
# -----------------------------------------------------

def sequential(*args):
    """
    # ===================================
    # Advanced nn.Sequential
    # reform nn.Sequentials and nn.Modules
    # to a single nn.Sequential
    # https://github.com/cszn/KAIR
    # ===================================
    :param args: tuple
    :return: nn.Sequentials object
    """

    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR'):
    """
    :param in_channels: int
    :param out_channels: int
    :param kernel_size: (int, int) or int
    :param stride: int
    :param padding: int
    :param bias: bool True|False
    :param mode:
    :return: Conv + BN + ReLU  nn.Sequential object
    """
    conv_blk = list()
    for t in mode:
        if t == 'C':
            conv_blk.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
            )
        elif t == 'T':
            conv_blk.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
            )
        elif t == 'B':
            conv_blk.append(
                nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True)
            )
        elif t == 'R':
            conv_blk.append(nn.ReLU(inplace=True))
        elif t == 'r':
            conv_blk.append(nn.ReLU(inplace=False))
        elif t == '2':
            conv_blk.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            conv_blk.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            conv_blk.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            conv_blk.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            conv_blk.append(nn.Upsample(scale_factor=3, mode='nearest'))
        else:
            raise NotImplementedError('Undefined type: '.format(t))

    return sequential(*conv_blk)


def splits(a, sf):
    """split a into sf x sf distinct blocks
    Args:
        a: NxCxWxHx2
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x2x(sf^2)
    """
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b


def c2c(x):
    return torch.from_numpy(np.stack([np.float32(x.real), np.float32(x.imag)], axis=-1))


def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)


def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c ** 2 + d ** 2
    return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)


def crdiv(x, y):
    # complex/real division
    a, b = x[..., 0], x[..., 1]
    return torch.stack([a / y, b / y], -1)


def csum(x, y):
    # complex + real
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
    # modulus of a complex number
    return torch.pow(x[..., 0] ** 2 + x[..., 1] ** 2, 0.5)


def cabs2(x):
    return x[..., 0] ** 2 + x[..., 1] ** 2


def cmul(t1, t2):
    """complex multiplication
    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2, complex tensor
    Returns:
        output: NxCxHxWx2
    """
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def crmul(x, y):
    """
    Add by Shengjiang Kong, sjkongxd@gmail.com.
    Date: Mar 10, 2022.
    real * complex multiplication
    :param x: NxCxHxWx2, complex tensor
    :param y: NxCxHxW   real tensor
    :return: NxCxHxWx2 complex tensor
    """
    real, imag = x[..., 0], x[..., 1]
    return torch.stack([real * y, imag * y], dim=-1)


def cconj(t, inplace=False):
    """complex's conjugation
    Args:
        t: NxCxHxWx2
        inplace:
    Returns:
        output: NxCxHxWx2
    """
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def rfft(t):
    # Real-to-complex Discrete Fourier Transform
    return torch.rfft(t, 2, onesided=False)


def irfft(t):
    # Complex-to-real Inverse Discrete Fourier Transform
    return torch.irfft(t, 2, onesided=False)


def fft(t):
    # Complex-to-complex Discrete Fourier Transform
    return torch.fft(t, 2)


def ifft(t):
    # Complex-to-complex Inverse Discrete Fourier Transform
    return torch.ifft(t, 2)


def p2o(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: N x C x h x w
        shape: [H, W]
    Returns:
        otf: N x C x H x W x 2
    """
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops * 2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def convert_psf2otf(ker, size):
    psf = torch.zeros(size).cuda()
    # circularly shift
    # Modified by Shengjiang Kong for processing rectangle kernel.
    # -----------------------------------------------------------------------------
    centre0 = ker.shape[2:][0] // 2 + 1
    centre1 = ker.shape[2:][1] // 2 + 1
    psf[:, :, :centre0, :centre1] = ker[:, :, (centre0-1):, (centre1-1):]
    psf[:, :, :centre0, -(centre1-1):] = ker[:, :, (centre0-1):, :(centre1-1)]
    psf[:, :, -(centre0-1):, :centre1] = ker[:, :, : (centre0-1), (centre1-1):]
    psf[:, :, -(centre0-1):, -(centre1-1):] = ker[:, :, :(centre0-1), :(centre1-1)]
    # ------------------------------------------------------------------------------
    # compute the otf
    otf = torch.rfft(psf, 3, onesided=False)
    return otf


def upsample(x, sf=3):
    """s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    """
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    """s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    """
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


# -------------------------------------------------------
# strideconv + relu
# -------------------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R'):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return down1


# -------------------------------------------------------
# convTranspose + relu
# -------------------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R'):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return up1


# -----------------------------------------------------
# useful classes
# -----------------------------------------------------


class Conv3x3(nn.Module):
    """
    (3 x 3)Conv+ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, dilation=1):
        """
        Constructor
        :param in_channels: int, the number of input channels.
        :param out_channels: int, the number of output channels.
        :param padding: int, the size of input padding
        :return the object of Conv3x3.
        """
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation
            ),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv3x3(x)


class Conv1x1(nn.Module):
    """
    (1 x 1)Conv+ReLU
    """

    def __init__(self, in_channels, out_channels):
        """
        Constructor
        :param in_channels: int, the number of input channels.
        :param out_channels: int, the number of output channels.
        :return the object of Conv1x1.
        """
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv1x1(x)


class ResConvBlock(nn.Module):
    """
    Implement residual conv block.
    """

    def __init__(self, in_channels, out_channels, bias=True):
        """
        :param in_channels:  the size of input channel
        :param out_channels:  the size of output channel
        """
        super().__init__()
        self.conv = conv(in_channels=in_channels, out_channels=out_channels, bias=bias, mode='CRC')

    def forward(self, x):
        return f.relu(self.conv(x) + x)


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)

    def forward(self, x):
        res = self.res(x)
        return x + res


class ConvReLUConv(nn.Module):
    """
    (3 x 3)Conv+ReLU + (3 x 3)Conv
    """

    def __init__(self, in_channels, out_channels):
        """
        Constructor
        :param in_channels: int, the number of input channels.
        :param out_channels: int, the number of output channels.
        :return the object of ConvReLUConv.
        """
        super().__init__()
        self.conv_blk = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        return self.conv_blk(x)


class ResUnet(nn.Module):
    """
    Implement res-U-Net
    """

    def __init__(self, in_nc=64, out_nc=64, nc=None, nb=2, act_mode='R'):
        if nc is None:
            nc = [64, 128, 256, 512]

        super().__init__()
        self.m_head = conv(in_nc, nc[0], bias=False, mode='C')
        # down-sample
        downsample_block = downsample_strideconv

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body = sequential(
            *[ResBlock(nc[3], nc[3], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])

        # upsample
        upsample_block = upsample_convtranspose
        self.m_up3 = sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'),
                                *[ResBlock(nc[2], nc[2], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'),
                                *[ResBlock(nc[1], nc[1], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'),
                                *[ResBlock(nc[0], nc[0], bias=False, mode='C' + act_mode + 'C') for _ in range(nb)])
        self.m_tail = conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        return x


"""
Following codes from https://gitlab.mpi-klsb.mpg.de/jdong/dwdn
"""


class Conv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=1, padding=0, bias=True, bn=False, act=False):
        super(Conv, self).__init__()
        m = list()
        m.append(nn.Conv2d(input_channels, n_feats, kernel_size, stride, padding, bias=bias))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        if act:
            m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class Deconv(nn.Module):
    def __init__(self, input_channels, n_feats, kernel_size, stride=2, padding=0, output_padding=0 , bias=True, act=False):
        super(Deconv, self).__init__()
        m = []
        m.append(nn.ConvTranspose2d(input_channels, n_feats, kernel_size, stride=stride, padding=padding,output_padding=output_padding, bias=bias))
        if act: m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)

    def forward(self, inputs):
        return self.body(inputs)


class ResidualBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, padding=0, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResidualBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, padding=padding, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


def main():
    # c = conv(mode='CBR'*2)
    # print(c)
    # conv1, conv2_4 = conv_4_blk()
    # print(conv1)
    # print(conv2_4)
    pass


if __name__ == '__main__':
    main()
