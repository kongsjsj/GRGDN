# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Shengjiang Kong,School of mathematics and statistics, Xidian University.
Email: sjkongxd@gmail.com.
Date: 2021/09/23 17:48.
This Module implement ours network.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from .basicblocks import ConvBlock, UNetRes, HyPaNet
from utils.deblurtools import model_init
# compute the gradient of data fitting term
# Modified GradDataFitting


class GradDataFitting(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, k):
        """
        InPUTS:
               # k kernel N x 1 x h x w, tensor.float32
               # y Observed input N x C x H x W, tensor.float32
               # x is initialize of y, N x C x H x W, tensor.float32
        """
        # k kernel
        # kt kernel's transpose
        # Add comments in 2021/9/8
        n_size = x.size()[0]
        k_size = k.size()[2]
        padding = int(k_size / 2)
        x1 = x.transpose(1, 0)  # x1: C x N x H x W
        y1 = y.transpose(1, 0)
        kx_y = F.conv2d(x1, k, padding=padding, groups=n_size)  # Ax
        kx_y.sub_(y1)  # Ax-y
        res = kx_y.transpose(1, 0)  # h3: N x C x H x W
        return res


class RGDN(nn.Module):
    """Gradient descent based optimizer model for uniform
    x^{k+1} = x^{k} - \alpha^{k}*D^{k}*(1/\lambda*A^T*(Ax^k-y)+\lambda*\nabla*R(x^k)).
    Cite: Learning Deep Gradient Descent Optimization for Image Deconvolution.
    """

    def __init__(self, num_steps=8, stop_epsilon=1e-6):
        super().__init__()
        #
        self.num_steps = num_steps
        self.stop_epsilon = stop_epsilon
        self.grad_data_fitting = GradDataFitting()
        self.r = UNetRes(in_nc=3, out_nc=3)  # R module
        self.h = ConvBlock()  # H module
        self.d_alpha = ConvBlock()  # \alpha_t*D^t module

    def forward(self, y, k, kt):
        n_size = y.size()[0]
        k_size = k.size()[2]
        padding = int(k_size / 2)
        # init x
        x_init = y
        output_list = []
        # K^T
        # optimization init
        error_prev = 0
        error_reltv = 0
        for i in range(self.num_steps):
            # single step operation
            grad_loss = self.grad_data_fitting(x_init, y, k)  # Ax - y
            grad_loss = grad_loss.transpose(1, 0)
            at_ax_y = F.conv2d(grad_loss, kt, padding=padding, groups=n_size)
            at_ax_y = at_ax_y.transpose(1, 0)
            # H()
            h_at_ax_y = self.h(at_ax_y)
            # R(x)
            r_x = self.r(x_init)

            # D_a()
            grad_scaled = self.d_alpha(h_at_ax_y + r_x)

            # update x
            x_init = x_init + grad_scaled
            # -end- single step operation

            # output
            output_list += [x_init]

            # check stopping condition, only for testing
            if self.stop_epsilon < float("inf"):
                error = fitting_error_cal(x_init, y, k)
                if i == 0:
                    error_prev = error
                    error_0 = error
                else:
                    error_prev_past = error_prev
                    error_reltv_past = error_reltv
                    # error_reltv = (abs(error - error_prev) + 1e-10) / (error_0 + 1e-10)
                    error_reltv = (abs(error - error_prev) + 1e-10) / (error_prev + 1e-10)
                    error_prev = error
                    # print('ite: %d; error_reltv: %f error_prev: %f' % (i, error_reltv, error_prev))
                    # if i > 4:
                    #     if error_prev_past - error_prev < 0.05 and error_reltv_past - error_reltv < 0.01:
                    #         # print('stop opt.')
                    #         break
                    if i > 4:
                        if error_reltv_past - error_reltv < 0.001:
                            break

        return output_list[-1]


class RGDNNonUniform(nn.Module):
    """Gradient descent based optimizer model for non-uniform
    x^{k+1} = x^{k} - \alpha^{k}*D^{k}*(A^T*\Sigma*(Ax^k-y)+\lambda*\nabla*R(x^k)).
    """

    def __init__(self, num_steps=10, stop_epsilon=1e-3):
        super().__init__()
        #
        self.num_steps = num_steps
        self.stop_epsilon = stop_epsilon
        self.grad_data_fitting = GradDataFitting()
        self.r = UNetRes(in_nc=3, out_nc=3)  # R module
        self.h = ConvBlock()  # H module
        self.d_alpha = ConvBlock()  # \alpha_t*D^t module

    def forward(self, y, k, kt):
        n_size = y.size()[0]
        k_size = k.size()[2]
        padding = int(k_size / 2)
        # init x
        # x_init = y
        # init x by using sample FFT
        x_init = model_init(y, k.squeeze(dim=1), 1)

        output_list = []
        # K^T
        # optimization init
        error_prev = 0
        error_reltv = 0
        for i in range(self.num_steps):
            # single step operation
            grad_loss = self.grad_data_fitting(x_init, y, k)  # Ax - y
            # H()
            h_ax_y = self.h(grad_loss)
            h_ax_y1 = h_ax_y.transpose(1, 0)
            # A^TH(Ax-y)
            at_h_ax_y = F.conv2d(h_ax_y1, kt, padding=padding, groups=n_size)
            at_h_ax_y = at_h_ax_y.transpose(1, 0)
            # R(x)
            r_x = self.r(x_init)

            # D_a()
            grad_scaled = self.d_alpha(at_h_ax_y + r_x)

            # update x
            x_init = x_init + grad_scaled
            # -end- single step operation

            # output
            output_list += [x_init]

            # check stopping condition, only for testing
            if self.stop_epsilon < float("inf"):
                error = fitting_error_cal(x_init, y, k)
                if i == 0:
                    error_prev = error
                    error_0 = error
                else:
                    error_prev_past = error_prev
                    error_reltv_past = error_reltv
                    # error_reltv = (abs(error - error_prev) + 1e-10) / (error_0 + 1e-10)
                    error_reltv = (abs(error - error_prev) + 1e-10) / (error_prev + 1e-10)
                    error_prev = error
                    # print('ite: %d; error_reltv: %f error_prev: %f' % (i, error_reltv, error_prev))
                    # if i > 4:
                    #     if error_prev_past - error_prev < 0.05 and error_reltv_past - error_reltv < 0.01:
                    #         # print('stop opt.')
                    #         break
                    if i > 4:
                        if error_reltv_past - error_reltv < 0.001:
                            break

        return output_list[-1]


class RGDNNonUniformY0(nn.Module):
    """Gradient descent based optimizer model for non-uniform
    x^{k+1} = x^{k} - \alpha^{k}*D^{k}*(A^T*\Sigma*(Ax^k-y)+\lambda*\nabla*R(x^k)).
    """

    def __init__(self, num_steps=10, stop_epsilon=1e-3):
        super().__init__()
        #
        self.num_steps = num_steps
        self.stop_epsilon = stop_epsilon
        self.grad_data_fitting = GradDataFitting()
        self.r = UNetRes(in_nc=3, out_nc=3)  # R module
        self.h = ConvBlock()  # H module
        self.d_alpha = ConvBlock()  # \alpha_t*D^t module

    def forward(self, y, k, kt):
        n_size = y.size()[0]
        k_size = k.size()[2]
        padding = int(k_size / 2)
        # init x
        x_init = y

        output_list = []
        # K^T
        # optimization init
        error_prev = 0
        error_reltv = 0
        for i in range(self.num_steps):
            # single step operation
            grad_loss = self.grad_data_fitting(x_init, y, k)  # Ax - y
            # H()
            h_ax_y = self.h(grad_loss)
            h_ax_y1 = h_ax_y.transpose(1, 0)
            # A^TH(Ax-y)
            at_h_ax_y = F.conv2d(h_ax_y1, kt, padding=padding, groups=n_size)
            at_h_ax_y = at_h_ax_y.transpose(1, 0)
            # R(x)
            r_x = self.r(x_init)

            # D_a()
            grad_scaled = self.d_alpha(at_h_ax_y + r_x)

            # update x
            x_init = x_init + grad_scaled
            # -end- single step operation

            # output
            output_list += [x_init]

            # check stopping condition, only for testing
            if self.stop_epsilon < float("inf"):
                error = fitting_error_cal(x_init, y, k)
                if i == 0:
                    error_prev = error
                    error_0 = error
                else:
                    error_prev_past = error_prev
                    error_reltv_past = error_reltv
                    # error_reltv = (abs(error - error_prev) + 1e-10) / (error_0 + 1e-10)
                    error_reltv = (abs(error - error_prev) + 1e-10) / (error_prev + 1e-10)
                    error_prev = error
                    # print('ite: %d; error_reltv: %f error_prev: %f' % (i, error_reltv, error_prev))
                    # if i > 4:
                    #     if error_prev_past - error_prev < 0.05 and error_reltv_past - error_reltv < 0.01:
                    #         # print('stop opt.')
                    #         break
                    if i > 4:
                        if error_reltv_past - error_reltv < 0.001:
                            break

        return output_list[-1]


def fitting_error_cal(x, y, k):
    # only used during testing
    n_size = x.size()[0]
    k_size = k.size()[2]
    padding = int(k_size / 2)
    x1 = x.transpose(1, 0)
    y1 = y.transpose(1, 0)
    kx_y = F.conv2d(x1, k, padding=padding, groups=n_size)  # Ax
    kx_y.sub_(y1)  # Ax-y
    return torch.norm(kx_y.transpose(1, 0), 'fro') / 2
