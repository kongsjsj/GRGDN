# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Shengjiang Kong,School of mathematics and statistics, Xidian University.
Email: sjkongxd@gmail.com.
Date: 2022/7/24 20:13.
"""
import os
import random
import numpy as np
import cv2
from math import floor
from scipy.io import savemat
from matplotlib import pyplot as plt
from skimage import img_as_float, img_as_ubyte


def generate_sigma_map(patch_size=128, sigma_min=0, sigma_max=75):
    """
    Generate gauss kernel
    """
    center = [random.uniform(0, patch_size), random.uniform(0, patch_size)]
    scale = random.uniform(patch_size/4, patch_size/4*3)
    kernel = gaussian_kernel(patch_size, patch_size, center, scale)
    up = random.uniform(sigma_min/255.0, sigma_max/255.0)
    down = random.uniform(sigma_min/255.0, sigma_max/255.0)
    if up < down:
        up, down = down, up
    up += 5 / 255.0
    sigma_map = down + (kernel - kernel.min()) / (kernel.max() - kernel.min()) * (up - down)
    sigma_map = sigma_map.astype(np.float32)
    return sigma_map[:, :, np.newaxis]


def gaussian_kernel(height, width, center, scale):
    center_h = center[0]
    center_w = center[1]
    x_data, y_data = np.meshgrid(np.arange(width), np.arange(height))
    z_data = 1 / (2*np.pi*scale**2) * np.exp((-(x_data-center_h)**2-(y_data-center_w)**2)/(2*scale**2))
    return z_data


def sincos_kernel():
    # Nips Version
    [xx, yy] = np.meshgrid(np.linspace(1, 10, 256), np.linspace(1, 20, 256))
    # [xx, yy] = np.meshgrid(np.linspace(1, 10, 256), np.linspace(-10, 15, 256))
    zz = np.sin(xx) + np.cos(yy)
    return zz


def generate_gauss_kernel_mix(H, W):
    '''
    Generate a H x W mixture Gaussian kernel with mean (center) and std (scale).
    Input:
        H, W: interger
        center: mean value of x axis and y axis
        scale: float value
    '''
    pch_size = 32
    K_H = floor(H / pch_size)
    K_W = floor(W / pch_size)
    K = K_H * K_W
    # prob = np.random.dirichlet(np.ones((K,)), size=1).reshape((1,1,K))
    centerW = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_W = np.arange(K_W) * pch_size
    centerW += ind_W.reshape((1, -1))
    centerW = centerW.reshape((1,1,K)).astype(np.float32)
    centerH = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_H = np.arange(K_H) * pch_size
    centerH += ind_H.reshape((-1, 1))
    centerH = centerH.reshape((1,1,K)).astype(np.float32)
    scale = np.random.uniform(low=pch_size/2, high=pch_size, size=(1,1,K))
    scale = scale.astype(np.float32)
    XX, YY = np.meshgrid(np.arange(0, W), np.arange(0,H))
    XX = XX[:, :, np.newaxis].astype(np.float32)
    YY = YY[:, :, np.newaxis].astype(np.float32)
    ZZ = 1./(2*np.pi*scale**2) * np.exp( (-(XX-centerW)**2-(YY-centerH)**2)/(2*scale**2) )
    out = ZZ.sum(axis=2, keepdims=False) / K

    return out


def peaks(n):
    """
    Implementation the peak function of matlab.
    """
    X = np.linspace(-3, 3, n)
    Y = np.linspace(-3, 3, n)
    [XX, YY] = np.meshgrid(X, Y)
    ZZ = 3 * (1-XX)**2 * np.exp(-XX**2 - (YY+1)**2) \
            - 10 * (XX/5.0 - XX**3 -YY**5) * np.exp(-XX**2-YY**2) - 1/3.0 * np.exp(-(XX+1)**2 - YY**2)
    return ZZ


if __name__ == '__main__':
    case = 4
    sigma_map = None
    # Generate the sigma map
    if case == 1:
        # Test case 1
        sigma_map = peaks(256)
    elif case == 2:
        # Test case 2
        sigma_map = sincos_kernel()
    elif case == 3:
        # Test case 3
        sigma_map = generate_gauss_kernel_mix(256, 256)
    elif case == 4:
        sigma_map = generate_sigma_map(patch_size=256)
    else:
        pass

    img_path = r"E:\kong\second_work\dataset\testsets\CBSD68\14037.jpg"
    img_gt = img_as_float(cv2.imread(img_path))
    h, w, c = img_gt.shape
    if case != 4:
        sig_av = 10
        sigma_map = sig_av/255.0 + (sigma_map-sigma_map.min())/(sigma_map.max()-sigma_map.min()) * ((75-sig_av)/255.0)

    d = {"sigma_map": sigma_map}
    savemat('sigma_map.mat', d)

    sigma_map = cv2.resize(sigma_map, (w, h))
    noise = np.random.randn(h, w, c) * sigma_map[:, :, np.newaxis]
    img_noisy = (img_gt + noise).astype(np.float32)

    # img_noisy = img_as_ubyte(img_noisy.clip(0, 1))
    # img_gt = img_as_ubyte(img_gt.clip(0, 1))
    # sigma_map = img_as_ubyte(sigma_map.clip(0, 1))

    img_noisy = img_as_float(img_noisy.clip(0, 1))
    img_gt = img_as_float(img_gt.clip(0, 1))
    sigma_map = img_as_float(sigma_map.clip(0, 1))

    plt.subplot(131)
    plt.imshow(img_gt)
    plt.title('Groundtruth')
    plt.subplot(132)
    plt.imshow(img_noisy)
    plt.title('Noisy Image')
    plt.subplot(133)
    plt.imshow(sigma_map)
    plt.title('sigma map')
    plt.show()



