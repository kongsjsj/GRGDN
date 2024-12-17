# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Shengjiang Kong,School of mathematics and statistics, Xidian University.
Email: sjkongxd@gmail.com
Date: JUL 16, 2020.
This Script is main for processing images toolbox
"""

from collections import OrderedDict
import logging
import math
import os
import platform
import random

import cv2
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.io import imread

import scipy
import torch

from math import cos, sin
from numpy import zeros, ones, prod, array, pi, log, min, mod, arange, sum, mgrid, exp, pad, round
from numpy.random import randn, rand
from scipy.signal import convolve2d
from scipy import fftpack


def weights_init_kaiming(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1:
        # nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def add_noise(img, noise_level, noise_range, noise_type):
    """
    :param img: tensor, N * C * H * W, the clean image data
    :param noise_level: float, the size of noise level, eg: 15., 25., 55. etc.
    :param noise_range: list, the range of noise level eg: [0, 55] if noise_type=B
    :param noise_type: string, 'S' for known noise level, 'B' for unknown
                       noise level or single model of known noise level map

    :return: tensor: img_noise(N * C * H * W), noise(N * C * H * W), noise_map(N * 1 * H * W)
    """

    noise = None
    noise_map = None
    batch_size, c, h, w = img.size()

    if noise_type == 'S':
        noise = torch.FloatTensor(batch_size, c, h, w).normal_(mean=0, std=noise_level / 255.)
        noise_map = torch.tensor(noise_level / 255.).repeat(batch_size, 1, h, w)
    if noise_type == 'B':
        noise = torch.zeros(batch_size, c, h, w)
        noise_map = torch.zeros(batch_size, 1, h, w)  # noise level map has only one channels
        std_n = np.random.uniform(noise_range[0], noise_range[1], size=batch_size) / 255.
        for n in range(batch_size):
            noise[n, :, :, :] = torch.FloatTensor(c, h, w).normal_(mean=0, std=std_n[n])
            noise_map[n, :, :, :] = torch.from_numpy(np.array(std_n[n])).float().repeat(1, h, w)  # 1 * h * w

    return img + noise, noise, noise_map


# --------------------------------------------
# PSNR
# --------------------------------------------


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    """calculate SSIM
    the same outputs as matlab's
    img1, img2: [0, 255]
    """
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = list()
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) *
                                                            (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def batch_psnr(img, img_clean, data_range):
    img_new = img.data.cpu().numpy().astype(np.float32)
    img_clean_new = img_clean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_new.shape[0]):
        psnr += compare_psnr(img_clean_new[i, :, :, :], img_new[i, :, :, :], data_range=data_range)

    return psnr / img_new.shape[0]


def batch_ssim(img, img_clean):
    """
    :param img: tensor, N X C X H X W
    :param img_clean: tensor, N X C X H X W
    :return: psnr value
    """
    img_new = img.data.cpu().numpy().transpose(0, 2, 3, 1)  # N X H X W x C
    img_clean_new = img_clean.data.cpu().numpy().transpose(0, 2, 3, 1)
    img_new = np.clip(img_new * 255, 0, 255)
    img_clean_new = np.clip(img_clean_new * 255, 0, 255)
    ssim_ = 0.
    for i in range(img_new.shape[0]):
        # ssim_ += compare_ssim(img_clean_new[i, :, :, :], img_new[i, :, :, :],
        #                       data_range=[0, 255],
        #                       multichannel=True)
        ssim_ += calculate_ssim(img_clean_new[i, :, :, :], img_new[i, :, :, :])

    return ssim_ / img_new.shape[0]


def data_augmentation(image, mode):
    """
    Data augmentation function
    :param image: numpy array, C X H X W
    :param mode: int, value: 0, 1, 2, ..., 7.
    :return: augmentation image, C X H X W.
    """
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counter-wise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def variable_to_cv2_image(var_img):
    """
    Converts a torch.auto-grad.Variable to an OpenCV image
    :param var_img: a torch.auto-grad.Variable
    :return: res
    """
    num_channels = var_img.size()[1]
    if num_channels == 1:
        res = (var_img.data.cpu().numpy()[0, 0, :] * 255.).clip(0, 255).astype(np.uint8)
    elif num_channels == 3:
        res = var_img.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        res = (res * 255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color_more channels not supported')
    return res


def init_logger_ipol(file_name, obj_name):
    """
    Initializes a logging.Logger in order to log
    the results_hight after testing a model
    :param: file_name, the name of file
    :return: path to the folder with the denoising results_hight
    """
    logger = logging.getLogger(obj_name)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(file_name, mode='w')
    formatter = logging.Formatter('%(message)s')
    # formatter = logging.Formatter('[%(asctime)s] ------: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def is_rgb(im_path):
    """
    Returns True if the image in im_path is an RGB image
    :param im_path:
    :return: rgb
    """
    rgb = False
    im = imread(im_path)
    if len(im.shape) == 3:
        if not (np.allclose(im[..., 0], im[..., 1]) and np.allclose(im[..., 2], im[..., 1])):
            rgb = True
    return rgb


def remove_data_parallel_wrapper(state_dict):
    """
    Converts a DataParallel model to a normal one by
    removing the "module.
    wrapper in the module dictionary
    :param state_dict: a torch.nn.DataParallel state dictionary
    :return:
    """
    new_state_dict = OrderedDict()
    for k, vl in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel
        new_state_dict[name] = vl

    return new_state_dict


def extract_file_name(ap):
    """
    This function is used for extracting a file
    name with absolute path
    :param ap: absolute path file, eg. '/data/data_testsets/001.png'
    :return: filename, eg. '/data/data_testsets/001.png' return '001.png'
    Writen by Shengjiang Kong, Xidian University,
    school of mathematics and statistics.
    Date: Dec 17, 2019.

    """
    if platform.system() == 'Windows':
        return ap.split('.')[-2].split('\\')[-1]
    elif platform.system() == 'Linux' or 'Mac':
        return ap.split('.')[-2].split('/')[-1]
    else:
        print('PC platform is not Windows, Linux or Mac!\n')
        print('Extracted failed!')
        return ap


def make_dirs(dir_name):
    """
    Make a dir in current path for save files
    :param dir_name: the name of dir
    :return: None
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def set_random_seed(seed=10, deterministic=False, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True

    if benchmark:
        torch.backends.cudnn.benchmark = True


def imread_uint8(img_path, n_channels=1):
    """
    Use cv2.imread to load img_path
    :param img_path: string, The path of image
    :param n_channels: int, 1 or 3.
    :return: HxWx3(RGB or GGG), or HxWx1 (G)
    """
    img = None
    if n_channels == 1:
        img = cv2.imread(img_path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

    return img


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


def add_non_uniform_noise(img, noise_levels, region='four'):
    """Add non-uniform Additive Write Gaussian Noise (AWGN)
    Author: Shengjiang Kong.
    Date: Sep 23, 2021.
    :param: img: numpy -> C X H X W
    :param: noise_levels: list -> [sigma_1, sigma_2, sigma_3, sigma_4].
    :param: region: 'four'(default)|'top_bottom'|'left_right'|'one'.
    :return: img: numpy -> C X H X W
    """
    # img = img / 255.0
    ndim, high, width = img.shape
    sigma = []
    if len(noise_levels) == 1 and region == 'one':
        sigma = noise_levels * 4
    elif len(noise_levels) == 2:
        if region == 'top_bottom':
            sigma = [noise_levels[0], noise_levels[0], noise_levels[1], noise_levels[1]]
        elif region == 'left_right':
            sigma = [noise_levels[0], noise_levels[1], noise_levels[0], noise_levels[1]]
    elif len(noise_levels) == 4 and region == 'four':
        sigma = noise_levels
    else:
        raise Exception("Not implementation!")

    # width
    if np.mod(width, 2) != 0:
        width1 = (width - np.mod(width, 2)) // 2
        width2 = width - width1
    else:
        width1 = width // 2
        width2 = width1
    # high
    if np.mod(high, 2) != 0:
        high1 = (high - np.mod(high, 2)) // 2
        high2 = high - high1
    else:
        high1 = high // 2
        high2 = high1

    # Add noise
    # Upper left
    noise_img = np.random.normal(loc=0.0, scale=sigma[0], size=(ndim, high1, width1))
    img[:, 0: high1, 0: width1] = img[:, 0: high1, 0: width1] + noise_img
    # Upper right
    noise_img = np.random.normal(loc=0.0, scale=sigma[1], size=(ndim, high1, width2))
    img[:, 0:high1, width1:width] = img[:, 0:high1, width1:width] + noise_img
    # Lower left
    noise_img = np.random.normal(loc=0.0, scale=sigma[2], size=(ndim, high2, width1))
    img[:, high1:high, 0:width1] = img[:, high1:high, 0:width1] + noise_img
    # Lower right
    noise_img = np.random.normal(loc=0.0, scale=sigma[3], size=(ndim, high2, width2))
    img[:, high1:high, width1:width] = img[:, high1:high, width1:width] + noise_img

    return img


"""
codes from https://github.com/cszn/KAIR
"""
"""
Created on Thu Jan 18 15:36:32 2018
@author: italo
https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
"""

"""
Syntax
h = fspecial(type)
h = fspecial('average',hsize)
h = fspecial('disk',radius)
h = fspecial('gaussian',hsize,sigma)
h = fspecial('laplacian',alpha)
h = fspecial('log',hsize,sigma)
h = fspecial('motion',len,theta)
h = fspecial('prewitt')
h = fspecial('sobel')
"""


def fspecial_average(hsize=3):
    """Smoothing filter"""
    return np.ones((hsize, hsize)) / hsize ** 2


def fspecial_disk(radius):
    """Disk filter"""
    raise (NotImplemented)
    rad = 0.6
    crad = np.ceil(rad - 0.5)
    [x, y] = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))
    maxxy = np.zeros(x.shape)
    maxxy[abs(x) >= abs(y)] = abs(x)[abs(x) >= abs(y)]
    maxxy[abs(y) >= abs(x)] = abs(y)[abs(y) >= abs(x)]
    minxy = np.zeros(x.shape)
    minxy[abs(x) <= abs(y)] = abs(x)[abs(x) <= abs(y)]
    minxy[abs(y) <= abs(x)] = abs(y)[abs(y) <= abs(x)]
    m1 = (rad ** 2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (minxy - 0.5) + \
         (rad ** 2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * \
         np.sqrt((rad ** 2 + 0j) - (maxxy + 0.5) ** 2)
    m2 = (rad ** 2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5) + \
         (rad ** 2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * \
         np.sqrt((rad ** 2 + 0j) - (maxxy - 0.5) ** 2)
    h = None
    return h


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha, 1])])
    h1 = alpha / (alpha + 1)
    h2 = (1 - alpha) / (alpha + 1)
    h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial_log(hsize, sigma):
    raise (NotImplemented)


def fspecial_motion(motion_len, theta):
    raise (NotImplemented)


def fspecial_prewitt():
    return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])


def fspecial_sobel():
    return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'average':
        return fspecial_average(*args, **kwargs)
    if filter_type == 'disk':
        return fspecial_disk(*args, **kwargs)
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)
    if filter_type == 'log':
        return fspecial_log(*args, **kwargs)
    if filter_type == 'motion':
        return fspecial_motion(*args, **kwargs)
    if filter_type == 'prewitt':
        return fspecial_prewitt(*args, **kwargs)
    if filter_type == 'sobel':
        return fspecial_sobel(*args, **kwargs)


def fspecial_gauss(size, sigma):
    x, y = mgrid[-size // 2 + 1: size // 2 + 1, -size // 2 + 1: size // 2 + 1]
    g = exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def blurkernel_synthesis(h=37, w=None):
    # https://github.com/tkkcc/prior/blob/879a0b6c117c810776d8cc6b63720bf29f7d0cc4/util/gen_kernel.py
    w = h if w is None else w
    kdims = [h, w]
    x = randomTrajectory(250)
    k = None
    while k is None:
        k = kernelFromTrajectory(x)

    # center pad to kdims
    pad_width = ((kdims[0] - k.shape[0]) // 2, (kdims[1] - k.shape[1]) // 2)
    pad_width = [(pad_width[0],), (pad_width[1],)]

    if pad_width[0][0] < 0 or pad_width[1][0] < 0:
        k = k[0:h, 0:h]
    else:
        k = pad(k, pad_width, "constant")
    x1, x2 = k.shape
    if np.random.randint(0, 4) == 1:
        k = cv2.resize(k, (random.randint(x1, 5 * x1), random.randint(x2, 5 * x2)), interpolation=cv2.INTER_LINEAR)
        y1, y2 = k.shape
        k = k[(y1 - x1) // 2: (y1 - x1) // 2 + x1, (y2 - x2) // 2: (y2 - x2) // 2 + x2]

    if sum(k) < 0.1:
        k = fspecial_gaussian(h, 0.1 + 6 * np.random.rand(1))
    k = k / sum(k)
    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()
    return k


def kernelFromTrajectory(x):
    h = 5 - log(rand()) / 0.15
    h = round(min([h, 27])).astype(int)
    h = h + 1 - h % 2
    w = h
    k = zeros((h, w))

    xmin = min(x[0])
    xmax = max(x[0])
    ymin = min(x[1])
    ymax = max(x[1])
    xthr = arange(xmin, xmax, (xmax - xmin) / w)
    ythr = arange(ymin, ymax, (ymax - ymin) / h)

    for i in range(1, xthr.size):
        for j in range(1, ythr.size):
            idx = (
                    (x[0, :] >= xthr[i - 1])
                    & (x[0, :] < xthr[i])
                    & (x[1, :] >= ythr[j - 1])
                    & (x[1, :] < ythr[j])
            )
            k[i - 1, j - 1] = sum(idx)
    if sum(k) == 0:
        return
    k = k / sum(k)
    k = convolve2d(k, fspecial_gauss(3, 1), "same")
    k = k / sum(k)
    return k


def randomTrajectory(T):
    x = zeros((3, T))
    v = randn(3, T)
    r = zeros((3, T))
    trv = 1 / 1
    trr = 2 * pi / T
    for t in range(1, T):
        F_rot = randn(3) / (t + 1) + r[:, t - 1]
        F_trans = randn(3) / (t + 1)
        r[:, t] = r[:, t - 1] + trr * F_rot
        v[:, t] = v[:, t - 1] + trv * F_trans
        st = v[:, t]
        st = rot3D(st, r[:, t])
        x[:, t] = x[:, t - 1] + st
    return x


def rot3D(x, r):
    Rx = array([[1, 0, 0], [0, cos(r[0]), -sin(r[0])], [0, sin(r[0]), cos(r[0])]])
    Ry = array([[cos(r[1]), 0, sin(r[1])], [0, 1, 0], [-sin(r[1]), 0, cos(r[1])]])
    Rz = array([[cos(r[2]), -sin(r[2]), 0], [sin(r[2]), cos(r[2]), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    x = R @ x
    return x


def gen_kernel(k_size=np.array([25, 25]), min_var=0.6, max_var=12., noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    sf = 1
    scale_factor = np.array([sf, sf])
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = 0  # -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5 * (scale_factor - 1)  # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    # raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    # kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


"""
Reducing boundary artifacts
"""


def opt_fft_size(n):
    '''
    Kai Zhang (github: https://github.com/cszn)
    03/03/2019
    #  opt_fft_size.m
    # compute an optimal data length for Fourier transforms
    # written by Sunghyun Cho (sodomau@postech.ac.kr)
    # persistent opt_fft_size_LUT;
    '''

    LUT_size = 2048
    # print("generate opt_fft_size_LUT")
    opt_fft_size_LUT = np.zeros(LUT_size)

    e2 = 1
    while e2 <= LUT_size:
        e3 = e2
        while e3 <= LUT_size:
            e5 = e3
            while e5 <= LUT_size:
                e7 = e5
                while e7 <= LUT_size:
                    if e7 <= LUT_size:
                        opt_fft_size_LUT[e7 - 1] = e7
                    if e7 * 11 <= LUT_size:
                        opt_fft_size_LUT[e7 * 11 - 1] = e7 * 11
                    if e7 * 13 <= LUT_size:
                        opt_fft_size_LUT[e7 * 13 - 1] = e7 * 13
                    e7 = e7 * 7
                e5 = e5 * 5
            e3 = e3 * 3
        e2 = e2 * 2

    nn = 0
    for i in range(LUT_size, 0, -1):
        if opt_fft_size_LUT[i - 1] != 0:
            nn = i - 1
        else:
            opt_fft_size_LUT[i - 1] = nn + 1

    m = np.zeros(len(n))
    for c in range(len(n)):
        nn = n[c]
        if nn <= LUT_size:
            m[c] = opt_fft_size_LUT[nn - 1]
        else:
            m[c] = -1
    return m


def wrap_boundary_liu(img, img_size):
    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    if img.ndim == 2:
        ret = wrap_boundary(img, img_size)
    elif img.ndim == 3:
        ret = [wrap_boundary(img[:, :, i], img_size) for i in range(3)]
        ret = np.stack(ret, 2)
    return ret


def wrap_boundary(img, img_size):
    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    (H, W) = np.shape(img)
    H_w = int(img_size[0]) - H
    W_w = int(img_size[1]) - W

    # ret = np.zeros((img_size[0], img_size[1]));
    alpha = 1
    HG = img[:, :]

    r_A = np.zeros((alpha * 2 + H_w, W))
    r_A[:alpha, :] = HG[-alpha:, :]
    r_A[-alpha:, :] = HG[:alpha, :]
    a = np.arange(H_w) / (H_w - 1)
    # r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)
    r_A[alpha:-alpha, 0] = (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0]
    # r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end)
    r_A[alpha:-alpha, -1] = (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1]

    r_B = np.zeros((H, alpha * 2 + W_w))
    r_B[:, :alpha] = HG[:, -alpha:]
    r_B[:, -alpha:] = HG[:, :alpha]
    a = np.arange(W_w) / (W_w - 1)
    r_B[0, alpha:-alpha] = (1 - a) * r_B[0, alpha - 1] + a * r_B[0, -alpha]
    r_B[-1, alpha:-alpha] = (1 - a) * r_B[-1, alpha - 1] + a * r_B[-1, -alpha]

    if alpha == 1:
        A2 = solve_min_laplacian(r_A[alpha - 1:, :])
        B2 = solve_min_laplacian(r_B[:, alpha - 1:])
        r_A[alpha - 1:, :] = A2
        r_B[:, alpha - 1:] = B2
    else:
        A2 = solve_min_laplacian(r_A[alpha - 1:-alpha + 1, :])
        r_A[alpha - 1:-alpha + 1, :] = A2
        B2 = solve_min_laplacian(r_B[:, alpha - 1:-alpha + 1])
        r_B[:, alpha - 1:-alpha + 1] = B2
    A = r_A
    B = r_B

    r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w))
    r_C[:alpha, :] = B[-alpha:, :]
    r_C[-alpha:, :] = B[:alpha, :]
    r_C[:, :alpha] = A[:, -alpha:]
    r_C[:, -alpha:] = A[:, :alpha]

    if alpha == 1:
        C2 = C2 = solve_min_laplacian(r_C[alpha - 1:, alpha - 1:])
        r_C[alpha - 1:, alpha - 1:] = C2
    else:
        C2 = solve_min_laplacian(r_C[alpha - 1:-alpha + 1, alpha - 1:-alpha + 1])
        r_C[alpha - 1:-alpha + 1, alpha - 1:-alpha + 1] = C2
    C = r_C
    # return C
    A = A[alpha - 1:-alpha - 1, :]
    B = B[:, alpha:-alpha]
    C = C[alpha:-alpha, alpha:-alpha]
    ret = np.vstack((np.hstack((img, B)), np.hstack((A, C))))
    return ret


def solve_min_laplacian(boundary_image):
    (H, W) = np.shape(boundary_image)

    # Laplacian
    f = np.zeros((H, W))
    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0
    j = np.arange(2, H) - 1
    k = np.arange(2, W) - 1
    f_bp = np.zeros((H, W))
    f_bp[np.ix_(j, k)] = -4 * boundary_image[np.ix_(j, k)] + boundary_image[np.ix_(j, k + 1)] + boundary_image[
        np.ix_(j, k - 1)] + boundary_image[np.ix_(j - 1, k)] + boundary_image[np.ix_(j + 1, k)]

    del (j, k)
    f1 = f - f_bp  # subtract boundary points contribution
    del (f_bp, f)

    # DST Sine Transform algo starts here
    f2 = f1[1:-1, 1:-1]
    del (f1)

    # compute sine tranform
    if f2.shape[1] == 1:
        tt = fftpack.dst(f2, type=1, axis=0) / 2
    else:
        tt = fftpack.dst(f2, type=1) / 2

    if tt.shape[0] == 1:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1, axis=0) / 2)
    else:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1) / 2)
    del (f2)

    # compute Eigen Values
    [x, y] = np.meshgrid(np.arange(1, W - 1), np.arange(1, H - 1))
    denom = (2 * np.cos(np.pi * x / (W - 1)) - 2) + (2 * np.cos(np.pi * y / (H - 1)) - 2)

    # divide
    f3 = f2sin / denom
    del (f2sin, x, y)

    # compute Inverse Sine Transform
    if f3.shape[0] == 1:
        tt = fftpack.idst(f3 * 2, type=1, axis=1) / (2 * (f3.shape[1] + 1))
    else:
        tt = fftpack.idst(f3 * 2, type=1, axis=0) / (2 * (f3.shape[0] + 1))
    del (f3)
    if tt.shape[1] == 1:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt) * 2, type=1) / (2 * (tt.shape[0] + 1)))
    else:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt) * 2, type=1, axis=0) / (2 * (tt.shape[1] + 1)))
    del (tt)

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image
    img_direct[1:-1, 1:-1] = 0
    img_direct[1:-1, 1:-1] = img_tt
    return img_direct


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16., 128., 128.]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


EXT = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif')


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(EXT)]


def crop_img2patch(path=None, channels=1, patch_sizes=None):
    """Crop images into small patches
    patch_sizes: list or tuple 1*1|1*2, eg. [3, 3], (3, 3), (3, ), [3] and or int.
    Date: Mar 05, 2021.
    Author: Shengjiang Kong
    """
    save_path = ''.join([path, '_more'])

    m_row, n_col = None, None
    if patch_sizes is None:
        # Default value
        m_row, n_col = 3, 3
    elif isinstance(patch_sizes, tuple) or isinstance(patch_sizes, list):
        if len(patch_sizes) == 1:
            m_row, n_col = patch_sizes[0], patch_sizes[0]
        if len(patch_sizes) == 2:
            m_row, n_col = patch_sizes[0], patch_sizes[1]
    elif isinstance(patch_sizes, int):
        m_row, n_col = patch_sizes, patch_sizes
    else:
        raise NotImplementedError('patch_sizes is not configure!')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if path is not None:
        im_list = get_imlist(path)
        with tqdm(total=len(im_list), desc='Cropping', leave=True, ncols=100, unit='B', unit_scale=True) as bar:
            for img_file in im_list:
                # Update progressbar
                bar.update(1)
                img_name = os.path.basename(img_file).split('.')[0]
                img = imread_uint8(img_file, channels)
                # Crop into m x n patches
                height, width, dim = img.shape
                em = math.floor(height / m_row)
                en = math.floor(width / n_col)
                for ii in range(m_row):
                    for jj in range(n_col):
                        pic = img[ii * em + 1:(ii + 1) * em, jj * en + 1:(jj + 1) * en, :]
                        save_img_name = ''.join([img_name, '_', str(ii + 1), str(jj + 1), '.png'])
                        cv2.imwrite(os.path.join(save_path, save_img_name), pic)

    else:
        raise NotImplementedError('path must be not empty!')


def add_impulse_noise(img=None, pc=0.01, noise_type='sp'):
    """Add salt and pepper noise/random impulse noise to an input image.
    Date: Mar 18, 2023.
    :param img: clean image, size: c * m * n.
    :param pc: the percentage of the impulse noise. (0~1)
    :param noise_type: 'sp(salt and pepper) | rd(random impulse noise)'
    :return: Noisy image.
    # -------------------------------------------------------------------------
    We reference to following code from https://www.math.hkust.edu.hk/~jfcai/
    Paper: J.-F. Cai, R.H. Chan, and M. Nikolova,
    Two-phase Approach for Deblurring Images Corrupted by Impulse Plus Gaussian Noise,
    Inverse Probl. Imaging, 2(2):187--204, 2008.
    # -------------------------------------------------------------------------
    """
    np.random.seed(seed=0)  # for reproducibility
    h, w = img.shape[:2]
    new_img = img
    ran = np.random.rand(h, w)
    r = pc / 2.

    h1, w1 = new_img[(ran <= r)].shape[:2]
    if noise_type == 'sp':
        new_img[(ran <= r)] = 0
    elif noise_type == 'rd':
        new_img[(ran <= r)] = 255 * np.random.rand(h1, w1)

    h2, w2 = new_img[(ran >= 1-r)].shape[:2]
    if noise_type == 'sp':
        new_img[(ran >= 1-r)] = 255
    elif noise_type == 'rd':
        new_img[(ran >= 1-r)] = 255 * np.random.rand(h2, w2)

    return new_img


# ------------------------------------------------------------------------
# truncate_image
# code come from: https://github.com/donggong1/learn-optimizer-rgdn
# ------------------------------------------------------------------------
def truncate_image(img, s):
    # s: truncate size
    if(s>0):
        if(len(img.shape)==3):
            # F or C x H x W
            return img[:, s:(-s), s:(-s)]
        elif(len(img.shape)==4):
            # F x C x H x W
            return img[:, :, s:(-s), s:(-s)]
    else:
        return img


if __name__ == '__main__':
    img = imread_uint8('../1.jpg', 3)
    n_img = add_impulse_noise(img=img, pc=0.1, noise_type='sp')
    print(n_img.shape)
    img_noise_ = n_img.clip(0, 255).astype(np.uint8)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img_noise_)
    plt.show()


