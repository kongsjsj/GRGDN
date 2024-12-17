# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Shengjiang Kong,School of mathematics and statistics, Xidian University.
Email: sjkongxd@gmail.com.
Date: 2021/09/17 17:38
"""

import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from scipy import ndimage
from utils import (get_image_paths,
                   imread_uint,
                   augment_img,
                   uint2single,
                   single2tensor3,
                   uint2tensor3,
                   blurkernel_synthesis,
                   gen_kernel,
                   generate_sigma_map,
                   generate_gauss_kernel_mix)
from utils.othertools import add_impulse_noise, add_non_uniform_noise


class DatasetNRLGDNDeblurring(Dataset):
    """
    Dataset of NRL-GDN for non-blind deblurring.
    """

    def __init__(
            self,
            root_dir='',
            patch_size=128,
            kernels=None,
            num_channels=3,
            sigma_max=25,
            nl=2.55,
            kernel_index=0,
            is_train=True
    ):
        super().__init__()
        self.is_train = is_train
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.sigma_max = sigma_max
        self.kernels = kernels
        self.kernel_index = kernel_index
        self.val_noise_level = nl
        self.high_img_paths = get_image_paths(root_dir)

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        # Get high quality image
        h_path = self.high_img_paths[index]
        h_img = imread_uint(h_path, self.num_channels)
        # l_path = h_path  # low quality image

        if self.is_train:
            height, width, _ = h_img.shape

            # ----------------------------
            #  1) randomly crop the patch
            # ----------------------------

            rnd_h = random.randint(0, max(0, height - self.patch_size))
            rnd_w = random.randint(0, max(0, width - self.patch_size))
            h_patch = h_img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            h_patch = augment_img(h_patch, mode=np.random.randint(0, 8))

            # ---------------------------
            #  2) kernel
            # ---------------------------
            r_value = np.random.randint(0, 20)
            if 4 <= r_value < 17:
                kernel = blurkernel_synthesis(h=25)  # motion blur
            else:
                kernel = gen_kernel()  # Gaussian blur
                kernel = augment_img(kernel, mode=np.random.randint(0, 8))

            kernel /= np.sum(kernel)

            # kernel transpose 180 degree.
            kernel_t = kernel[::-1, ::-1]

            # Set noise level
            noise_level = np.random.uniform(0., self.sigma_max) / 255.0
            # ---------------------------
            # Low-quality image
            # ---------------------------
            # blur and down-sample
            l_img = ndimage.filters.convolve(h_patch, np.expand_dims(kernel, axis=2), mode='wrap')

            r_value = np.random.randint(0, 10)
            # ----------------------------------------------
            # (1) add Gaussian noise
            # ----------------------------------------------
            l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)
            h_img = h_patch

        else:
            kernel = self.kernels[0, self.kernel_index].astype(np.float64)  # validation kernel
            kernel /= np.sum(kernel)
            kernel_t = kernel[::-1, ::-1]
            noise_level = self.val_noise_level / 255.0  # validation noise level
            l_img = ndimage.filters.convolve(h_img, np.expand_dims(kernel, axis=2), mode='wrap')  # blur
            l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)

        kernel = single2tensor3(np.expand_dims(np.float32(kernel), axis=2))
        kernel_t = single2tensor3(np.expand_dims(np.float32(kernel_t), axis=2))
        h_img, l_img = uint2tensor3(h_img), single2tensor3(l_img)

        return h_img, l_img, kernel, kernel_t

    def __len__(self):
        return len(self.high_img_paths)


class DatasetNRLGDNDeblurringNonUniform(Dataset):
    """
    Dataset of NRL-GDN for non-blind deblurring of non-uniform noise.
    """

    def __init__(
            self,
            root_dir='',
            patch_size=128,
            kernels=None,
            num_channels=3,
            sigma_max=25,
            nl=2.55,
            kernel_index=0,
            is_train=True
    ):
        super().__init__()
        self.is_train = is_train
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.sigma_max = sigma_max
        self.kernels = kernels
        self.kernel_index = kernel_index
        self.val_noise_level = nl
        self.high_img_paths = get_image_paths(root_dir)

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        # Get high quality image
        h_path = self.high_img_paths[index]
        h_img = imread_uint(h_path, self.num_channels)
        # l_path = h_path  # low quality image

        if self.is_train:

            height, width, _ = h_img.shape

            # ----------------------------
            #  1) randomly crop the patch
            # ----------------------------

            rnd_h = random.randint(0, max(0, height - self.patch_size))
            rnd_w = random.randint(0, max(0, width - self.patch_size))
            h_patch = h_img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            h_patch = augment_img(h_patch, mode=np.random.randint(0, 8))

            # ---------------------------
            #  2) kernel
            # ---------------------------
            r_value = np.random.randint(0, 20)
            if 4 <= r_value < 17:
                kernel = blurkernel_synthesis(h=25)  # motion blur
            else:
                kernel = gen_kernel()  # Gaussian blur
                kernel = augment_img(kernel, mode=np.random.randint(0, 8))

            kernel /= np.sum(kernel)
            # kernel transpose 180 degree.
            kernel_t = kernel[::-1, ::-1]

            # Set noise level
            # noise_level = np.random.uniform(0., self.sigma_max) / 255.0

            # ---------------------------
            # Low-quality image
            # ---------------------------
            # blur and down-sample
            l_img = ndimage.filters.convolve(h_patch, np.expand_dims(kernel, axis=2), mode='wrap')

            # --------------------------------------
            # Method One
            # --------------------------------------
            # add non-uniform Gaussian noise
            # Date: 2022-08-23
            # sigma_map = generate_sigma_map(patch_size=self.patch_size, sigma_max=self.sigma_max)
            # noise = torch.randn(l_img.shape).numpy() * sigma_map
            # # im_noisy = im_gt + noise.astype(np.float32)
            # l_img = uint2single(l_img) + noise

            # -------------------------------------
            # Method two
            # -------------------------------------
            r_value1 = np.random.randint(0, 9)
            if 0 <= r_value1 <= 5:
                region = 'one'
                nls = [np.random.uniform(0.0, self.sigma_max) / 255.0]
            elif r_value1 >= 6:
                region = 'four'
                nls = np.random.uniform(0.0, self.sigma_max, size=4) / 255.0
            else:
                raise Exception('Not implementation!')
            l_img = add_non_uniform_noise(img=(l_img / 255.0).transpose(2, 0, 1), noise_levels=nls, region=region)
            l_img = l_img.transpose(1, 2, 0)
            # print('=================******_____********=================')
            h_img = h_patch

        else:
            kernel = self.kernels[0, self.kernel_index].astype(np.float64)  # validation kernel
            kernel /= np.sum(kernel)
            kernel_t = kernel[::-1, ::-1]
            # noise_level = self.val_noise_level / 255.0  # validation noise level
            l_img = ndimage.filters.convolve(h_img, np.expand_dims(kernel, axis=2), mode='wrap')  # blur
            # l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)
            # Add non-uniform noise
            sigma_map = generate_gauss_kernel_mix(256, 256)
            sig_av = 5
            sigma_map = sig_av / 255.0 + (sigma_map - sigma_map.min()) / (sigma_map.max() - sigma_map.min()) * ((self.sigma_max - sig_av) / 255.0)
            h, w, c = l_img.shape
            sigma_map = cv2.resize(sigma_map, (w, h))
            noise = np.random.randn(h, w, c) * sigma_map[:, :, np.newaxis]
            l_img = uint2single(l_img) + noise

        kernel = single2tensor3(np.expand_dims(np.float32(kernel), axis=2))
        kernel_t = single2tensor3(np.expand_dims(np.float32(kernel_t), axis=2))
        h_img, l_img = uint2tensor3(h_img), single2tensor3(l_img)

        return h_img, l_img, kernel, kernel_t

    def __len__(self):
        return len(self.high_img_paths)


class DatasetNRLGDNDeblurringPoissonGaussian(Dataset):
    """
    Dataset of NRL-GDN for non-blind deblurring.
    """

    def __init__(
            self,
            root_dir='',
            patch_size=128,
            kernels=None,
            num_channels=3,
            sigma_max=25,
            nl=2.55,
            kernel_index=0,
            is_train=True
    ):
        super().__init__()
        self.is_train = is_train
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.sigma_max = sigma_max
        self.kernels = kernels
        self.kernel_index = kernel_index
        self.val_noise_level = nl
        self.high_img_paths = get_image_paths(root_dir)

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        # Get high quality image
        h_path = self.high_img_paths[index]
        h_img = imread_uint(h_path, self.num_channels)
        # l_path = h_path  # low quality image

        if self.is_train:
            height, width, _ = h_img.shape

            # ----------------------------
            #  1) randomly crop the patch
            # ----------------------------

            rnd_h = random.randint(0, max(0, height - self.patch_size))
            rnd_w = random.randint(0, max(0, width - self.patch_size))
            h_patch = h_img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            h_patch = augment_img(h_patch, mode=np.random.randint(0, 8))

            # ---------------------------
            #  2) kernel
            # ---------------------------
            r_value = np.random.randint(0, 20)
            if 4 <= r_value < 17:
                kernel = blurkernel_synthesis(h=25)  # motion blur
            else:
                kernel = gen_kernel()  # Gaussian blur
                kernel = augment_img(kernel, mode=np.random.randint(0, 8))

            kernel /= np.sum(kernel)

            # kernel transpose 180 degree.
            kernel_t = kernel[::-1, ::-1]

            # Set noise level
            noise_level = np.random.uniform(0., self.sigma_max) / 255.0
            # ---------------------------
            # Low-quality image
            # ---------------------------
            # blur and down-sample
            l_img = ndimage.filters.convolve(h_patch, np.expand_dims(kernel, axis=2), mode='wrap')

            r_value = np.random.randint(0, 10)
            # ----------------------------------------------
            # (1) add Gaussian noise
            # ----------------------------------------------
            l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)

            h_img = h_patch

        else:

            kernel = self.kernels[0, self.kernel_index].astype(np.float64)  # validation kernel
            kernel /= np.sum(kernel)
            kernel_t = kernel[::-1, ::-1]
            noise_level = self.val_noise_level / 255.0  # validation noise level
            l_img = ndimage.filters.convolve(h_img, np.expand_dims(kernel, axis=2), mode='wrap')  # blur
            l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)

        kernel = single2tensor3(np.expand_dims(np.float32(kernel), axis=2))
        kernel_t = single2tensor3(np.expand_dims(np.float32(kernel_t), axis=2))
        h_img, l_img = uint2tensor3(h_img), single2tensor3(l_img)

        return h_img, l_img, kernel, kernel_t

    def __len__(self):
        return len(self.high_img_paths)


class DatasetNRLGDNDeblurringImpulseNoise(Dataset):
    """
    Dataset of NRL-GDN for non-blind deblurring in
    the presence of impulse noise.
    """

    def __init__(
            self,
            root_dir='',
            patch_size=128,
            kernels=None,
            num_channels=3,
            kernel_index=0,
            is_train=True
    ):
        super().__init__()
        self.is_train = is_train
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.kernels = kernels
        self.kernel_index = kernel_index
        self.high_img_paths = get_image_paths(root_dir)

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        # Get high quality image
        h_path = self.high_img_paths[index]
        h_img = imread_uint(h_path, self.num_channels)
        # l_path = h_path  # low quality image

        if self.is_train:

            height, width, _ = h_img.shape
            # ----------------------------
            #  1) randomly crop the patch
            # ----------------------------
            rnd_h = random.randint(0, max(0, height - self.patch_size))
            rnd_w = random.randint(0, max(0, width - self.patch_size))
            h_patch = h_img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            h_patch = augment_img(h_patch, mode=np.random.randint(0, 8))

            # ---------------------------
            #  2) kernel
            # ---------------------------
            r_value = np.random.randint(0, 20)
            if 4 <= r_value < 17:
                kernel = blurkernel_synthesis(h=25)  # motion blur
            else:
                kernel = gen_kernel()  # Gaussian blur
                kernel = augment_img(kernel, mode=np.random.randint(0, 8))

            kernel /= np.sum(kernel)
            # kernel transpose 180 degree.
            kernel_t = kernel[::-1, ::-1]

            # Set noise level
            # noise_level = np.random.uniform(0., self.sigma_max) / 255.0
            # noise_level = 2.55 / 255.0
            # ---------------------------
            # Low-quality image
            # ---------------------------
            # blur and down-sample
            l_img = ndimage.filters.convolve(h_patch, np.expand_dims(kernel, axis=2), mode='wrap')
            # add Gaussian noise
            # l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)
            # add JPEG compression.
            # jpeg_quality_factors = [50, 60, 70, 80, 90]
            # qf_value = np.random.randint(0, len(jpeg_quality_factors))
            # jpeg_quality_factor = jpeg_quality_factors[qf_value]
            # _, encimg = cv2.imencode('.jpg', 255.0 * l_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality_factor])
            # l_img = cv2.imdecode(encimg, 3)
            # l_img = l_img / 255.
            # -----------------------------------------------------------
            # Add impulse noise (salt&peper noise/random impulse noise)
            # -----------------------------------------------------------
            pcs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            pc_vav_indx = np.random.randint(0, len(pcs))
            pc = pcs[pc_vav_indx]
            noise_type = 'sp'
            if r_value >= 8:
                noise_type = 'sp'
            else:
                noise_type = 'rd'
            l_img = add_impulse_noise(l_img, pc=pc, noise_type=noise_type)
            l_img = l_img / 255.

            h_img = h_patch
            # print('...............................')
            # print(jpeg_quality_factor)
            # print('...............................')
        else:
            kernel = self.kernels[0, self.kernel_index].astype(np.float64)  # validation kernel
            kernel /= np.sum(kernel)
            kernel_t = kernel[::-1, ::-1]
            l_img = ndimage.filters.convolve(h_img, np.expand_dims(kernel, axis=2), mode='wrap')  # blur
            # Add impulse noise.
            pc = 0.03
            l_img = add_impulse_noise(l_img, pc=pc, noise_type='sp')
            l_img = l_img / 255.

        kernel = single2tensor3(np.expand_dims(np.float32(kernel), axis=2))
        kernel_t = single2tensor3(np.expand_dims(np.float32(kernel_t), axis=2))
        h_img, l_img = uint2tensor3(h_img), single2tensor3(l_img)

        return h_img, l_img, kernel, kernel_t

    def __len__(self):
        return len(self.high_img_paths)
