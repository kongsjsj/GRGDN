# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Shengjiang Kong,School of mathematics and statistics, Xidian University.
Email: sjkongxd@gmail.com.
Data: Mar 4, 2023.
Different Algorithms For Image Restoration Tasks.
"""

from abc import ABC, abstractmethod

import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from scipy import ndimage
from tqdm import tqdm
from models.modeloption import return_model_info
from trainnet import train_deblurring_nrl_gdn
from utils import generate_gauss_kernel_mix, sincos_kernel, peaks
from utils.othertools import (weights_init_kaiming,
                              batch_psnr,
                              batch_ssim,
                              init_logger_ipol,
                              imread_uint8,
                              remove_data_parallel_wrapper,
                              variable_to_cv2_image,
                              extract_file_name,
                              add_impulse_noise,
                              make_dirs,
                              truncate_image)


class AlgorithmBase(ABC):
    """
    Basic class for all algorithms
    """

    def __init__(self, params):
        self.params = params  # parameters for algorithms

    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def test(self, *args):
        pass

    @abstractmethod
    def test_real(self, *args):
        pass

    @abstractmethod
    def test_nonuniform(self, *args):
        pass

    @abstractmethod
    def test_poisson_gaussian(self, *args):
        pass

    @abstractmethod
    def test_impulse(self, *args):
        pass


"""
# ------------------------------------------------------
# Algorithm NRL-GDN for non-blind deblurring
# ------------------------------------------------------
"""


class AlgorithmNRLGDNDeblurring(AlgorithmBase):
    """
    Main implement of NRL-GDN Algorithm for Non-Blind Deblurring.
    """

    def __init__(self, params):
        super().__init__(params)

    def train(self, img_deblur_handle):
        """
        Training NRL-GDN
        :param img_deblur_handle: the object of ImageDeblur class.
        :return:
        """

        # -------------------------------------------
        # Build network
        # -------------------------------------------
        # num_channels = self.params.num_channels
        net = self.get_net()
        print(net)

        # -------------------------------------------
        # Initialize network
        # -------------------------------------------
        net.apply(weights_init_kaiming)
        # print(criterion)
        criterion = nn.L1Loss().to(device=self.params.device)

        # -------------------------------------------
        # Move to GPU
        # -------------------------------------------
        model = nn.DataParallel(net, device_ids=self.params.device_ids).cuda()
        criterion.cuda()

        # -------------------------------------------
        # Optimizer
        # -------------------------------------------
        optimizer = optim.Adam(model.parameters(), lr=self.params.lr)

        # -------------------------------------------
        # training
        # -------------------------------------------
        train_set = img_deblur_handle.train_set
        validate_set = img_deblur_handle.validate_set
        train_deblurring_nrl_gdn(model, train_set, validate_set, criterion, optimizer, self.params)

    def test(self, img_deblur_handle):
        """
        Test trained NRL-GDN
        :param : img_deblur_handle, the object of ImageDeblur class.
        :return:
        """
        print('testing on {} data ...\n'.format(self.params.test_data))

        # ------------------------------------------------------
        # 1) Initial
        # ------------------------------------------------------
        psnr_test = 0
        ssim_test = 0
        all_psnr_test = 0
        all_ssim_test = 0

        # -------------------------------------------------------
        # 2) load test dataset
        # -------------------------------------------------------
        test_set = img_deblur_handle.test_set

        # -------------------------------------------------------
        # 3) Absolute path to model file
        # -------------------------------------------------------
        net_file = self.get_net_file()
        # -------------------------------------------------------
        # 4) Create model and Load saved weights
        # -------------------------------------------------------
        num_channels = self.params.num_channels
        model = self.get_model(net_file, num_channels)
        # --------------------------------------------------------
        # load kernel shape: (1, n)
        # --------------------------------------------------------
        blur_kernels = img_deblur_handle.kernels

        concreate_dir = ''.join([self.params.test_data, '_', str(self.params.test_noiseL)])
        make_dirs(os.path.join('results', concreate_dir))

        log_all_psnr_ssim_name = os.path.join('results', concreate_dir, 'logger_summary_PSNR_SSIM_' + self.params.model_name.split('.')[0] + '.txt')
        logger_all = init_logger_ipol(file_name=log_all_psnr_ssim_name, obj_name='all')
        logger_all.info('Testing on {}\'s dataset'.format(self.params.test_data))

        for kk in range(blur_kernels.size):
            # --------------------------------------------------------
            # 7) main test for specific kernel
            # --------------------------------------------------------
            # make save directory
            make_dirs(os.path.join('results', concreate_dir, ''.join(['kernel', str(kk)])))

            # Init logger
            # Single kernel logger
            log_file_name = os.path.join('results',
                                         concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         str(kk) + '_logger_' + self.params.model_name.split('.')[0] + '.txt')
            logger = init_logger_ipol(file_name=log_file_name, obj_name=''.join(['avg', str(kk)]))

            print('*' * (len(net_file) + 10))
            print('Testing on {}th kernel'.format(kk))
            print('*' * (len(net_file) + 10))

            logger.info("\n")
            logger.info('*' * (len(net_file) + 10))
            logger.info('Testing on {}th kernel'.format(kk))
            logger.info('*' * (len(net_file) + 10))
            logger_all.info("\n")
            logger_all.info('*' * (len(net_file) + 10))
            logger_all.info('Testing on {}th kernel'.format(kk))
            logger_all.info('*' * (len(net_file) + 10))

            for f in test_set:
                np.random.seed(seed=0)  # for reproducibility
                # Open image
                img_clean = imread_uint8(f, n_channels=num_channels)  # HR image, int8,RGB
                img_clean = img_clean / 255.

                # generate degraded LR image
                blur_kernel = blur_kernels[0, kk].astype(np.float64)
                blur_kernel /= np.sum(blur_kernel)
                blur_kernel_t = blur_kernel[::-1, ::-1]
                img_blur = ndimage.filters.convolve(img_clean, np.expand_dims(blur_kernel, axis=2), mode='wrap')  # blur

                # -----------------------------------------------------------------
                # add A_W_G_N noise or signal-dependent noise(Poisson-Gaussian)
                # Modified at June 19, 2021.
                # -----------------------------------------------------------------
                test_noise_level = self.params.test_noiseL / 255.0
                img_noise = img_blur + np.random.normal(0.0, test_noise_level, img_blur.shape)
                h, w = img_noise.shape[:2]

                # expand dim
                img_clean_ = torch.from_numpy(img_clean).float().permute(2, 0, 1).unsqueeze(dim=0)
                img_noise_ = torch.from_numpy(img_noise).float().permute(2, 0, 1).unsqueeze(dim=0)
                blur_kernel_ = torch.from_numpy(np.array(blur_kernel[np.newaxis, ...])).float().unsqueeze(dim=0)
                blur_kernel_t_ = torch.from_numpy(np.array(blur_kernel_t[np.newaxis, ...])).float().unsqueeze(dim=0)

                if torch.cuda.is_available():
                    img_clean_, img_noise_, blur_kernel_, blur_kernel_t_ = \
                        [ii.cuda() for ii in [img_clean_, img_noise_, blur_kernel_, blur_kernel_t_]]

                # Test
                with torch.no_grad():  # this can save much memory
                    blur_kernel_t_ = blur_kernel_
                    blur_kernel_ = torch.rot90(blur_kernel_, 2, (-1, -2))
                    img_deblurred = torch.clamp(model(img_noise_, blur_kernel_, blur_kernel_t_), 0., 1.)

                # resize to original
                img_deblurred = img_deblurred[..., :h, :w]
                img_noise_ = img_noise_[..., :h, :w]

                # Add code in Dec 11, 2022.
                if self.params.is_gray_scale:
                    logger.info("*** Gray-scale debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))
                    img_deblurred = (img_deblurred[:, 0, :, :] * 0.299 + img_deblurred[:, 1, :, :] * 0.587 + img_deblurred[:, 2, :,:] * 0.114).unsqueeze(dim=0)
                    img_clean_ = (img_clean_[:, 0, :, :] * 0.299 + img_clean_[:, 1, :, :] * 0.587 + img_clean_[:, 2,:,:] * 0.114).unsqueeze(dim=0)
                    img_noise_ = (img_noise_[:, 0, :, :] * 0.299 + img_noise_[:, 1, :, :] * 0.587 + img_noise_[:, 2,:,:] * 0.114).unsqueeze(dim=0)
                else:
                    logger.info("*** sRGB debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))

                psnr = batch_psnr(img_deblurred, img_clean_, 1.)
                ssim = batch_ssim(img_deblurred, img_clean_)
                psnr_noisy = batch_psnr(img_noise_, img_clean_, 1.)
                ssim_noisy = batch_ssim(img_noise_, img_clean_)
                logger.info("\tPSNR deblurred {0:0.2f} dB".format(psnr))
                logger.info("\tSSIM {0:0.4f}".format(ssim))
                logger.info("\tPSNR noisy {0:0.2f} dB".format(psnr_noisy))
                logger.info("\tSSIM noisy {0:0.4f}".format(ssim_noisy))
                logger.info("-" * 60)
                logger.info("\n")

                # Save images
                img_noise_ = variable_to_cv2_image(img_noise_)
                img_deblurred = variable_to_cv2_image(img_deblurred)

                file_name = extract_file_name(f)
                file_name_noisy = '{}_{}_noisy.png'.format(concreate_dir, file_name)
                file_name_deblurred = '{}_{}_{}_deblurred.png'.format(concreate_dir,
                                                                    file_name,
                                                                    self.params.model_name)

                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_noisy), img_noise_)
                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_deblurred), img_deblurred)

                # Sum all psnr and compute avg
                psnr_test += psnr
                ssim_test += ssim
                print("%s PSNR %0.2f SSIM %0.4f" % (f, psnr, ssim))

            psnr_test /= len(test_set)
            ssim_test /= len(test_set)
            print("\nPSNR on test data {0:0.2f}dB".format(psnr_test))
            print("SSIM on test data {0:0.4f}".format(ssim_test))
            logger.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger.info("SSIM on test data {0:0.4f}\n".format(ssim_test))
            logger_all.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger_all.info("SSIM on test data {0:0.4f}".format(ssim_test))

            # Statistic all psnr and ssim
            all_psnr_test += psnr_test
            all_ssim_test += ssim_test

            # Reset psnr_test and ssim_test
            psnr_test = 0
            ssim_test = 0

        # Calculate average psnr and ssim for all kernels
        all_psnr_test /= blur_kernels.size
        all_ssim_test /= blur_kernels.size
        print('Average PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        print('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))
        logger_all.info('\nAverage PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        logger_all.info('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))

    def test_real(self, img_deblur_handle):
        """
        Test trained NRL-GDN for real-world dataset
        :param : img_deblur_handle, the object of ImageDeblur class.
        :return:
        """
        print('testing on {} data ...\n'.format(self.params.test_data))

        # -------------------------------------------------------
        # 2) load test dataset
        # -------------------------------------------------------
        test_set = img_deblur_handle.test_set
        # -------------------------------------------------------
        # 3) Absolute path to model file
        # -------------------------------------------------------
        net_file = self.get_net_file()
        # -------------------------------------------------------
        # 4) Create model and Load saved weights
        # -------------------------------------------------------
        num_channels = self.params.num_channels
        model = self.get_model(net_file, num_channels)
        # --------------------------------------------------------
        # load kernel shape: (1, n)
        # --------------------------------------------------------
        blur_kernels = img_deblur_handle.kernels
        # --------------------------------------------------------
        # 7) main test for specific kernel
        # --------------------------------------------------------
        # make save directory
        make_dirs(os.path.join('results', self.params.test_data, 'realDeblur'))

        with tqdm(total=len(test_set), desc='De-blurring', ncols=100) as bar:
            for image_name in range(len(test_set)):
                bar.update(1)
                # Open image
                f = test_set[image_name]
                img_blur = imread_uint8(f, n_channels=num_channels)  # HR image, int8,RGB
                img_blur = img_blur / 255.

                # blur kernel
                blur_kernel = cv2.imread(blur_kernels[image_name], 0)  # cv2.IMREAD_GRAYSCALE
                blur_kernel = (blur_kernel / 255.).astype(np.float64)
                blur_kernel /= np.sum(blur_kernel)
                blur_kernel_t = blur_kernel[::-1, ::-1]

                # expand dim
                img_blur_ = torch.from_numpy(img_blur).float().permute(2, 0, 1).unsqueeze(dim=0)
                blur_kernel_ = torch.from_numpy(np.array(blur_kernel[np.newaxis, ...])).float().unsqueeze(dim=0)
                blur_kernel_t_ = torch.from_numpy(np.array(blur_kernel_t[np.newaxis, ...])).float().unsqueeze(dim=0)

                if torch.cuda.is_available():
                    img_blur_, blur_kernel_, blur_kernel_t_ = [ii.cuda() for ii in
                                                               [img_blur_, blur_kernel_, blur_kernel_t_]]
                h, w = img_blur.shape[:2]

                # Test
                with torch.no_grad():  # this can save much memory
                    blur_kernel_t_ = blur_kernel_
                    blur_kernel_ = torch.rot90(blur_kernel_, 2, (-1, -2))
                    k_size = blur_kernel_.size()[2]
                    padding_size = int((k_size / 2) * 2)
                    img_blur_ = F.pad(img_blur_, [padding_size, padding_size, padding_size, padding_size], mode='replicate')
                    img_deblurred = torch.clamp(model(img_blur_, blur_kernel_, blur_kernel_t_), 0., 1.)
                    img_deblurred = truncate_image(img_deblurred, padding_size)

                # resize to original
                img_deblurred = img_deblurred[..., :h, :w]

                # Save images
                img_deblurred = variable_to_cv2_image(img_deblurred)

                file_name = extract_file_name(f)
                file_name_deblurred = '{}_{}_deblurred.png'.format(file_name, self.params.model_name)

                cv2.imwrite(os.path.join('results', self.params.test_data,
                                         'realDeblur',
                                         file_name_deblurred), img_deblurred)

    def test_nonuniform(self, img_deblur_handle):
        """
        Test trained NRL-GDN
        :param : img_deblur_handle, the object of ImageDeblur class.
        :return:
        """
        print('testing on {} data ...\n'.format(self.params.test_data))

        # ------------------------------------------------------
        # 1) Initial
        # ------------------------------------------------------
        psnr_test = 0
        ssim_test = 0
        all_psnr_test = 0
        all_ssim_test = 0

        # -------------------------------------------------------
        # 2) load test dataset
        # -------------------------------------------------------
        test_set = img_deblur_handle.test_set

        # -------------------------------------------------------
        # 3) Absolute path to model file
        # -------------------------------------------------------
        net_file = self.get_net_file()
        # -------------------------------------------------------
        # 4) Create model and Load saved weights
        # -------------------------------------------------------
        num_channels = self.params.num_channels
        model = self.get_model(net_file, num_channels)
        # --------------------------------------------------------
        # load kernel shape: (1, n)
        # --------------------------------------------------------
        blur_kernels = img_deblur_handle.kernels
        concreate_dir = ''.join([self.params.test_data, '_', str(self.params.sigma_map_type)])
        make_dirs(os.path.join('results', concreate_dir))
        log_all_psnr_ssim_name = os.path.join('results', concreate_dir, 'logger_summary_PSNR_SSIM_' +
                                              self.params.model_name.split('.')[0] + '_'
                                              + str(self.params.sigma_map_av) + '_'
                                              + str(self.params.sigma_map_index) + '.txt')
        logger_all = init_logger_ipol(file_name=log_all_psnr_ssim_name, obj_name='all')
        logger_all.info('Testing on {}\'s dataset'.format(self.params.test_data))

        for kk in range(blur_kernels.size):
            # --------------------------------------------------------
            # 7) main test for specific kernel
            # --------------------------------------------------------

            # make save directory
            make_dirs(os.path.join('results', concreate_dir, ''.join(['kernel', str(kk)])))

            # Init logger
            # Single kernel logger
            log_file_name = os.path.join('results',
                                         concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         str(kk) + '_logger_' + self.params.model_name.split('.')[0] + '_' + \
                                         str(self.params.sigma_map_av) + '_' + \
                                         str(self.params.sigma_map_index) + '.txt')
            logger = init_logger_ipol(file_name=log_file_name, obj_name=''.join(['avg', str(kk)]))

            print('*' * (len(net_file) + 10))
            print('Testing on {}th kernel'.format(kk))
            print('*' * (len(net_file) + 10))

            logger.info("\n")
            logger.info('*' * (len(net_file) + 10))
            logger.info('Testing on {}th kernel'.format(kk))
            logger.info('*' * (len(net_file) + 10))
            logger_all.info("\n")
            logger_all.info('*' * (len(net_file) + 10))
            logger_all.info('Testing on {}th kernel'.format(kk))
            logger_all.info('*' * (len(net_file) + 10))

            for f in test_set:
                np.random.seed(seed=0)  # for reproducibility

                # Open image
                img_clean = imread_uint8(f, n_channels=num_channels)  # HR image, int8,RGB
                img_clean = img_clean / 255.

                # generate degraded LR image
                blur_kernel = blur_kernels[0, kk].astype(np.float64)
                blur_kernel /= np.sum(blur_kernel)
                blur_kernel_t = blur_kernel[::-1, ::-1]
                img_blur = ndimage.filters.convolve(img_clean, np.expand_dims(blur_kernel, axis=2), mode='wrap')  # blur

                # -----------------------------------------------------------------
                # Add non-uniform gauss noise
                # Modified at Aug 25, 2022.
                # -----------------------------------------------------------------

                # ---------------------------------------------
                # Method one
                # ---------------------------------------------
                sigma_map = None
                if self.params.sigma_map_type == 'peaks':
                    sigma_map = peaks(256)
                elif self.params.sigma_map_type == 'gauss':
                    sigma_map = sincos_kernel()
                elif self.params.sigma_map_type == 'gauss_mix':
                    sigma_map = generate_gauss_kernel_mix(256, 256)

                sig_av = 5
                sigma_map = sig_av / 255.0 + (sigma_map - sigma_map.min()) / (sigma_map.max() - sigma_map.min()) * (
                            (25 - sig_av) / 255.0)

                # ---------------------------------------------
                # Method two
                # ---------------------------------------------
                # sigma_maps = loadmat(self.params.sigma_map_path)['smaps']
                # sigma_map = sigma_maps[self.params.sigma_map_av, self.params.sigma_map_index, ...].astype(np.float64)
                # sigma_map = sigma_map / 255.0

                h, w, c = img_blur.shape
                sigma_map = cv2.resize(sigma_map, (w, h))
                noise = np.random.randn(h, w, c) * sigma_map[:, :, np.newaxis]
                img_noise = img_blur + noise
                # h, w = img_noise.shape[:2]
                # expand dim
                img_clean_ = torch.from_numpy(img_clean).float().permute(2, 0, 1).unsqueeze(dim=0)
                img_noise_ = torch.from_numpy(img_noise).float().permute(2, 0, 1).unsqueeze(dim=0)
                blur_kernel_ = torch.from_numpy(np.array(blur_kernel[np.newaxis, ...])).float().unsqueeze(dim=0)
                blur_kernel_t_ = torch.from_numpy(np.array(blur_kernel_t[np.newaxis, ...])).float().unsqueeze(dim=0)

                if torch.cuda.is_available():
                    img_clean_, img_noise_, blur_kernel_, blur_kernel_t_ = \
                        [ii.cuda() for ii in [img_clean_, img_noise_, blur_kernel_, blur_kernel_t_]]

                # Test
                with torch.no_grad():  # this can save much memory
                    blur_kernel_t_ = blur_kernel_
                    blur_kernel_ = torch.rot90(blur_kernel_, 2, (-1, -2))
                    # k_size = blur_kernel_.size()[2]
                    # padding_size = int((k_size / 2) * 4)
                    # img_noise_ = F.pad(img_noise_, [padding_size, padding_size, padding_size, padding_size], mode='replicate')
                    img_deblurred = torch.clamp(model(img_noise_, blur_kernel_, blur_kernel_t_), 0., 1.)

                    # img_deblurred = truncate_image(img_deblurred, padding_size)
                    # img_noise_ = truncate_image(img_noise_, padding_size)

                # resize to original
                img_deblurred = img_deblurred[..., :h, :w]
                img_noise_ = img_noise_[..., :h, :w]

                # Add code in Dec 11, 2022.
                if self.params.is_gray_scale:
                    logger.info("*** Gray-scale debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))
                    img_deblurred = (img_deblurred[:, 0, :, :] * 0.299 + img_deblurred[:, 1, :, :] * 0.587 + img_deblurred[:, 2,:,:] * 0.114).unsqueeze(dim=0)
                    img_clean_ = (img_clean_[:, 0, :, :] * 0.299 + img_clean_[:, 1, :, :] * 0.587 + img_clean_[:, 2, :,:] * 0.114).unsqueeze(dim=0)
                    img_noise_ = (img_noise_[:, 0, :, :] * 0.299 + img_noise_[:, 1, :, :] * 0.587 + img_noise_[:, 2, :,:] * 0.114).unsqueeze(dim=0)
                else:
                    logger.info("*** sRGB debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))

                psnr = batch_psnr(img_deblurred, img_clean_, 1.)
                ssim = batch_ssim(img_deblurred, img_clean_)
                psnr_noisy = batch_psnr(img_noise_, img_clean_, 1.)
                ssim_noisy = batch_ssim(img_noise_, img_clean_)
                logger.info("\tPSNR deblurred {0:0.2f} dB".format(psnr))
                logger.info("\tSSIM {0:0.4f}".format(ssim))
                logger.info("\tPSNR noisy {0:0.2f} dB".format(psnr_noisy))
                logger.info("\tSSIM noisy {0:0.4f}".format(ssim_noisy))
                logger.info("-" * 60)
                logger.info("\n")

                # Save images
                img_noise_ = variable_to_cv2_image(img_noise_)
                img_deblurred = variable_to_cv2_image(img_deblurred)

                file_name = extract_file_name(f)
                file_name_noisy = '{}_{}_nonuniform_av{}_index{}_noisy.png'.format(concreate_dir,
                                                                                   file_name,
                                                                                   self.params.sigma_map_av,
                                                                                   self.params.sigma_map_index)
                file_name_deblurred = '{}_{}_{}_nonuniform_av{}_index{}_deblurred.png'.format(concreate_dir,
                                                                                            file_name,
                                                                                            self.params.model_name,
                                                                                            self.params.sigma_map_av,
                                                                                            self.params.sigma_map_index)

                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_noisy), img_noise_)
                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_deblurred), img_deblurred)

                # Sum all psnr and compute avg
                psnr_test += psnr
                ssim_test += ssim
                print("%s PSNR %0.2f SSIM %0.4f" % (f, psnr, ssim))

            psnr_test /= len(test_set)
            ssim_test /= len(test_set)
            print("\nPSNR on test data {0:0.2f}dB".format(psnr_test))
            print("SSIM on test data {0:0.4f}".format(ssim_test))
            logger.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger.info("SSIM on test data {0:0.4f}\n".format(ssim_test))
            logger_all.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger_all.info("SSIM on test data {0:0.4f}".format(ssim_test))

            # Statistic all psnr and ssim
            all_psnr_test += psnr_test
            all_ssim_test += ssim_test

            # Reset psnr_test and ssim_test
            psnr_test = 0
            ssim_test = 0

        # Calculate average psnr and ssim for all kernels
        all_psnr_test /= blur_kernels.size
        all_ssim_test /= blur_kernels.size
        print('Average PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        print('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))
        logger_all.info('\nAverage PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        logger_all.info('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))

    def test_poisson_gaussian(self, img_deblur_handle):
        """
        Test trained NRL-GDN
        :param : img_deblur_handle, the object of ImageDeblur class.
        :return:
        """
        print('testing on {} data ...\n'.format(self.params.test_data))

        # ------------------------------------------------------
        # 1) Initial
        # ------------------------------------------------------
        psnr_test = 0
        ssim_test = 0
        all_psnr_test = 0
        all_ssim_test = 0

        # -------------------------------------------------------
        # 2) load test dataset
        # -------------------------------------------------------
        test_set = img_deblur_handle.test_set

        # -------------------------------------------------------
        # 3) Absolute path to model file
        # -------------------------------------------------------
        net_file = self.get_net_file()
        # -------------------------------------------------------
        # 4) Create model and Load saved weights
        # -------------------------------------------------------
        num_channels = self.params.num_channels
        model = self.get_model(net_file, num_channels)
        # --------------------------------------------------------
        # load kernel shape: (1, n)
        # --------------------------------------------------------
        blur_kernels = img_deblur_handle.kernels
        # Poisson noise with different peak values.
        peak_val = 128  # 256/512/1024
        concreate_dir = ''.join([self.params.test_data, '_', str(self.params.test_noiseL), '_', str(peak_val)])
        make_dirs(os.path.join('results', concreate_dir))

        log_all_psnr_ssim_name = os.path.join('results', concreate_dir, 'logger_summary_PSNR_SSIM_' + self.params.model_name.split('.')[0] + '.txt')
        logger_all = init_logger_ipol(file_name=log_all_psnr_ssim_name, obj_name='all')
        logger_all.info('Testing on {}\'s dataset'.format(self.params.test_data))

        for kk in range(blur_kernels.size):
            # --------------------------------------------------------
            # 7) main test for specific kernel
            # --------------------------------------------------------
            # make save directory
            make_dirs(os.path.join('results', concreate_dir, ''.join(['kernel', str(kk)])))

            # Init logger
            # Single kernel logger
            log_file_name = os.path.join('results',
                                         concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         str(kk) + '_logger_' + self.params.model_name.split('.')[0] + '.txt')
            logger = init_logger_ipol(file_name=log_file_name, obj_name=''.join(['avg', str(kk)]))

            print('*' * (len(net_file) + 10))
            print('Testing on {}th kernel'.format(kk))
            print('*' * (len(net_file) + 10))

            logger.info("\n")
            logger.info('*' * (len(net_file) + 10))
            logger.info('Testing on {}th kernel'.format(kk))
            logger.info('*' * (len(net_file) + 10))
            logger_all.info("\n")
            logger_all.info('*' * (len(net_file) + 10))
            logger_all.info('Testing on {}th kernel'.format(kk))
            logger_all.info('*' * (len(net_file) + 10))

            for f in test_set:
                np.random.seed(seed=0)  # for reproducibility
                # Open image
                img_clean = imread_uint8(f, n_channels=num_channels)  # HR image, int8,RGB
                img_clean = img_clean / 255.

                # generate degraded LR image
                blur_kernel = blur_kernels[0, kk].astype(np.float64)
                blur_kernel /= np.sum(blur_kernel)
                blur_kernel_t = blur_kernel[::-1, ::-1]
                img_blur = ndimage.filters.convolve(img_clean, np.expand_dims(blur_kernel, axis=2), mode='wrap')  # blur

                # -----------------------------------------------------------------
                # signal-dependent noise (Poisson-Gaussian)
                # Modified at June 19, 2021.
                # Ref: Alessandro Foi. Clipped Noisy Images: Heteroskedastic Modeling and Practical Denoising.
                # Signal Processing, 89(12):2609–2629, 2009.
                # -----------------------------------------------------------------
                test_noise_level = self.params.test_noiseL / 255.0
                chi = peak_val
                rng = np.random.default_rng()
                z = rng.poisson(chi * img_blur) / chi + np.random.normal(0.0, test_noise_level, img_blur.shape)
                img_noise = np.maximum(0, np.minimum(1, z))

                h, w = img_noise.shape[:2]

                # expand dim
                img_clean_ = torch.from_numpy(img_clean).float().permute(2, 0, 1).unsqueeze(dim=0)
                img_noise_ = torch.from_numpy(img_noise).float().permute(2, 0, 1).unsqueeze(dim=0)
                blur_kernel_ = torch.from_numpy(np.array(blur_kernel[np.newaxis, ...])).float().unsqueeze(dim=0)
                blur_kernel_t_ = torch.from_numpy(np.array(blur_kernel_t[np.newaxis, ...])).float().unsqueeze(dim=0)

                if torch.cuda.is_available():
                    img_clean_, img_noise_, blur_kernel_, blur_kernel_t_ = \
                        [ii.cuda() for ii in [img_clean_, img_noise_, blur_kernel_, blur_kernel_t_]]

                # Test
                with torch.no_grad():  # this can save much memory
                    blur_kernel_t_ = blur_kernel_
                    blur_kernel_ = torch.rot90(blur_kernel_, 2, (-1, -2))
                    img_deblurred = torch.clamp(model(img_noise_, blur_kernel_, blur_kernel_t_), 0., 1.)

                # resize to original
                img_deblurred = img_deblurred[..., :h, :w]
                img_noise_ = img_noise_[..., :h, :w]

                # Add code in Dec 11, 2022.
                if self.params.is_gray_scale:
                    logger.info("*** Gray-scale debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))
                    img_deblurred = (img_deblurred[:, 0, :, :] * 0.299 + img_deblurred[:, 1, :, :] * 0.587 + img_deblurred[:, 2, :,:] * 0.114).unsqueeze(dim=0)
                    img_clean_ = (img_clean_[:, 0, :, :] * 0.299 + img_clean_[:, 1, :, :] * 0.587 + img_clean_[:, 2,:,:] * 0.114).unsqueeze(dim=0)
                    img_noise_ = (img_noise_[:, 0, :, :] * 0.299 + img_noise_[:, 1, :, :] * 0.587 + img_noise_[:, 2,:,:] * 0.114).unsqueeze(dim=0)
                else:
                    logger.info("*** sRGB debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))

                psnr = batch_psnr(img_deblurred, img_clean_, 1.)
                ssim = batch_ssim(img_deblurred, img_clean_)
                psnr_noisy = batch_psnr(img_noise_, img_clean_, 1.)
                ssim_noisy = batch_ssim(img_noise_, img_clean_)
                logger.info("\tPSNR deblurred {0:0.2f} dB".format(psnr))
                logger.info("\tSSIM {0:0.4f}".format(ssim))
                logger.info("\tPSNR noisy {0:0.2f} dB".format(psnr_noisy))
                logger.info("\tSSIM noisy {0:0.4f}".format(ssim_noisy))
                logger.info("-" * 60)
                logger.info("\n")

                # Save images
                img_noise_ = variable_to_cv2_image(img_noise_)
                img_deblurred = variable_to_cv2_image(img_deblurred)

                file_name = extract_file_name(f)
                file_name_noisy = '{}_{}_noisy.png'.format(concreate_dir, file_name)
                file_name_deblurred = '{}_{}_{}_deblurred.png'.format(concreate_dir,
                                                                    file_name,
                                                                    self.params.model_name)

                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_noisy), img_noise_)
                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_deblurred), img_deblurred)

                # Sum all psnr and compute avg
                psnr_test += psnr
                ssim_test += ssim
                print("%s PSNR %0.2f SSIM %0.4f" % (f, psnr, ssim))

            psnr_test /= len(test_set)
            ssim_test /= len(test_set)
            print("\nPSNR on test data {0:0.2f}dB".format(psnr_test))
            print("SSIM on test data {0:0.4f}".format(ssim_test))
            logger.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger.info("SSIM on test data {0:0.4f}\n".format(ssim_test))
            logger_all.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger_all.info("SSIM on test data {0:0.4f}".format(ssim_test))

            # Statistic all psnr and ssim
            all_psnr_test += psnr_test
            all_ssim_test += ssim_test

            # Reset psnr_test and ssim_test
            psnr_test = 0
            ssim_test = 0

        # Calculate average psnr and ssim for all kernels
        all_psnr_test /= blur_kernels.size
        all_ssim_test /= blur_kernels.size
        print('Average PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        print('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))
        logger_all.info('\nAverage PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        logger_all.info('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))

    def test_impulse(self, img_deblur_handle):
        print('test_noise_intensity...{}'.format(self.params.test_noise_intensity))
        """
        Test trained NRL-GDN
        :param : img_deblur_handle, the object of ImageDeblur class.
        :return:
        """
        print('testing on {} data ...\n'.format(self.params.test_data))
        # ------------------------------------------------------
        # 1) Initial
        # ------------------------------------------------------
        psnr_test = 0
        ssim_test = 0
        all_psnr_test = 0
        all_ssim_test = 0

        # -------------------------------------------------------
        # 2) load test dataset
        # -------------------------------------------------------
        test_set = img_deblur_handle.test_set

        # -------------------------------------------------------
        # 3) Absolute path to model file
        # -------------------------------------------------------
        net_file = self.get_net_file()
        # -------------------------------------------------------
        # 4) Create model and Load saved weights
        # -------------------------------------------------------
        num_channels = self.params.num_channels
        model = self.get_model(net_file, num_channels)
        # --------------------------------------------------------
        # load kernel shape: (1, n)
        # --------------------------------------------------------
        blur_kernels = img_deblur_handle.kernels

        concreate_dir = ''.join([self.params.test_data, '_', str(self.params.test_noise_intensity)])
        make_dirs(os.path.join('results', concreate_dir))

        log_all_psnr_ssim_name = os.path.join('results', concreate_dir,
                                              'logger_summary_PSNR_SSIM_' + self.params.model_name.split('.')[
                                                  0] + '.txt')
        logger_all = init_logger_ipol(file_name=log_all_psnr_ssim_name, obj_name='all')
        logger_all.info('Testing on {}\'s dataset'.format(self.params.test_data))

        for kk in range(blur_kernels.size):
            # --------------------------------------------------------
            # 7) main test for specific kernel
            # --------------------------------------------------------

            # make save directory
            make_dirs(os.path.join('results', concreate_dir, ''.join(['kernel', str(kk)])))

            # Init logger
            # Single kernel logger
            log_file_name = os.path.join('results',
                                         concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         str(kk) + '_logger_' + self.params.model_name.split('.')[0] + '.txt')
            logger = init_logger_ipol(file_name=log_file_name, obj_name=''.join(['avg', str(kk)]))

            print('*' * (len(net_file) + 10))
            print('Testing on {}th kernel'.format(kk))
            print('*' * (len(net_file) + 10))

            logger.info("\n")
            logger.info('*' * (len(net_file) + 10))
            logger.info('Testing on {}th kernel'.format(kk))
            logger.info('*' * (len(net_file) + 10))
            logger_all.info("\n")
            logger_all.info('*' * (len(net_file) + 10))
            logger_all.info('Testing on {}th kernel'.format(kk))
            logger_all.info('*' * (len(net_file) + 10))

            for f in test_set:
                np.random.seed(seed=0)  # for reproducibility

                # Open image
                img_clean = imread_uint8(f, n_channels=num_channels)  # HR image, int8,RGB
                img_clean = img_clean / 255.

                # generate degraded LR image
                blur_kernel = blur_kernels[0, kk].astype(np.float64)
                blur_kernel /= np.sum(blur_kernel)
                blur_kernel_t = blur_kernel[::-1, ::-1]
                img_blur = ndimage.filters.convolve(img_clean, np.expand_dims(blur_kernel, axis=2), mode='wrap')  # blur
                # -----------------------------------------------------------------
                # Add impulse noise with noise_level_intensity
                # -----------------------------------------------------------------
                pc = self.params.test_noise_intensity
                img_noise = add_impulse_noise(255 * img_blur, pc=pc, noise_type='rd')
                img_noise = img_noise / 255.

                h, w = img_noise.shape[:2]
                # expand dim
                img_clean_ = torch.from_numpy(img_clean).float().permute(2, 0, 1).unsqueeze(dim=0)
                img_noise_ = torch.from_numpy(img_noise).float().permute(2, 0, 1).unsqueeze(dim=0)
                blur_kernel_ = torch.from_numpy(np.array(blur_kernel[np.newaxis, ...])).float().unsqueeze(dim=0)
                blur_kernel_t_ = torch.from_numpy(np.array(blur_kernel_t[np.newaxis, ...])).float().unsqueeze(dim=0)

                if torch.cuda.is_available():
                    img_clean_, img_noise_, blur_kernel_, blur_kernel_t_ = \
                        [ii.cuda() for ii in [img_clean_, img_noise_, blur_kernel_, blur_kernel_t_]]

                # Test
                with torch.no_grad():  # this can save much memory
                    blur_kernel_t_ = blur_kernel_
                    blur_kernel_ = torch.rot90(blur_kernel_, 2, (-1, -2))
                    # k_size = blur_kernel_.size()[2]
                    # padding_size = int((k_size / 2) * 2)
                    # img_noise_ = F.pad(img_noise_, [padding_size, padding_size, padding_size, padding_size],
                    #                    mode='replicate')
                    img_deblurred = torch.clamp(model(img_noise_, blur_kernel_, blur_kernel_t_), 0., 1.)

                    # img_deblurred = truncate_image(img_deblurred, padding_size)
                    # img_noise_ = truncate_image(img_noise_, padding_size)

                # resize to original
                img_deblurred = img_deblurred[..., :h, :w]
                img_noise_ = img_noise_[..., :h, :w]

                # Add code in Dec 11, 2022.
                if self.params.is_gray_scale:
                    logger.info("*** Gray-scale debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))
                    img_deblurred = (
                                img_deblurred[:, 0, :, :] * 0.299 + img_deblurred[:, 1, :, :] * 0.587 + img_deblurred[:, 2,
                                                                                                      :,
                                                                                                      :] * 0.114).unsqueeze(
                        dim=0)
                    img_clean_ = (img_clean_[:, 0, :, :] * 0.299 + img_clean_[:, 1, :, :] * 0.587 + img_clean_[:, 2, :,
                                                                                                    :] * 0.114).unsqueeze(
                        dim=0)
                    img_noise_ = (img_noise_[:, 0, :, :] * 0.299 + img_noise_[:, 1, :, :] * 0.587 + img_noise_[:, 2, :,
                                                                                                    :] * 0.114).unsqueeze(
                        dim=0)
                else:
                    logger.info("*** sRGB debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))

                psnr = batch_psnr(img_deblurred, img_clean_, 1.)
                ssim = batch_ssim(img_deblurred, img_clean_)
                psnr_noisy = batch_psnr(img_noise_, img_clean_, 1.)
                ssim_noisy = batch_ssim(img_noise_, img_clean_)
                logger.info("\tPSNR deblurred {0:0.2f} dB".format(psnr))
                logger.info("\tSSIM {0:0.4f}".format(ssim))
                logger.info("\tPSNR noisy {0:0.2f} dB".format(psnr_noisy))
                logger.info("\tSSIM noisy {0:0.4f}".format(ssim_noisy))
                logger.info("-" * 60)
                logger.info("\n")

                # Save images
                img_noise_ = variable_to_cv2_image(img_noise_)
                img_deblurred = variable_to_cv2_image(img_deblurred)

                file_name = extract_file_name(f)
                file_name_noisy = '{}_{}_noisy.png'.format(concreate_dir, file_name)
                file_name_deblurred = '{}_{}_{}_deblurred.png'.format(concreate_dir,
                                                                    file_name,
                                                                    self.params.model_name)

                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_noisy), img_noise_)
                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_deblurred), img_deblurred)

                # Sum all psnr and compute avg
                psnr_test += psnr
                ssim_test += ssim
                print("%s PSNR %0.2f SSIM %0.4f" % (f, psnr, ssim))

            psnr_test /= len(test_set)
            ssim_test /= len(test_set)
            print("\nPSNR on test data {0:0.2f}dB".format(psnr_test))
            print("SSIM on test data {0:0.4f}".format(ssim_test))
            logger.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger.info("SSIM on test data {0:0.4f}\n".format(ssim_test))
            logger_all.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger_all.info("SSIM on test data {0:0.4f}".format(ssim_test))

            # Statistic all psnr and ssim
            all_psnr_test += psnr_test
            all_ssim_test += ssim_test

            # Reset psnr_test and ssim_test
            psnr_test = 0
            ssim_test = 0

        # Calculate average psnr and ssim for all kernels
        all_psnr_test /= blur_kernels.size
        all_ssim_test /= blur_kernels.size
        print('Average PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        print('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))
        logger_all.info('\nAverage PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        logger_all.info('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))

    def get_net(self):
        _, net = return_model_info(model_select_name=self.params.model_select)
        return net

    def get_net_file(self):
        file_name = str(self.params.model_name)
        net_file = os.path.join(self.params.model_save_file, ''.join(file_name))
        print('-' * (len(net_file) + 10))
        print('net file: {}'.format(net_file))
        print('-' * (len(net_file) + 10))

        return net_file

    def get_model(self, net_file, num_channels):
        net = self.get_net()
        device_ids = [0]

        # -------------------------------------------------------
        # 5) Load saved weights
        # -------------------------------------------------------
        if torch.cuda.is_available():
            state_dict = torch.load(net_file)
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
        else:
            state_dict = torch.load(net_file, map_location=torch.device('cpu'))
            # CPU mode: remove the DataParallel wrapper
            state_dict = remove_data_parallel_wrapper(state_dict)
            model = net

        model.load_state_dict(state_dict)

        # -------------------------------------------------------
        # 6) Sets the model in evaluation mode (e.g. it removes BN)
        # -------------------------------------------------------
        model.eval()

        return model
