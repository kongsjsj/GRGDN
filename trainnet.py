# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Shengjiang Kong,School of mathematics and statistics, Xidian University.
Email: sjkongxd@gmail.com.
Date: 2021/12/27 18:28.
"""

import os
import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from utils.othertools import batch_psnr, make_dirs, init_logger_ipol, Averager, truncate_image

"""
# ---------------------------------------------------------------
# train deblurring net
# ---------------------------------------------------------------
"""

def train_deblurring_nrl_gdn(model, loader_train, dataset_validation, criterion, optimizer, params):
    """
    This function is main for training the networks
    :param model: the main network model
    :param loader_train: the object of DataLoader
    :param dataset_validation: the validation of dataset
    :param criterion: loss function
    :param optimizer: the object of torch.optim
    :param params: the object of parser, contain training arguments
    :return: None
    """
    make_dirs('checkpoints')
    start_epoch = -1

    # -------------------------------------------------
    # checkpoint resume
    # -------------------------------------------------
    if params.resume:
        path_checkpoint = os.path.join('checkpoints', ''.join(params.resume_model_name))
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    writer = SummaryWriter()
    step = 0
    img_noise = None
    img_clean = None
    blur_kernel = None
    logger = init_logger_ipol(params.log_file, obj_name='train_log')

    train_loss = Averager()

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9, last_epoch=-1)
    
    # -------------------------------------------------
    # epochs iter
    # -------------------------------------------------
    for epoch in range(start_epoch + 1, params.epochs):
        # if epoch < params.milestone:
        #     current_lr = params.lr
        # else:
        #     current_lr = params.lr / 10.
        #
        # # ---------------------------------------------
        # # set learning rate
        # # ---------------------------------------------
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        # print('learning rate %f' % current_lr)

        # ---------------------------------------------
        # train every epoch
        # ---------------------------------------------
        for i, (img_clean, img_noise, blur_kernel, blur_kernel_t) in enumerate(loader_train):
            # -----------------------------------------
            # training step
            # -----------------------------------------
            model.train()
            optimizer.zero_grad()

            # noise = img_noise - img_clean

            if torch.cuda.is_available():
                img_clean, img_noise = img_clean.cuda(), img_noise.cuda()  # image patch pairs
                # Residual learning
                # noise = noise.cuda()
                blur_kernel = blur_kernel.cuda()
                blur_kernel_t = blur_kernel_t.cuda()

            # -----------------------------------------
            # forward and backward
            # -----------------------------------------
            # k_size = blur_kernel.size()[2]
            # padding_size = int((k_size / 2) * 1.5)
            # img_noise = F.pad(img_noise, [padding_size, padding_size, padding_size, padding_size], mode='replicate')
            blur_kernel_t = blur_kernel
            blur_kernel = torch.rot90(blur_kernel, 2, (-1, -2))
            img_train_out = model(img_noise, blur_kernel, blur_kernel_t)  # output size is N X C X H X W
            # img_train_out = truncate_image(img_train_out, padding_size)
            # img_train_out = img_train_out[-1]
            loss = criterion(img_train_out, img_clean) / (img_train_out.size()[0] * 2)
            train_loss.add(loss.item())
            loss.backward()

            # -----------------------------------------
            # update parameters
            # -----------------------------------------
            optimizer.step()

            # -----------------------------------------
            # net eval
            # -----------------------------------------
            model.eval()
            img_train_out = torch.clamp(model(img_noise, blur_kernel, blur_kernel_t), 0., 1.)
            # img_train_out = truncate_image(img_train_out, padding_size)
            psnr_train = batch_psnr(img_train_out, img_clean, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f"
                  % (epoch + 1, i + 1, len(loader_train), train_loss.item(), psnr_train))

            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 100 == 0:
                # Log the scalar values
                writer.add_scalar('loss', train_loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)

            step += 1

        # the end of each epoch
        scheduler.step()

        # ---------------------------------------------
        # Validating in trained every epoch
        # ---------------------------------------------
        model.eval()
        psnr_validation = 0

        with torch.no_grad():
            for k in range(len(dataset_validation)):
                img_clean_val, img_noise_val, blur_kernel_val, blur_kernel_val_t = dataset_validation[k]
                # add dim
                img_clean_val = torch.unsqueeze(img_clean_val, 0)  # 1 X C X H X W
                img_noise_val = torch.unsqueeze(img_noise_val, 0)  # 1 X C X H X W
                blur_kernel_val = torch.unsqueeze(blur_kernel_val, 0)  # 1 X 1 X H X W
                blur_kernel_val_t = torch.unsqueeze(blur_kernel_val_t, 0)  # 1 X 1 X H X W

                if torch.cuda.is_available():
                    img_clean_val, img_noise_val = img_clean_val.cuda(), img_noise_val.cuda()  # image patch pairs
                    blur_kernel_val = blur_kernel_val.cuda()
                    blur_kernel_val_t = blur_kernel_val_t.cuda()

                blur_kernel_val_t = blur_kernel_val
                blur_kernel_val = torch.rot90(blur_kernel_val, 2, (-1, -2))
                # k_size = blur_kernel_val.size()[2]
                # padding_size = int((k_size / 2) * 1.5)
                # img_noise_val = F.pad(img_noise_val, [padding_size, padding_size, padding_size, padding_size], mode='replicate')
                img_val_out = torch.clamp(model(img_noise_val, blur_kernel_val, blur_kernel_val_t), 0., 1.)
                # img_val_out = truncate_image(img_val_out, padding_size)
                psnr_validation += batch_psnr(img_val_out, img_clean_val, 1.)

            psnr_validation /= len(dataset_validation)
            logger.info("[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_validation))
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_validation))
            writer.add_scalar('PSNR on validation data', psnr_validation, epoch)

        # ---------------------------------------------
        # save the images
        # ---------------------------------------------
        img_train_out = torch.clamp(model(img_noise, blur_kernel, blur_kernel_t), 0., 1.)
        img_cleaned = make_grid(img_clean.data, nrow=8, normalize=True, scale_each=True)
        img_noised = make_grid(img_noise.data, nrow=8, normalize=True, scale_each=True)
        img_reconstructed = make_grid(img_train_out.data, nrow=8, normalize=True, scale_each=True)
        print('save clean image ..')
        writer.add_image('clean image', img_cleaned, epoch)
        print('save noisy image ..')
        writer.add_image('noisy image', img_noised, epoch)
        print('save reconstructed image ..')
        writer.add_image('reconstructed image', img_reconstructed, epoch)

        # ---------------------------------------------
        # save model
        # ---------------------------------------------
        file_name = params.model_name
        torch.save(model.state_dict(), os.path.join(params.model_save_file, ''.join(file_name)))

        # ---------------------------------------------
        # save checkpoint
        # ---------------------------------------------
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        file_name = ['ckp_', params.model_name.split('.')[0], '_%s.pth' % (str(epoch))]
        torch.save(checkpoint, os.path.join('checkpoints', ''.join(file_name)))

    writer.close()
