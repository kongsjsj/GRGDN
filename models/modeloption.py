# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Shengjiang Kong,School of mathematics and statistics, Xidian University.
Email: sjkongxd@163.com
Date: 2023/2/23 10:29
# -------------------------------------------------------------
# Modified Date: Mar 21, 2023.
# To facilitate the management of various model collections.
# -------------------------------------------------------------
"""
import torch
from models import RGDN, RGDNNonUniform, RGDNNonUniformY0


def return_model_info(model_select_name='NRL-GDN'):
    """
    """
    model_name = None
    net = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_select_name == 'rgdn_uniform':
        model_name = 'rgdn_uniform.pth'  # rgdn + drunet
        net = RGDN(num_steps=40).to(device=device)
    elif model_select_name == 'NRL_GDN':  # NRL_GDN initializer: Eq.22
        model_name = 'NRL_GDN.pth'
        net = RGDNNonUniform(num_steps=40).to(device=device)
    elif model_select_name == 'NRL_GDN_Y0': # NRL_GDN initializer: x0 = y
        model_name = 'NRL_GDN_Y0.pth'
        net = RGDNNonUniformY0(num_steps=40).to(device=device)
    else:
        print("model select name is not valid!")

    return model_name, net
