#!/bin/bash
#JSUB -q gpu
#JSUB -m gpu26
#JSUB -gpgpu "2 type=NVIDIAA100-PCIE-40GB gmem=200"
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J my_job
/home/userID/.conda/envs/pytorch/bin/python deblurring.py>logger_train_NRL-GDN.txt 2>&1
