#!/bin/bash
#JSUB -q gpu
#JSUB -gpgpu 1
#JSUB -m gpu21
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J my_job
/home/usrID/.conda/envs/pytorch/bin/python deblurring.py>logger_test_Set12_2.55.txt 2>&1
