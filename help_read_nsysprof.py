#!/usr/bin/env python
import csv
# Python code to
# demonstrate readlines()


# Using readlines()
file1 = open('tune_sdxl_ds_finetune_cuda_mts20.log', 'r')
Lines = file1.readlines()

count = 0
# Strips the newline character
stats = {'GEMM': [0.0, 0.0],
         'NCCL': [0.0, 0.0],
         'elementwise': [0.0, 0.0],
         'flash_atten': [0.0, 0.0],
         'softmax': [0.0, 0.0],
         'others': [0.0, 0.0]
         }

for line in Lines:
    count += 1
    if count >= 3:
        #print("Line{}: {}".format(count, line.strip()))
        tmp = line.strip()
        left, kernal_name = tmp[:90], tmp[90:]
        tmp = left.split()
        if 'gemm' in kernal_name:
            stats['GEMM'][0] += float(tmp[0])
            stats['GEMM'][1] += float(tmp[1])

        elif 'nccl' in kernal_name:
            stats['NCCL'][0] += float(tmp[0])
            stats['NCCL'][1] += float(tmp[1])

        elif 'elementwise' in kernal_name:
            stats['elementwise'][0] += float(tmp[0])
            stats['elementwise'][1] += float(tmp[1])

        elif 'flash' in kernal_name:
            stats['flash_atten'][0] += float(tmp[0])
            stats['flash_atten'][1] += float(tmp[1])

        elif 'softmax' in kernal_name:
            stats['softmax'][0] += float(tmp[0])
            stats['softmax'][1] += float(tmp[1])

        else:
            stats['others'][0] += float(tmp[0])
            stats['others'][1] += float(tmp[1])

for key, val in stats.items():
    print(key, val[0], val[1])
