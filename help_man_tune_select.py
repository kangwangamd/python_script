#!/usr/bin/env python
'''
help_man_tune.py
1. Input: man_tune_res.log, the log file from rocblas-bench
2. Do: 
    (1) get the min_time from all the rocblas-bench (cmd, time) pair
    (2) compare from rocblas (cmd, time) to best time from hipblaslt
3. Output: tuned-8.csv the manual tuned result of problem ('N', 'N', 28672, 65536, 7168) in tuned.csv format for further combination use
'''
import csv
path = '74.log'

file1 = open(path, 'r')
Lines = file1.readlines()
visited = set()
cur_min = float('inf')
cnt = 0
find = 0
track = list()
for line in Lines:
    if find == 1:
        find = 0
        tmp = line.strip()
        tmp = tmp.split(',')
        track.append(float(tmp[-1]))
        cnt += 1
        if float(tmp[-1]) < cur_min:
            cur_min = float(tmp[-1])
            cur_idx = cnt

    if 'hipblaslt-Gflops' in line.strip():
        find = 1


print(cnt, cur_min)
for tc in track:
    print(tc)

'''
    if './rocblas-bench' in line.strip():
        tmp = line.strip()
        tmp = tmp.split()
        solidx = int(tmp[-3])
'''

'''
libtype = 'rocblas'
with open('tuned_hip_only.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        if i > 0 and cur_min > float(row[-2]):
            cur_idx = int(row[-3])
            cur_min = float(row[-2])
            libtype = 'hipblaslt'
'''
'''
with open('tuned-8.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    row = ['','TA','TB','M','N','K','libtype','solidx','soltimems','dtype']
    writer.writerow(row)
    if libtype != 'rocblas':
        row = ['0', 'N', 'N', 28672, 65536, 7168, 'rocblas', cur_idx, cur_min, 'torch.float16']
    else:
        row = ['0', 'N', 'N', 28672, 65536, 7168, 'hipblaslt', cur_idx, cur_min, 'torch.float16']

    writer.writerow(row)
    
print('res=', cur_idx, cur_min)
'''
