#!/usr/bin/env python
import os

mapping = dict()

def read_all_files_in_folder(folder_path):
    # Loop through all the files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if 'batch' in file_path:
            tmp = file_path.split('_')
            print()
            bs = tmp[3].split('batch')
            bs = int(bs[1])
            input_ = tmp[4].split('input')
            input_ = int(input_[1])
            output_ = tmp[5].split('output')
            output_ = int(output_[1])
            tp = tmp[6].split('tp')
            tp = tp[1].split('.txt')
            tp = int(tp[0])
            key = (bs, input_, output_, tp)
            val = file_path
            mapping[key] = val



read_all_files_in_folder('./')

print(mapping.keys())
print((1, 128, 128, 8) in mapping)
for tp in [8, 4]:
    for il in [128, 2048]:
        for ol in [128, 2048]:
            for bs in range(1, 9):
                if (2**bs, il, ol, tp) in mapping:
                    path = mapping[(2**bs, il, ol, tp)]
                    print((2**bs, il, ol, tp))

                    file1 = open(path, 'r')
                    Lines = file1.readlines()
                    cnt = 0
                    for line in Lines:
                        if 'median latency' in line.strip():
                            cnt += 1
                            tmp = line.strip()
                            tmp = tmp.split()
                            if cnt % 2 == 0:
                                print(float(tmp[3]) * 1000)
