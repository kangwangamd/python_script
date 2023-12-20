#!/usr/bin/env python
import csv
profile_path = 'blas_problem03.stats.csv'

def get_csv_name_calls(file_path):
    res = dict()
    with open(file_path, newline='') as csvfile:
        profreader = csv.reader(csvfile, delimiter=',')
        next(profreader, None)
        for row in profreader:
            key = row[0]
            calls = int(row[1])
            duration = int(row[2])
            res[key] = [calls, duration]
    return res


def update_data(file1, file2):
    stats= dict()
    with open(file1, newline='') as csvfile:
        profreader = csv.reader(csvfile, delimiter=',')
        next(profreader, None)
        for row in profreader:
            key = row[0]
            calls = int(row[1])
            TotalDuration = int(row[2])
            stats[key] = [calls, TotalDuration]
    with open(file2, newline='') as csvfile:
        profreader = csv.reader(csvfile, delimiter=',')
        next(profreader, None)
        for row in profreader:
            key = row[0]
            calls = int(row[1])
            TotalDuration = int(row[2])
            stats[key][0] += calls
            stats[key][1] += TotalDuration
    res = dict()
    for key, val in stats.items():
        res[key] = val[1] / val[0]
    return res

import collections
def classify(prof):
    res = {'RCCL': [0, 0],
            'GEMM': [0, 0],
            'Flash_attention': [0, 0],
            'element_wise': [0, 0],
            'reduce_kernel': [0, 0],
            'layer_norm': [0, 0],
            'CatArrayBatchedCopy': [0, 0],
            'barrier': [0, 0],
            'others': [0, 0]
            }
    for key, val in prof.items():
        call, duration = val
        if 'ccl' in key.lower():
            res['RCCL'][0] += call
            res['RCCL'][1] += duration
        elif 'cijk' in key.lower():
            res['GEMM'][0] += call
            res['GEMM'][1] += duration
        elif 'gemm_softmax_gemm' in key.lower():
            res['Flash_attention'][0] += call
            res['Flash_attention'][1] += duration
        elif 'elementwise' in key.lower():
            res['element_wise'][0] += call
            res['element_wise'][1] += duration
        #elif 'reduce_kernel' in key.lower():
        #    res['reduce_kernel'][0] += call
        #    res['reduce_kernel'][1] += duration
        elif 'layer_norm' in key.lower():
            res['layer_norm'][0] += call
            res['layer_norm'][1] += duration
        elif 'barrier' in key.lower():
            res['barrier'][0] += call
            res['barrier'][1] += duration
        elif 'catarraybatchedcopy' in key.lower():
            res['CatArrayBatchedCopy'][0] += call
            res['CatArrayBatchedCopy'][1] += duration
        else:
            res['others'][0] += call
            res['others'][1] += duration
    return res

prof1_calls = get_csv_name_calls(profile_path)
print(prof1_calls)
order = ['RCCL', 'GEMM', 'Flash_attention', 'element_wise', 'layer_norm', 'CatArrayBatchedCopy', 'barrier', 'others'] # 'reduce_kernel',

for key in order:
    val = classify(prof1_calls)[key]
    print(key, val[0], val[1])


'''
prof2_calls = get_csv_name_calls(path + profile_2)
for key in order:
    val = classify(prof2_calls)[key]
    print(key, val[0], val[1])
'''

'''
new_stats = update_data(path + profile_1, path + profile_2)

#print(len(new_stats))
# usd prof_with_2_batch remove the one in prof_with_1batch
for key in prof2_calls:
    if key not in prof1_calls:
        print('ERROR')
    else:
        val_2 = prof2_calls[key]
        val_1 = prof1_calls[key]
        if val_2 >= val_1:
            prof2_calls[key] = val_2 - val_1

one_batch_no_overlap = {
        'RCCL': 0.0,
        'GEMM': 0.0,
        'Flash_attention': 0.0,
        'element_wise': 0.0,
        'reduce_kernel': 0.0,
        'layer_norm': 0.0,
        'others': 0.0
        }

for key, calls in prof2_calls.items():
    #one_batch_no_overlap[key] = new_stats[key] * calls
    if 'ccl' in key.lower():
        one_batch_no_overlap['RCCL'] += new_stats[key] * calls
    elif 'cijk' in key.lower():
        one_batch_no_overlap['GEMM'] += new_stats[key] * calls
    elif 'gemm_softmax_gemm' in key.lower():
        one_batch_no_overlap['Flash_attention'] += new_stats[key] * calls
    elif 'elementwise' in key.lower():
        one_batch_no_overlap['element_wise'] += new_stats[key] * calls
    elif 'reduce_kernel' in key.lower():
        one_batch_no_overlap['reduce_kernel'] += new_stats[key] * calls
    elif 'layer_norm' in key.lower():
        one_batch_no_overlap['layer_norm'] += new_stats[key] * calls
    elif 'barrier' not in key.lower():
        one_batch_no_overlap['others'] += new_stats[key] * calls
    else:
        continue
        # print('only barrier can go HERE!!')
        # print('key=', key)
total_time = sum(one_batch_no_overlap.values())
print(total_time)
for key, val in one_batch_no_overlap.items():
    print(key, val, val/total_time)
'''
