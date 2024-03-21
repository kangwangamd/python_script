#!/usr/bin/env python
path = 'sdxl_rocblas_bench_0320.log'

file1 = open(path, 'r')
Lines = file1.readlines()
visited = dict()
for line in Lines:
    if './rocblas-bench' in line.strip() and 'Profiling' not in line.strip():
        tmp = line.strip()
        if tmp not in visited:
            visited[tmp] = 1
        else:
            visited[tmp] += 1

print(len(visited))
cnt = 0
yaml = {'rocblas_function':'',
        'a_type': '',
        'b_type': '',
        'c_type': '',
        'd_type': '',
        'compute_type': '',
        'transA': '',
        'transB': '',
        'alpha': '',
        'beta': '',
        'initialization': "trig_float",
        'M': 0,
        'N': 0,
        'K': 0,
        'initialization': "trig_float",
        'cold_iters': 2,
        'iters': 10}
res = list()
for key in visited:
    cnt += 1
    if cnt <= 10:
        tmp = key.split('rocblas-bench')
        tmp = tmp[1].split()
        dict_ = {'rocblas_function': tmp[1]}
        for i in range(2, len(tmp), 2):
            if '-' in tmp[i]:
                tmp_i = tmp[i].split('-')
                dict_[tmp_i[-1]] = tmp[i+1]
            else:
                dict_[tmp[i]] = tmp[i+1]
        #print(dict_)
        for key in yaml:
            if key in dict_:
                print(key, dict_[key])
            elif key == 'transA':
                print(key, dict_['transposeA'])
            elif key == 'tranB':
                print(key, dict_['transposeB'])
            elif key in {'m', 'n', 'k'}:
                print(key, dict_[key.upper()])
            else:
                print(key, yaml[key])
