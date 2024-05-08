#!/usr/bin/env python
path = 'check_gpt2_XL_GEMM_8_gcd.log'

file1 = open(path, 'r')
Lines = file1.readlines()
visited = dict()
cnt = 0
cnt_gemm_ex = 0
cnt_gemm_strided_batched_ex = 0
visited_batch = dict()
for line in Lines:
    if './rocblas-bench' in line.strip() and 'Profiling' not in line.strip():
        if 'gemm_ex' in line.strip():
            cnt_gemm_ex += 1

            tmp = line.strip()
            tmp = line.split('rocblas-bench')
            if tmp[1] not in visited:
                visited[tmp[1]] = 1
            else:
                visited[tmp[1]] += 1

        if 'gemm_strided_batched_ex' in line.strip():
            cnt_gemm_strided_batched_ex += 1
            tmp_b = line.strip()
            tmp_b = line.split('rocblas-bench')
            if tmp_b[1] not in visited_batch:
                visited_batch[tmp_b[1]] = 1
            else:
                visited_batch[tmp_b[1]] += 1


        if tmp[1] not in visited:
            visited[tmp[1]] = 1
        else:
            visited[tmp[1]] += 1
print('cnt_gemm_ex=', cnt_gemm_ex)
print('cnt_gemm_strided_batched_ex=', cnt_gemm_strided_batched_ex)


print(len(visited))
for key in visited:
    print(key)




res = list()
for key in visited:
    tmp = key.split()
    print(key)
    key_ = (tmp[4-1], tmp[6-1], int(tmp[8-1]), int(tmp[10-1]), int(tmp[12-1]))
    tmp = (visited[key], key_)
    res.append(tmp)
res.sort(reverse=True)

for r in res:
    print(r)

print(len(visited))
'''
print(len(visited_batch))
for key in visited_batch:
    print(key)
res = list()
for key in visited_batch:
    tmp = key.split()
    #print(tmp[-9: ])
    key_ = (tmp[4-1], tmp[6-1], int(tmp[8-1]), int(tmp[10-1]), int(tmp[12-1]), int(tmp[-9]))
    tmp = (visited_batch[key], key_)
    res.append(tmp)
res.sort(reverse=True)

print(len(visited))

for r in res:
    print(r)
'''
