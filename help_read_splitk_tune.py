#!/usr/bin/env python
path = '4.res.log'

file1 = open(path, 'r')
Lines = file1.readlines()
visited = set()
find = 0
cnt = 0
cur_min = float('inf')
cur_min_idx = 256
for line in Lines:
    if find == 1:
        find = 0
        tmp = line.strip()
        cnt += 1
        tmp = tmp.split()
        print(tmp[-1])
        if float(tmp[-1]) < cur_min:
            cur_min = float(tmp[-1])
            cur_min_idx = 256 - cnt
        #print(255-cnt, tmp[-1])
        #if tmp not in visited:
        #    visited.add(tmp)

    if 'transA,transB' in line.strip():
        find = 1

print(cur_min_idx, cur_min)

'''
print(len(visited))
res = list()
for vi in visited:
    tmp = vi.split()
    key = (tmp[4], tmp[6], int(tmp[8]), int(tmp[10]), int(tmp[12]))
    res.append(key)
res.sort()
for r in res:
    print(r)
'''
