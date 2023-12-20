#!/usr/bin/env python
path = 'bloom7b_rocblas_6.log'

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

res = list()
for key in visited:
    tmp = key.split()
    print(tmp)
    key_ = (tmp[4], tmp[6], int(tmp[8]), int(tmp[10]), int(tmp[12]))
    tmp = (visited[key], key_)
    res.append(tmp)
res.sort(reverse=True)

for r in res:
    print(r)

print(len(visited))
