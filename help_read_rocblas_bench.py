path = 'debug.log'

file1 = open(path, 'r')
Lines = file1.readlines()
visited = set()
for line in Lines:
    if './rocblas-bench' in line.strip():
        tmp = line.strip()
        if tmp not in visited:
            visited.add(tmp)

print(len(visited))
res = list()
for vi in visited:
    tmp = vi.split()
    key = (tmp[4], tmp[6], int(tmp[8]), int(tmp[10]), int(tmp[12]))
    res.append(key)
res.sort()
for r in res:
    print(r)
