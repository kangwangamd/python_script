# Using readlines()
for IL, OL in [(100, 100), (2048, 128)]:
    for BS in [1, 4, 8, 16, 32, 64, 128, 256]:
        path = 'latency_llama3_tuned-' + str(BS) + '-' + str(IL) + '-' + str(OL) + '.log'
        #print(path)
        file1 = open(path, 'r')
        Lines = file1.readlines()

        count = 0
        # Strips the newline character
        for line in Lines:
            if 'Avg latency:' in line:
                tmp = line.strip()
                res = tmp.split()
                print(res[2])
