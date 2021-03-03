#!/usr/bin/env python3

n = 32+1
m = 32+1

arr = [[0]]*n
for i in range(n):
    arr[i] = ["**"]*m

def show():
    for i in range(n):
        for j in range(m):
            if j != 0: print(" ",end='')
            print(f"{arr[i][j]: >5}", end='')
        print()

def idx(j):
    return j + (1 if (j//16) > 0 else 0)

for i in range(n):
    for j in range(m):
        arr[i][j] = f"{(i*m+j)%16: 2}"

show()
for i in range(n):
    arr[i] = ["**"]*m

for i in range(32):
    for j in range(32):
        arr[idx(i)][idx(j)] = f"{i:2}:{j:2}"

print("================")

show()
