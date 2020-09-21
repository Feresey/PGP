#!/bin/bash

set -e

BLOCKSIZE=(1 32 128 1024)
THREADS=(1 32 128 1024)

make debug

TESTS=(1000 100000 1000000)

for i in ${TESTS[@]}; do
    ./testgen.py --size "$i" -o "$i.t"
done

for t in "${TESTS[@]}"
do
    ./main_cpu < "$t.t" > /dev/null
    for block in "${BLOCKSIZE[@]}"
    do
        for thread in "${THREADS[@]}"
        do
            ./main -threads "$thread" -blocks "$block" < "$t.t" > /dev/null
        done
    done
done