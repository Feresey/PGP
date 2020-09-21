#!/usr/bin/env python3

import argparse

from random import random as rd

args = argparse.ArgumentParser()

args.add_argument("--size", help="number of elements to generate",
                  action="store",
                  dest="size", default=1000,
                  type=int)
args.add_argument("-o", help="output file",
                  action="store",
                  dest="output", default="out.txt",
                  type=str)
args = args.parse_args()

with open(args.output, "w") as f:
    f.write(str(args.size)+"\n")
    for i in range(args.size):
        f.write(str(rd()*float(1<<25))+" ")
    f.write("\n")
