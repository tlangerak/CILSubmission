import os
import random
import shutil

SPLIT = 0.8

dir = ['test_renamed']
subsdir = ['/']

for d in dir:
    for sd in subsdir:
        wd = d+sd
        files = [file for file in os.listdir(wd) if file.endswith('.png')]
        for i, f in enumerate(files):
            print(d+sd+f)
            shutil.move(d+sd+f, d+sd+str(i).zfill(5)+".png")
