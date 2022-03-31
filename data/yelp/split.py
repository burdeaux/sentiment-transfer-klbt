import os
import subprocess
import random
import torch
import numpy as np
import pandas as pd

from collections import defaultdict
if __name__ == '__main__':
    save_dir = "/home/rtx2070s/exp/deep-latent-sequence-model/data/yelp"
    n_0=176000
    n_1=267000
    random_seed=random.randint(0,176000/2)
    ftrain_t = open(os.path.join(save_dir, "train_s8.txt"), "w")
    ftrain_a = open(os.path.join(save_dir, "train_s8.attr"), "w")
    i_n=n_0*0.8
    i_p=n_1*0.8
    i=0
    p=0
    with open(os.path.join(save_dir, "train_0.txt"), "r") as fin:
        for line in fin:
            p+=1
            if p>random_seed:
                ftrain_t.write(line)
                ftrain_a.write("negative\n")
                i+= 1
                if i==i_n:
                    break

    i=0
    p=0
    with open(os.path.join(save_dir, "train_1.txt"), "r") as fin:
         for line in fin:
            p+=1
            if p>random_seed:
                ftrain_t.write(line)
                ftrain_a.write("positive\n")
                i+= 1
                if i==i_p:
                    break

    ftrain_t.close()
    ftrain_a.close()