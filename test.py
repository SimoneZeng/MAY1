import numpy as np
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import random
import os, sys, shutil
import pandas as pd
import math

# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_ccl_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_ccl_rainbow_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_cl_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_cl_rainbow_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_rainbow_linear.py')

for i in range(5):
    print(i)
T= torch.tensor([[[1,1,1],[2,2,2],[3,3,3]],
                 [[4,4,4],[5,5,5],[6,6,6]]])
print(T.shape)
print(T.permute((1,0,2)))
print(T.permute((1,0,2)).shape)
print(T.reshape(2,3))