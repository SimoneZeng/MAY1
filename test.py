import numpy as np
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import random
import os, sys, shutil
import pandas as pd
import math
import re
from collections import deque

# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_ccl_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_ccl_rainbow_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_cl_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_cl_rainbow_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_rainbow_linear.py')

queue = deque(maxlen=0)
queue.append(1)
print(queue.__len__())