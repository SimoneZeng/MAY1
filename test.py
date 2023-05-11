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

# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_ccl_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_ccl_rainbow_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_cl_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_cl_rainbow_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_linear.py')
# os.system('python ./trainer/pdqn_train_5l_tlr_fluc_rainbow_linear.py')

def temp():
    return 1, 2


# T= torch.tensor([[[1,1,1],[2,2,2],[3,3,3]],
#                  [[4,4,4],[5,5,5],[6,6,6]]])

# 待匹配的字符串
text = "This is a_rainbow_linear_test string."

# 正则表达式模式
pattern = r"linear"

# 使用re模块的search函数进行匹配
match = re.search(pattern, text)

# 如果找到匹配项，输出其位置和匹配的字符串
if match:
    print("Match found at position %d: %s" % (match.start(), match.group()))
else:
    print("No match found.")
