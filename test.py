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

def get_dataset(record_dir):
    cols = ['stage', 'epo', 'train_step', 'position_y', 'target_direc', 'lane','speed', 
            'lc_int', 'fact_acc', 'acc', 'change_lane', 'ttc', 'tail_car_acc', 'r', 
            'r_safe', 'r_eff', 'r_com', 'r_tl', 'r_fluc', 'other_record', 'done',
            's', 's_']
    df_all = pd.DataFrame(columns = cols)
    for file_name in os.listdir(record_dir):
        if re.search(r'df_record_epo', file_name):
            one_file = pd.read_csv(f"{record_dir}/{file_name}")
            if len(one_file)>=3:
                df_all=pd.concat([df_all, one_file], axis=0)
                #print(df_all.info)
    
    return df_all

tl_list = [[0,1,0,0,0,0,1], [1,1,0,1,1,1,0], [1,0,1,1,0,0,0]]
df_all=get_dataset('./dataset').sample(128)
for i in df_all['lc_int']:
    print(tl_list[i])