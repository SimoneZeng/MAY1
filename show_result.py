# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:04:40 2022

绘图 训练结果

@author: Skye
"""


import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 整理每个epo的数据，获得df_all_epo
cols = ['epo', 'train_step', 'position_y', 'target_lane', 'lane', 'r_avg', 'r_sum', 'dis_to_target_lane']
df_all_epo = pd.DataFrame(columns = cols)

fail_cnt = 0 # one_file中没有结果
csv_cnt = len([n for n in os.listdir('result_record_tmp')]) # 统计文件夹下的文件个数

for i in range(csv_cnt):
    if os.path.getsize(f"result_record_tmp/df_record_epo_{i}.csv") > 0:
        one_file = pd.read_csv(f"result_record_tmp/df_record_epo_{i}.csv")
        one_file = one_file.drop(index = one_file[(one_file['r']==-1)].index.tolist()) # 去掉撞墙时r为-1的记录，否则某些time step重复记录了两次
    
        if len(one_file) >= 1:
            last_line = one_file.iloc[-1,:]
            last_line['r_sum'] = one_file['r'].sum() # float
            last_line['dis_to_target_lane'] = abs(int(last_line['lane'][-1]) - last_line['target_lane']) # int
            last_line['r_avg'] = last_line['r_sum'] / (last_line['train_step']+1) # 每个回合的平均reward
            last_line = last_line[cols] # 选取需要的数据
            
            df_all_epo = df_all_epo.append(last_line)
        else:
            df_all_epo = df_all_epo.append({'epo':0, 'train_step':0,'position_y':0,
                                            'target_lane':0,'lane':0, 'r_avg':0, 'r_sum':0, 
                                            'dis_to_target_lane':0}, ignore_index=True)
    else:
        fail_cnt += 1

del i, last_line

df_all_epo.to_csv("all_final_record.csv", index = False)

# 绘图
df_all_epo = pd.read_csv("all_final_record.csv")
epo_list = df_all_epo['epo'].to_list()
r_avg_list = df_all_epo['r_avg'].to_list()
r_sum_list = df_all_epo['r_sum'].to_list()
train_step_mean = df_all_epo['train_step'].mean()
#plt.scatter(epo_list, r_avg_list)

# 2. 每100个回合算reward平均值
df_100 = pd.DataFrame(columns = ['epo', 'r_avg_100'])
for i in range(0, len(df_all_epo)-100+1):
    avg = df_all_epo.iloc[i:i+100]['r_avg'].mean() # 从i到1+100-1的r_sum平均值
    df_100 = df_100.append({'epo':i,'r_avg_100':avg},ignore_index=True)
    

x = df_100['epo'].to_list()
y = df_100['r_avg_100'].to_list()
#plt.scatter(x, y)

# 3. 计算什么时候开始学习
step_cnt = 0
start_train = 0
for i in range(len(df_all_epo)):
    step_cnt = step_cnt + df_all_epo['train_step'][i]
    if step_cnt >= 20000:
        print(i)
        start_train = i
        break

# 绘图
plt.figure(figsize=(12,6))
plt.xlabel("epo", fontsize=14) # x y轴含义
plt.ylabel("r_avg", fontsize=14)
#plt.scatter(epo_list, r_avg_list, label='每个回合的r_avg的')
#plt.scatter(x, y, marker='^', label='每100个回合的r_avg的平均值')
type1 = plt.scatter(epo_list, r_avg_list)
type2 = plt.scatter(x, y, marker='^')
plt.legend((type1, type2), ('r_avg per episode', 'average of 100 episodes\' r_avg'), fontsize=14)
plt.axvline(start_train, color='red', linewidth=2) # 开始训练的位置
plt.tick_params(labelsize=14)
plt.show()




