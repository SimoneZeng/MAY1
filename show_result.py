# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:04:40 2022

绘图 训练结果
统计每个epo的最终结果 last_line，拼接成所有回合的大表 df_all_epo
在统计df_all_epo的一些观测值，例如每100个回合算reward平均值，获取观测值随着不同回合的变化

@author: Skye
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


# 1. 整理每个epo的数据，获得df_all_epo
#cols = ['epo', 'train_step', 'position_y', 'target_lane', 'lane', 'r_avg', 'r_sum', 'dis_to_target_lane']
#df_all_epo = pd.DataFrame(columns = cols)
#
#fail_cnt = 0 # one_file中没有结果
#csv_cnt = len([n for n in os.listdir('./1012/result_record_sp8_ttc_co100')]) # 统计文件夹下的文件个数
#
#for i in range(csv_cnt-2): # 减去两个模型保存文件的数量
#    if os.path.getsize(f"./1012/result_record_sp8_ttc_co100") > 0:
#        one_file = pd.read_csv(f"./1012/result_record_sp8_ttc_co100/df_record_epo_{i}.csv")
#        one_file = one_file.drop(index = one_file[(one_file['r']==-1)].index.tolist()) # 去掉撞墙时r为-1的记录，否则某些time step重复记录了两次
#        one_file = one_file.drop(index = one_file[(one_file['r']==-3)].index.tolist()) # 去掉最后一条为-3的撞墙记录
#    
#        if len(one_file) >= 1:
#            last_line = one_file.iloc[-1,:]
#            last_line['r_sum'] = one_file['r'].sum() # float
#            last_line['dis_to_target_lane'] = abs(int(last_line['lane'][-1]) - last_line['target_lane']) # int
#            last_line['r_avg'] = last_line['r_sum'] / (last_line['train_step']+1) # 每个回合的平均reward
#            last_line = last_line[cols] # 选取需要的数据
#            
#            df_all_epo = df_all_epo.append(last_line)
#        else:
#            df_all_epo = df_all_epo.append({'epo':0, 'train_step':0,'position_y':0,
#                                            'target_lane':0,'lane':0, 'r_avg':0, 'r_sum':0, 
#                                            'dis_to_target_lane':0}, ignore_index=True)
#    else:
#        fail_cnt += 1
#
#del i, last_line
#
#df_all_epo.to_csv("result_record_sp8_thw_co3/all_final_record.csv", index = False)

# 绘图
df_all_epo = pd.read_csv("./1015/result_record_sp8_ttc_co100/all_final_record.csv")
df_all_epo = df_all_epo.drop(index = df_all_epo[(df_all_epo['position_y']==0)].index.tolist()) # 去掉没有行驶到1100m的


# 2. 每100个回合算reward平均值
df_100 = pd.DataFrame(columns = ['epo', 'r_avg_100', 'y_avg_100'])
for i in range(0, len(df_all_epo)-100+1):
    r_avg = df_all_epo.iloc[i:i+100]['r_avg'].mean() # 从i到1+100-1的 r_avg 平均值
    y_avg = df_all_epo.iloc[i:i+100]['position_y'].mean()
    df_100 = df_100.append({'epo':i,'r_avg_100':r_avg, 'y_avg_100': y_avg},ignore_index=True)
    

# 3. 计算什么时候开始学习
step_cnt = 0
start_train = 0 # 开始训练的位置
#for i in range(len(df_all_epo)):  有些epo被删去了，只能按照epo中的来
for i in df_all_epo['epo'].to_list():
#    step_cnt = step_cnt + df_all_epo['train_step'][i] # df_all_epo['train_step'][0]这里有问题，返回object，两行两列
    step_cnt = step_cnt + df_all_epo.iloc[i]['train_step']
    if step_cnt >= 20000:
        print(i)
        start_train = i
        break

# 绘图 统计r_avg per episode 和 average of 100 episodes\' r_avg
plt.figure(figsize=(12,6))
plt.xlabel("epo", fontsize=14) # x y轴含义
plt.ylabel("r_avg", fontsize=14)
type1 = plt.scatter(df_all_epo['epo'].to_list(), df_all_epo['r_avg'].to_list())
type2 = plt.scatter(df_100['epo'].to_list(), df_100['r_avg_100'].to_list(), marker='^')
plt.legend((type1, type2), ('r_avg per episode', 'average of 100 episodes\' r_avg'), fontsize=14)
plt.axvline(start_train, color='red', linewidth=2) # 开始训练的位置
plt.tick_params(labelsize=14) # 坐标轴字体
#plt.savefig('./1012/result_record_sp8_ttc_co100.jpg') # 先save再show；反之保存的图片为空
plt.show()

# 绘图 统计 position_y per episode 和 average of 100 episodes\' position_y
plt.figure(figsize=(12,6))
plt.xlabel("epo", fontsize=14) # x y轴含义
plt.ylabel("position_y", fontsize=14)
type1 = plt.scatter(df_all_epo['epo'].to_list(), df_all_epo['position_y'].to_list())
type2 = plt.scatter(df_100['epo'].to_list(), df_100['y_avg_100'].to_list(), marker='^')
plt.legend((type1, type2), ('y per episode', 'average of 100 episodes\' y_avg'), fontsize=14)
plt.axvline(start_train, color='red', linewidth=2) # 开始训练的位置
plt.tick_params(labelsize=14) # 坐标轴字体
#plt.savefig('./1012/y_avg-result_record_sp8_ttc_co100.jpg') # 先save再show；反之保存的图片为空
plt.show()

del r_avg, y_avg, i, step_cnt # 删掉一些没用的变量，方便查看其他有用的变量



# ===================================其他统计========================================！！！
# df_drop_ramdom_search 不包含start_train之前的随机探索epo；
df_drop_ramdom_search = df_all_epo[(df_all_epo['epo'] >= start_train)] # 这里不能直接根据start_train 切片，因为有些 epo 没有

# 1. 车辆行驶距离和start_index的关系
st_in = df_drop_ramdom_search[['epo', 'position_y', 'r_avg', 'r_sum']]
st_in['start_index'] = st_in['epo'].apply(lambda x: int(x) % 100)
epo_analysis = pd.DataFrame([i for i in range(100)]) # 关于epo的统计特征
# 求每一类start_index的position_y平均值，默认生成结果的列明为mean
epo_analysis['start_index_y'] = st_in.groupby(['start_index'])['position_y'].agg(['mean'])
# 求每一类start_index的 r_avg_y 平均值
epo_analysis['r_avg_y'] = st_in.groupby(['start_index'])['r_avg'].agg(['mean'])

# 画图 start_index_y
plt.figure(figsize=(12,6), dpi=100)
plt.xlabel("start_index", fontsize=14) # x y轴含义
plt.ylabel("position_y", fontsize=14)
plt.tick_params(labelsize=14) # 坐标轴字体
plt.plot(epo_analysis.index, epo_analysis['start_index_y'], linewidth=2, label = 'start_index_y')
#plt.savefig('./1012/start_index_y-result_record_sp8_ttc_co100.jpg')
plt.show()

# 画图 r_avg_y
plt.figure(figsize=(12,6), dpi=100)
plt.xlabel("start_index", fontsize=14) # x y轴含义
plt.ylabel("r_avg", fontsize=14)
plt.tick_params(labelsize=14) # 坐标轴字体
plt.plot(epo_analysis.index, epo_analysis['r_avg_y'], linewidth=2, label = 'c')
#plt.savefig('./1012/r_avg_y-result_record_sp8_ttc_co100.jpg')
plt.show()



'''
统计撞车的原因
'''
cols = ['epo', 'train_step', 'position_y', 'r_safe', 'collision_reason']
co_all_epo = pd.DataFrame(columns = cols)

fail_cnt = 0 # one_file中没有结果
csv_cnt = len([n for n in os.listdir('./1015/result_record_sp8_ttc_co100')]) # 统计文件夹下的文件个数

for i in tqdm(range(csv_cnt-3)): # 减去两个模型保存文件的数量
    if os.path.getsize(f"./1015/result_record_sp8_ttc_co100") > 0:
        one_file = pd.read_csv(f"./1015/result_record_sp8_ttc_co100/df_record_epo_{i}.csv")
        
        # 1.如果epo中有2条以上记录
        if len(one_file) >= 2:
            last_line = one_file.iloc[-1,:]
            second_last_line = one_file.iloc[-2,:]
            
            if last_line['position_y'] > 3100:
                last_line['collision_reason'] = 'no collision'
            elif last_line['change_lane'] != second_last_line['change_lane']:
                last_line['collision_reason'] = 'change lane collision'
            elif second_last_line['r_safe'] != 0 and second_last_line['r_safe'] != -1: # 看倒数第2行的数据，-1也能是撞墙
                last_line['collision_reason'] = 'hit leading vehicle'
            else:
                last_line['collision_reason'] = 'unknown'
        
            last_line = last_line[cols] # 选取需要的数据
            co_all_epo = co_all_epo.append(last_line)
        # 2.如果epo中只有1条记录
        elif len(one_file) == 1:
            last_line = one_file.iloc[-1,:]
            last_line['collision_reason'] = 'beginning collision'
            last_line = last_line[cols] # 选取需要的数据
            co_all_epo = co_all_epo.append(last_line)
        # 2.如果epo中没有记录
        else:
            co_all_epo = co_all_epo.append({'epo':0, 'train_step':0,'position_y':0,
                                            'r_safe':0, 'collision_reason': '0'}, ignore_index=True)


co_all_epo_drop_random = co_all_epo[(co_all_epo['epo'] >= start_train)]
co_all_epo_drop_random['collision_reason'].value_counts() # 统计每一类 collision_reason 的个数


