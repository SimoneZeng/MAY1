# -*- coding: utf-8 -*-
"""
Created on Tue May  16 10:34:40 2023

统计pdqn 的训练和测试结果

包括 6 和 宏观指标， 3 个微观指标
AvgA-C average affection times 还没有计算

@author: Simone
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pprint as pp

memory_size = 40000 # 20000 200000 bmem 40000
TRAIN = False # True False
 
# name = 'pdqn_5l_rainbow_linear_mp'
# record_dir = './0516/result_pdqn_5l_linear_mp'
record_dir = './result_pdqn_5l_lstm_bs_mp'
# record_dir = './0516'

def get_df_all_epo(record_dir):
# if __name__ == '__main__':
    '''
    整理每个 epo 的数据，获得 df_all，一个 epo 对应 df_all 中的一条数据
    macroscopic metrics:
        - Success
        - AvgT average travelling time
        - AvgLC average lane changing times
        - AvgC-C average rate of collision with conventional vehicle # 撞车
        - AvgRI average rate of off-road infraction # 撞墙
        - AvgA-C average affection times that the autonomous vehicle has negative impact on its following 
    microscopic metrics:
        - AvgTTC-A
        - AvgV-A
        - AvgJ-A
    '''
    cols = ['epo', 'total_step', 'epo_step', 'target_direc', 'position_y', 'lane','avg_r', 'sum_r', 'dis_to_target_lane',
            'success', 'lane_change_times', 'co_vehicle', 'co_road', 'affect_times', 
            'avg_ttc', 'avg_velocity', 'avg_jerk', 'rg_times']
    df_all = pd.DataFrame(columns = cols)
    fail_cnt = 0 # one_file中没有结果
    csv_cnt = len([n for n in os.listdir(record_dir)]) # 统计文件夹下的文件个数

    for i in tqdm(range(csv_cnt - 20)): # 去掉一定数量的非 record 文件
        if os.path.getsize(f"{record_dir}/df_record_epo_{i}.csv") > 0:
            one_file = pd.read_csv(f"{record_dir}/df_record_epo_{i}.csv")
            
            # 除了表头还有2条以上的记录
            if len(one_file) >= 3:
                ori_last_line = one_file.iloc[-1,:] # 原始最后一行记录，可能超出3100m
                new_one_file = one_file.copy()
                rg_times = new_one_file['done'].sum()
                # 不能直接删掉最后一行，有碰撞时，需要考虑碰撞惩罚；无碰撞时，可以去掉最后大于3100m 的数据
                new_one_file.drop(index = new_one_file[(new_one_file['position_y'] < 0)].index.tolist(), inplace=True) # 碰撞后 position_y 为负的小bug
                new_one_file.drop(index = new_one_file[(new_one_file['position_y'] >= 3100)].index.tolist(), inplace=True)
                # 去除中间修正数据，不包括最后一行 即 0 : -1
                new_one_file.drop(index = new_one_file.iloc[0:-1,:][(new_one_file.iloc[0:-1,:]['done'] == 1)].index.tolist(), inplace=True) 
                last_line = new_one_file.iloc[-1,:] 
                
                epo_analysis = last_line
                epo_analysis['epo'] = last_line['epo']
                epo_analysis['total_step'] = last_line['train_step']
                epo_analysis['target_direc'] = last_line['target_direc']
                epo_analysis['rg_times'] = rg_times
                epo_analysis['epo_step'] = len(new_one_file)
                
                # 完成了
                if last_line['position_y'] >= 3080: # 最后一条的位置超出3100不准确
                    epo_analysis['position_y'] = 3100
                    # 宏观 4 5
                    epo_analysis['co_vehicle'] = 0
                    epo_analysis['co_road'] = 0  
                # 没完成，有碰撞
                else:
                    epo_analysis['position_y'] = last_line['position_y']
                    epo_analysis['rg_times'] = epo_analysis['rg_times'] - 1 # 去除最后一条碰撞的 done=1
                    if 'another_co_id' in ori_last_line['other_record']: # 判断碰撞原因
                        epo_analysis['co_vehicle'] = 1
                        epo_analysis['co_road'] = 0
                    else:
                        epo_analysis['co_vehicle'] = 0
                        epo_analysis['co_road'] = 1
                    
                epo_analysis['lane'] = last_line['lane']
                epo_analysis['sum_r'] = new_one_file['r'].sum()
                epo_analysis['avg_r'] = epo_analysis['sum_r'] / epo_analysis['epo_step']
                
                # 驶入下一段路时，用 second_last_line 判断 dis_to_target_lane
                if epo_analysis['target_direc'] == 0: # 0 是右转
                    epo_analysis['dis_to_target_lane'] = abs(int(last_line['lane'][-1]) - 0) # int
                elif epo_analysis['target_direc'] == 1:# 1 是直行
                    epo_analysis['dis_to_target_lane'] = min(abs(int(last_line['lane'][-1]) - 1),
                                                          abs(int(last_line['lane'][-1]) - 2),
                                                          abs(int(last_line['lane'][-1]) - 3))
                elif epo_analysis['target_direc'] == 2:# 1 是左转
                    epo_analysis['dis_to_target_lane'] = min(abs(int(last_line['lane'][-1]) - 3),
                                                          abs(int(last_line['lane'][-1]) - 4))
                # 宏观 1 
                if epo_analysis['position_y'] == 3100 and epo_analysis['dis_to_target_lane'] == 0:
                    epo_analysis['success'] = 1
                else:
                    epo_analysis['success'] = 0
                # 宏观 3
                epo_analysis['lane_change_times'] = 0
                lane_change_cnt = new_one_file['change_lane'].value_counts()
                if 'left' in dict(lane_change_cnt).keys():
                    epo_analysis['lane_change_times'] += lane_change_cnt['left']
                if 'right' in dict(lane_change_cnt).keys():
                    epo_analysis['lane_change_times'] += lane_change_cnt['right']
                # 宏观 6
                epo_analysis['affect_times'] = 0
                # epo_analysis['affect_times'] = len(new_one_file[(new_one_file['tail_car_acc'] >= 0.5)].index.tolist())
                
                # 微观 1
                # 筛选 ttc 在 0~50的数据
                #epo_ttc = new_one_file[(new_one_file['ttc'] > 0) & (new_one_file['ttc'] <= 50)]
            
                
                epo_analysis['avg_ttc'] = 0
                # if len(epo_ttc) > 0:
                #     epo_analysis['avg_ttc'] = epo_ttc['ttc'].mean()
                # 微观 2
                epo_analysis['avg_velocity'] = new_one_file['speed'].mean()
                # 微观 3
                epo_analysis['avg_jerk'] = abs(new_one_file['acc']).mean()
                    
                
                epo_analysis = epo_analysis[cols] # 选取需要的数据
                df_all = df_all.append(epo_analysis)

        else:
            fail_cnt += 1
    
    
    return df_all

def print_metric(df_all):
    df_all_epo = df_all.copy()
    result_metric = {}
    result_metric['Success'] = df_all_epo['success'].mean()
    result_metric['AvgT'] = 0.5 * df_all_epo['epo_step'].mean() # 每个时间步是 0.5 s
    result_metric['AvgLC'] = df_all_epo['lane_change_times'].mean()
    result_metric['AvgC-C'] = df_all_epo['co_vehicle'].mean()
    result_metric['AvgRI'] = df_all_epo['co_road'].mean()
    result_metric['AvgA-C'] = df_all_epo['affect_times'].mean()
    
    result_metric['AvgTTC-A'] = df_all_epo['avg_ttc'].mean()
    result_metric['AvgV-A'] = df_all_epo['avg_velocity'].mean()
    result_metric['AvgJ-A'] = df_all_epo['avg_jerk'].mean()
    print(result_metric)

def smooth(data, sm=1):
    pri_sum = []
    sum_i = 0
    for d in data:
        sum_i = sum_i + d
        pri_sum.append(sum_i)
    smooth_data = []
    for i in range(len(data)):
        if i >= sm * 2:
            smooth_data.append((pri_sum[i]-pri_sum[i-sm * 2]) / (sm * 2))
    return smooth_data

def draw_epo_reward(df_all_epo, sm_size ): 
    df = df_all_epo.copy()
    sm_size = sm_size
    epo_reward = smooth(df['avg_r'].to_list(), sm = sm_size) # 从 2 倍sm_size开始才有数据
    
    # 绘图 epo 和平滑后的 reward
    plt.figure(figsize=(12,6))
    plt.xlabel("epo", fontsize=14) # x y轴含义
    plt.ylabel("average reward per episode", fontsize=14)
    plt.plot(df["epo"][2 * sm_size:,], epo_reward, 's-', color = 'g', label = 'model') # s- 方形， o- 圆形
    plt.tick_params(labelsize=14) # 坐标轴字体
    #plt.savefig('./1012/result_record_sp8_ttc_co100.jpg') # 先save再show；反之保存的图片为空
    plt.show()
    

if __name__ == '__main__':
    '''
    macroscopic metrics:
        - Success
        - AvgT average travelling time
        - AvgLC average lane changing times
        - AvgC-C average rate of collision with conventional vehicle # 撞车
        - AvgRI average rate of off-road infraction # 撞墙
        - AvgA-C average affection times that the autonomous vehicle has negative impact on its following 
    microscopic metrics:
        - AvgTTC-A
        - AvgV-A
        - AvgJ-A
    '''
    # df_all_epo = get_df_all_epo(record_dir)
    # df_all_epo.to_csv(f"{record_dir}/all_epo.csv", index = False)
    # print_metric(df_all_epo)
    # draw_epo_reward(df_all_epo, sm_size = 10)
    
    # method_name = ['pdqn_5l_linear_mp', 'pdqn_5l_cl2_rg_rainbow_linear_mp', 
    #                 'pdqn_5l_cl2_rainbow_linear_mp', 'pdqn_5l_ccl2_rainbow_linear_mp',
    #                 'pdqn_5l_cl1_rg_rainbow_linear_mp', 'pdqn_5l_cl1_rainbow_linear_mp',
    #                 'pdqn_5l_ccl1_rainbow_linear_mp']
    method_name = ['pdqn_5l_lstm_bs_mp', 'pdqn_5l_lstm_bs_mp_1']
    
    
    sm_size = 100
    
    plt.figure(figsize=(16,8))
    plt.xlabel("epo", fontsize=14) # x y轴含义
    plt.ylabel("average reward per episode", fontsize=14) 
    
    for name in method_name:
        print(name)
        df = pd.read_csv(f"./result_{name}/all_epo.csv")
        print_metric(df)
        plt.plot(df["epo"][2 * sm_size:,], 
                  smooth(df['avg_r'].to_list(), sm = sm_size), 
                  label = name)
    plt.tick_params(labelsize=14) # 坐标轴字体
    # plt.savefig('./0516/result1.jpg') # 先save再show；反之保存的图片为空
    plt.legend()
    plt.show()
    




