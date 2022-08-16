# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:26:55 2022
主文件
包括(1)使用TraCI接口获取模拟环境中的信息，以及(2)训练RL模型

dependencies:
traci:1.14.0
sumolib:1.14.0

@author: Skye
"""

import numpy as np
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import random
#from traci_generation import generate_routefile
from dqn_model import DQN
#from queue import Queue
import os, sys
import time
import traci
from sumolib import checkBinary
import pandas as pd


# 引入地址 
sumo_path = os.environ['SUMO_HOME'] # "D:\\sumo\\sumo1.13.0"
project_path = "E:\condaProject\sumo_test01\sumo"
cfg_path = "E:\condaProject\sumo_test01\sumo\one_way.sumocfg"
sys.path.append(sumo_path)
sys.path.append(sumo_path + "tools")
sys.path.append(sumo_path + "/tools/xml")

# 各种参数
#parser = argparse.ArgumentParser(description='delay RL velocity control')
#parser.add_argument('--discount', default=0.9995, type=float, help='')
#parser.add_argument('--m_size', default=20000, type=int, help='memory size')
#parser.add_argument('--seed', default=42, type=int, help='')
#args = parser.parse_args()
#if args.seed > 0:
#    np.random.seed(args.seed)

# 是否打开gui
gui = False
if gui == 1:
    sumoBinary = checkBinary('sumo-gui') # 方法一
#    sumoBinary = sumo_path + "/bin/sumo-gui" # 方法二：后面添加路径中的/与sumo_path中的\可能不匹配，不建议使用
else:
    sumoBinary = checkBinary('sumo')
#    sumoBinary = sumo_path + "/bin/sumo"

sumoCmd = [sumoBinary, "-c", cfg_path]
#sumoCmd = [sumoBinary, "-c", cfg_path, '--lanechange.duration','0.5','--route-steps','0','--seed','42']

# 记录车辆信息
map_ve = {}
# 包括前一时刻的后方第一辆车和当前时刻的后方第一辆车
back_id = [' ', ' ']

auto_vehicle_a = 0
# 统计总的训练次数
step = 0
# 存储奖励值的数组，分epo存的
all_reward = []
cols = ['epo', 'train_step', 'position_y', 'lane', 's', 'a', 'r', 's_']
df_record = pd.DataFrame(columns = cols)


def get_all(control_vehicle, select_dis):
    """
    该函数部分代码没有改动
    :param control_vehicle: 自动驾驶车的ID
    :param select_dis: 探测车辆的范围，[-select_dis, select_dis]
    :return:
    """

    # 将自动驾驶车要用到的周围车的信息保存在map_ve字典里，key是周围车辆的ID，每个周围车辆的数据为
    # [该车相对于自动驾驶车的相对纵向距离，相对横向距离，相对速度]
    global map_ve
    vehicle_list = traci.vehicle.getIDList()
    for vehicle in vehicle_list:
        if vehicle not in map_ve.keys():
            map_ve[vehicle]=[traci.vehicle.getPosition(vehicle)[0]-traci.vehicle.getPosition(control_vehicle)[0],
                             traci.vehicle.getPosition(vehicle)[1]-traci.vehicle.getPosition(control_vehicle)[1],
                             traci.vehicle.getSpeed(vehicle)-traci.vehicle.getSpeed(control_vehicle)]
        else:
            map_ve[vehicle]=[traci.vehicle.getPosition(vehicle)[0]-traci.vehicle.getPosition(control_vehicle)[0],
                             traci.vehicle.getPosition(vehicle)[1]-traci.vehicle.getPosition(control_vehicle)[1],
                             traci.vehicle.getSpeed(vehicle)-traci.vehicle.getSpeed(control_vehicle)]

    # 获取自动驾驶车的Y坐标和X坐标和纵向速度和车道
    y_pos = traci.vehicle.getPosition(control_vehicle)[0]
    x_pos = traci.vehicle.getPosition(control_vehicle)[1]
    y_speed = traci.vehicle.getSpeed(control_vehicle)
    lane = traci.vehicle.getLaneID(control_vehicle)

    # 总的需要收集的车辆信息的数组
    Id_list = []
    # 六个方向的车辆信息，这里收集的信息主要是为了根据距离找到最近的车的ID
    up = []
    upright = []
    upleft = []
    down = []
    downright = []
    downleft = []

    # 将六个方向的车收集到分别的数组里面去
    for select_vehicle in vehicle_list:
        if select_vehicle == control_vehicle:
            continue
        # 正前方
        if y_pos + select_dis > traci.vehicle.getPosition(select_vehicle)[0] > y_pos and np.abs(traci.vehicle.getPosition(select_vehicle)[1]-x_pos) < 1:
            up.append([traci.vehicle.getPosition(select_vehicle)[0] - y_pos, select_vehicle])
        # 右前方
        if y_pos + select_dis > traci.vehicle.getPosition(select_vehicle)[0] > y_pos and 2 < x_pos - traci.vehicle.getPosition(select_vehicle)[1] < 4:
            upright.append([traci.vehicle.getPosition(select_vehicle)[0] - y_pos, select_vehicle])
        # 左前方
        if y_pos + select_dis > traci.vehicle.getPosition(select_vehicle)[0] > y_pos and 2 < traci.vehicle.getPosition(select_vehicle)[1]-x_pos < 4:
            upleft.append([traci.vehicle.getPosition(select_vehicle)[0] - y_pos, select_vehicle])
        # 正后方，防止select_dis范围内没有后车，所有不设范围
        if traci.vehicle.getPosition(select_vehicle)[0] < y_pos and np.abs(traci.vehicle.getPosition(select_vehicle)[1]-x_pos) < 1:
            down.append([traci.vehicle.getPosition(select_vehicle)[0] - y_pos, select_vehicle])
        # 右后方
        if y_pos - select_dis < traci.vehicle.getPosition(select_vehicle)[0] < y_pos and 2 < x_pos - traci.vehicle.getPosition(select_vehicle)[1] < 4:
            downright.append([traci.vehicle.getPosition(select_vehicle)[0] - y_pos, select_vehicle])
        # 左后方
        if y_pos - select_dis < traci.vehicle.getPosition(select_vehicle)[0] < y_pos and 2 < traci.vehicle.getPosition(select_vehicle)[1]-x_pos < 4:
            downleft.append([traci.vehicle.getPosition(select_vehicle)[0] - y_pos, select_vehicle])

    # 排序操作是为了找到那个方向离自动驾驶车最近的那辆车，（每个方向只选了最近的那辆）
    up = sorted(up, key=lambda x: x[0])
    # 如果该方向有车 则将最近的车的信息加到ID_list里面里面。
    # 如果该方向没车，对于自己车道上的则用远的mask车辆代替，如果是旁边车道的得判断到底有没有旁边车道，有的话也是用远的mask来代替，否则用纵向距离为0的车辆代替，表示这个方向没车道，不能去
    # up[0][-1]指车辆ID, map_ve[up[0][-1]]是要收集的车辆信息
    if len(up) >= 1:
        Id_list.append(map_ve[up[0][-1]])
    else:
        Id_list.append([200, 0, 0])

    upright = sorted(upright, key=lambda x: x[0])
    if len(upright) >= 1:
        Id_list.append(map_ve[upright[0][-1]])
    else:
        if '0' in lane:
            Id_list.append([0, -3.2, 0])
        else:
            Id_list.append([200, -3.2, 0])

    upleft = sorted(upleft, key=lambda x: x[0])
    if len(upleft) >= 1:
        Id_list.append(map_ve[upleft[0][-1]])
    else:
        if '2' in lane:
            Id_list.append([0, 3.2, 0])
        else:
            Id_list.append([200, 3.2, 0])

    down = sorted(down, key=lambda x: x[0])
    if len(down) >= 1:
        Id_list.append(map_ve[down[-1][-1]])
        back_v = down[-1][-1]
    else:
        Id_list.append([-200, 0, 0])
        back_v = 0

    downright = sorted(downright, key=lambda x: x[0])
    if len(downright) >= 1:
        Id_list.append(map_ve[downright[-1][-1]])
    else:
        if '0' in lane:
            Id_list.append([0, -3.2, 0])
        else:
            Id_list.append([-200, -3.2, 0])

    downleft = sorted(downleft, key=lambda x: x[0])
    if len(downleft) >= 1:
        Id_list.append(map_ve[downleft[-1][-1]])
    else:
        if '2' in lane:
            Id_list.append([0, 3.2, 0])
        else:
            Id_list.append([-200, 3.2, 0])

    # 更新后方车的ID
    global back_id
    back_id[0] = back_id[1]
    back_id[1] = back_v

    # 得到自动驾驶车自己所在的车道
    ego_lane=traci.vehicle.getLaneID(control_vehicle)
    if '0' in ego_lane:
        ego_l=-1
    elif '1' in ego_lane:
        ego_l=0
    else:
        ego_l=1
    global auto_vehicle_a
    if np.abs(traci.vehicle.getAcceleration(control_vehicle)) < 0.0001:
        cal_a = auto_vehicle_a
    else:
        cal_a = traci.vehicle.getAcceleration(control_vehicle)
    Id_list.append([traci.vehicle.getSpeed(control_vehicle)/25, cal_a/3, ego_l])

    # 归一化，需要，要不然容易边界值
    for i in range(6):
        Id_list[i][0] = Id_list[i][0]/50
        Id_list[i][1] = Id_list[i][1]/3.2
        Id_list[i][2] = Id_list[i][2]/25

    # 为了得到与前车的相对纵向距离和相对纵向速度
    relspace = 100000
    relspeed = 0
    for select_vehicle in vehicle_list:
        if select_vehicle == control_vehicle:
            continue
        if traci.vehicle.getPosition(select_vehicle)[0] - y_pos > 0 and traci.vehicle.getLaneID(select_vehicle) == traci.vehicle.getLaneID(control_vehicle):
            if traci.vehicle.getPosition(select_vehicle)[0] - y_pos < relspace:
                relspace = traci.vehicle.getPosition(select_vehicle)[0] - y_pos
                relspeed = traci.vehicle.getSpeed(select_vehicle) - y_speed
    # 周围车信息，相对距离，相对速度
    return Id_list, relspace, relspeed


def train(agent, all_vehicle, control_vehicle, episode, train_step, index):
    action_int = agent.choose_action(np.array(all_vehicle)) # 返回第action_int个动作 0-8
    # 012位指加速 345指不变速 678指减速；然后036指左变道，147指不变道，258指右变道
    if action_int in [0, 1, 2]:
        acc= 3
    if action_int in [3, 4, 5]:
        acc= 0
    if action_int in [6, 7, 8]:
        acc= -3
        
    # 0车道右车道在-8.0；1车道在-4.8；2车道左车道在-1.6
    if action_int in [0, 3, 6]:
        change_lane = 'left'
    if action_int in [1, 4, 7]:
        change_lane = 'keep'
    if action_int in [2, 5, 8]:
        change_lane = 'right'
    
    # 随机探索
    
    # 记录当前车辆的数据
    global auto_vehicle_a
    pre_ego_info_dict = {"speed": traci.vehicle.getSpeed(control_vehicle), 
                         "acc":auto_vehicle_a, 
                         "LaneID": traci.vehicle.getLaneID(control_vehicle), 
                         "position": traci.vehicle.getPosition(control_vehicle)}
    
    # 计算速度，控制限速
    sp = traci.vehicle.getSpeed(control_vehicle) + acc*0.5
    sp = np.array([sp]).clip(0, 25)[0] # 将sp转为array，裁剪取值范围后取其float值
    traci.vehicle.setSpeed(control_vehicle, sp) # 将速度设置好
    
    # 变道处理
    if change_lane=='left':
        if '1' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'E0_2', traci.vehicle.getLanePosition(control_vehicle))
        elif '0' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'E0_1', traci.vehicle.getLanePosition(control_vehicle))
    elif change_lane=='right':
        if '1' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'E0_0', traci.vehicle.getLanePosition(control_vehicle))
        elif '2' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'E0_1', traci.vehicle.getLanePosition(control_vehicle))
    
    # ================================执行 ==================================
    traci.simulationStep()
    
    # 查询自动驾驶车是否发生碰撞
    collision=0
    if control_vehicle in traci.simulation.getCollidingVehiclesIDList():
        print('=====================collision==', traci.simulation.getCollidingVehiclesIDList(), control_vehicle)
        print("==========================发生了撞车=========================")
        collision=1
        return collision
    
    # 获取动作执行后的状态
    new_all_vehicle, rel_space, rel_speed = get_all(control_vehicle, 200)
    cur_ego_info_dict = {"speed": traci.vehicle.getSpeed(control_vehicle), 
                     "acc":traci.vehicle.getAcceleration(control_vehicle), 
                     "LaneID": traci.vehicle.getLaneID(control_vehicle), 
                     "position": traci.vehicle.getPosition(control_vehicle)}
      
    # 避免分母为0
    e = 0.000001
    if 0 <= rel_speed < e:
        rel_speed = e
    if -e < rel_speed < 0:
        rel_speed = -e
    
    # 计算reward
    y_ttc=-rel_space/rel_speed
    r_efficiency = cur_ego_info_dict['speed']/25*0.8
    
    if 0 < y_ttc < 4:
        r_safe = np.log(y_ttc/4)
    else:
        r_safe = 0
    
    auto_vehicle_a = cur_ego_info_dict['acc']
    r_comfort = ((np.abs(pre_ego_info_dict['acc']-cur_ego_info_dict['acc'])/0.1) ** 2) / 3600
    
    cur_reward = r_safe + r_efficiency - r_comfort
    global all_reward
    all_reward.append([cur_reward, episode, index])
    global df_record
    df_record = df_record.append(pd.DataFrame([[episode, train_step, pre_ego_info_dict['position'][0], 
                                    pre_ego_info_dict['LaneID'], all_vehicle, action_int, 
                                    cur_reward, new_all_vehicle]], columns = cols))
    
    # agent存储
    agent.store_transition(all_vehicle, action_int, cur_reward, new_all_vehicle)
    
    return collision


def main_train():
    a_dim = 9
    # 状态就是自己+六辆周围车的状态
    s_dim = 3*7
    #模型加载
    agent = DQN(s_dim, a_dim)
    
#    epo = 0 # 之后再用epo循环
    for epo in range(1):
        traci.start(sumoCmd)
        ego_index = 20 + epo % 100   # 选取中间车道第index辆出发的车为我们的自动驾驶车
        ego_index_str = '1_'+str(ego_index) # 自动驾驶车辆的id为'1_$index$', 如index为20,id='1_20'
        control_vehicle = '' # 自动驾驶车辆的id
        ego_show = False # 自动驾驶车辆是否出现过
        global auto_vehicle_a
        auto_vehicle_a = 0
        train_step = 0
        
        while traci.simulation.getMinExpectedNumber() > 0:
            # 1. 得到道路上所有的车辆ID
            vehicle_list = traci.vehicle.getIDList()
            
            # 2. 找到我们控制的自动驾驶车辆
            # 2.1 如果此时自动驾驶车辆已出现，设置其为灰色, id为'1_$ego_index$'
            if ego_index_str in vehicle_list:
                control_vehicle = ego_index_str
                ego_show = True
                traci.vehicle.setColor(control_vehicle, (128,128,128,255))
            # 2.2 如果此时自动驾驶车辆还未出现
            if ego_show == False:
                traci.simulationStep() # 2个step出现1辆车
                continue
            # 2.3 如果已经出现了而且撞了就退出
            if ego_show and control_vehicle not in vehicle_list:
                print("=====================已经出现了而且撞了================")
                break
            
            # 3 在非RL控制路段中采取其他行驶策略，控制的路段为1100-3100这2000m的距离
            # 3.1 在0-1100m是去掉模拟器自带算法中的变道，但暂时保留速度控制
            traci.vehicle.setLaneChangeMode(control_vehicle, 0b000000000000)
            print("自动驾驶车的位置====================", traci.vehicle.getPosition(control_vehicle)[0])     
            if traci.vehicle.getPosition(control_vehicle)[0] < 1100:
                traci.simulationStep()
                continue
            # 3.2 在大于3100m
            if traci.vehicle.getPosition(control_vehicle)[0] > 3100:
                print("=======================距离超过3100====================")
                break
    
            # 4 在RL控制路段中收集自动驾驶车周围车辆的信息，并设置周围车辆
            # 4.1 获取周围车辆信息
            all_vehicle, re_sp, relspeed = get_all(control_vehicle, 200) # 前后200m距离
    
            # 4.2 将所有后方车辆设置为不变道
            for vehicle in vehicle_list:
                if traci.vehicle.getPosition(vehicle)[0] < traci.vehicle.getPosition(control_vehicle)[0]:
                    traci.vehicle.setLaneChangeMode(vehicle, 0b000000000000)
    
            # 4.3 去除自动驾驶车默认的跟车和换道模型，为模型训练做准备
            traci.vehicle.setSpeedMode(control_vehicle, 00000)
            traci.vehicle.setLaneChangeMode(control_vehicle, 0b000000000000)
            
            # 5 模型训练
            collision = train(agent, all_vehicle, control_vehicle, epo, train_step, ego_index)
            if collision:
                break
            train_step = train_step + 1
            global step
            step = step + 1
#        np.save("reward/"+str(epo)+"r.npy", np.array(all_reward))
        traci.close(wait=True)
        # 保存
#        agent.save_net()



if __name__ == '__main__':
    main_train() 
    df_record.to_csv("df_record.csv", index = False)


