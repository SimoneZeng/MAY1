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
from dqn_model import DQN
import os, sys
import pandas as pd


# 引入地址 
sumo_path = os.environ['SUMO_HOME'] # "D:\\sumo\\sumo1.13.0"
#project_path = "E:\condaProject\sumo_intersection\sumo_test01\sumo" # 暂时没啥用
#cfg_path = "E:\condaProject\sumo_intersection\sumo_test01\sumo\one_way.sumocfg" # 1.在本地用这个cfg_path
cfg_path = "/home/zengximu/sumo_inter/sumo_test01/sumo/one_way.sumocfg" # 2. 在服务器上用这个cfg_path
sys.path.append(sumo_path)
sys.path.append(sumo_path + "/tools")
sys.path.append(sumo_path + "/tools/xml")
import traci # 在ubuntu中，traci和sumolib需要在tools地址引入之后import
from sumolib import checkBinary

# 各种参数
#parser = argparse.ArgumentParser(description='delay RL velocity control')
#parser.add_argument('--discount', default=0.9995, type=float, help='')
#parser.add_argument('--m_size', default=20000, type=int, help='memory size')
#parser.add_argument('--seed', default=42, type=int, help='')
#args = parser.parse_args()
#if args.seed > 0:
#    np.random.seed(args.seed)


os.environ['CUDA_VISIBLE_DEVICES']='0, 1'  # 显卡使用

# 是否打开gui
gui = False
if gui == 1:
    sumoBinary = checkBinary('sumo-gui') # 方法一
#    sumoBinary = sumo_path + "/bin/sumo-gui" # 方法二：后面添加路径中的/与sumo_path中的\可能不匹配，不建议使用
else:
    sumoBinary = checkBinary('sumo')
#    sumoBinary = sumo_path + "/bin/sumo"

sumoCmd = [sumoBinary, "-c", cfg_path]

# 记录车辆信息
map_ve = {}
# 包括前一时刻的后方第一辆车和当前时刻的后方第一辆车
back_id = [' ', ' ']
auto_vehicle_a = 0
# 统计总的训练次数
step = 0
# 存储奖励值的数组，分epo存的
cols = ['epo', 'train_step', 'position_y', 'target_lane', 'lane', 'speed', 
         'a', 'acc', 'change_lane', 'r','r_safe', 'r_eff','r_com', 's', 's_']
df_record = pd.DataFrame(columns = cols)


def get_all(control_vehicle, select_dis):
    """
    该函数部分代码稍微改了一下逻辑
    :param control_vehicle: 自动驾驶车的ID
    :param select_dis: 探测车辆的范围，[-select_dis, select_dis]
    :return: Id_list, rel_up, flow 周围车信息，相对距离，相对速度
    """
    # 获取自动驾驶车的Y坐标和X坐标和纵向速度和车道
    y_pos = traci.vehicle.getPosition(control_vehicle)[0]
    x_pos = traci.vehicle.getPosition(control_vehicle)[1]
    y_speed = traci.vehicle.getSpeed(control_vehicle)
    ego_lane = traci.vehicle.getLaneID(control_vehicle)
    
    # 将自动驾驶车要用到的周围车的信息保存在map_ve字典里，key是周围车辆的ID，每个周围车辆的数据为
    # [该车相对于自动驾驶车的相对纵向距离，相对横向距离，相对速度]
    global map_ve
    vehicle_list = traci.vehicle.getIDList()
    for v in vehicle_list:
        if v not in map_ve.keys():
            map_ve[v]=[traci.vehicle.getPosition(v)[0] - y_pos,
                             traci.vehicle.getPosition(v)[1] - x_pos,
                             traci.vehicle.getSpeed(v) - y_speed]
        else:
            map_ve[v]=[traci.vehicle.getPosition(v)[0] - y_pos,
                             traci.vehicle.getPosition(v)[1] - x_pos,
                             traci.vehicle.getSpeed(v) - y_speed]

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
    for v in vehicle_list:
        if v == control_vehicle:
            continue
        # 正前方
        if y_pos + select_dis > traci.vehicle.getPosition(v)[0] > y_pos and np.abs(traci.vehicle.getPosition(v)[1]-x_pos) < 1:
            up.append([traci.vehicle.getPosition(v)[0] - y_pos, v])
        # 右前方
        if y_pos + select_dis > traci.vehicle.getPosition(v)[0] > y_pos and 2 < x_pos - traci.vehicle.getPosition(v)[1] < 4:
            upright.append([traci.vehicle.getPosition(v)[0] - y_pos, v])
        # 左前方
        if y_pos + select_dis > traci.vehicle.getPosition(v)[0] > y_pos and 2 < traci.vehicle.getPosition(v)[1]-x_pos < 4:
            upleft.append([traci.vehicle.getPosition(v)[0] - y_pos, v])
        # 正后方，防止select_dis范围内没有后车，所有不设范围
        if traci.vehicle.getPosition(v)[0] < y_pos and np.abs(traci.vehicle.getPosition(v)[1]-x_pos) < 1:
            down.append([traci.vehicle.getPosition(v)[0] - y_pos, v])
        # 右后方
        if y_pos - select_dis < traci.vehicle.getPosition(v)[0] < y_pos and 2 < x_pos - traci.vehicle.getPosition(v)[1] < 4:
            downright.append([traci.vehicle.getPosition(v)[0] - y_pos, v])
        # 左后方
        if y_pos - select_dis < traci.vehicle.getPosition(v)[0] < y_pos and 2 < traci.vehicle.getPosition(v)[1]-x_pos < 4:
            downleft.append([traci.vehicle.getPosition(v)[0] - y_pos, v])

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
        if '0' in ego_lane:
            Id_list.append([0, -3.2, 0])
        else:
            Id_list.append([200, -3.2, 0])

    upleft = sorted(upleft, key=lambda x: x[0])
    if len(upleft) >= 1:
        Id_list.append(map_ve[upleft[0][-1]])
    else:
        if '2' in ego_lane:
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
        if '0' in ego_lane:
            Id_list.append([0, -3.2, 0])
        else:
            Id_list.append([-200, -3.2, 0])

    downleft = sorted(downleft, key=lambda x: x[0])
    if len(downleft) >= 1:
        Id_list.append(map_ve[downleft[-1][-1]])
    else:
        if '2' in ego_lane:
            Id_list.append([0, 3.2, 0])
        else:
            Id_list.append([-200, 3.2, 0])

    # 更新后方车的ID
    global back_id
    back_id[0] = back_id[1]
    back_id[1] = back_v

    # 得到自动驾驶车自己所在的车道
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
    # 前方车辆可能不在200m内，不能直接用up中的数据
    relspace = 100000
    relspeed = 0
    for v in vehicle_list:
        if v == control_vehicle:
            continue
        if traci.vehicle.getPosition(v)[0] - y_pos > 0 and traci.vehicle.getLaneID(v) == ego_lane:
            if traci.vehicle.getPosition(v)[0] - y_pos < relspace:
                relspace = traci.vehicle.getPosition(v)[0] - y_pos
                relspeed = traci.vehicle.getSpeed(v) - y_speed
    
    # 前后200m范围内容的车道级流量
    flow_left = len(upleft) + len(downleft)
    flow_middle = len(up) + len(down)
    flow_right = len(upright) + len(downright)
    flow = {'flow_left': flow_left, 'flow_middle': flow_middle, 'flow_right': flow_right}
    
    rel_up = {'relspace': relspace, 'relspeed':relspeed}
    
    # 周围车信息，相对距离，相对速度
    return Id_list, rel_up, flow


def train(agent, control_vehicle, episode, train_step, target_lane):
    '''
    该函数使用agent根据state得出action，执行对应的速度控制和变道控制
    计算出reward，包括 r_safe + r_efficiency - r_comfort + r_target_lane
    存储信息到df_record以及 agent.store_transition
    :return: collision 
    '''
    all_vehicle, rel_up, _ = get_all(control_vehicle, 200)
    action_int = agent.choose_action(np.array(all_vehicle)) # 返回第action_int个动作 0-8
    
    # 随机探索
    global step
    if step<20000:
        action_int = random.randint(0, 9)
        step = step+1
    
    # 012位指加速 345指不变速 678指减速；然后036指左变道，147指不变道，258指右变道
    acc = 0 # 需要先赋值初始值
    if action_int in [0, 1, 2]:
        acc= 1
    if action_int in [3, 4, 5]:
        acc= 0
    if action_int in [6, 7, 8]:
        acc= -1
        
    # 0车道右车道在-8.0；1车道在-4.8；2车道左车道在-1.6
    change_lane = ''
    if action_int in [0, 3, 6]:
        change_lane = 'left'
    if action_int in [1, 4, 7]:
        change_lane = 'keep'
    if action_int in [2, 5, 8]:
        change_lane = 'right'
    

    
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
    
    # 撞墙处理，车道从左到右是2,1,0
    if 'E0_0' == pre_ego_info_dict["LaneID"] and change_lane=='right':
        # 将撞墙的数据存到经验池里面去
        agent.store_transition(all_vehicle, action_int, -1, np.zeros((7,3)))
        global df_record
        df_record = df_record.append(pd.DataFrame([[episode, train_step, pre_ego_info_dict['position'][0], 
                                                    target_lane, pre_ego_info_dict['LaneID'], 
                                                    pre_ego_info_dict['speed'], action_int, acc, change_lane, 
                                                    -1, -1, -1, -1, all_vehicle, np.zeros((7,3))]], columns = cols))
        print("====================右右右右车道撞墙墙墙墙===================")
        change_lane='keep'
    if 'E0_2' == pre_ego_info_dict["LaneID"] and change_lane=='left':
        # 将撞墙的数据存到经验池里面去
        agent.store_transition(all_vehicle, action_int, -1, np.zeros((7,3)))
#            global df_record
        df_record = df_record.append(pd.DataFrame([[episode, train_step, pre_ego_info_dict['position'][0], 
                                                    target_lane, pre_ego_info_dict['LaneID'], 
                                                    pre_ego_info_dict['speed'], action_int, acc, change_lane, 
                                                    -1, -1, -1, -1, all_vehicle, np.zeros((7,3))]], columns = cols))
        print("====================左左左左车道撞墙墙墙墙===================")
        change_lane='keep'
    
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
    new_all_vehicle, new_rel_up, _ = get_all(control_vehicle, 200)
    cur_ego_info_dict = {"speed": traci.vehicle.getSpeed(control_vehicle), 
                     "acc":traci.vehicle.getAcceleration(control_vehicle), 
                     "LaneID": traci.vehicle.getLaneID(control_vehicle), 
                     "position": traci.vehicle.getPosition(control_vehicle)}
      
    # 避免分母为0
    e = 0.000001
    if 0 <= new_rel_up['relspeed'] < e:
        new_rel_up['relspeed'] = e
    if -e < new_rel_up['relspeed'] < 0:
        new_rel_up['relspeed'] = -e
    
    # 计算reward
    y_ttc=-new_rel_up['relspace']/new_rel_up['relspeed'] # time to collision
    r_efficiency = cur_ego_info_dict['speed']/25*0.8
    
    if 0 < y_ttc < 4: 
        r_safe = np.log(y_ttc/4)
    else:
        r_safe = 0
    
    auto_vehicle_a = cur_ego_info_dict['acc']
    r_comfort = ((np.abs(pre_ego_info_dict['acc']-cur_ego_info_dict['acc'])/0.1) ** 2) / 3600
#    r_target_lane = -0.5 * abs(int(cur_ego_info_dict['LaneID'][-1]) - target_lane) # 与目标车道的差距绝对值，*0.5，分别为 0 -0.5 -1
    
#    cur_reward = r_safe + r_efficiency - r_comfort + r_target_lane
    cur_reward = r_safe + r_efficiency - r_comfort
    
#    global df_record
    df_record = df_record.append(pd.DataFrame([[episode, train_step, cur_ego_info_dict['position'][0], 
                                                target_lane, cur_ego_info_dict['LaneID'], 
                                                cur_ego_info_dict['speed'], action_int, acc, change_lane,
                                                cur_reward, r_safe, r_efficiency, r_comfort, 
                                                all_vehicle, new_all_vehicle]], columns = cols))
    
    # agent存储
    agent.store_transition(all_vehicle, action_int, cur_reward, new_all_vehicle)
    
    return collision


def main_train():
    a_dim = 9
    # 状态就是自己+六辆周围车的状态
    s_dim = 3*7
    #模型加载
    agent = DQN(s_dim, a_dim)
    
    for epo in range(20000): # 测试时可以调小epo回合次数 
        traci.start(sumoCmd)
        ego_index = 20 + epo % 100   # 选取中间车道第index辆出发的车为我们的自动驾驶车
        ego_index_str = '1_'+str(ego_index) # ego的id为'1_$index$', 如index为20,id='1_20'
        control_vehicle = '' # ego车辆的id
        ego_show = False # ego车辆是否出现过
        target_lane = random.randint(0, 2) # ego的变道目标车道，从0 1 2中取
        global auto_vehicle_a
        auto_vehicle_a = 0
        train_step = 0
        global df_record
        df_record = pd.DataFrame(columns = cols)
        
        print("++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"++++++++++++++++++++{epo}+++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++")
        
        
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
#            all_vehicle, rel_up, _ = get_all(control_vehicle, 200) # 前后200m距离
    
            # 4.2 将所有后方车辆设置为不变道
            for vehicle in vehicle_list:
                if traci.vehicle.getPosition(vehicle)[0] < traci.vehicle.getPosition(control_vehicle)[0]:
                    traci.vehicle.setLaneChangeMode(vehicle, 0b000000000000)
    
            # 4.3 去除自动驾驶车默认的跟车和换道模型，为模型训练做准备
            traci.vehicle.setSpeedMode(control_vehicle, 00000)
            traci.vehicle.setLaneChangeMode(control_vehicle, 0b000000000000)
            
            # 5 模型训练
            collision = train(agent, control_vehicle, epo, train_step, target_lane) # 模拟一个时间步
            if collision:
                break
            train_step = train_step + 1
            global step
            step = step + 1

        traci.close(wait=True)
        # 保存
        df_record.to_csv(f"result_record_exp/df_record_epo_{epo}.csv", index = False)
#        agent.save_net() # 一定要保存网络，这样结果才有提升
        torch.save(agent.eval_net, './result_record_exp/eval_net.pkl')
        torch.save(agent.eval_net.state_dict(), './result_record_exp/eval_net_params.pkl')



if __name__ == '__main__':
    main_train() 
#    pass





