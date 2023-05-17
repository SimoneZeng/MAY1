# -*- coding: utf-8 -*-
"""
Created on May Thu 4 22:19:26 2023

- 5车道场景，含有curriculum learning
- 无rule-based guidance，无LSTM，无撞车撞墙的规则限制

stage设计：
reward一样，每个stage都有r_tl，切换时stage区别跨度较小
    - 一共5条lane，模拟2条lane，低车流密度，车道编码不变，中间车道都是车
    - 一共5条lane，低车流密度
    - 一共5条lane，中车流密度
    - 一共5条lane，高车流密度

reward权重：
    - efficiency [0, 1]
    - safe [-2, 0]
    - comfort [-1, 0]
    - tl [-2, 0]

@author: Simone
"""


import numpy as np
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import random
import os, sys, shutil
import pandas as pd
import math, re
import pprint as pp
curPath=os.path.abspath(os.path.dirname(__file__))
rootPath=os.path.split(os.path.split(curPath)[0])[0]
sys.path.append(rootPath+'/sumo_test01')

#from model.pdqn_model_5tl_lstm import PDQNAgent
from model.pdqn_model_5tl_rainbow_linear import PDQNAgent as RainbowPDQNAgent
from model.pdqn_model_5tl_linear import PDQNAgent as LinearPDQNAgent
from model.pdqn_model_5tl_lstm import PDQNAgent as LstmPDQNAgent


# 引入地址 
sumo_path = os.environ['SUMO_HOME'] # "D:\\sumo\\sumo1.13.0"
# sumo_dir = "C:\--codeplace--\sumo_inter\sumo_test01\sumo\\" # 1.在本地用这个cfg_path
# sumo_dir = "D:\Git\MAY1\sumo\\" # 1.在本地用这个cfg_path
sumo_dir = "/data1/zengximu/sumo_test01/sumo/" # 2. 在服务器上用这个cfg_path
OUT_DIRs=["../0516/result_pdqn_5l_ccl1_rainbow_linear_mp","../0516/result_pdqn_5l_ccl2_rainbow_linear_mp"]
OUT_DIR=""
sys.path.append(sumo_path)
sys.path.append(sumo_path + "/tools")
sys.path.append(sumo_path + "/tools/xml")
import traci # 在ubuntu中，traci和sumolib需要在tools地址引入之后import
from sumolib import checkBinary

TRAIN = False # False True
gui = False # False True # 是否打开gui
if gui == 1:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')

cols = ['stage','epo', 'train_step', 'position_y', 'target_direc', 'lane', 'speed', 
         'lc_int', 'fact_acc', 'acc', 'change_lane', 'ttc', 'tail_car_acc','r','r_safe', 'r_eff',
         'r_com', 'r_tl', 'r_fluc','other_record', 'done', 's', 's_']
df_record = pd.DataFrame(columns = cols) # 存储transition等信息的dataframe，每个epo建立一个dataframe

np.random.seed(0)
random.seed(0)
torch.manual_seed(5)

# PRE_LANE = None
RL_CONTROL = 1100 # Rl agent take control after 1100 meters
DEVICE = torch.device("cuda:0")
# DEVICE = torch.device("cpu")

def get_all(control_vehicle, select_dis):
    """
    该函数部分代码稍微改了一下逻辑
    :param control_vehicle: 自动驾驶车的ID
    :param select_dis: 探测车辆的范围，[-select_dis, select_dis]
    :return: Id_list, rel_up, Id_dict 周围车信息，相对距离，周围车的id
    """
    # 获取自动驾驶车的Y坐标和X坐标和纵向速度和车道
    y_pos = traci.vehicle.getPosition(control_vehicle)[0]
    x_pos = traci.vehicle.getPosition(control_vehicle)[1]
    y_speed = traci.vehicle.getSpeed(control_vehicle)
    ego_lane = traci.vehicle.getLaneID(control_vehicle)
    
    # 将自动驾驶车要用到的周围车的信息保存在map_ve字典里，key是周围车辆的ID，每个周围车辆的数据为
    # [该车相对于自动驾驶车的相对纵向距离，相对横向距离，相对速度]
    map_ve = {} # 记录车辆信息
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
    Id_dict = {}
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
    # 如果该方向没车，对于自己车道上的则用远的mask车辆代替，如果是旁边车道的得判断到底有没有旁边车道，
    # 有的话也是用远的mask来代替，否则用纵向距离为0的车辆代替，表示这个方向没车道，不能去
    # up[0][-1]指车辆ID, map_ve[up[0][-1]]是要收集的车辆信息
    if len(up) >= 1:
        Id_list.append(map_ve[up[0][-1]])
        Id_dict['up'] = up[0][-1]
    else:
        Id_list.append([200, 0, 0])
        Id_dict['up'] = ''

    upright = sorted(upright, key=lambda x: x[0])
    if len(upright) >= 1:
        Id_list.append(map_ve[upright[0][-1]])
        Id_dict['upright'] = upright[0][-1]
    else:
        if '0' in ego_lane:
            Id_list.append([0, -3.2, 0])
            Id_dict['upright'] = ''
        else:
            Id_list.append([200, -3.2, 0])
            Id_dict['upright'] = ''

    upleft = sorted(upleft, key=lambda x: x[0])
    if len(upleft) >= 1:
        Id_list.append(map_ve[upleft[0][-1]])
        Id_dict['upleft'] = upleft[0][-1]
    else:
        if '4' in ego_lane:
            Id_list.append([0, 3.2, 0])
            Id_dict['upleft'] = ''
        else:
            Id_list.append([200, 3.2, 0])
            Id_dict['upleft'] = ''

    down = sorted(down, key=lambda x: x[0])
    if len(down) >= 1:
        Id_list.append(map_ve[down[-1][-1]])
        Id_dict['down'] = down[-1][-1]
    else:
        Id_list.append([-200, 0, 0])
        Id_dict['down'] = ''

    downright = sorted(downright, key=lambda x: x[0])
    if len(downright) >= 1:
        Id_list.append(map_ve[downright[-1][-1]])
        Id_dict['downright'] = downright[-1][-1]
    else:
        if '0' in ego_lane:
            Id_list.append([0, -3.2, 0])
            Id_dict['downright'] = ''
        else:
            Id_list.append([-200, -3.2, 0])
            Id_dict['downright'] = ''

    downleft = sorted(downleft, key=lambda x: x[0])
    if len(downleft) >= 1:
        Id_list.append(map_ve[downleft[-1][-1]])
        Id_dict['downleft'] = downleft[-1][-1]
    else:
        if '4' in ego_lane:
            Id_list.append([0, 3.2, 0])
            Id_dict['downleft'] = ''
        else:
            Id_list.append([-200, 3.2, 0])
            Id_dict['downleft'] = ''

    # # 得到自动驾驶车自己所在的车道
    if '0' in ego_lane:
        ego_l=-1.0
    elif '1' in ego_lane:
        ego_l=-0.5
    elif '2' in ego_lane:
        ego_l=0
    elif '3' in ego_lane:
        ego_l=0.5
    elif '4' in ego_lane:
        ego_l=1
    else:
        ego_l=5
        print("CODE LOGIC ERROR!")

    # Id_list.append([y_speed/25, cal_a/3, ego_l]) # 其他论文里也是这样加ego车辆数据
    Id_list.append([y_pos/3100, ego_l, y_speed/25])

    # 归一化，需要，要不然容易边界值
    for i in range(6):
        Id_list[i][0] = Id_list[i][0]/select_dis
        Id_list[i][1] = Id_list[i][1]/3.2
        Id_list[i][2] = Id_list[i][2]/25

    # 为了得到与前车的相对纵向距离和相对纵向速度，前方车辆可能不在200m内，不能直接用up中的数据
    relspace = 100000
    relspeed = 0
    for v in vehicle_list:
        if v == control_vehicle:
            continue
        if traci.vehicle.getPosition(v)[0] - y_pos > 0 and traci.vehicle.getLaneID(v) == ego_lane:
            if traci.vehicle.getPosition(v)[0] - y_pos < relspace:
                relspace = traci.vehicle.getPosition(v)[0] - y_pos - 5 #vehicle-length: 5
                relspeed = traci.vehicle.getSpeed(v) - y_speed
    
    rel_up = {'relspace': relspace, 'relspeed':relspeed}
    
    # 周围车的3个相对信息，相对距离，周围车分方向记录id
    return Id_list, rel_up, Id_dict


def train(agent, control_vehicle, episode, target_dir, CL_Stage):
    '''
    - 根据不同stage以及get_all中ego所在lane，修改其target lanes
    - choose_action 得到动作 change_lane, action_acc
    - 记录pre数据
    - 根据 change_lane判断是否撞墙，若撞墙，结束回合
    - 根据change_lane, action_acc变道变速
    - 执行，simulateionStep
    - 记录cur数据
    - 计算reward
    - 查询是否发生碰撞

    :return: collision 
    '''
    print()
    print(f"+++++++++++++++ epo: {episode}  CL_Stage: {CL_Stage} ++++++++++++++++")
    print(f"++++++++++++++ {OUT_DIR} ++++++++++++++++")
    global TRAIN
    global df_record
    stage = CL_Stage # 当前train是在哪个CL的stage
    
    # 1. 根据不同stage以及get_all中ego所在lane，修改其target lanes
    all_vehicle, rel_up, v_dict = get_all(control_vehicle, 200)
    print("$ v_dict", v_dict)
    
    # target_dir_inits 编码
    tl_list = [[0,1,0,0,0,0,1], [1,1,0,1,1,1,0], [1,0,1,1,0,0,0]] # 0 是前方右转，1是直行，2是前方左转
    tl_code = tl_list[target_dir] # 根据target_dir获得对应direction的tl_code
    
    # 2. choose_action 得到动作 change_lane, action_acc
    if TRAIN:
        action_lc_int, action_acc, all_action_parameters = agent.choose_action(np.array(all_vehicle), tl_code) # 返回离散lane change ，连续acc，参数
    else:
        action_lc_int, action_acc, all_action_parameters = agent.choose_action(np.array(all_vehicle), tl_code, train = False)
    
    inf = -10 # 撞墙惩罚
    inf_car = -10 # 撞车惩罚
    done = 0 # 回合结束标志
    
    action_change_dict = {0: 'left', 1: 'keep', 2:'right'}
    change_lane = action_change_dict[action_lc_int] # 0车道右车道在-8.0；1车道在-4.8；2车道左车道在-1.6
    
    # 3. 记录pre数据
    pre_ego_info_dict = {"speed": traci.vehicle.getSpeed(control_vehicle), 
                         "acc":traci.vehicle.getAcceleration(control_vehicle), 
                         "LaneID": traci.vehicle.getLaneID(control_vehicle),
                         # "LaneIndex": traci.vehicle.getLaneIndex(control_vehicle), 
                         "position": traci.vehicle.getPosition(control_vehicle)}
    print("$ pre_ego_info_dict")
    pp.pprint(pre_ego_info_dict, indent = 5)
    
    get_all_info = [] # 记录与前后车的距离
    get_all_info.append('v_dict')
    if v_dict['up'] != '':
        get_all_info.append(("up", v_dict['up'], traci.vehicle.getPosition(v_dict['up'])[0] - pre_ego_info_dict['position'][0]))
    if v_dict['upright'] != '':
        get_all_info.append(("upright", v_dict['upright'], traci.vehicle.getPosition(v_dict['upright'])[0] - pre_ego_info_dict['position'][0]))
    if v_dict['upleft'] != '':
        get_all_info.append(("upleft", v_dict['upleft'], traci.vehicle.getPosition(v_dict['upleft'])[0] - pre_ego_info_dict['position'][0]))
    if v_dict['down'] != '':
        get_all_info.append(("down", v_dict['down'], pre_ego_info_dict['position'][0] - traci.vehicle.getPosition(v_dict['down'])[0]))
    if v_dict['downright'] != '':
        get_all_info.append(("downright", v_dict['downright'], pre_ego_info_dict['position'][0] - traci.vehicle.getPosition(v_dict['downright'])[0]))
    if v_dict['downleft'] != '':
        get_all_info.append(("downleft", v_dict['downleft'], pre_ego_info_dict['position'][0] - traci.vehicle.getPosition(v_dict['downleft'])[0]))
            
    # 4. 根据 change_lane判断是否撞墙，若撞墙，结束回合
    collision=0
    loss_actor = 0
    Q_loss = 0
    if 'EA_0' == pre_ego_info_dict["LaneID"] and change_lane=='right':
        collision=1
        done = 1
        train_step = agent._step
        print(f"\t ====== train_step:{train_step}  target_dir:{target_dir} ======")

        print('$ transition')
        pp.pprint({'obs': all_vehicle, 'act_lc': action_lc_int, 'act_param': all_action_parameters, 
                  'rew': inf, 'next_obs': np.zeros((7,3)), 'done': done}, indent = 5)
        agent.store_transition(all_vehicle, tl_code, action_lc_int, all_action_parameters, inf, np.zeros((7,3)), tl_code, done)
        df_record = df_record.append(pd.DataFrame([[stage,episode, train_step, pre_ego_info_dict['position'][0], 
                                                    target_dir, pre_ego_info_dict['LaneID'], 
                                                    pre_ego_info_dict['speed'], action_lc_int, pre_ego_info_dict['acc'], action_acc, change_lane,
                                                    inf, traci.vehicle.getAcceleration(v_dict['down']) if v_dict['down'] != '' else inf,  
                                                    inf, 0, 0, 0, 0, 0, 0, done, all_vehicle, np.zeros((7,3))]], columns = cols))
        print("====================右右右右车道撞墙墙墙墙===================")
        return collision, loss_actor, Q_loss

    if 'EA_4' == pre_ego_info_dict["LaneID"] and change_lane=='left':
        collision=1
        done = 1
        train_step = agent._step
        print(f"\t ====== train_step:{train_step}  target_dir:{target_dir} ======")
        print('$ transition')
        pp.pprint({'obs': all_vehicle, 'act_lc': action_lc_int, 'act_param': all_action_parameters, 
                  'rew': inf, 'next_obs': np.zeros((7,3)), 'done': done}, indent = 5)
        agent.store_transition(all_vehicle, tl_code, action_lc_int, all_action_parameters, inf, np.zeros((7,3)), tl_code, done)
        df_record = df_record.append(pd.DataFrame([[stage,episode, train_step, pre_ego_info_dict['position'][0], 
                                                    target_dir, pre_ego_info_dict['LaneID'], 
                                                    pre_ego_info_dict['speed'], action_lc_int, pre_ego_info_dict['acc'], action_acc, change_lane, 
                                                    inf, traci.vehicle.getAcceleration(v_dict['down']) if v_dict['down'] != '' else inf,  
                                                    inf, 0, 0, 0, 0, 0, 0, done, all_vehicle, np.zeros((7,3))]], columns = cols))
        print("====================左左左左车道撞墙墙墙墙===================")
        return collision, loss_actor, Q_loss
    
    # 5. 根据change_lane, action_acc变道变速
    # 计算速度，没有限速控制
    sp = traci.vehicle.getSpeed(control_vehicle) + action_acc*0.5 # 0.5s simulate一次
    traci.vehicle.setSpeed(control_vehicle, sp) # 将速度设置好
    print('$ speed ', sp)
    
    # 变道处理
    if change_lane=='left':
        if '0' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'EA_1', traci.vehicle.getLanePosition(control_vehicle))
        elif '1' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'EA_2', traci.vehicle.getLanePosition(control_vehicle))
        elif '2' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'EA_3', traci.vehicle.getLanePosition(control_vehicle))
        elif '3' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'EA_4', traci.vehicle.getLanePosition(control_vehicle))
    elif change_lane=='right':
        if '1' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'EA_0', traci.vehicle.getLanePosition(control_vehicle))
        elif '2' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'EA_1', traci.vehicle.getLanePosition(control_vehicle))
        elif '3' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'EA_2', traci.vehicle.getLanePosition(control_vehicle))
        elif '4' in pre_ego_info_dict["LaneID"]:
            traci.vehicle.moveTo(control_vehicle, 'EA_3', traci.vehicle.getLanePosition(control_vehicle))    
    
    # 6. 执行
    # ================================执行 ==================================
    traci.simulationStep()
    train_step = agent._step
    print("\t ################ 执行 ###################")
    print(f"\t ====== train_step:{train_step}  target_dir:{target_dir} ======")
    
    # 7. 记录cur数据
    new_all_vehicle, new_rel_up, new_v_dict = get_all(control_vehicle, 200)
    print("$ new_v_dict", new_v_dict)
    # print("\t ", new_v_dict)

    cur_ego_info_dict = {"speed": traci.vehicle.getSpeed(control_vehicle), 
                     "acc":traci.vehicle.getAcceleration(control_vehicle), 
                     "LaneID": traci.vehicle.getLaneID(control_vehicle), 
                     # "LaneIndex": traci.vehicle.getLaneIndex(control_vehicle), 
                     "position": traci.vehicle.getPosition(control_vehicle)}
    
    # print("cur_ego_info_dict", cur_ego_info_dict)
    print("$ cur_ego_info_dict")
    pp.pprint(cur_ego_info_dict, indent = 5)
    
    # 8. 计算reward
    e = 0.000001 # 避免分母为0
    if 0 <= new_rel_up['relspeed'] < e:
        new_rel_up['relspeed'] = e
    if -e < new_rel_up['relspeed'] < 0:
        new_rel_up['relspeed'] = -e
    
    y_ttc=-new_rel_up['relspace']/new_rel_up['relspeed'] # time to collision
    max_speed = 25
    if cur_ego_info_dict['speed'] > max_speed:
        r_efficiency = math.exp(max_speed - cur_ego_info_dict['speed'])
    else:
        r_efficiency = cur_ego_info_dict['speed'] / max_speed
    
    if 0 < y_ttc < 4: 
        r_safe = np.log(y_ttc/4)
    else:
        r_safe = 0
    
    if r_safe < -2: # 对r_safe 进行裁剪
        r_safe = -2
        
    r_comfort = ((np.abs(pre_ego_info_dict['acc']-cur_ego_info_dict['acc'])/0.1) ** 2) / 3600 

    r_tl = 0
    if cur_ego_info_dict['LaneID'] != '':
        if target_dir == 0:
            r_tl = -(0.0005 * (pre_ego_info_dict['position'][0] - RL_CONTROL) ) * abs(int(cur_ego_info_dict['LaneID'][-1]) - 0) *1/4
        elif target_dir == 1:
            if int(cur_ego_info_dict['LaneID'][-1]) == 4 or int(cur_ego_info_dict['LaneID'][-1]) == 0:
                r_tl = -(0.0005 * (pre_ego_info_dict['position'][0] - RL_CONTROL) ) *1/4
            else:
                r_tl = 0
        else:
            if int(cur_ego_info_dict['LaneID'][-1]) == 4 or int(cur_ego_info_dict['LaneID'][-1]) == 3:
                r_tl = 0
            else:
                r_tl = -(0.0005 * (pre_ego_info_dict['position'][0] - RL_CONTROL) ) * abs(int(cur_ego_info_dict['LaneID'][-1]) - 3) *1/4

    # # add penalty to discourage lane_change behavior fluctuation
    # if PRE_LANE == None:
    #     r_fluc = 0
    # else:
    #     r_fluc = -abs(cur_ego_info_dict['LaneIndex'] - PRE_LANE) * (1-abs(r_tl)) * 0.1
    # r_fluc=0
    # globals()['PRE_LANE'] = cur_ego_info_dict['LaneIndex']
    r_fluc = 0 # 取消r_fluc的作用
    
    get_all_info.append("new_v_dict")
    if new_v_dict['up'] != '':
        get_all_info.append(("up", new_v_dict['up'], traci.vehicle.getPosition(new_v_dict['up'])[0] - cur_ego_info_dict['position'][0]))
    if new_v_dict['upright'] != '':
        get_all_info.append(("upright", new_v_dict['upright'], traci.vehicle.getPosition(new_v_dict['upright'])[0] - cur_ego_info_dict['position'][0]))
    if new_v_dict['upleft'] != '':
        get_all_info.append(("upleft", new_v_dict['upleft'], traci.vehicle.getPosition(new_v_dict['upleft'])[0] - cur_ego_info_dict['position'][0]))
    if new_v_dict['down'] != '':
        get_all_info.append(("down", new_v_dict['down'], cur_ego_info_dict['position'][0] - traci.vehicle.getPosition(new_v_dict['down'])[0]))
    if new_v_dict['downright'] != '':
        get_all_info.append(("downright", new_v_dict['downright'], cur_ego_info_dict['position'][0] - traci.vehicle.getPosition(new_v_dict['downright'])[0]))
    if new_v_dict['downleft'] != '':
        get_all_info.append(("downleft", new_v_dict['downleft'], cur_ego_info_dict['position'][0] - traci.vehicle.getPosition(new_v_dict['downleft'])[0]))
    
    cur_reward = r_safe + r_efficiency - r_comfort + r_fluc + r_tl * 2
    
    # 9. 查询自动驾驶车是否发生碰撞
    collision=0
    loss_actor = 0
    Q_loss = 0
    if control_vehicle in traci.simulation.getCollidingVehiclesIDList():
        print('=====================collision==', traci.simulation.getCollidingVehiclesIDList(), control_vehicle) # 第一个是自动驾驶车辆
        print("==========================发生了撞车=========================")
        if control_vehicle == traci.simulation.getCollidingVehiclesIDList()[1]:
            print("与前方车辆撞")
            y_ttc = 0
            another_co_id = traci.simulation.getCollidingVehiclesIDList()[0]
        elif control_vehicle == traci.simulation.getCollidingVehiclesIDList()[0]:
            print("与后方车辆撞")
            another_co_id = traci.simulation.getCollidingVehiclesIDList()[1]
        get_all_info.append(("another_co_id", another_co_id))
        print("$ another_co_id ", another_co_id)
        # print("**** collision ego ****", traci.vehicle.getPosition(control_vehicle)[0], # ego
        #       traci.vehicle.getPosition(control_vehicle)[1], 
        #       "pre_ego", pre_ego_info_dict['position'][0], pre_ego_info_dict['position'][1])
        # print("**** collision another****", traci.vehicle.getPosition(another_co_id)[0], # 另一个碰撞的vehicle
        #       traci.vehicle.getPosition(another_co_id)[1])
        collision = 1
        done = 1
        print('$ transition')
        pp.pprint({'obs': all_vehicle, 'act_lc': action_lc_int, 'act_param': all_action_parameters, 
                  'rew': [inf_car, ('r_safe', r_safe), ('r_efficiency', r_efficiency), ('r_comfort', r_comfort), ('r_tl', r_tl)], 
                  'next_obs': new_all_vehicle, 'done': done}, indent = 5)
        agent.store_transition(all_vehicle, tl_code, action_lc_int, all_action_parameters, inf_car, new_all_vehicle, tl_code, done)
        df_record = df_record.append(pd.DataFrame([[stage,episode, train_step, cur_ego_info_dict['position'][0], 
                                            target_dir, cur_ego_info_dict['LaneID'], 
                                            cur_ego_info_dict['speed'], action_lc_int, cur_ego_info_dict['acc'], action_acc, change_lane,
                                            y_ttc, traci.vehicle.getAcceleration(v_dict['down']) if v_dict['down'] != '' else inf,  
                                            inf_car, r_safe, r_efficiency, r_comfort, r_tl, r_fluc, get_all_info, done, 
                                            all_vehicle, new_all_vehicle]], columns = cols))
        return collision, loss_actor, Q_loss
    

    print('$ transition')
    pp.pprint({'obs': all_vehicle, 'act_lc': action_lc_int, 'act_param': all_action_parameters, 
              'rew': [cur_reward, ('r_safe', r_safe), ('r_efficiency', r_efficiency), ('r_comfort', r_comfort), ('r_tl', r_tl)], 
              'next_obs': new_all_vehicle, 'done': done}, indent = 5)
    agent.store_transition(all_vehicle, tl_code, action_lc_int, all_action_parameters, cur_reward, new_all_vehicle, tl_code, done)
    df_record = df_record.append(pd.DataFrame([[stage,episode, train_step, cur_ego_info_dict['position'][0], 
                                                target_dir, cur_ego_info_dict['LaneID'], 
                                                cur_ego_info_dict['speed'], action_lc_int, cur_ego_info_dict['acc'], action_acc, change_lane,
                                                y_ttc, traci.vehicle.getAcceleration(v_dict['down']) if v_dict['down'] != '' else inf,  
                                                cur_reward, r_safe, r_efficiency, r_comfort, r_tl, r_fluc, get_all_info, done, 
                                                all_vehicle, new_all_vehicle]], columns = cols))
    
    if TRAIN and (len(agent.memory) > agent.minimal_size):
    # if TRAIN and (agent._step > agent.memory_size):
        loss_actor, Q_loss = agent.learn()
        print('$ actor的loss ', loss_actor, 'q的loss ', Q_loss)
    else:
        loss_actor = Q_loss = None
    
    return collision, loss_actor, Q_loss


def main_train():
    a_dim = 1 # one parameter for a continous action
    s_dim = 3 * 7    # ego vehicle + 6 surrounding vehicle
    agent_param={
        "s_dim": s_dim,
        "a_dim": a_dim,
        "acc3": True,
        "Kaiming_normal": False,
        "memory_size": 40000,
        "minimal_size": 5000,
        "batch_size": 128,
        "n_step": 3,
        "device": DEVICE
    }

    for out_dir in OUT_DIRs:
        if re.search(r"rainbow_linear", out_dir):
            # rainbow_linear model test
            agent = RainbowPDQNAgent(
                state_dim=agent_param["s_dim"],
                action_dim=agent_param["a_dim"],
                acc3=agent_param["acc3"],
                Kaiming_normal=agent_param["Kaiming_normal"],
                memory_size=agent_param["memory_size"],
                minimal_size=agent_param["minimal_size"],
                batch_size=agent_param["batch_size"],
                n_step=agent_param["n_step"],
                device=agent_param["device"])
        elif re.search(r"linear", out_dir):
            # linear model test
            agent = LinearPDQNAgent(
                state_dim=agent_param["s_dim"],
                action_dim=agent_param["a_dim"],
                acc3=agent_param["acc3"],
                Kaiming_normal=agent_param["Kaiming_normal"],
                memory_size=agent_param["memory_size"],
                minimal_size=agent_param["minimal_size"],
                batch_size=agent_param["batch_size"],
                device=agent_param["device"])
        elif re.search(r"lstm", out_dir):
            # lstm model test
            agent = LstmPDQNAgent(
                state_dim=agent_param["s_dim"],
                action_dim=agent_param["a_dim"],
                acc3=agent_param["acc3"],
                Kaiming_normal=agent_param["Kaiming_normal"],
                memory_size=agent_param["memory_size"],
                minimal_size=agent_param["minimal_size"],
                batch_size=agent_param["batch_size"],
                device=agent_param["device"])

        losses_actor = [] # 不需要看第一个memory 即前20000步
        losses_episode = [] # 存一个episode的loss，一个episode结束后清除内容
        switch_cnt = 0 # 某个stage中epo的数量
        swicth_min = 50 # 一个stage中最少要训练swicth_min个epo
            
        # (1) 区分train和test的参数设置，以及output位置
        episode_num = 400 # test的episode上限
        CL_Stage = 4 # test都在最后一个stage进行
        agent.load_state_dict(torch.load(f"{out_dir}/net_params.pth", map_location=DEVICE)) 
        globals()['RL_CONTROL']=1100
        globals()['OUT_DIR']=f"{out_dir}/test" 
        
        # 创建output文件夹
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        else:
            shutil.rmtree(OUT_DIR, ignore_errors=True)
            os.makedirs(OUT_DIR)
        
        # (2) 分episode进行 train / test
        for epo in range(episode_num): 
            if re.search(r"lstm", out_dir):
                agent.init_hidden()
            truncated = False # 撞车
            target_dir_init = None # 初始的target_dir，目标转向方向
            
            # (3) 根据不同的CL_Stage启动对应的sumoCmd
            # stage 1 在5车道中模拟2车道，之后的stage都是5车道
            cfg_path = f"{sumo_dir}cfg_CL2_high.sumocfg"
            sumoCmd = [sumoBinary, "-c", cfg_path, "--log", f"{OUT_DIR}/logfile_{CL_Stage}.txt"]
            traci.start(sumoCmd)
            ego_index = 5 + epo % 20   # 选取随机车道第index辆出发的车为我们的自动驾驶车
            if CL_Stage == 1:            
                departLane=np.random.choice([0,1,3,4]) # 除2车道外，随机初始ego的lane
                ego_index_str = str(departLane) + '_' + str(ego_index)
                if departLane == 0 or departLane == 1: # 根据ego生成的位置赋予target_dir_init
                    target_dir_init = random.randint(0, 1)
                else:
                    target_dir_init = random.randint(1, 2)
            else:
                ego_index_str = str(np.random.randint(0,5)) + '_' + str(ego_index) # ego的id为'1_$index$', 如index为20,id='1_20'
                target_dir_init = random.randint(0, 2) # ego的变道方向，从0 1 2中取
                
            control_vehicle = '' # ego车辆的id
            ego_show = False # ego车辆是否出现过

            global df_record
            df_record = pd.DataFrame(columns = cols)
            
            print(f"+++++++{epo}  STAGE:{CL_Stage} +++++++++++++")
            print(f"++++++++++++++++++ {OUT_DIR} +++++++++++++++++++++++")
            
            # (4) 一个episode中的交互
            while traci.simulation.getMinExpectedNumber() > 0:
                # 1. 得到道路上所有的车辆ID
                vehicle_list = traci.vehicle.getIDList()
                if CL_Stage == 1:
                    for v in vehicle_list:
                        if traci.vehicle.getTypeID(v) == 'CarA':
                            traci.vehicle.setLaneChangeMode(v, 0b000000000000) # 2车道的车不能变道，其他车道的车可以变道
                
                # 2. 找到我们控制的自动驾驶车辆
                # 2.1 如果此时自动驾驶车辆已出现，设置其为绿色, id为'1_$ego_index$'
                if ego_index_str in vehicle_list:
                    control_vehicle = ego_index_str
                    ego_show = True
                    traci.vehicle.setColor(control_vehicle,  (0,225,0,255))
                # 2.2 如果此时自动驾驶车辆还未出现
                if ego_show == False:
                    traci.simulationStep() # 2个step出现1辆车
                    continue
                # 2.3 如果已经出现了而且撞了就退出
                if ego_show and control_vehicle not in vehicle_list:
                    print("=====================已经出现了而且撞了================")
                    break
                
                # 3 在非RL控制路段中采取其他行驶策略，控制的路段为RL_CONTROL-3100这2000m的距离
                # 3.1 在0-RL_CONTROLm是去掉模拟器自带算法中的变道，但暂时保留速度控制
                traci.vehicle.setLaneChangeMode(control_vehicle, 0b000000000000)
                # print("自动驾驶车的位置====================", traci.vehicle.getPosition(control_vehicle)[0])     
                if traci.vehicle.getPosition(control_vehicle)[0] < RL_CONTROL:
                    traci.simulationStep()
                    continue
                # 3.2 在大于3100m
                if traci.vehicle.getPosition(control_vehicle)[0] > 3100:
                    print("=======================距离超过3100====================")
                    break
        
                # 4 在RL控制路段中收集自动驾驶车周围车辆的信息，并设置周围车辆
                # 4.2 将所有后方车辆设置为不变道
                # for vehicle in vehicle_list:
                #     if traci.vehicle.getPosition(vehicle)[0] < traci.vehicle.getPosition(control_vehicle)[0]:
                #         traci.vehicle.setLaneChangeMode(vehicle, 0b000000000000)
        
                # 4.3 去除自动驾驶车默认的跟车和换道模型，为模型训练做准备
                traci.vehicle.setSpeedMode(control_vehicle, 00000)
                traci.vehicle.setLaneChangeMode(control_vehicle, 0b000000000000)
                
                # 5 模型训练
                collision, loss_actor, _ = train(agent, control_vehicle, epo,  target_dir_init, CL_Stage) # 模拟一个时间步
                if collision:
                    truncated = True
                    break

                if loss_actor is not None:
                    losses_actor.append(loss_actor)
                    losses_episode.append(loss_actor)
                
            # (5) 判断在某个stage的训练情况，收敛了就进入下一个stage
            # globals()['PRE_LANE']=None
            traci.close(wait=True)
            switch_cnt += 1
            if TRAIN and not truncated and switch_cnt >= swicth_min and np.average(losses_episode)<=0.05:
                if CL_Stage == 1:
                    CL_Stage = 2
                    switch_cnt = 0 # 进入下一个stage，switch_cnt清0
                elif CL_Stage == 2:
                    CL_Stage = 3
                    switch_cnt = 0
                elif CL_Stage == 3:
                    CL_Stage = 4
                    switch_cnt = 0
                # elif CL_Stage == 4:
                #     CL_Stage = 1
                #     switch_cnt = 0

            losses_episode.clear()
            
            # 保存
            df_record.to_csv(f"{OUT_DIR}/df_record_epo_{epo}.csv", index = False)
            if TRAIN:
                torch.save(agent.state_dict(), f"./{OUT_DIR}/net_params.pth") 
                pd.DataFrame(data=losses_actor).to_csv(f"./{OUT_DIR}/losses.csv")


if __name__ == '__main__':
    main_train() 
#    pass




