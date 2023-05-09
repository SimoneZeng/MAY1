# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:08:26 2023

在d3qn_train_3r.py的基础上，将 DQN 模型改为 PDQN 模型
删掉了一些没用的代码

在pdqn_train_3r 的基础上，修改bug，侧方有车时的store。
修改惩罚都为-10

使用one_way2.sumocfg
Kaiming_normal
撞墙改为-10 
速度权重改为0.4
去掉撞车、撞墙的控制
把存储记录改为current的信息，例如cur_ego_info_dict['position'][0]


修改模型的 ego vehicle输入信息
使用tlcode 和 tl_reward
加上ttc 的 clip [-2, 0]
增加撞车id的记录
还没有增加相对加速度

efficiency 从[0, 0.4]
r_tl 从[-0.2,0]调整到[-1, 0]
comfort 保持[-1, 0]
r_safe保持clip [-2, 0]

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
import math
curPath=os.path.abspath(os.path.dirname(__file__))
rootPath=os.path.split(os.path.split(curPath)[0])[0]
sys.path.append(rootPath+'/sumo_test01')

#from model.pdqn_model_5tl_lstm import PDQNAgent
from model.pdqn_model_5tl_linear import PDQNAgent


# 引入地址 
sumo_path = os.environ['SUMO_HOME'] # "D:\\sumo\\sumo1.13.0"
# cfg_path1 = "D:\Git\MAY1\sumo\one_way_2l.sumocfg" # 1.在本地用这个cfg_path
# cfg_path2 = "D:\Git\MAY1\sumo\one_way_5l.sumocfg" # 1.在本地用这个cfg_path
cfg_path1 = "/data1/zengximu/sumo_test01/sumo/one_way_2l.sumocfg" # 2. 在服务器上用这个cfg_path
cfg_path2 = "/data1/zengximu/sumo_test01/sumo/one_way_5l.sumocfg" # 2. 在服务器上用这个cfg_path
OUT_DIR="result_pdqn_5l_ccl_linear"
sys.path.append(sumo_path)
sys.path.append(sumo_path + "/tools")
sys.path.append(sumo_path + "/tools/xml")
import traci # 在ubuntu中，traci和sumolib需要在tools地址引入之后import
from sumolib import checkBinary

# os.environ['CUDA_VISIBLE_DEVICES']='0, 1'  # 显卡使用
EPISODE_NUM=20000
TRAIN = True # False True
gui = False # False True # 是否打开gui
if gui == 1:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')
sumoCmd0 = [sumoBinary, "-c", cfg_path1, "--log", f"{OUT_DIR}/logfile.txt"]
sumoCmd1 = [sumoBinary, "-c", cfg_path2, "--log", f"{OUT_DIR}/logfile.txt"]

map_ve = {} # 记录车辆信息
auto_vehicle_a = 0
step = 0 # 统计总的训练次数
# 存储奖励值的数组，分epo存的
cols = ['stage','epo', 'train_step', 'position_y', 'target_direc', 'lane', 'speed', 
         'lc_int', 'fact_acc', 'acc', 'change_lane', 'r','r_safe', 'r_eff',
         'r_com', 'r_tl', 'r_fluc','other_record', 'done', 's', 's_']
df_record = pd.DataFrame(columns = cols)
action_change_dict = {0: 'left', 1: 'keep', 2:'right'}

np.random.seed(0)
random.seed(0)
torch.manual_seed(5)
tl_list = [[0,1,0,0,0,0,1], [1,1,0,1,1,1,0], [1,0,1,1,0,0,0]] # 0 是右车道
# different curriculum stages
# state 1: only ego vehicle, no surrounding vehicles
# state 2: ego + surrounding vehicles
# state 3: ego + surrounding vehicles + target lane
CURRICULUM_STAGE = 1
SWITCH_COUNT = 50 # the minimal episode count
PRE_LANE = None
RL_CONTROL = 500 # Rl agent take control after 500 meters
DEVICE = torch.device("cuda:3")

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
    # global auto_vehicle_a
    # if np.abs(traci.vehicle.getAcceleration(control_vehicle)) < 0.0001:
    #     cal_a = auto_vehicle_a
    # else:
    #     cal_a = traci.vehicle.getAcceleration(control_vehicle)
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
    
    # 周围车信息，相对距离，周围车id
    return Id_list, rel_up, Id_dict


def train(agent, control_vehicle, episode, target_lane):
    '''
    1. 该函数使用agent根据state得出lane change action，对应的 action_acc，和 all_action_parameters
    2. 执行速度控制
    3. 执行变道控制。其中，撞墙done为0，撞车done 为 1
    4. 计算出reward，包括 r_safe + r_efficiency - r_comfort + r_target_lane
    5. 存储信息到df_record以及 agent.store_transition，注意agent需要存all_action_parameters
    :return: collision 
    '''
    print()
    global TRAIN
    global tl_list
    
    #get surrounding vehicles information
    all_vehicle, rel_up, v_dict = get_all(control_vehicle, 200)
    #change ego vehicle information for curriculum stage 1 and stage 2
    if CURRICULUM_STAGE != 3:
        if all_vehicle[6][1]==-1:
            target_lane=0
        elif all_vehicle[6][1]==-0.5 or all_vehicle[6][1]==0:
            target_lane=1
        elif all_vehicle[6][1]==0.5:
            target_lane=random.choice([1, 2])
        else:
            target_lane=2
    print("v_dict", v_dict)
    tl_code = tl_list[target_lane]

    if TRAIN:
        action_lc_int, action_acc, all_action_parameters = agent.choose_action(np.array(all_vehicle), tl_code) # 离散lane change ，连续acc，参数
    else:
        action_lc_int, action_acc, all_action_parameters = agent.choose_action(np.array(all_vehicle), tl_code, train = False)
    
    inf = -10 # 撞墙惩罚
    inf_car = -10 # 撞车惩罚
    done = 0 # 回合结束标志
    
    global action_change_dict
    change_lane = action_change_dict[action_lc_int] # 0车道右车道在-8.0；1车道在-4.8；2车道左车道在-1.6
    r_side = [] # 记录与前后车的距离
    
    # 1. 记录当前车辆的数据
    global auto_vehicle_a
    pre_ego_info_dict = {"speed": traci.vehicle.getSpeed(control_vehicle), 
                         "acc":traci.vehicle.getAcceleration(control_vehicle), 
                         "LaneID": traci.vehicle.getLaneID(control_vehicle),
                         "LaneIndex": traci.vehicle.getLaneIndex(control_vehicle), 
                         "position": traci.vehicle.getPosition(control_vehicle)}
    print("pre_ego_info_dict", pre_ego_info_dict)

    r_side.append('v_dict')
    if v_dict['up'] != '':
        r_side.append(("up", v_dict['up'], traci.vehicle.getPosition(v_dict['up'])[0] - pre_ego_info_dict['position'][0]))
    if v_dict['upright'] != '':
        r_side.append(("upright", v_dict['upright'], traci.vehicle.getPosition(v_dict['upright'])[0] - pre_ego_info_dict['position'][0]))
    if v_dict['upleft'] != '':
        r_side.append(("upleft", v_dict['upleft'], traci.vehicle.getPosition(v_dict['upleft'])[0] - pre_ego_info_dict['position'][0]))
    if v_dict['down'] != '':
        r_side.append(("down", v_dict['down'], pre_ego_info_dict['position'][0] - traci.vehicle.getPosition(v_dict['down'])[0]))
    if v_dict['downright'] != '':
        r_side.append(("downright", v_dict['downright'], pre_ego_info_dict['position'][0] - traci.vehicle.getPosition(v_dict['downright'])[0]))
    if v_dict['downleft'] != '':
        r_side.append(("downleft", v_dict['downleft'], pre_ego_info_dict['position'][0] - traci.vehicle.getPosition(v_dict['downleft'])[0]))
    
    # print("++ 上一步与周围车的距离 ++")
    # if v_dict['up'] != '':
    #     print("up", traci.vehicle.getPosition(v_dict['up'])[0] - pre_ego_info_dict['position'][0])
    # if v_dict['upright'] != '':
    #     print("upright", traci.vehicle.getPosition(v_dict['upright'])[0] - pre_ego_info_dict['position'][0])
    # if v_dict['upleft'] != '':
    #     print("upleft", traci.vehicle.getPosition(v_dict['upleft'])[0] - pre_ego_info_dict['position'][0])
    # if v_dict['down'] != '':
    #     print("down", pre_ego_info_dict['position'][0] - traci.vehicle.getPosition(v_dict['down'])[0])
    # if v_dict['downright'] != '':
    #     print("downright", pre_ego_info_dict['position'][0] - traci.vehicle.getPosition(v_dict['downright'])[0])
    # if v_dict['downleft'] != '':
    #     print("downleft", pre_ego_info_dict['position'][0] - traci.vehicle.getPosition(v_dict['downleft'])[0])
    # 计算速度，控制限速
    sp = traci.vehicle.getSpeed(control_vehicle) + action_acc*0.5 # 0.5s simulate一次
    # if sp> 25:
    #     sp = 25
    # if sp < 0:
    #     sp = 0
    traci.vehicle.setSpeed(control_vehicle, sp) # 将速度设置好
    
    # print('@@@@@@ action_lc_int', action_lc_int, 'action_acc', action_acc, 'all_action_parameters', all_action_parameters)
    print('@@@@@@ speed ', sp)
    
    global df_record
        
    # 2. 撞墙处理，车道从左到右是2,1,0
    collision=0
    loss_actor = 0
    Q_loss = 0
    if 'EA_0' == pre_ego_info_dict["LaneID"] and change_lane=='right':
        collision=1
        done = 1
        train_step = agent._step
        print(f"---- train_step:{train_step}  target_lane:{target_lane} ----")
        print(f"before store---obs:{all_vehicle} \n"
            f"act:{action_lc_int} act_param:{all_action_parameters} \n" 
            f"rew:{inf}\n"
            f"next_obs:{np.zeros((7,3))} \ndone:{done}" )
        agent.store_transition(all_vehicle, tl_code, action_lc_int, all_action_parameters, inf, np.zeros((7,3)), tl_code, done)
        df_record = df_record.append(pd.DataFrame([[CURRICULUM_STAGE,episode, train_step, pre_ego_info_dict['position'][0], 
                                                    target_lane, pre_ego_info_dict['LaneID'], 
                                                    pre_ego_info_dict['speed'], action_lc_int, pre_ego_info_dict['acc'], action_acc, change_lane, 
                                                    inf, 0, 0, 0, 0, 0, 0, done, all_vehicle, np.zeros((7,3))]], columns = cols))
        print("====================右右右右车道撞墙墙墙墙===================")
        return collision, loss_actor, Q_loss

    if 'EA_4' == pre_ego_info_dict["LaneID"] and change_lane=='left':
        collision=1
        done = 1
        train_step = agent._step
        print(f"---- train_step:{train_step}  target_lane:{target_lane} ----")
        print(f"before store---obs:{all_vehicle} \n"
            f"act:{action_lc_int} act_param:{all_action_parameters} \n" 
            f"rew:{inf}\n"
            f"next_obs:{np.zeros((7,3))} \ndone:{done}" )
        agent.store_transition(all_vehicle, tl_code, action_lc_int, all_action_parameters, inf, np.zeros((7,3)), tl_code, done)
        df_record = df_record.append(pd.DataFrame([[CURRICULUM_STAGE,episode, train_step, pre_ego_info_dict['position'][0], 
                                                    target_lane, pre_ego_info_dict['LaneID'], 
                                                    pre_ego_info_dict['speed'], action_lc_int, pre_ego_info_dict['acc'], action_acc, change_lane, 
                                                    inf, 0, 0, 0, 0, 0, 0, done, all_vehicle, np.zeros((7,3))]], columns = cols))
        print("====================左左左左车道撞墙墙墙墙===================")
        return collision, loss_actor, Q_loss
    
    # 3. 变道处理
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
    
    # ================================执行 ==================================
    traci.simulationStep()
    # print("\n \n")
    print("################ 执行 ###################")
    
    # 4. 获取动作执行后的状态
    new_all_vehicle, new_rel_up, new_v_dict = get_all(control_vehicle, 200)
    print("new_v_dict", new_v_dict)
    cur_ego_info_dict = {"speed": traci.vehicle.getSpeed(control_vehicle), 
                     "acc":traci.vehicle.getAcceleration(control_vehicle), 
                     "LaneID": traci.vehicle.getLaneID(control_vehicle), 
                     "LaneIndex": traci.vehicle.getLaneIndex(control_vehicle), 
                     "position": traci.vehicle.getPosition(control_vehicle)}
    
    print("cur_ego_info_dict", cur_ego_info_dict)
    # 避免分母为0
    e = 0.000001
    if 0 <= new_rel_up['relspeed'] < e:
        new_rel_up['relspeed'] = e
    if -e < new_rel_up['relspeed'] < 0:
        new_rel_up['relspeed'] = -e
    
    # 计算reward
    y_ttc=-new_rel_up['relspace']/new_rel_up['relspeed'] # time to collision
    # r_efficiency = cur_ego_info_dict['speed']/25*0.8 - 0.8 # 范围是[-0.8, 0]
    # r_efficiency = cur_ego_info_dict['speed']/25*0.4 # 0.4 0.8
    max_speed = 25
    if cur_ego_info_dict['speed'] > max_speed:
        # fEff = 1
        r_efficiency = math.exp(max_speed - cur_ego_info_dict['speed'])
    else:
        r_efficiency = cur_ego_info_dict['speed'] / max_speed
    
    if 0 < y_ttc < 4: 
        r_safe = np.log(y_ttc/4)
    else:
        r_safe = 0
    
    if r_safe < -2: # 对r_safe 进行裁剪
        r_safe = -2
        
    auto_vehicle_a = cur_ego_info_dict['acc']
    r_comfort = ((np.abs(pre_ego_info_dict['acc']-cur_ego_info_dict['acc'])/0.1) ** 2) / 3600 

    r_tl = 0
    if cur_ego_info_dict['LaneID'] != '':
        if target_lane == 0:
            r_tl = -(0.0005 * (pre_ego_info_dict['position'][0] - RL_CONTROL) ) * abs(int(cur_ego_info_dict['LaneID'][-1]) - 0) *1/4
        elif target_lane == 1:
            if int(cur_ego_info_dict['LaneID'][-1]) == 4 or int(cur_ego_info_dict['LaneID'][-1]) == 0:
                r_tl = -(0.0005 * (pre_ego_info_dict['position'][0] - RL_CONTROL) ) *1/4
            else:
                r_tl = 0
        else:
            if int(cur_ego_info_dict['LaneID'][-1]) == 4 or int(cur_ego_info_dict['LaneID'][-1]) == 3:
                r_tl = 0
            else:
                r_tl = -(0.0005 * (pre_ego_info_dict['position'][0] - RL_CONTROL) ) * abs(int(cur_ego_info_dict['LaneID'][-1]) - 3) *1/4

    # add penalty to discourage lane_change behavior fluctuation
    if PRE_LANE == None:
        r_fluc = 0
    else:
        r_fluc = -abs(cur_ego_info_dict['LaneIndex'] - PRE_LANE) * (1-abs(r_tl)) * 0.1
    r_fluc = 0
    globals()['PRE_LANE'] = cur_ego_info_dict['LaneIndex']
    
    # r_side = [] # 记录与前后车的距离
    r_side.append("new_v_dict")
    if new_v_dict['up'] != '':
        r_side.append(("up", new_v_dict['up'], traci.vehicle.getPosition(new_v_dict['up'])[0] - cur_ego_info_dict['position'][0]))
    if new_v_dict['upright'] != '':
        r_side.append(("upright", new_v_dict['upright'], traci.vehicle.getPosition(new_v_dict['upright'])[0] - cur_ego_info_dict['position'][0]))
    if new_v_dict['upleft'] != '':
        r_side.append(("upleft", new_v_dict['upleft'], traci.vehicle.getPosition(new_v_dict['upleft'])[0] - cur_ego_info_dict['position'][0]))
    if new_v_dict['down'] != '':
        r_side.append(("down", new_v_dict['down'], cur_ego_info_dict['position'][0] - traci.vehicle.getPosition(new_v_dict['down'])[0]))
    if new_v_dict['downright'] != '':
        r_side.append(("downright", new_v_dict['downright'], cur_ego_info_dict['position'][0] - traci.vehicle.getPosition(new_v_dict['downright'])[0]))
    if new_v_dict['downleft'] != '':
        r_side.append(("downleft", new_v_dict['downleft'], cur_ego_info_dict['position'][0] - traci.vehicle.getPosition(new_v_dict['downleft'])[0]))
        
    # print("++ 当前与周围车的距离 ++")
    # if new_v_dict['up'] != '':
    #     print("up", traci.vehicle.getPosition(new_v_dict['up'])[0] - cur_ego_info_dict['position'][0])
    # if new_v_dict['upright'] != '':
    #     print("upright", traci.vehicle.getPosition(new_v_dict['upright'])[0] - cur_ego_info_dict['position'][0])
    # if new_v_dict['upleft'] != '':
    #     print("upleft", traci.vehicle.getPosition(new_v_dict['upleft'])[0] - cur_ego_info_dict['position'][0])
    # if new_v_dict['down'] != '':
    #     print("down", cur_ego_info_dict['position'][0] - traci.vehicle.getPosition(new_v_dict['down'])[0])
    # if new_v_dict['downright'] != '':
    #     print("downright", cur_ego_info_dict['position'][0] - traci.vehicle.getPosition(new_v_dict['downright'])[0])
    # if new_v_dict['downleft'] != '':
    #     print("downleft", cur_ego_info_dict['position'][0] - traci.vehicle.getPosition(new_v_dict['downleft'])[0])
    
    
    # cur_reward = r_safe + r_efficiency - r_comfort
    if CURRICULUM_STAGE == 1:
        cur_reward = r_safe + r_efficiency - r_comfort + r_fluc
        r_tl = 0
    elif CURRICULUM_STAGE == 2:
        cur_reward = r_safe + r_efficiency - r_comfort + r_fluc
        r_tl = 0
    elif CURRICULUM_STAGE == 3:
        cur_reward = r_safe + r_efficiency - r_comfort + r_fluc + r_tl*2
    else:
        print("CODE LOGIC ERROR!")
    
    # 5. 查询自动驾驶车是否发生碰撞
    collision=0
    loss_actor = 0
    Q_loss = 0
    if control_vehicle in traci.simulation.getCollidingVehiclesIDList():
        print('=====================collision==', traci.simulation.getCollidingVehiclesIDList(), control_vehicle) # 第一个是自动驾驶车辆
        print("==========================发生了撞车=========================")
        if control_vehicle == traci.simulation.getCollidingVehiclesIDList()[1]:
            print("与前方车辆撞")
            another_co_id = traci.simulation.getCollidingVehiclesIDList()[0]
        elif control_vehicle == traci.simulation.getCollidingVehiclesIDList()[0]:
            print("与后方车辆撞")
            another_co_id = traci.simulation.getCollidingVehiclesIDList()[1]
        # r_side.append(traci.vehicle.getPosition(traci.simulation.getCollidingVehiclesIDList()[0])[0])
        r_side.append(("another_co_id", another_co_id))
        # another_co_direction = ''
        # print("v_dict", v_dict)
        # for key in v_dict:
        #     if v_dict[key] == another_co_id:
        #         another_co_direction = key
        print("**** collision ego ****", traci.vehicle.getPosition(control_vehicle)[0], # ego
              traci.vehicle.getPosition(control_vehicle)[1], 
              "pre_ego", pre_ego_info_dict['position'][0], pre_ego_info_dict['position'][1])
        print("**** collision another****", traci.vehicle.getPosition(another_co_id)[0], # 另一个碰撞的vehicle
              traci.vehicle.getPosition(another_co_id)[1])
        collision=1
        done = 1
        train_step = agent._step
        print(f"---- train_step:{train_step}  target_lane:{target_lane} ----")
        print(f"before store---obs:{all_vehicle} \n"
            f"act:{action_lc_int} act_param:{all_action_parameters} \n" 
            f"rew:{cur_reward} safe:{r_safe} efficiency:{r_efficiency} comfort:{r_comfort} target_lane_reward:{r_tl} fluctuation:{r_fluc}\n"
            f"next_obs:{new_all_vehicle} \ndone:{done}" )
        agent.store_transition(all_vehicle, tl_code, action_lc_int, all_action_parameters, inf_car, new_all_vehicle, tl_code, done)
        df_record = df_record.append(pd.DataFrame([[CURRICULUM_STAGE,episode, train_step, cur_ego_info_dict['position'][0], 
                                            target_lane, cur_ego_info_dict['LaneID'], 
                                            cur_ego_info_dict['speed'], action_lc_int, cur_ego_info_dict['acc'], action_acc, change_lane, 
                                            inf_car, r_safe, r_efficiency, r_comfort, r_tl, r_fluc, r_side, done, 
                                            all_vehicle, new_all_vehicle]], columns = cols))
        return collision, loss_actor, Q_loss
    
    train_step = agent._step
    print(f"---- train_step:{train_step}  target_lane:{target_lane} ----")
    print(f"before store---obs:{all_vehicle} \n"
        f"act:{action_lc_int} act_param:{all_action_parameters} \n" 
        f"rew:{cur_reward} safe:{r_safe} efficiency:{r_efficiency} comfort:{r_comfort} target_lane_reward:{r_tl} fluctuation:{r_fluc}\n"
        f"next_obs:{new_all_vehicle} \ndone:{done}" )
    agent.store_transition(all_vehicle, tl_code, action_lc_int, all_action_parameters, cur_reward, new_all_vehicle, tl_code, done)
    df_record = df_record.append(pd.DataFrame([[CURRICULUM_STAGE,episode, train_step, cur_ego_info_dict['position'][0], 
                                                target_lane, cur_ego_info_dict['LaneID'], 
                                                cur_ego_info_dict['speed'], action_lc_int, cur_ego_info_dict['acc'], action_acc, change_lane,
                                                cur_reward, r_safe, r_efficiency, r_comfort, r_tl, r_fluc, r_side, done, 
                                                all_vehicle, new_all_vehicle]], columns = cols))
    
    if TRAIN and (agent._step > agent.minimal_size):
    # if TRAIN and (agent._step > agent.batch_size):
        loss_actor, Q_loss = agent.learn()
        print('!!!!!!! actor的loss ', loss_actor, 'q的loss ', Q_loss)
    else:
        loss_actor = Q_loss = None
    
    return collision, loss_actor, Q_loss


def main_train():
    a_dim = 1 # 1个连续动作
    s_dim = 3*7    # 状态就是自己+六辆周围车的状态

    agent = PDQNAgent(
        s_dim, 
        a_dim,
        acc3 = True,
        Kaiming_normal = False,
        memory_size = 40000,
        device=DEVICE)
    losses_actor = [] # 不需要看第一个memory 即前20000步
    losses_episode = []
    
    if not TRAIN:
        globals()['EPISODE_NUM']=400
        globals()['CURRICULUM_STAGE']=3
        globals()['RL_CONTROL']=1100
        agent.load_state_dict(torch.load(f"{OUT_DIR}/net_params.pth", map_location=DEVICE))
        globals()['OUT_DIR']=f"./{OUT_DIR}/test"
    else:
        #load pre-trained model params for further training
        if os.path.exists(f"./model_params/{OUT_DIR}_net_params.pth"):
            agent.load_state_dict(torch.load(f"./model_params/{OUT_DIR}_net_params.pth", map_location=DEVICE))

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    else:
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        #os.removedirs(OUT_DIR)
        os.makedirs(OUT_DIR)
    
    switch_count=1
    for epo in range(EPISODE_NUM): # 测试时可以调小epo回合次数
        truncated = False 
        target_lane = None
        if CURRICULUM_STAGE == 1:
            traci.start(sumoCmd0)
            departLane=np.random.choice([0,1,3,4])
            traci.vehicle.add(vehID="0_0",routeID="r1",typeID="CarB", depart="5.000000", departLane=str(departLane), 
                              departPos="0", departSpeed="20", arrivalPos="3100")
            ego_index_str = "0_0"
            if departLane == 0 or departLane == 1:
                target_lane = random.randint(0, 1)
            else:
                target_lane = random.randint(1, 2)
        else:
            traci.start(sumoCmd1)
            # ego_index = 20 + epo % 100   # 选取中间车道第index辆出发的车为我们的自动驾驶车
            ego_index = 5 + epo % 20   # 选取中间车道第index辆出发的车为我们的自动驾驶车
            ego_index_str = str(np.random.randint(0,5))+'_'+str(ego_index) # ego的id为'1_$index$', 如index为20,id='1_20'
            target_lane = random.randint(0, 2) # ego的变道方向，从0 1 2中取

        control_vehicle = '' # ego车辆的id
        ego_show = False # ego车辆是否出现过
        global auto_vehicle_a
        auto_vehicle_a = 0
        global df_record
        df_record = pd.DataFrame(columns = cols)
        
        print(f"+++++++{epo}  STAGE:{CURRICULUM_STAGE} +++++++++++++")
        print(f"++++++++++++++++++ {OUT_DIR} +++++++++++++++++++++++")
        
        
        while traci.simulation.getMinExpectedNumber() > 0:
            # 1. 得到道路上所有的车辆ID
            vehicle_list = traci.vehicle.getIDList()
            if CURRICULUM_STAGE == 1:
                for vehicle in vehicle_list:
                    traci.vehicle.setLaneChangeMode(vehicle, 0b000000000000)
            
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
            # 4.1 获取周围车辆信息
#            all_vehicle, rel_up, _ = get_all(control_vehicle, 200) # 前后200m距离
    
            # 4.2 将所有后方车辆设置为不变道
            # for vehicle in vehicle_list:
            #     if traci.vehicle.getPosition(vehicle)[0] < traci.vehicle.getPosition(control_vehicle)[0]:
            #         traci.vehicle.setLaneChangeMode(vehicle, 0b000000000000)
    
            # 4.3 去除自动驾驶车默认的跟车和换道模型，为模型训练做准备
            traci.vehicle.setSpeedMode(control_vehicle, 00000)
            traci.vehicle.setLaneChangeMode(control_vehicle, 0b000000000000)
            
            # 5 模型训练
            collision, loss_actor, _ = train(agent, control_vehicle, epo,  target_lane) # 模拟一个时间步
            if collision:
                truncated = True
                break

            global step
            step = step + 1
            if loss_actor is not None:
                losses_actor.append(loss_actor)
                losses_episode.append(loss_actor)
            
        if TRAIN and not truncated and len(losses_episode)>0 and np.average(losses_episode)<=0.02:
            if CURRICULUM_STAGE == 1 and switch_count >= SWITCH_COUNT:
                switch_count = 1
                globals()['CURRICULUM_STAGE'] = 2
            elif CURRICULUM_STAGE == 2 and switch_count >= SWITCH_COUNT:
                switch_count = 1
                globals()['CURRICULUM_STAGE'] = 3
            elif CURRICULUM_STAGE == 3 and switch_count >= SWITCH_COUNT:
                switch_count = 1
                globals()['CURRICULUM_STAGE'] = 1
        globals()['PRE_LANE']=None
        losses_episode.clear()
        traci.close(wait=True)
        switch_count+=1
        
        # 保存
        df_record.to_csv(f"{OUT_DIR}/df_record_epo_{epo}.csv", index = False)
        if TRAIN:
            torch.save(agent.state_dict(), f"./{OUT_DIR}/net_params.pth") 
            pd.DataFrame(data=losses_actor).to_csv(f"./{OUT_DIR}/losses.csv")


if __name__ == '__main__':
    main_train() 
#    pass




