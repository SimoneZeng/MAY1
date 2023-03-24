# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:08:26 2023

在d3qn_train_3r.py的基础上，将 DQN 模型改为 PDQN 模型
删掉了一些没用的代码

在pdqn_train_3r 的基础上，修改bug，侧方有车时的store。
修改惩罚都为-10

@author: Simone
"""


import numpy as np
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import random
from pdqn_model import PDQNAgent
import os, sys
import pandas as pd


# 引入地址 
sumo_path = os.environ['SUMO_HOME'] # "D:\\sumo\\sumo1.13.0"
# cfg_path = "C:\--codeplace--\sumo_inter\sumo_test01\sumo\one_way.sumocfg" # 1.在本地用这个cfg_path
cfg_path = "/home/zengximu/sumo_inter/sumo_test01/sumo/one_way.sumocfg" # 2. 在服务器上用这个cfg_path
sys.path.append(sumo_path)
sys.path.append(sumo_path + "/tools")
sys.path.append(sumo_path + "/tools/xml")
import traci # 在ubuntu中，traci和sumolib需要在tools地址引入之后import
from sumolib import checkBinary

TRAIN = True # False True

# os.environ['CUDA_VISIBLE_DEVICES']='0, 1'  # 显卡使用

# 是否打开gui
gui = False # False True
if gui == 1:
    sumoBinary = checkBinary('sumo-gui') # 方法一
#    sumoBinary = sumo_path + "/bin/sumo-gui" # 方法二：后面添加路径中的/与sumo_path中的\可能不匹配，不建议使用
else:
    sumoBinary = checkBinary('sumo')

sumoCmd = [sumoBinary, "-c", cfg_path]

# 记录车辆信息
map_ve = {}
# 包括前一时刻的后方第一辆车和当前时刻的后方第一辆车
back_id = [' ', ' ']
auto_vehicle_a = 0
# 统计总的训练次数s
step = 0 # 0 20000
# 存储奖励值的数组，分epo存的
cols = ['epo', 'train_step', 'position_y', 'target_lane', 'lane', 'speed', 
         'lc_int', 'fact_acc', 'acc', 'change_lane', 'r','r_safe', 'r_eff','r_com', 'r_back', 'done', 's', 's_']
df_record = pd.DataFrame(columns = cols)
action_change_dict = {0: 'left', 1: 'keep', 2:'right'}

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
    # 如果该方向没车，对于自己车道上的则用远的mask车辆代替，如果是旁边车道的得判断到底有没有旁边车道，有的话也是用远的mask来代替，否则用纵向距离为0的车辆代替，表示这个方向没车道，不能去
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
        if '2' in ego_lane:
            Id_list.append([0, 3.2, 0])
            Id_dict['upleft'] = ''
        else:
            Id_list.append([200, 3.2, 0])
            Id_dict['upleft'] = ''

    down = sorted(down, key=lambda x: x[0])
    if len(down) >= 1:
        Id_list.append(map_ve[down[-1][-1]])
        Id_dict['down'] = down[-1][-1]
        back_v = down[-1][-1]
    else:
        Id_list.append([-200, 0, 0])
        Id_dict['down'] = ''
        back_v = 0

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
        if '2' in ego_lane:
            Id_list.append([0, 3.2, 0])
            Id_dict['downleft'] = ''
        else:
            Id_list.append([-200, 3.2, 0])
            Id_dict['downleft'] = ''

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
    Id_list.append([y_speed/25, cal_a/3, ego_l]) # 其他论文里也是这样加ego车辆数据

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
#    flow_left = len(upleft) + len(downleft)
#    flow_middle = len(up) + len(down)
#    flow_right = len(upright) + len(downright)
#    flow = {'flow_left': flow_left, 'flow_middle': flow_middle, 'flow_right': flow_right} # 暂时还没有用到
    
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
    global TRAIN
    all_vehicle, rel_up, v_dict = get_all(control_vehicle, 200)
    if TRAIN:
        action_lc_int, action_acc, all_action_parameters = agent.choose_action(np.array(all_vehicle)) # 离散lane change ，连续acc，参数
    else:
        action_lc_int, action_acc, all_action_parameters = agent.choose_action(np.array(all_vehicle), train = False)
    
    inf = -10 # 撞墙惩罚
    inf_car = -10 # 撞车惩罚
    done = 0 # 回合结束标志
    
    global action_change_dict
    change_lane = action_change_dict[action_lc_int] # 0车道右车道在-8.0；1车道在-4.8；2车道左车道在-1.6
    
    # 记录当前车辆的数据
    global auto_vehicle_a
    pre_ego_info_dict = {"speed": traci.vehicle.getSpeed(control_vehicle), 
                         "acc":auto_vehicle_a, 
                         "LaneID": traci.vehicle.getLaneID(control_vehicle), 
                         "position": traci.vehicle.getPosition(control_vehicle)}
    
    # 计算速度，控制限速
    sp = traci.vehicle.getSpeed(control_vehicle) + action_acc*0.5 # 0.5s simulate一次
    if sp> 25:
        sp = 25
    if sp < 0:
        sp = 0
    traci.vehicle.setSpeed(control_vehicle, sp) # 将速度设置好
    
    print('@@@@@@ action_lc_int', action_lc_int, 'action_acc', action_acc, 'all_action_parameters', all_action_parameters)
    print('@@@@@@ speed ', sp)
    
    global df_record
    
    # 撞墙处理，车道从左到右是2,1,0
    if 'E0_0' == pre_ego_info_dict["LaneID"] and change_lane=='right':
        # 将撞墙的数据存到经验池里面去
        done = 1
        print("右右车道撞墙墙 ", 'all_vehicle', all_vehicle, 'action_lc_int', action_lc_int, 
              'all_action_parameters', all_action_parameters, 'inf', inf, 
              'all_vehicle_next', np.zeros((7,3)), 'done', done)
        train_step = agent._step
        print('train_step ', train_step)
        agent.store_transition(all_vehicle, action_lc_int, all_action_parameters, inf, np.zeros((7,3)), done)
        
        df_record = df_record.append(pd.DataFrame([[episode, train_step, pre_ego_info_dict['position'][0], 
                                                    target_lane, pre_ego_info_dict['LaneID'], 
                                                    pre_ego_info_dict['speed'], action_lc_int, pre_ego_info_dict['acc'], action_acc, change_lane, 
                                                    inf, 0, 0, 0, 0, done, all_vehicle, np.zeros((7,3))]], columns = cols))
        print("====================右右右右车道撞墙墙墙墙===================")
        change_lane='keep'
        done = 0 # 撞墙存储完了之后，done为0，ego车辆继续跑
        # 修改action_int 到 keep的范围中
        action_lc_int = 1

    if 'E0_2' == pre_ego_info_dict["LaneID"] and change_lane=='left':
        # 将撞墙的数据存到经验池里面去
        done = 1
        print("左左车道撞墙墙 ", 'all_vehicle', all_vehicle, 'action_lc_int', action_lc_int, 
              'all_action_parameters', all_action_parameters, 'inf', inf, 
              'all_vehicle_next', np.zeros((7,3)), 'done', done)
        train_step = agent._step
        print('train_step ', train_step)
        agent.store_transition(all_vehicle, action_lc_int, all_action_parameters, inf, np.zeros((7,3)), done)
#            global df_record
        df_record = df_record.append(pd.DataFrame([[episode, train_step, pre_ego_info_dict['position'][0], 
                                                    target_lane, pre_ego_info_dict['LaneID'], 
                                                    pre_ego_info_dict['speed'], action_lc_int, pre_ego_info_dict['acc'], action_acc, change_lane, 
                                                    inf, 0, 0, 0, 0, done, all_vehicle, np.zeros((7,3))]], columns = cols))
        print("====================左左左左车道撞墙墙墙墙===================")
        change_lane='keep'
        done = 0
        # 修改action_int 到 keep的范围中
        action_lc_int = 1
    
    # 变道撞车控制 惩罚为 inf_car
    dis_downleft = 0
    dis_downright = 0
    dis_upleft = 0
    dis_upright = 0
    
    if v_dict['downleft'] != '':
        dis_downleft = pre_ego_info_dict['position'][0] - traci.vehicle.getPosition(v_dict['downleft'])[0]
    if v_dict['downright'] != '':
        dis_downright = pre_ego_info_dict['position'][0] - traci.vehicle.getPosition(v_dict['downright'])[0]
    if v_dict['upleft'] != '':
        dis_upleft = traci.vehicle.getPosition(v_dict['upleft'])[0] - pre_ego_info_dict['position'][0]
    if v_dict['upright'] != '':
        dis_upright = traci.vehicle.getPosition(v_dict['upright'])[0] - pre_ego_info_dict['position'][0]
    
    # 侧方有车时，不能变道
    if ((0 < dis_downleft <= 5 or 0 < dis_upleft < 5) and change_lane=='left') or ((0 < dis_downright <= 5 or 0 < dis_upright <= 5) and change_lane=='right'):
        done = 1
        train_step = agent._step
        print('train_step ', train_step)
        agent.store_transition(all_vehicle, action_lc_int, all_action_parameters, inf_car, np.zeros((7,3)), done)
        df_record = df_record.append(pd.DataFrame([[episode, train_step, pre_ego_info_dict['position'][0], 
                                                    target_lane, pre_ego_info_dict['LaneID'], 
                                                    pre_ego_info_dict['speed'], action_lc_int, pre_ego_info_dict['acc'], action_acc, change_lane, 
                                                    inf_car, 0, 0, 0, 0, done, all_vehicle, np.zeros((7,3))]], columns = cols))
        if change_lane=='left':
            print("====================左左左左变道撞车车车车===================")
            print("左左变道撞车车 ", 'all_vehicle', all_vehicle, 'action_lc_int', action_lc_int, 
                  'all_action_parameters', all_action_parameters, 'inf_car', inf_car, 
                  'all_vehicle_next', np.zeros((7,3)), 'done', done)
            action_lc_int = 1
        if change_lane=='right':
            print("====================右右右右变道撞车车车车===================")
            print("右右变道撞车车 ", 'all_vehicle', all_vehicle, 'action_lc_int', action_lc_int, 
                  'all_action_parameters', all_action_parameters, 'inf_car', inf_car, 
                  'all_vehicle_next', np.zeros((7,3)), 'done', done)
            action_lc_int = 1
        change_lane='keep'
        done = 0
        
    
    
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
    print("################ 执行 ###################")
    
    # 查询自动驾驶车是否发生碰撞
    collision=0
    loss_actor = 0
    Q_loss = 0
    if control_vehicle in traci.simulation.getCollidingVehiclesIDList():
        print('=====================collision==', traci.simulation.getCollidingVehiclesIDList(), control_vehicle)
        print("==========================发生了撞车=========================")
        collision=1
        done = 1
        # 存储碰撞数据
        print("碰碰碰撞撞撞 ", 'all_vehicle', all_vehicle, 'action_lc_int', action_lc_int, 
              'all_action_parameters', all_action_parameters, 'inf_car', inf_car, 
              'all_vehicle_next', np.zeros((7,3)), 'done', done)
        train_step = agent._step
        print('train_step ', train_step)
        agent.store_transition(all_vehicle, action_lc_int, all_action_parameters, inf_car, np.zeros((7,3)), done)
        df_record = df_record.append(pd.DataFrame([[episode, train_step, pre_ego_info_dict['position'][0], 
                                            target_lane, pre_ego_info_dict['LaneID'], 
                                            pre_ego_info_dict['speed'], action_lc_int, pre_ego_info_dict['acc'], action_acc, change_lane, 
                                            inf_car, 0, 0, 0, 0, done, all_vehicle, np.zeros((7,3))]], columns = cols))
        return collision, loss_actor, Q_loss
    
    # 获取动作执行后的状态
    new_all_vehicle, new_rel_up, v_dict = get_all(control_vehicle, 200)
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
    r_efficiency = cur_ego_info_dict['speed']/25*0.8 - 0.8 # 范围是[-0.8, 0]
    
    if 0 < y_ttc < 4: 
        r_safe = np.log(y_ttc/4)
    else:
        r_safe = 0
     
    auto_vehicle_a = cur_ego_info_dict['acc']
    r_comfort = ((np.abs(pre_ego_info_dict['acc']-cur_ego_info_dict['acc'])/0.1) ** 2) / 3600    
    r_side = 0

    cur_reward = r_safe + r_efficiency - r_comfort
    
#    global df_record
    df_record = df_record.append(pd.DataFrame([[episode, train_step, pre_ego_info_dict['position'][0], 
                                                target_lane, pre_ego_info_dict['LaneID'], 
                                                pre_ego_info_dict['speed'], action_lc_int, pre_ego_info_dict['acc'], action_acc, change_lane,
                                                cur_reward, r_safe, r_efficiency, r_comfort, r_side, done, 
                                                all_vehicle, new_all_vehicle]], columns = cols))
        
    # agent存储
    print("正正正常常常 ", 'all_vehicle', all_vehicle, 'action_lc_int', action_lc_int, 
          'all_action_parameters', all_action_parameters, 'cur_reward', cur_reward, 
          'all_vehicle_next', new_all_vehicle, 'done', done)
    train_step = agent._step
    print('train_step ', train_step)
    agent.store_transition(all_vehicle, action_lc_int, all_action_parameters, cur_reward, new_all_vehicle, done)
    if TRAIN and (agent._step > agent.memory_size):
    # if TRAIN and (agent._step > agent.batch_size):
        loss_actor, Q_loss = agent.learn()
        print('!!!!!!! actor的loss ', loss_actor, 'q的loss ', Q_loss)
    
    return collision, loss_actor, Q_loss


def main_train():
    a_dim = 1 # 1个连续动作
    # 状态就是自己+六辆周围车的状态
    s_dim = 3*7
    #模型加载
    agent = PDQNAgent(
        s_dim, 
        a_dim,
        acc3 = True,
        Kaiming_normal = False,
        )
    # epsilons = [] # 不需要看前20000步
    losses_actor = [] # 不需要看第一个memory 即前20000步
    
    os.mkdir("result_record_pdqn_3r_inf_init")
    if not TRAIN:
        agent.load_state_dict(torch.load('./0318/result_record_pdqn_3r_inf_acc1/net_params.pth'))
    
    for epo in range(20000): # 测试时可以调小epo回合次数 
        traci.start(sumoCmd)
        # ego_index = 20 + epo % 100   # 选取中间车道第index辆出发的车为我们的自动驾驶车
        ego_index = 5 + epo % 20   # 选取中间车道第index辆出发的车为我们的自动驾驶车
        ego_index_str = '1_'+str(ego_index) # ego的id为'1_$index$', 如index为20,id='1_20'
        control_vehicle = '' # ego车辆的id
        ego_show = False # ego车辆是否出现过
        target_lane = random.randint(0, 2) # ego的变道目标车道，从0 1 2中取
        global auto_vehicle_a
        auto_vehicle_a = 0
        # train_step = 0
        global df_record
        df_record = pd.DataFrame(columns = cols)
        
        print("++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"++++++++++++++++++++{epo}+++++++++++++++++++++++++")
        print("++++++++++++++++++ result_record_pdqn_3r_inf_init +++++++++++++++++++++++")
        
        
        while traci.simulation.getMinExpectedNumber() > 0:
            # 1. 得到道路上所有的车辆ID
            vehicle_list = traci.vehicle.getIDList()
            
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
            collision, loss_actor, _ = train(agent, control_vehicle, epo,  target_lane) # 模拟一个时间步
            if collision:
                break
            # train_step = train_step + 1
            global step
            step = step + 1
            
            losses_actor.append(loss_actor)
            # epsilons.append(agent._epsilon)

        traci.close(wait=True)
        # 保存
        df_record.to_csv(f"result_record_pdqn_3r_inf_init/df_record_epo_{epo}.csv", index = False)
        torch.save(agent.state_dict(), './result_record_pdqn_3r_inf_init/net_params.pth') 
        
        # agent.memory.acts_param_buf.to_csv('./result_record_pdqn_3r/acts_param_buf.csv')

        # pd.DataFrame(data=epsilons).to_csv('./result_record_pdqn_3r/epsilons.csv')
        pd.DataFrame(data=losses_actor).to_csv('./result_record_pdqn_3r_inf_init/losses.csv')


if __name__ == '__main__':
    main_train() 
#    pass




