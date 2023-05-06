# -*- coding: utf-8 -*-
"""
Created on Fri May 5 16:38:33 2023
第一种CL设计下的 stage1 车辆生成文件 CL1_s1
一共5条lane，模拟2条lane，低车流密度

- 低密度：
    lane_freq = 10 每10 * 0.5 = 5 s在同一个lane发车
    5m * 25m/s = 125m左右的间距
    每个车道发车 timestep / lane_freq = 50辆车
    调小了timestep，每个车道30辆车
@author: Simone
"""

lane_freq = 10 # 每条车道上出发频率，每5个时间步发出一辆车

timestep = 300 # 500 300 因为ego_vehile只会出现在5-25
v0_cnt = 0 # 每条车道上的车辆计数
v1_cnt = 0
v2_cnt = 0
v3_cnt = 0
v4_cnt = 0


with open("sumo/route_CL1_s1.rou.xml", "w") as routes:
    # 配置所有的车辆属性和所有的路线
    print("""<routes>
    <vType id="CarA" length="5" maxSpeed="25" carFollowModel="Krauss" width='1.6' tau="0.0001" minGap="0" accel="0.0001"  decel="0.004"/>
    <vType id="CarB" length="5" maxSpeed="25" carFollowModel="Krauss" width='1.6' tau="1" minGap="0" accel="3"  decel="3"/>
    <route id="r1" edges="EA EB" />""", file=routes)
    
    '''
    每条车辆信息内容
    vehicle_id 车辆id
    type 车辆类型： CarA是2车道的车流，CarB是其他车道的车流
    route 车辆路线：都是r1，从EA到EB
    depart 车辆从起点出发的时间：是真实时间，不是时间步
    departLane 车辆从起点出发的车道
    departSpeed 车辆从起点出发的速度
    arrivalLane 车辆抵达终点的车道：只有2车道的车流有该项
    
    '''
    for i in range(timestep):
        # 测试车所在车道从5条车道中随机选取
        # ts_cnt = ts_cnt + 1
        if i % lane_freq == 0: # 0车道在每lane_freq个时间步的第1个时间步发车
            v0_cnt += 1
            print('    <vehicle id="0_%i" type="CarB" route="r1" depart="%f" departLane="0" departSpeed="20" color="248,248,255"/>' 
                  % (v0_cnt, 0.5 * i), file=routes)
        if i % lane_freq == 2: # 1车道在每lane_freq个时间步的第3个时间步发车
            v1_cnt += 1
            print('    <vehicle id="1_%i" type="CarB" route="r1" depart="%f" departLane="1" departSpeed="20" color="248,248,255"/>' 
                  % (v1_cnt, 0.5 * i), file=routes)
        if i % lane_freq == 4: # 3车道在每lane_freq个时间步的第5个时间步发车
            v3_cnt += 1
            print('    <vehicle id="3_%i" type="CarB" route="r1" depart="%f" departLane="3" departSpeed="20" color="248,248,255"/>' 
                  % (v3_cnt, 0.5 * i), file=routes)
        if i % lane_freq == 6: # 4车道在每lane_freq个时间步的第7个时间步发车
            v4_cnt += 1
            print('    <vehicle id="4_%i" type="CarB" route="r1" depart="%f" departLane="4" departSpeed="20" color="248,248,255"/>' 
                  % (v4_cnt, 0.5 * i), file=routes)

        for j in range(500): # 2车道在每0.001s发一辆车，i + 0.001 * j，在一个time step 0.5s中生成的车
            v2_cnt = v2_cnt + 1
            print('    <vehicle id="2_%i" type="CarA" route="r1" depart="%f" departLane="2" departSpeed="20" arrivalLane="2" color="247,247,247" />' 
                    % (v2_cnt, 0.5 * i + 0.001 * j), file=routes)

    print("</routes>", file=routes)
        


