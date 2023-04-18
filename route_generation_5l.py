# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:23:33 2022

@author: Skye
"""

import numpy as np

# 每条车道上出发频率
p0=5
p1=5
p2=5
p3=5
p4=5

timestep=500
ts_cnt = 0 # 时间步计数
v0_cnt = 0 # 每条车道上的车辆计数
v1_cnt = 0
v2_cnt = 0
v3_cnt = 0
v4_cnt = 0


with open("sumo/one_way_5L.rou.xml", "w") as routes:
    # 配置所有的车辆属性和所有的路线.
    print("""<routes>
    <vType id="CarA" length="5" maxSpeed="25" carFollowModel="Krauss" width='1.6' tau="1" minGap="0" accel="3"  decel="3" />
    <route id="r1" edges="EA EB" />""", file=routes)
    
    for i in range(timestep):
        # 测试车所在车道从5条车道中随机选取
        ts_cnt = ts_cnt + 1
        #if i % p0 == 0:
        if np.random.randint(0, p0) == 0:
            v0_cnt = v0_cnt + 1
            print('    <vehicle id="0_%i" type="CarA" route="r1" depart="%f" departLane="0" departSpeed="20" color="247,247,247"/>' 
                  % (v0_cnt, i), file=routes)
        #if i % p1 == 1:
        if np.random.randint(0, p1) == 0:
            v1_cnt = v1_cnt + 1
            print('    <vehicle id="1_%i" type="CarA" route="r1" depart="%f" departLane="1" departSpeed="20" color="247,247,247" />' 
                  % (v1_cnt, i), file=routes)
        #if i % p2 == 2:
        if np.random.randint(0, p2) == 0:
            v2_cnt = v2_cnt + 1
            print('    <vehicle id="2_%i" type="CarA" route="r1" depart="%f" departLane="2" departSpeed="20" color="247,247,247" />' 
                  % (v2_cnt, i), file=routes)
        #if i % p3 == 3:
        if np.random.randint(0, p3) == 0:
            v3_cnt = v3_cnt + 1
            print('    <vehicle id="3_%i" type="CarA" route="r1" depart="%f" departLane="3" departSpeed="20" color="247,247,247" />' 
                  % (v3_cnt, i), file=routes)
        #if i % p4 == 4:
        if np.random.randint(0, p4) == 0:
            v4_cnt = v4_cnt + 1
            print('    <vehicle id="4_%i" type="CarA" route="r1" depart="%f" departLane="4" departSpeed="20" color="247,247,247" />' 
                  % (v4_cnt, i), file=routes)
    print("</routes>", file=routes)
        


