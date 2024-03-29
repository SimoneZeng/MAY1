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

timestep=5000
ts_cnt = 0 # 时间步计数
v0_cnt = 0 # 每条车道上的车辆计数
v1_cnt = 0
v2_cnt = 0
v3_cnt = 0
v4_cnt = 0


with open("sumo/one_way_2L.rou.xml", "w") as routes:
    # 配置所有的车辆属性和所有的路线.
    print("""<routes>
    <vType id="CarA" length="5" maxSpeed="25" carFollowModel="Krauss" width='1.6' tau="0.0001" minGap="0" accel="0.0001"  decel="0.004"/>
    <vType id="CarB" length="5" maxSpeed="25" carFollowModel="Krauss" width='1.6' tau="1" minGap="0" accel="3"  decel="3"/>
    <route id="r1" edges="EA EB" />""", file=routes)
    
    for i in range(timestep):
        # 测试车所在车道从5条车道中随机选取
        ts_cnt = ts_cnt + 1
        v2_cnt = v2_cnt + 1
        print('    <vehicle id="2_%i" type="CarA" route="r1" depart="%f" departLane="2" departSpeed="25" arrivalLane="2" color="247,247,247" />' 
                % (v2_cnt, float(i)/1000), file=routes)

    print("</routes>", file=routes)
        


