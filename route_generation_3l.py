# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:23:33 2022

每个lane每6个step 3s发出一辆车
一共200个step发车
tau设为3s
minGap="5"
均匀发车

@author: Skye
"""

import numpy as np

# 每条车道上出发频率
p0=6
p1=6
p2=6

timestep=200
ts_cnt = 0 # 时间步计数
v0_cnt = 0 # 每条车道上的车辆计数
v1_cnt = 0
v2_cnt = 0


with open("sumo/one_way_py.rou.xml", "w") as routes:
    # 配置所有的车辆属性和所有的路线.
    print("""<routes>
    <vType id="CarA" length="5" maxSpeed="25" carFollowModel="Krauss" width='1.6' tau="3" minGap="5" accel="3"  decel="3" />
    <route id="r1" edges="E0 E1" />""", file=routes)
    
    for i in range(timestep):
        # ts_cnt = ts_cnt + 1
        if i % p0 == 0:
        # if np.random.randint(0, p0) == 0:
            v0_cnt = v0_cnt + 1
            print('    <vehicle id="0_%i" type="CarA" route="r1" depart="%f" departLane="0" departSpeed="20" color="248,248,255"/>' % (v0_cnt, 0.5 * i), file=routes)
        # 测试车所在车道
        if i % p0 == 2:
        # if np.random.randint(0, p1) == 0:
            v1_cnt = v1_cnt + 1
            # mark autonomous vehicle controlled by our model.
            print('    <vehicle id="1_%i" type="CarA" route="r1" depart="%f" departLane="1" departSpeed="20" color="0,0,255" />' % (v1_cnt, 0.5 * i), file=routes)
        if i % p0 == 4:
        # if np.random.randint(0, p2) == 0:
            v2_cnt = v2_cnt + 1
            print('    <vehicle id="2_%i" type="CarA" route="r1" depart="%f" departLane="2" departSpeed="20" color="248,248,255" />' % (v2_cnt, 0.5 * i), file=routes)
    print("</routes>", file=routes)
        


