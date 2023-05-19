# -*- coding: utf-8 -*-
"""
Created on Fri May 6 16:38:33 2023
第二种CL设计下的 stage1，2，3，4 车辆生成文件 CL2_low
一共5条lane，+低+车流密度

- 低密度：
    lane_freq = 10 每10 * 0.5 = 5 s在同一个lane发车
    5s * 20m/s = 100m左右的间距
    每个车道发车 timestep / lane_freq = 50辆车
5个车道分别的发车时间步 0，2，4，6，8

第二种CL设计下的 stage3 车辆生成文件 CL2_mid
一共5条lane，+中+车流密度
- 中密度：
    lane_freq = 6 每 6 * 0.5 = 3 s在同一个lane发车
    3s * 20m/s = 60m左右的间距
    每个车道发车 timestep / lane_freq = 83辆车
5个车道分别的发车时间步 0，1，2，3，4

第二种CL设计下的 stage4 车辆生成文件 CL2_high
一共5条lane，+高+车流密度
- 高密度：
    lane_freq = 4 每 4 * 0.5 = 2 s在同一个lane发车
    3s * 20m/s = 40m左右的间距
    每个车道发车 timestep / lane_freq = 125辆车
5个车道分别的发车时间步 0，1，2，3，0

不同密度修改(1)lane_freq ， (2)with open 文件名，(3)if i % lane_freq == x:

@author: Simone
"""

# 每条车道上出发频率
# lane_freq = 10 # 低密度，每10个时间步发出一辆车
# lane_freq = 6 # 中密度，每6个时间步发出一辆车
lane_freq = 4 # 高密度，每4个时间步发出一辆车

timestep = 500
v0_cnt = 0 # 每条车道上的车辆计数
v1_cnt = 0
v2_cnt = 0
v3_cnt = 0
v4_cnt = 0


# with open("sumo/route_CL2_low.rou.xml", "w") as routes:
# with open("sumo/route_CL2_mid.rou.xml", "w") as routes:
with open("sumo/network_5l_high.rou.xml", "w") as routes:
    # 配置所有的车辆属性和所有的路线
    print("""<routes>
    <vType id="CarA" length="5" maxSpeed="25" carFollowModel="Krauss" width='1.6' tau="0.0001" minGap="0" accel="0.0001"  decel="0.004"/>
    <vType id="CarB" length="5" maxSpeed="25" carFollowModel="Krauss" width='1.6' tau="1" minGap="0" accel="3"  decel="3"/>
    <route id="r1" edges="EA EB EC ED EF EG" />""", file=routes)
    
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
        if i % lane_freq == 0: # 0车道在每lane_freq个时间步的第1个时间步发车
            v0_cnt += 1
            print('    <vehicle id="0_%i" type="CarB" route="r1" depart="%f" departLane="0" departSpeed="20" color="248,248,255"/>' 
                  % (v0_cnt, 0.5 * i), file=routes)
        # if i % lane_freq == 2: # 1车道在每lane_freq个时间步的第3个时间步发车
        if i % lane_freq == 1:
            v1_cnt += 1
            print('    <vehicle id="1_%i" type="CarB" route="r1" depart="%f" departLane="1" departSpeed="20" color="248,248,255"/>' 
                  % (v1_cnt, 0.5 * i), file=routes)
        # if i % lane_freq == 4: # 3车道在每lane_freq个时间步的第5个时间步发车
        if i % lane_freq == 2:
            v2_cnt += 1
            print('    <vehicle id="2_%i" type="CarB" route="r1" depart="%f" departLane="2" departSpeed="20" color="248,248,255"/>' 
                  % (v2_cnt, 0.5 * i), file=routes)
        # if i % lane_freq == 6: # 3车道在每lane_freq个时间步的第7个时间步发车
        if i % lane_freq == 3:
            v3_cnt = v3_cnt + 1
            print('    <vehicle id="3_%i" type="CarB" route="r1" depart="%f" departLane="3" departSpeed="20" color="248,248,255"/>' 
                  % (v3_cnt, 0.5 * i), file=routes)
        # if i % lane_freq == 8: # 4车道在每lane_freq个时间步的第9个时间步发车
        # if i % lane_freq == 4:
        if i % lane_freq == 0:
            v4_cnt = v4_cnt + 1
            print('    <vehicle id="4_%i" type="CarB" route="r1" depart="%f" departLane="4" departSpeed="20" color="248,248,255"/>' 
                  % (v4_cnt, 0.5 * i), file=routes)

    print("</routes>", file=routes)
        


