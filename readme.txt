1. route_generation.py
是用Python代码生成路由文件，即one_way_py.rou.xml

2. dqn_model.py
是修改过后的dqn模型代码

3. dqn_mode00.py
是莫烦Python中pytorch部分关于dqn的代码，用于借鉴学习

4. dqn_train.py
是主文件，包括使用TraCI接口获取模拟环境中的信息，以及训练RL模型

5. sumo文件夹
（1）路由文件.rou.xml，其中one_way_py.rou.xml是python生成的，
one_way_netedit.rou.xml是netedit生成的，empty是空的路由文件。
one_way_netedit和empty不需要被用到
（2）地图文件one_way.net.xml，可以用netedit打开，由两条直线的三车道相连
（3）配置文件one_way.sumocfg，简单的配置文件，直接写的

-----------------------------------------------------------------------------

第三方库的版本
numpy:1.21.6
torch:1.11.0
traci:1.14.0
sumolib:1.14.0





