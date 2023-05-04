# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:45:19 2023

借鉴师兄的pdqn
github中的 mpdqn

库版本
python 3.8.15
conda 本地 conda 4.12.0 服务器 conda 22.9.0
torch 本地 1.11.0+cpu 服务器 1.13.1
numpy 本地 1.21.6 服务器 1.23.4
pandas 本地 1.3.5 服务器 1.3.5
matplotlib 3.6.2


@author: Skye
"""

import os
from typing import Dict, List, Tuple
import random, collections
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.replay_buffer import ReplayBuffer


class QActor(nn.Module):
    '''
    params:
        state_size, state space
        action_size, discrete action space
        action_param_size, the parameter of continuous action
    return:
        all q values of discrete actions
    '''
    def __init__(self, state_size, action_size, action_param_size):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_param_size = action_param_size
        
        inputSize = self.state_size + self.action_param_size
        
        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(inputSize, 128), 
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size),
        )
        
    def forward(self, state, action_parameters):
        x = torch.reshape(state, (-1, 21))  # 7*3变为1*21 torch.Size([1, 21])
        x = x.float() 
        x = torch.cat((x, action_parameters), dim=1)
        q = self.feature_layer(x)
        
        return q
       
        
class ParamActor(nn.Module):
    '''
    params:
        state_size, state space
        action_param_size, the parameter of continuous action
    return:
        all the optimal parameter of continuous action
    '''
    def __init__(self, state_size, action_param_size):
        super(ParamActor, self).__init__()
        self.state_size = state_size
        self.action_param_size = action_param_size
        
        inputSize = self.state_size
        
        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(inputSize, 128), 
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_param_size),
        )
        
    def forward(self, state):
        x = torch.reshape(state, (-1, 21))
        x = x.float() # 否则报错expected scalar type Float but found Double
        action = self.feature_layer(x)
        action = torch.tanh(action) # n * 3维的action
        
        return action


class QActor_init(nn.Module):
    '''
    params:
        state_size, state space
        action_size, discrete action space
        action_param_size, the parameter of continuous action
    return:
        all q values of discrete actions
    '''
    def __init__(self, state_size, action_size, action_param_size):
        super(QActor_init, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_param_size = action_param_size
        
        inputSize = self.state_size + self.action_param_size
        
        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(inputSize, 128), 
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size),
        )
        
        for layer in self.feature_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
        
    def forward(self, state, action_parameters):
        x = torch.reshape(state, (-1, 21))  # 7*3变为1*21 torch.Size([1, 21])
        x = x.float() 
        x = torch.cat((x, action_parameters), dim=1)
        q = self.feature_layer(x)
        # print("qqqqqqq ", q)
        
        return q
       
        
class ParamActor_init(nn.Module):
    '''
    params:
        state_size, state space
        action_param_size, the parameter of continuous action
    return:
        all the optimal parameter of continuous action
    '''
    def __init__(self, state_size, action_param_size):
        super(ParamActor_init, self).__init__()
        self.state_size = state_size
        self.action_param_size = action_param_size
        
        inputSize = self.state_size
        
        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(inputSize, 128), 
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_param_size),
        )
        for layer in self.feature_layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                
    def forward(self, state):
        x = torch.reshape(state, (-1, 21))
        x = x.float() # 否则报错expected scalar type Float but found Double
        action = self.feature_layer(x)
        action = torch.tanh(action) # n * 3维的action
        
        return action        

class PDQNAgent(nn.Module):
    def __init__(
            self, 
            state_dim = 3*7, 
            action_dim: int =1, 
            memory_size: int = 20000,
            batch_size: int = 128, # former 32
            epsilon_initial=1.0,
            epsilon_final=0.05,
            epsilon_decay=2000,
            gamma=0.9, # former 0.99
            lr_actor=0.001,
            lr_param=0.0001,
            acc3 = True, # action_acc = 3 * action_parameters
            NormalNoise = False, # 高斯噪声
            Kaiming_normal = False, # 网络参数初始化
    ):
        super(PDQNAgent, self).__init__()
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print('device ', self.device)
        self.action_dim = action_dim # 1 维，输出1个连续动作 acc
        self.state_dim = state_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.lr_actor, self.lr_param = lr_actor, lr_param
        self.gamma = gamma
        self.clip_grad = 10
        self.tau_actor = 0.01
        self.tau_param = 0.001
        
        self.acc3 = acc3
        self.NormalNoise = NormalNoise
        self.Kaiming_normal = Kaiming_normal
        
        self._epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self._step = 0
        self._learn_step = 0
        
        self.num_action = 3 # 3 kinds of discrete action
        self.action_param_sizes = np.array([self.action_dim, self.action_dim, self.action_dim]) # 1, 1, 1
        self.action_param_size = int(self.action_param_sizes.sum()) # 3
        self.action_param_max_numpy = np.array([1, 1, 1]) # np.array([3, 3, 3]) 
        self.action_param_min_numpy = np.array([-1, -1, -1]) # np.array([3, 3, 3]) 
        self.action_param_range = torch.from_numpy((self.action_param_max_numpy - self.action_param_min_numpy)).float().to(self.device)
        self.action_param_max = torch.from_numpy(self.action_param_max_numpy).float().to(self.device)
        self.action_param_min = torch.from_numpy(self.action_param_min_numpy).float().to(self.device)
        self.action_param_offsets = self.action_param_sizes.cumsum() # 不知道干嘛的
        self.action_param_offsets = np.insert(self.action_param_offsets, 0, 0)
        self.memory = ReplayBuffer(self.state_dim, self.action_param_size, self.memory_size, self.batch_size)
        
        if self.Kaiming_normal:
            self.actor = QActor_init(self.state_dim, self.num_action, self.action_param_size).to(self.device)
            self.actor_target = QActor_init(self.state_dim, self.num_action, self.action_param_size).to(self.device)
        else:
            self.actor = QActor(self.state_dim, self.num_action, self.action_param_size).to(self.device)
            self.actor_target = QActor(self.state_dim, self.num_action, self.action_param_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()  # 不启用 BatchNormalization 和 Dropout
        if self.Kaiming_normal:
            self.param = ParamActor_init(self.state_dim, self.action_param_size,).to(self.device)
            self.param_target = ParamActor_init(self.state_dim, self.action_param_size,).to(self.device)
        else:
            self.param = ParamActor(self.state_dim, self.action_param_size,).to(self.device)
            self.param_target = ParamActor(self.state_dim, self.action_param_size,).to(self.device)
        self.param_target.load_state_dict(self.param.state_dict())
        self.param_target.eval()
        
        self.loss_func = F.smooth_l1_loss
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.param_optimizer = torch.optim.Adam(self.param.parameters(), lr=self.lr_param)
        

    def choose_action(self, state, train=True):
        if train:
            # epsilon 更新
            self._epsilon = max(
                    self.epsilon_final, 
                    self._epsilon - (self.epsilon_initial - self.epsilon_final) / self.epsilon_decay
                    )

            with torch.no_grad(): # 不生成计算图，减少显存开销
                state = torch.tensor(state, device=self.device)
                all_action_parameters = self.param.forward(state) # 1*3 维连续 param 
                
                if self._epsilon > np.random.random(): # 探索率随着迭代次数增加而减小
                    # if self._step < self.batch_size:    
                    if self._step < self.memory_size: # 开始学习前，变道随机，acc随机
                        action = np.random.randint(0, self.num_action) # 离散 action 随机
                        all_action_parameters = np.random.uniform(self.action_param_min_numpy, self.action_param_max_numpy)
                    else: # 开始学习后，变道 不 随机
                        if self.NormalNoise: # acc加噪声
                            Q_value = self.actor.forward(state, all_action_parameters) 
                            Q_value = Q_value.detach().cpu().numpy()
                            action = np.argmax(Q_value)
                            print("all_action_parameters before", all_action_parameters)
                            all_action_parameters = all_action_parameters.squeeze()
                            all_action_parameters = all_action_parameters.cpu().data.numpy()
                            all_action_parameters = np.clip(np.random.normal(all_action_parameters,scale=0.02,size=3), -3, 3)
                            print("all_action_parameters after", all_action_parameters)
                        if not self.NormalNoise: # acc 随机
                            all_action_parameters = torch.from_numpy(
                                    np.random.uniform(self.action_param_min_numpy, self.action_param_max_numpy)).to(self.device)
                            all_action_parameters = all_action_parameters.unsqueeze(0).to(torch.float32) # np是64的精度，转为32的精度
                            Q_value = self.actor.forward(state, all_action_parameters)
                            Q_value = Q_value.detach().cpu().numpy()
                            action = np.argmax(Q_value) # 得到随机 all_action_parameters 下的离散action
                            all_action_parameters = all_action_parameters.squeeze() # 变为3维 连续 param
                            all_action_parameters = all_action_parameters.cpu().data.numpy()
                else: # select maximum action
                    Q_value = self.actor.forward(state, all_action_parameters) # 1*3 维 所有离散动作的Q_value
                    Q_value = Q_value.detach().cpu().numpy() # tensor 转换为 numpy格式
                    action = np.argmax(Q_value)
                    all_action_parameters = all_action_parameters.squeeze() # 变为3维 连续 param
                    all_action_parameters = all_action_parameters.cpu().data.numpy()

                action_parameters = all_action_parameters[action] # all_action_parameters从1*3维，从第1维中选
        if not train:
            with torch.no_grad(): 
                state = torch.tensor(state, device=self.device)
                all_action_parameters = self.param.forward(state) # 1*3 维连续 param
                Q_value = self.actor.forward(state, all_action_parameters) # 1*3 维 所有离散动作的Q_value
                Q_value = Q_value.detach().cpu().numpy() # tensor 转换为 numpy格式
                action = np.argmax(Q_value)
                all_action_parameters = all_action_parameters.squeeze() # 变为3维 连续 param
            
                all_action_parameters = all_action_parameters.cpu().data.numpy()
                action_parameters = all_action_parameters[action] # all_action_parameters从1*3维，从第1维中选
        
        if self.acc3:
            action_parameters = 3 * action_parameters
        return action, action_parameters, all_action_parameters
                    

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True):
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_param_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[self.action_param_offsets[a]:self.action_param_offsets[a+1]] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.
        return grad

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_param_max
            min_p = self.action_param_min
            rnge = self.action_param_range
        else:
            raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad
    
    # def step(self, obs, act, act_param, rew, next_obs, done):
    #     self._step += 1
        
    #     self.store_transition(obs, act, act_param, rew, next_obs, done)
    #     if self._step >= self.batch_size:
    #         self.learn()
    #         self._learn_step += 1
        
    def store_transition(self, obs, act, act_param, rew, next_obs, done):
        self._step += 1
        
        obs = np.reshape(obs, (-1, 1)) # 列数为1，行数-1根据列数来确定
        obs = np.squeeze(obs)
        act = np.reshape(act, (-1, 1))
        act = np.squeeze(act)
        act_param = np.reshape(act_param, (-1, 1))
        act_param = np.squeeze(act_param)
        next_obs = np.reshape(next_obs, (-1, 1))
        next_obs = np.squeeze(next_obs)
       
        self.transition = [obs, act, act_param, rew, next_obs, done]
        self.memory.store(*self.transition) # store 没有返回值

    def learn(self):
        self._learn_step += 1        
        samples = self.memory.sample_batch()
        
        # compute loss
        device = self.device  # for shortening the following lines
        b_state = torch.FloatTensor(samples["obs"]).to(device)
        b_next_state = torch.FloatTensor(samples["next_obs"]).to(device) # b_next_state torch.Size([batch, 21])
        b_action = torch.LongTensor(samples["act"].reshape(-1, 1)).to(device)
        b_action_param = torch.FloatTensor(samples["act_param"]).to(device)
        b_reward = torch.FloatTensor(samples["rew"].reshape(-1, 1)).to(device) # b_reward torch.Size([128, 1])
        b_done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # -----------------------optimize Q actor------------------------
        with torch.no_grad():
            next_action_parameters = self.param_target.forward(b_next_state) # next_action_parameters torch.Size([batch, 3])
            next_q_value = self.actor_target(b_next_state, next_action_parameters) # next_q_value torch.Size([batch, 3])
            q_prime = torch.max(next_q_value, 1, keepdim=True)[0] # q_prime torch.Size([128, 1])
            # Compute the TD error
            target = b_reward + (1 - b_done) * self.gamma * q_prime # target torch.Size([128, 1])
        
        # Compute current Q-values using policy network
        q_values = self.actor(b_state, b_action_param) # q_values torch.Size([128, 3])
        y_predicted = q_values.gather(1, b_action.view(-1, 1)) # y_predicted torch.Size([128, 1])
        loss_actor = self.loss_func(y_predicted, target) # loss 是torch.Tensor的形式
        ret_loss_actor = loss_actor.detach().cpu().numpy()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        if self.clip_grad > 0:  # clip防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)

        self.actor_optimizer.step()

        # ------------------------optimize param net------------------------------
        with torch.no_grad():
            action_params = self.param(b_state)
        action_params.requires_grad = True
        Q_val = self.actor(b_state, action_params)
        Q_loss = torch.mean(torch.sum(Q_val, 1)) # 这里不知道为什么？？
        self.actor.zero_grad()
        Q_loss.backward()
        
        # ==============================
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        action_params = self.param(Variable(b_state))
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        # if self.zero_index_gradients:
        #     delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=b_action, inplace=True)
        
        out = -torch.mul(delta_a, action_params)
        self.param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.param.parameters(), self.clip_grad)
        
        self.param_optimizer.step()
        
        self.soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        self.soft_update_target_network(self.param, self.param_target, self.tau_param)
        
        return ret_loss_actor, Q_loss

    def soft_update_target_network(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)





