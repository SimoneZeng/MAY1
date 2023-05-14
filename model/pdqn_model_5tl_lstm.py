# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:45:19 2023

借鉴师兄的pdqn
github中的 mpdqn

库版本
python 3.8.15
conda 本地 conda 4.12.0 服务器 conda 22.9.0
torch 本地 1.11.0+cpu 服务器 1.13.1
numpy 本地 1.21.6 服务器 1.23.4
pandas 本地 1.3.5 服务器 1.3.5
matplotlib 3.6.2

在 pdqn_model 的基础上，增加了 5 维的 target lane code
尝试把 tl_code 和state 一起输入，而不是在之后网络中间加入 tl_code
需要修改：
1. ReplayBuffer 中增加tl_buf next_tl_buf
2. QActor ParamActor 增加tl_code
3. choose_action 增加 tl_code
4. 

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
from copy import deepcopy
from model.replay_buffer import RecurrentReplayBuffer, RecurrentPrioritizedReplayBuffer


class QActor(nn.Module):
    '''
    params:
        state_size, state space
        tl_size, dimension of targe lane code
        action_size, discrete action space
        action_param_size, the parameter of continuous action
    return:
        all q values of discrete actions
    '''
    def __init__(self, state_size: int, tl_size: int, action_size: int, action_param_size: int, kaiming_normal: bool = False):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_param_size = action_param_size
        self.tl_size = tl_size
        self.hidden_state = None
        
        # set common feature layer
        self.feature_layer = nn.Linear(self.state_size + self.tl_size + self.action_param_size, 128)
        self.lstm_layer = nn.LSTM(128, 128, batch_first=True)
        self.output = nn.Linear(128, self.action_size)
                
    def forward(self, state, tl_code, action_parameters, init_hidden=False):
        x = torch.reshape(state, (-1, 21))  # 7*3变为1*21 torch.Size([1, 21])
        batch_size = x.shape[0]
        x = x.float().reshape((-1, 1, 21))
        tl_code = torch.reshape(tl_code, (-1, 1, 7)) # 5 维变为 1*5
        action_parameters = torch.reshape(action_parameters, (-1, 1, 3))
        x = torch.cat((x, tl_code, action_parameters), dim=2)

        if init_hidden or self.hidden_state is None:
             #nn.LSTM以张量作为隐藏状态
            self.hidden_state=(torch.zeros((1, batch_size, 128), device=next(self.lstm_layer.parameters()).device),
                torch.zeros((1, batch_size, 128), device=next(self.lstm_layer.parameters()).device))
        
        feature = self.feature_layer(x)
        y, self.hidden_state = self.lstm_layer(F.relu(feature), self.hidden_state)
        q = self.output(y).squeeze(1)
        
        return q
    
    def init_hidden(self, batch_size, H=None, C=None):
        if H is None or C is None:
            self.hidden_state=(torch.zeros((1, batch_size, 128), device=next(self.lstm_layer.parameters()).device),
                torch.zeros((1, batch_size, 128), device=next(self.lstm_layer.parameters()).device))
        else:
            self.hidden_state=(H, C)

    def get_hidden(self):
        if self.hidden_state is None:
            return np.zeros((1, 1, 128), dtype=np.float32), np.zeros((1, 1, 128), dtype=np.float32)
        else:
            return self.hidden_state[0].clone().detach().cpu().numpy(), self.hidden_state[1].clone().detach().cpu().numpy()
       
        
class ParamActor(nn.Module):
    '''
    params:
        state_size, state space
        action_param_size, the parameter of continuous action
    return:
        all the optimal parameter of continuous action
    '''
    def __init__(self, state_size: int, tl_size: int, action_param_size: int, kaiming_normal: bool = False):
        super(ParamActor, self).__init__()
        self.state_size = state_size
        self.tl_size = tl_size
        self.action_param_size = action_param_size
        self.hidden_state=None
        
        self.feature_layer = nn.Linear(self.state_size + self.tl_size, 128)
        self.lstm_layer = nn.LSTM(128, 128, batch_first=True)
        self.output = nn.Linear(128, self.action_param_size)
        
    def forward(self, state, tl_code, init_hidden=False):
        x = torch.reshape(state, (-1, 21))
        batch_size = x.shape[0]
        x = x.float().reshape((-1, 1, 21))  # 否则报错expected scalar type Float but found Double
        tl_code = torch.reshape(tl_code, (-1, 1, 7)) # 5 维变为 1*5
        x = torch.cat((x, tl_code), dim=2)
        
        if init_hidden or self.hidden_state is None:
            #nn.LSTM以张量作为隐藏状态
            self.hidden_state=(torch.zeros((1, batch_size, 128), device=next(self.lstm_layer.parameters()).device),
                torch.zeros((1, batch_size, 128), device=next(self.lstm_layer.parameters()).device))

        feature = self.feature_layer(x)
        y, self.hidden_state = self.lstm_layer(F.relu(feature), self.hidden_state) # type(state)  <class 'tuple'> len  2 type(y)  <class 'torch.Tensor'>
        # print(state)
        # print('type(state) ', type(state), 'type(y) ', type(y)) # 
        # print('state len ', len(state), 'y.shape ', y.shape)
        action = self.output(y)
        action = torch.tanh(action) # n * 3维的action
        
        return action
    
    def init_hidden(self, batch_size, H=None, C=None):
        if H is None or C is None:
            self.hidden_state=(torch.zeros((1, batch_size, 128), device=next(self.lstm_layer.parameters()).device),
                torch.zeros((1, batch_size, 128), device=next(self.lstm_layer.parameters()).device))
        else:
            self.hidden_state=(H, C)
        
    def get_hidden(self):
        if self.hidden_state is None:
            return np.zeros((1, 1, 128), dtype=np.float32), np.zeros((1, 1, 128), dtype=np.float32)
        else:
            return self.hidden_state[0].clone().detach().cpu().numpy(), self.hidden_state[1].clone().detach().cpu().numpy()
       

class PDQNAgent(nn.Module):
    def __init__(
            self, 
            state_dim = 3*7, 
            action_dim: int =1, 
            tl_dim: int = 7,
            memory_size: int = 40000,
            minimal_size: int = 5000,
            batch_size: int = 128, # former 32
            epsilon_initial=1.0,
            epsilon_final=0.05,
            epsilon_decay=2000,
            gamma=0.9, # former 0.99
            lr_actor=0.001,
            lr_param=0.0001,
            acc3 = True, # action_acc = 3 * action_parameters
            NormalNoise = False, # 高斯噪声
            Kaiming_normal = False, # 网络参数初始化,
            n_step = 1, # n_step learning
            burn_in_step = 20, # burn in step for R2D2
             # PER parameters
            alpha: float = 0.6, #determines how much prioritization is used
            beta: float = 0.4,  #determines how much importance sampling is used
            prior_eps: float = 1e-3,    #guarantees every transition can be sampled
            per_flag: bool = False,
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
    ):
        super(PDQNAgent, self).__init__()
        
        self.device = device
        print('device ', self.device)
        self.action_dim = action_dim # 1 维，输出1个连续动作 acc
        self.state_dim = state_dim
        self.tl_dim = tl_dim
        self.memory_size = memory_size
        self.minimal_size = minimal_size
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
        self.n_step=n_step
        self.burn_in_step=burn_in_step
        
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

        # PER
        self.beta = beta
        self.beta_increment_per_sampling = 0.001
        self.prior_eps = prior_eps
        self.per_flag = per_flag
        if self.per_flag:
            self.memory = RecurrentPrioritizedReplayBuffer(self.state_dim, self.action_param_size, self.tl_dim, self.memory_size, self.batch_size, alpha, self.n_step, self.burn_in_step, self.gamma)
        else:
            self.memory = RecurrentReplayBuffer(self.state_dim, self.action_param_size, self.tl_dim, self.memory_size, self.batch_size, self.n_step, self.burn_in_step, self.gamma)
        # if self.n_step >1:
        #     self.memory_n = NSTEPReplayBuffer(self.state_dim, self.action_param_size, self.tl_dim, self.memory_size, self.batch_size, self.n_step, self.gamma)

        self.actor = QActor(self.state_dim, self.tl_dim, self.num_action, self.action_param_size, self.Kaiming_normal).to(self.device)
        self.actor_target = QActor(self.state_dim, self.tl_dim, self.num_action, self.action_param_size, self.Kaiming_normal).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()  # 不启用 BatchNormalization 和 Dropout
        self.param = ParamActor(self.state_dim, self.tl_dim, self.action_param_size, self.Kaiming_normal).to(self.device)
        self.param_target = ParamActor(self.state_dim, self.tl_dim, self.action_param_size, self.Kaiming_normal).to(self.device)
        self.param_target.load_state_dict(self.param.state_dict())
        self.param_target.eval()
        
        if self.per_flag:
            self.loss_func = nn.SmoothL1Loss(reduction='none')
        else:
            self.loss_func = nn.SmoothL1Loss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.param_optimizer = torch.optim.Adam(self.param.parameters(), lr=self.lr_param)


    def choose_action(self, state, tl_code, init_hidden=False, train=True):
        self._step += 1
        if train:
            # epsilon 更新
            self._epsilon = max(
                    self.epsilon_final, 
                    self._epsilon - (self.epsilon_initial - self.epsilon_final) / self.epsilon_decay
                    )

            with torch.no_grad(): # 不生成计算图，减少显存开销
                state = torch.tensor(state, device=self.device)
                tl_code = torch.tensor(tl_code, device=self.device)
                all_action_parameters = self.param.forward(state, tl_code, init_hidden) # 1*3 维连续 param
                print("Network output -- all_action_param: ",all_action_parameters)
                
                if self._epsilon > np.random.random(): # 探索率随着迭代次数增加而减小
                    # if self._step < self.batch_size:    
                    if self._step < self.minimal_size: # 开始学习前，变道随机，acc随机
                        action = np.random.randint(0, self.num_action) # 离散 action 随机
                        all_action_parameters = torch.from_numpy(np.random.uniform(
                            self.action_param_min_numpy, self.action_param_max_numpy)).to(self.device)
                        all_action_parameters = all_action_parameters.unsqueeze(0).to(torch.float32) # np是64的精度，转为32的精度
                    else: # 开始学习后，变道 不 随机
                        if self.NormalNoise: # acc加噪声
                            all_action_parameters = torch.clamp(torch.normal(mean=all_action_parameters, std=0.1), min=-3, max=3)
                        else: # acc 随机
                            all_action_parameters = torch.from_numpy(
                                    np.random.uniform(self.action_param_min_numpy, self.action_param_max_numpy)).to(self.device)
                            all_action_parameters = all_action_parameters.unsqueeze(0).to(torch.float32) # np是64的精度，转为32的精度

                Q_value = self.actor.forward(state, tl_code, all_action_parameters, init_hidden) # 1*3 维 所有离散动作的Q_value
                Q_value = Q_value.detach().cpu().numpy() # tensor 转换为 numpy格式
                action = np.argmax(Q_value)
                all_action_parameters = all_action_parameters.squeeze() # 变为3维 连续 param
                all_action_parameters = all_action_parameters.cpu().data.numpy()
                print("After Noise -- all_action_param:", all_action_parameters)
                print("Q values: ", Q_value)

                action_parameters = all_action_parameters[action] # all_action_parameters从1*3维，从第1维中选
        
        if not train:
            with torch.no_grad(): 
                state = torch.tensor(state, device=self.device)
                tl_code = torch.tensor(tl_code, device=self.device)
                all_action_parameters = self.param.forward(state, tl_code, init_hidden) # 1*3 维连续 param
                Q_value = self.actor.forward(state, tl_code, all_action_parameters, init_hidden) # 1*3 维 所有离散动作的Q_value
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
        
    def store_transition(self, obs, tl, act, act_param, rew, next_obs, next_tl, done):
        obs = np.reshape(obs, (-1, 1)) # 列数为1，行数-1根据列数来确定
        obs = np.squeeze(obs)
        tl = np.reshape(tl, (-1, 1)) # 列数为1，行数-1根据列数来确定
        tl = np.squeeze(tl)
        act = np.reshape(act, (-1, 1))
        act = np.squeeze(act)
        act_param = np.reshape(act_param, (-1, 1))
        act_param = np.squeeze(act_param)
        next_obs = np.reshape(next_obs, (-1, 1))
        next_obs = np.squeeze(next_obs)
        next_tl = np.reshape(next_tl, (-1, 1)) # 列数为1，行数-1根据列数来确定
        next_tl = np.squeeze(next_tl)
       
        self.transition = [obs, tl, act, act_param, rew, next_obs, next_tl, done]
        self.memory.store(*self.transition) # store 没有返回值

    def learn(self):
        self._learn_step += 1
        if self.per_flag:        
            samples = self.memory.sample_batch(self.beta)
            # PER needs beta to calculate weights
            weights = torch.FloatTensor(samples["weights"].reshape(-1,1)).to(self.device)
            indices = samples["indices"]
        else:
            samples = self.memory.sample_batch()
        
        # compute loss
        device = self.device  # for shortening the following lines
        b_state = torch.FloatTensor(samples["obs"]).to(device)
        b_next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        b_tl_code = torch.FloatTensor(samples["tl_code"]).to(device)
        b_next_tl_code = torch.FloatTensor(samples["next_tl_code"]).to(device)
        b_action = torch.LongTensor(samples["act"].reshape(-1, 1)).to(device)
        b_action_param = torch.FloatTensor(samples["act_param"]).to(device)
        b_reward = torch.FloatTensor(samples["rew"].reshape(-1, 1)).to(device)
        b_done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        b_prev_obs = torch.FloatTensor(samples["prev_obs"]).to(device).permute(1, 0, 2)
        b_prev_tl_codes = torch.FloatTensor(samples["prev_tl_code"]).to(device).permute(1, 0, 2)
        b_prev_actions = torch.LongTensor(samples["prev_acts"]).to(device).permute(1, 0, 2)
        b_prev_action_params = torch.FloatTensor(samples["prev_acts_param"]).to(device).permute(1, 0, 2)
        #print(samples["prev_obs"].shape, samples["prev_tl_code"].shape, samples["prev_acts"].shape, samples["prev_acts_param"].shape, sep='\n')
        #print(samples["obs"].shape, samples["tl_code"].shape, samples["act"].shape, samples["act_param"].shape, sep='\n')
        
        # retrieve 4 previous state in sequence
        self.init_hidden(self.batch_size)
        for i in range(self.burn_in_step):
            b_prev_ob = b_prev_obs[i, :, :]
            b_prev_tl_code = b_prev_tl_codes[i, :, :]
            b_prev_action = b_prev_actions[i, :, :].reshape(-1, 1)
            b_prev_action_param = b_prev_action_params[i, :, :]
            #print(b_prev_ob.shape, b_prev_tl_code.shape, b_prev_action.shape, b_prev_action_param.shape, sep='\n')
            
            self.param(b_prev_ob, b_prev_tl_code)
            self.param_target(b_prev_ob, b_prev_tl_code)
            self.actor( b_prev_ob, b_prev_tl_code, b_prev_action_param)
            self.actor_target(b_prev_ob, b_prev_tl_code, b_prev_action_param)

        actor_H, actor_C = deepcopy(torch.clone(self.actor.hidden_state[0]).detach()), \
            deepcopy(torch.clone(self.actor.hidden_state[1]).detach())
        actor_target_H, actor_target_C = deepcopy(torch.clone(self.actor_target.hidden_state[0]).detach()), \
            deepcopy(torch.clone(self.actor_target.hidden_state[1].detach()))
        param_H, param_C = deepcopy(torch.clone(self.param.hidden_state[0]).detach()), \
            deepcopy(torch.clone(self.param.hidden_state[1]).detach())
        param_target_H, param_target_C = deepcopy(torch.clone(self.param_target.hidden_state[0]).detach()), \
            deepcopy(torch.clone(self.param_target.hidden_state[1]).detach())

        # -----------------------optimize Q actor------------------------
        with torch.no_grad():
            next_action_parameters = self.param_target(b_next_state, b_next_tl_code) # b_next_state torch.Size([32, 21])
            next_q_value = self.actor_target(b_next_state, b_next_tl_code, next_action_parameters) # [32, 21] [32, 3]
            q_prime = torch.max(next_q_value, 1, keepdim=True)[0] # q_prime torch.Size([128, 1])
            # Compute the TD error
            gamma = self.gamma ** self.n_step
            target = b_reward + (1 - b_done) * gamma * q_prime # target torch.Size([128, 1])
        
        q_values = self.actor(b_state, b_tl_code, b_action_param) # [32, 21] [32, 3]
        y_predicted = q_values.gather(1, b_action.view(-1, 1)) # gather函数可以看作一种索引
        if self.per_flag:
            elementwise_loss = self.loss_func(y_predicted, target) 
            loss_actor = torch.mean(elementwise_loss * weights)
        else:
            loss_actor = self.loss_func(y_predicted, y_predicted)   # loss 是torch.Tensor的形式
        ret_loss_actor = loss_actor.detach().cpu().numpy()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        if self.clip_grad > 0:  # clip防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)

        self.actor_optimizer.step()

        # ------------------------optimize param net------------------------------
        with torch.no_grad():
            action_params = self.param(b_state, b_tl_code)
        action_params.requires_grad = True
        self.actor.hidden_state=(actor_H, actor_C)
        Q_val = self.actor(b_state, b_tl_code, action_params)
        Q_loss = torch.mean(torch.sum(Q_val, 1)) # 这里不知道为什么？？
        self.actor.zero_grad()
        Q_loss.backward()
        
        # ==============================
        delta_a = deepcopy(action_params.grad.data)
        self.param.hidden_state=(param_H, param_C)
        action_params = self.param(b_state, b_tl_code)
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
        
        #PER: update priorities
        if self.per_flag:
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(indices, new_priorities)
            self.beta = np.min([0.99, self.beta + self.beta_increment_per_sampling])  # max = 0.9
        
        return ret_loss_actor, Q_loss.detach().cpu().numpy()

    def soft_update_target_network(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    def init_hidden(self, batch_size=1):
        self.actor.init_hidden(batch_size)
        self.actor_target.init_hidden(batch_size)
        self.param.init_hidden(batch_size)
        self.param_target.init_hidden(batch_size)