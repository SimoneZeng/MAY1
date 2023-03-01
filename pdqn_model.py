# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:45:19 2023

借鉴学长的pdqn
github中的 mpdqn

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


class ReplayBuffer:
    """
    A simple numpy replay buffer.
    和d3qn中的ReplayBuffer 类似 多了一个acts_param_buf
    obs_dim: the dimension of the input
    size: the size of ReplayBuffer or memory
    batch_size
    """

    def __init__(self, obs_dim: int, param_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.acts_param_buf = np.zeros([size, param_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0 # self.ptr points the position to add a new line, self.size is the length of loaded lines

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        act_param: float,
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.acts_param_buf[self.ptr] = act_param
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False) # get a batch from loaded lines
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    act=self.acts_buf[idxs],
                    act_param = self.acts_param_buf[idxs],
                    rew=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


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
        

class PDQNAgent(nn.Module):
    def __init__(
            self, 
            state_dim = 3*7, 
            action_dim: int =1, 
            memory_size: int = 20000,
            batch_size: int = 32,
            epsilon_initial=1.0,
            epsilon_final=0.05,
            epsilon_decay=2000,
            gamma=0.99,
            lr_actor=0.001,
            lr_param=0.0001,
            TARGET_UPDATE_ITER: int = 1000,
    ):
        super(PDQNAgent, self).__init__()
        
        self.device = torch.device('cpu')
        self.action_dim = action_dim # 1 维，输出1个连续动作 acc
        self.state_dim = state_dim
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.lr_actor, self.lr_param = lr_actor, lr_param
        self.gamma = gamma
        self.clip_grad = 10
        self.tau_actor = 0.01
        self.tau_param = 0.001
        
        self._epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self._step = 0
        self._learn_step = 0
        
        self.num_action = 3 # 3 kinds of discrete action
        self.action_param_sizes = np.array([self.action_dim, self.action_dim, self.action_dim]) # 1, 1, 1
        self.action_param_size = int(self.action_param_sizes.sum()) # 3
        self.action_param_max_numpy = np.array([3, 3, 3])
        self.action_param_min_numpy = np.array([-3, -3, -3])
        self.action_param_range = torch.from_numpy((self.action_param_max_numpy - self.action_param_min_numpy)).float().to(self.device)
        self.action_param_max = torch.from_numpy(self.action_param_max_numpy).float().to(self.device)
        self.action_param_min = torch.from_numpy(self.action_param_min_numpy).float().to(self.device)
        self.action_param_offsets = self.action_param_sizes.cumsum() # 不知道干嘛的
        self.action_param_offsets = np.insert(self.action_param_offsets, 0, 0)
        self.memory = ReplayBuffer(self.state_dim, self.action_param_size, self.memory_size, self.batch_size)
        
        self.actor = QActor(self.state_dim, self.num_action, self.action_param_size).to(self.device)
        self.actor_target = QActor(self.state_dim, self.num_action, self.action_param_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()  # 不启用 BatchNormalization 和 Dropout
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
                    action = np.random.randint(0, self.num_action) # 离散 action 随机
                    all_action_parameters = torch.from_numpy( # 3 维连续 param 随机数
                            np.random.uniform(self.action_param_min_numpy, self.action_param_max_numpy))
                else:
                    # select maximum action
                    Q_value = self.actor.forward(state, all_action_parameters) # 1*3 维 所有离散动作的Q_value
                    Q_value = Q_value.detach().cpu().numpy() # tensor 转换为 numpy格式
                    action = np.argmax(Q_value)
                    all_action_parameters = all_action_parameters.squeeze() # 变为3维 连续 param

                all_action_parameters = all_action_parameters.cpu().data.numpy()
                action_parameters = all_action_parameters[action] # all_action_parameters从1*3维，从第1维中选
        else:
            with torch.no_grad(): 
                state = torch.tensor(state, device=self.device)
                all_action_parameters = self.param.forward(state) # 1*3 维连续 param
                Q_value = self.actor.forward(state, all_action_parameters) # 1*3 维 所有离散动作的Q_value
                Q_value = Q_value.detach().cpu().numpy() # tensor 转换为 numpy格式
                action = np.argmax(Q_value)
                all_action_parameters = all_action_parameters.squeeze() # 变为3维 连续 param
            
                all_action_parameters = all_action_parameters.cpu().data.numpy()
                action_parameters = all_action_parameters[action] # all_action_parameters从1*3维，从第1维中选
        
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
        b_next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        b_action = torch.LongTensor(samples["act"].reshape(-1, 1)).to(device)
        b_action_param = torch.FloatTensor(samples["act_param"]).to(device)
        b_reward = torch.FloatTensor(samples["rew"].reshape(-1, 1)).to(device)
        b_done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # -----------------------optimize Q actor------------------------
        with torch.no_grad():
            next_action_parameters = self.param_target.forward(b_next_state) # b_next_state torch.Size([32, 21])
            next_q_value = self.actor_target(b_next_state, next_action_parameters) # [32, 21] [32, 3]
            q_prime = torch.max(next_q_value, 1, keepdim=True)[0].squeeze()
            # Compute the TD error
            target = b_reward + (1 - b_done) * self.gamma * q_prime
        
        # Compute current Q-values using policy network
        q_values = self.actor(b_state, b_action_param) # [32, 21] [32, 3]
        y_predicted = q_values.gather(1, b_action.view(-1, 1)).squeeze() # gather函数可以看作一种索引
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





