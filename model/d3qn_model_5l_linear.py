# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:59:38 2022

d3qn_model 代码
借鉴
https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/04.dueling.ipynb
https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/02.double_q.ipynb

double dqn 只需要改一行代码


@author: Simone
"""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
from model.replay_buffer import ReplayBuffer

# class ReplayBuffer:
#     """A simple numpy replay buffer."""

#     def __init__(self, obs_dim: int, tl_dim: int, size: int, batch_size: int = 32):
#         self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         self.tl_buf = np.zeros([size, tl_dim], dtype=np.float32)
#         self.next_tl_buf = np.zeros([size, tl_dim], dtype=np.float32)
#         self.acts_buf = np.zeros([size], dtype=np.float32)
#         self.rews_buf = np.zeros([size], dtype=np.float32)
#         self.done_buf = np.zeros(size, dtype=np.float32)
#         self.max_size, self.batch_size = size, batch_size
#         self.ptr, self.size, = 0, 0

#     def store(
#         self,
#         obs: np.ndarray,
#         tl_code: np.ndarray,
#         act: np.ndarray, 
#         rew: float, 
#         next_obs: np.ndarray, 
#         next_tl_code: np.ndarray,
#         done: bool,
#     ):
#         # print('tl_code', tl_code)
#         # print(type(tl_code))
#         # print('obs', obs)
#         # print(type(obs))
#         self.obs_buf[self.ptr] = obs
#         self.next_obs_buf[self.ptr] = next_obs
#         self.tl_buf[self.ptr] = tl_code
#         self.next_tl_buf[self.ptr] = next_tl_code
#         self.acts_buf[self.ptr] = act
#         self.rews_buf[self.ptr] = rew
#         self.done_buf[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)

#     def sample_batch(self) -> Dict[str, np.ndarray]:
#         idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
#         return dict(obs=self.obs_buf[idxs],
#                     next_obs=self.next_obs_buf[idxs],
#                     tl_code=self.tl_buf[idxs],
#                     next_tl_code=self.next_tl_buf[idxs],
#                     acts=self.acts_buf[idxs],
#                     rews=self.rews_buf[idxs],
#                     done=self.done_buf[idxs])

#     def __len__(self) -> int:
#         return self.size


class Network(nn.Module):
    def __init__(self, in_dim: int, tl_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
        )

        self.tl_feature_layer = nn.Sequential(
            nn.Linear(128 + tl_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(),
        )
        
        # set advantage layer
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

        # set value layer
        self.value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, tl) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.reshape(x, (-1, 21)).float()  # 7*3变为1*21 torch.Size([1, 21])
        tl_code = torch.reshape(tl, (-1, 7)) # 5 维变为 1*7
        x = self.feature_layer(x)
        print(x.shape, tl.shape, sep='\t')
        
        x = torch.cat((x, tl_code), dim = 1)
        feature = self.tl_feature_layer(x)
        
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q


class DQNAgent(nn.Module):
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
        self, 
        N_STATES, 
        N_TL,
        N_ACTIONS,
        memory_size: int = 40000,
        minimal_size: int = 5000,
        batch_size: int = 128,
        epsilon_decay: float = 1 / 200,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        lr: float = 0.0001,
        gamma: float = 0.9,
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        super(DQNAgent, self).__init__()# 需要添加这个，同时继承自nn.Module 否则在保存agent.state_dict()是会显示has no attribute 'state_dict'
        # obs_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.n
        self.N_ACTIONS = N_ACTIONS
        self.N_TL = N_TL
        self.N_STATES = N_STATES
        
        # self.env = env
        self.memory = ReplayBuffer(obs_dim=self.N_STATES, param_dim=1, tl_dim=self.N_TL, size=memory_size, batch_size=batch_size)
        self.minimal_size = minimal_size    #The agent begins learn after replaly buffer reach minimal_size
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self._learn_step = 0                                     
        self._step = 0
        self.gamma = gamma
        self.tau_target = 0.01
        self.lr = lr      
        self.device = device

        # networks: dqn, dqn_target
        self.dqn = Network(self.N_STATES, self.N_TL, self.N_ACTIONS).to(self.device)
        self.dqn_target = Network(self.N_STATES, self.N_TL, self.N_ACTIONS).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr)

        # transition to store in memory
        self.transition = list()

    def choose_action(self, state: np.ndarray, tl, train=True) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        self._step += 1
        if train and self.epsilon > np.random.random(): # 探索率随着迭代次数增加而减小
            # selected_action = self.env.action_space.sample()
            selected_action = np.random.randint(0, self.N_ACTIONS)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            tl = torch.tensor(tl, dtype=torch.float32, device=self.device)
            selected_action = self.dqn(state, tl).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        return selected_action

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


    def learn(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # target net 参数更新
        self._learn_step += 1
        samples = self.memory.sample_batch()

        b_state = torch.FloatTensor(samples["obs"]).to(self.device)
        b_next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        b_tl_code = torch.FloatTensor(samples["tl_code"]).to(self.device)
        b_next_tl_code = torch.FloatTensor(samples["next_tl_code"]).to(self.device)
        b_action = torch.LongTensor(samples["act"].reshape(-1, 1)).to(self.device)
        b_reward = torch.FloatTensor(samples["rew"].reshape(-1, 1)).to(self.device)
        b_done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        curr_q_value = self.dqn(b_state, b_tl_code).gather(1, b_action)
        next_q_value = self.dqn_target(b_next_state, b_next_tl_code).gather(  # Double DQN
           1, self.dqn(b_next_state, b_next_tl_code).argmax(dim=1, keepdim=True)
        ).detach()
        target = (b_reward + self.gamma * next_q_value * (1 - b_done)).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        # DuelingNet: we clip the gradients to have their norm less than or equal to 10.
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # linearly decrease epsilon
        if self._learn_step % 50 == 0:
            self.epsilon = max(
                self.min_epsilon, 
                self.epsilon - ( self.max_epsilon - self.min_epsilon) * self.epsilon_decay
            )
        self.soft_update_target_network(self.dqn, self.dqn_target, self.tau_target)

        return loss.detach().cpu().numpy()
    
    # def train(self, num_frames: int, plotting_interval: int = 200):
    #     """Train the agent."""
    #     self.is_test = False
        
    #     state = self.env.reset()
    #     update_cnt = 0
    #     epsilons = []
    #     losses = []
    #     scores = []
    #     score = 0

    #     for frame_idx in range(1, num_frames + 1):
    #         action = self.select_action(state)
    #         next_state, reward, done = self.step(action)

    #         state = next_state
    #         score += reward

    #         # if episode ends
    #         if done:
    #             state = self.env.reset()
    #             scores.append(score)
    #             score = 0

    #         # if training is ready
    #         if len(self.memory) >= self.batch_size:
    #             loss = self.update_model()
    #             losses.append(loss)
    #             update_cnt += 1
                
    #             # linearly decrease epsilon
    #             self.epsilon = max(
    #                 self.min_epsilon, self.epsilon - (
    #                     self.max_epsilon - self.min_epsilon
    #                 ) * self.epsilon_decay
    #             )
    #             epsilons.append(self.epsilon)
                
    #             # if hard update is needed
    #             if update_cnt % self.target_update == 0:
    #                 self._target_hard_update()

    #         # plotting
    #         if frame_idx % plotting_interval == 0:
    #             self._plot(frame_idx, scores, losses, epsilons)
                
    #     self.env.close()

    def soft_update_target_network(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    

