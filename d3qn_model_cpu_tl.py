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
import torch.optim as optim
# from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, tl_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.tl_buf = np.zeros([size, tl_dim], dtype=np.float32)
        self.next_tl_buf = np.zeros([size, tl_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        tl_code: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        next_tl_code: np.ndarray,
        done: bool,
    ):
        # print('tl_code', tl_code)
        # print(type(tl_code))
        # print('obs', obs)
        # print(type(obs))
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.tl_buf[self.ptr] = tl_code
        self.next_tl_buf[self.ptr] = next_tl_code
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    tl_code=self.tl_buf[idxs],
                    next_tl_code=self.next_tl_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


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
        x = torch.reshape(x, (-1, 21))  # 7*3变为1*21 torch.Size([1, 21])
        x = self.feature_layer(x)
        
        x = torch.cat([x, tl], dim = 1)
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
        memory_size: int = 20000,
        batch_size: int = 32,
        target_update: int = 1000,
        epsilon_decay: float = 1 / 2000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
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
        self.memory = ReplayBuffer(self.N_STATES, self.N_TL, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.learn_step_counter = 0                                     # target 更新计数
        self.gamma = gamma
        
        # device: cpu / gpu
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu"
        # )
        self.device = torch.device("cpu")
        print(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(self.N_STATES, self.N_TL, self.N_ACTIONS).to(self.device)
        self.dqn_target = Network(self.N_STATES, self.N_TL, self.N_ACTIONS).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def choose_action(self, state: np.ndarray, tl) -> np.ndarray:
        """Select an action from the input state."""
        # # epsilon greedy policy
        # if self.epsilon > np.random.random(): # 探索率随着迭代次数增加而减小
        #     # selected_action = self.env.action_space.sample()
        #     selected_action = np.random.randint(0, self.N_ACTIONS)
        # else:
        #     # selected_action = self.dqn(
        #     #     torch.FloatTensor(state).to(self.device)
        #     # ).argmax()
        #     state = torch.unsqueeze(torch.FloatTensor(state), 0)
        #     tl = torch.unsqueeze(torch.FloatTensor(tl), 0)
        #     selected_action = self.dqn(state, tl).to(self.device).argmax()
            
        #     selected_action = selected_action.detach().cpu().numpy()
        
        # 测试
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        tl = torch.unsqueeze(torch.FloatTensor(tl), 0)
        selected_action = self.dqn(state, tl).to(self.device).argmax()
        
        selected_action = selected_action.detach().cpu().numpy()
        
        # if not self.is_test:
        #     self.transition = [state, selected_action]
        
        return selected_action

    def store_transition(self, s, tl, a, r, s_, tl_, done):
        s = np.reshape(s, (-1, 1)) # 列数为1，行数-1根据列数来确定
        s = np.squeeze(s)
        s_ = np.reshape(s_, (-1, 1))
        s_ = np.squeeze(s_)
        tl = np.array(tl)
        tl_ = np.array(tl_)
        # transition = np.hstack((s, [a, r], s_, done))
        # replace the old memory with new memory
        # index = self.memory_counter % MEMORY_CAPACITY
        # self.memory[index, :] = transition
        # self.memory_counter += 1
        self.transition = [s, tl, a, r, s_, tl_, done]
        # one_step_transition = self.transition
        self.memory.store(*self.transition) # store 没有返回值

    # def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
    #     """Take an action and return the response of the env."""
    #     next_state, reward, done, _ = self.env.step(action)

    #     if not self.is_test:
    #         self.transition += [reward, next_state, done]
    #         self.memory.store(*self.transition)
    
    #     return next_state, reward, done

    def learn(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # target net 参数更新
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.dqn_target.load_state_dict(self.dqn.state_dict())
        
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        # DuelingNet: we clip the gradients to have their norm less than or equal to 10.
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        print('loss.item() ', loss.item())

        return loss.item()
    
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
                
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        tl_code = torch.FloatTensor(samples["tl_code"]).to(device)
        next_tl_code = torch.FloatTensor(samples["next_tl_code"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state, tl_code).gather(1, action)
        # next_q_value = self.dqn_target(next_state).max(
        #     dim=1, keepdim=True
        # )[0].detach()
        next_q_value = self.dqn_target(next_state, next_tl_code).gather(  # Double DQN
           1, self.dqn(next_state, next_tl_code).argmax(dim=1, keepdim=True)
       ).detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    # def _target_hard_update(self):
    #     """Hard update: target <- local."""
    #     self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    # def _plot(
    #     self, 
    #     frame_idx: int, 
    #     scores: List[float], 
    #     losses: List[float], 
    #     epsilons: List[float],
    # ):
    #     """Plot the training progresses."""
    #     clear_output(True)
    #     plt.figure(figsize=(20, 5))
    #     plt.subplot(131)
    #     plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
    #     plt.plot(scores)
    #     plt.subplot(132)
    #     plt.title('loss')
    #     plt.plot(losses)
    #     plt.subplot(133)
    #     plt.title('epsilons')
    #     plt.plot(epsilons)
    #     plt.show()
        

