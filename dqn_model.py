# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 21:17:53 2022

DQN部分代码

dependencies:
numpy:1.21.6
torch:1.11.0


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.99                # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency
MEMORY_CAPACITY = 20000 # 20000
#env = gym.make('CartPole-v0')
#env = env.unwrapped
#N_ACTIONS = env.action_space.n
#N_STATES = env.observation_space.shape[0]
#ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(n_feature, 64)
        self.fc1.weight.data.normal_(0, 0.1)   # 初始化
        self.out = torch.nn.Linear(64, n_output)
        self.out.weight.data.normal_(0, 0.1)   # 初始化

    def forward(self, x):
        batch = x.shape[0]
        x = torch.reshape(x, (batch, 21)) # 3*7变为1*21
        x = F.relu(self.fc1(x))
        actions_value = self.out(x) # tensor([[-0.1769,  0.1941,  0.1093, 
#        actions_value = F.gumbel_softmax(actions_value, tau=1, hard=True) # tensor([[0., 0., 1.,
        return actions_value


class DQN(nn.Module): # 从object改为了nn.Module
    def __init__(self, N_STATES, N_ACTIONS):
        self.N_ACTIONS = N_ACTIONS
        self.N_STATES = N_STATES
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0                                     # target 更新计数
        self.memory_counter = 0                                         # 记忆库计数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) 
        self.loss_func = nn.CrossEntropyLoss()   # 回归问题用MSELoss()，分类问题用CrossEntropyLoss

    def choose_action(self, x):
        '''
        输入状态的观测值，选择动作
        '''
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 输入一个 sample
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy() # array([7], dtype=int64)
            action = action[0]  # 取int值
        else:   # 选随机动作
            action = np.random.randint(0, self.N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        s = np.reshape(s, (-1, 1))
        s = np.squeeze(s)
        s_ = np.reshape(s_, (-1, 1))
        s_ = np.squeeze(s_)
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # 根据b_s用eval_net得出所有动作的价值，获得b_a该动作的价值 shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # q_target这里不需要更新，q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # max返回的是[最大值，索引]，shape (batch, 1)
        loss = self.loss_func(q_eval, q_target) # loss_func 的参数中，预测值在前，真实值在后

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_net(self):
        torch.save(self.eval_net, 'eval_net.pkl')  # save entire net
        torch.save(self.eval_net.state_dict(), 'eval_net_params.pkl')   # save only the parameters




