import os
import random, collections, math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy

class Discriminator(nn.Module):
    def __init__(self, state_dim, tl_size, action_param_size):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + tl_size + action_param_size, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, state, tl_code, action_parametes):
        state = torch.reshape(state, (-1, 21)).float()
        tl_code = torch.reshape(tl_code, (-1, 7))
        cat = torch.cat([state, tl_code, action_parametes], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))
    
class GAIL:
    def __init__(self, agent, state_dim, tl_size, action_dim, 
                lr_d = 1e-3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.discriminator = Discriminator(state_dim, tl_size, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)
        self.agent = agent
        self.device = device

    def learn(self, expert_s, expert_tl, expert_a, agent_s, agent_tl, agent_a, next_s, next_tl, dones):
        expert_states = torch.tensor(expert_s, dtype=torch.float32).to(self.device)
        expert_tl_codes = torch.tensor(expert_tl, dtype=torch.float32).to(self.device)
        expert_actions = torch.tensor(expert_a, dtype=torch.float32).to(self.device)
        agent_states = torch.tensor(agent_s, dtype=torch.float32).to(self.device)
        agent_tl_codes = torch.tensor(agent_tl, dtype=torch.float32).to(self.device)
        agent_actions = torch.tensor(agent_a, dtype=torch.float32).to(self.device)
        expert_actions = F.one_hot(expert_actions, num_classes=2).float()
        agent_actions = F.one_hot(agent_actions, num_classes=2).float()

        expert_prob = self.discriminator(expert_states, expert_tl_codes, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_tl_codes, agent_actions)
        discriminator_loss = nn.BCELoss()(agent_prob, torch.ones_like(agent_prob)) + \
                            nn.BCELoss()(expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        self.agent.store_transition(agent_s, agent_tl, agent_a, rewards, next_s, next_tl, dones)
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': dones
        }
        self.agent.update(transition_dict)