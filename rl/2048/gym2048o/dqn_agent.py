#!/usr/bin/python3 

import torch 
from torch import nn 


class Net(nn.Module):
    def __init__(self, n, n_actions):
        super().__init__()
        n2 = n * n  
        self.net = nn.Sequential(
            nn.Linear(n2, n2), 
            nn.ReLU(), 
            nn.Linear(n2, n_actions), 
            nn.ReLU(), 
        )


    def forward(self, x): 
        return self.net(x)


class Dqn_Agent: 
    def __init__(self, env): 
        self.env = env 
        self.net = self.Net(env.n, env.action_space.n)

