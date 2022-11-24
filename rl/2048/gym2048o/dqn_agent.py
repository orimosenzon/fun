#!/usr/bin/python3 

import random 

import torch 
from torch import nn 


GAMMA = 0.9 

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
        self.epsilon = 1


    def gather_exprience(self, steps = -1): 
        o = self.env.reset()
        steps =[]
        while True: 
            if random.random() < self.epsilon:
                a = self.env.action_space.sample()
            else:
                action_vals = self.net(torch.FloatTensor(o)) 
                a = torch.argmax(action_vals).item()

            o1, r, d, info = self.env.step(a)
            steps.append((o.flatten(), a, r))
            o = o1 


    def train(self, n_episodes=100): 
        for ep_n in range(n_episodes): 

