#!/usr/bin/python3 

import random 
import collections 
import time 

import torch 
from torch import nn 

from env_2048 import Env2048

GAMMA = 0.9 
BATCH_SIZE = 10 


Step = collections.namedtuple('Step', field_names=['s', 'a', 'r', 's1'])

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
        self.net = Net(env.n, env.action_space.n)
        self.epsilon = 1

    def choose_action(self): 
        if random.random() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            action_vals = self.net(torch.FloatTensor(o)) 
            a = torch.argmax(action_vals).item()
        if self.epsilon > 0.2: 
            self.epsilon -= 0.00001
        return a 


    def gain_experience(self, batch_size = BATCH_SIZE): 
        s = self.env.reset().flatten()
        i = 0 
        steps = []
        while True: 
            if i == batch_size: 
                yield steps
                steps = [] 
                i = 0

            a = self.choose_action()

            s1, r, d, info = self.env.step(a)
            s1 = s1.flatten()
            steps.append(Step(s, a, r, s1))
            s = self.env.reset().flatten() if d else s1
            i += 1 



    def train(self, n_episodes=100): 
        for ep_n in range(n_episodes): 
            pass 


if __name__ == '__main__': 
    env = Env2048(2)
    env1 = Env2048(2)
    agent = Dqn_Agent(env)
    for i, batch in enumerate(agent.gain_experience()):
        if i == 2:
            break
        for step in batch:
            env1.brd = step.s.reshape((2,2))
            env1.render()
            time.sleep(2)
        print('*' * 40 + '\n\n')
