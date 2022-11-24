#!/usr/bin/python3 

import ossaudiodev
import random 
import collections 
import time 
import numpy as np 


import torch 
from torch import nn 

from torch.utils.tensorboard import SummaryWriter

from env_2048 import Env2048

GAMMA = 0.9 
BATCH_SIZE = 10 
LR = 1e-3


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
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(lr=LR)
        self.writer = SummaryWriter(comment=f'_dqn{env.n}X{env.n}_')


    def choose_action(self): 
        if random.random() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            action_vals = self.net(torch.FloatTensor(ossaudiodev)) 
            a = torch.argmax(action_vals).item()
        if self.epsilon > 0.2: 
            self.epsilon -= 0.00001
        return a 


    def gain_experience(self, batch_size=BATCH_SIZE): 
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

    @staticmethod
    def wrap_as_tensors():
            states = torch.Tensor(np.array(states))
            actions = torch.Tensor(np.array(actions))
            rewards = torch.Tensor(np.array(rewards))
            next_states = torch.Tensor(np.array(next_states))
            return states, actions, rewards, next_states


    def calc_loss(self, q_values, actions, rewards, ns_q_values):
        q_values_chosen = torch.gather(q_values, 1, actions.unsqueeze(1)).squeeze()
        belmann_values = rewards + GAMMA * ns_q_values
        return self.loss_fn(q_values_chosen, belmann_values)
        


    def train(self, n_batches=100): 
        for t, batch in enumerate(self.gain_experience()): 
            states, actions, rewards, next_states = zip(*batch)
            states, actions, rewards, next_states = \
                self.wrap_as_tensors(states, actions, rewards, next_states)
            
            q_values = self.net(states)
            ns_q_values = self.net(next_states).detach()
            self.optimizer.zero_grad()    
            loss = self.calc_loss(q_values, actions, rewards, ns_q_values)
            loss.backward() 
            self.optim.step()
            self.writer.add_scalar("loss", loss, t)






if __name__ == '__main__': 
    env = Env2048(2)
    env1 = Env2048(2)
    agent = Dqn_Agent(env)
    for i, batch in enumerate(agent.gain_experience()):
        if i == 2:
            break
        for step in batch:
            print(step)
            # env1.brd = step.s1.reshape((2,2))
            # env1.render()
            # time.sleep(2)
        print('*' * 40 + '\n\n')
