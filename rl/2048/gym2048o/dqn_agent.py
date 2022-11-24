#!/usr/bin/python3 

import random 
import collections 
import numpy as np 


import torch 
from torch import nn 

from torch.utils.tensorboard import SummaryWriter

from env_2048 import Env2048

GAMMA = 0.9 
BATCH_SIZE = 100 
LR = 1e-4
EPSILON_DECAY=1e-4

Step = collections.namedtuple('Step', field_names=['s', 'a', 'r', 's1', 'is_final'])

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
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=LR)
        self.writer = SummaryWriter(comment=f'_dqn{env.n}X{env.n}_')


    def choose_action(self, s): 
        if random.random() < self.epsilon:
            a = self.env.action_space.sample()
        else:
            action_vals = self.net(torch.tensor(s, dtype=torch.float32)) 
            a = torch.argmax(action_vals).item()
        if self.epsilon > 0.2: 
            self.epsilon -= EPSILON_DECAY
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

            a = self.choose_action(s)

            s1, r, d, info = self.env.step(a)
            s1 = s1.flatten()
            steps.append(Step(s, a, r, s1, d))
            s = self.env.reset().flatten() if d else s1
            i += 1 


    @staticmethod
    def wrap_as_tensors(states, actions, rewards, next_states, finals):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.int64)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        finals = np.array(finals)
        return states, actions, rewards, next_states, finals


    def calc_loss(self, states, actions, rewards, next_states, final_mask):
        q_values = self.net(states)

        ns_q_values = self.net(next_states).detach()
        ns_values = torch.max(ns_q_values, 1)[0]
        ns_values[final_mask] = 0 


        q_values_chosen = torch.gather(q_values, 1, actions.unsqueeze(1)).squeeze()

        belmann_values = rewards + GAMMA * ns_values
        return self.loss_fn(q_values_chosen, belmann_values)
        

    def train(self, n_batches=100): 
        test_env = Env2048(self.env.n)
        test_n = 3
        for t, batch in enumerate(self.gain_experience()): 
            if t == n_batches:
                break
            states, actions, rewards, next_states, finals = zip(*batch)
            states, actions, rewards, next_states, finals = \
                self.wrap_as_tensors(states, actions, rewards, next_states, finals)
            
            self.optimizer.zero_grad()    
            loss = self.calc_loss(states, actions, rewards, next_states, finals)
            loss.backward() 
            self.optimizer.step()
            self.writer.add_scalar('loss', loss.item(), t)
            if t % 100 == 0:
                total = 0 
                for _ in range(test_n):
                    g = self.play_episode(test_env)
                    total += g 
                avg = total / test_n
                print(t, loss.item(), f'avg reward:{avg}')
                self.writer.add_scalar('reward', avg, t)


    def play_episode(self, env): 
        total = 0 
        s = env.reset().flatten()
        for _ in range(200): 
            action_vals = self.net(torch.tensor(s, dtype=torch.float32)) 
            a = torch.argmax(action_vals).item()
            s1, r, d, _ = env.step(a)
            total += r 
            if d: 
                break 
            s = s1.flatten() 
        return total 


if __name__ == '__main__': 
    env = Env2048(4)
    agent = Dqn_Agent(env)
    agent.train(70000)

    # for i, batch in enumerate(agent.gain_experience()):
    #     if i == 2:
    #         break
    #     for step in batch:
    #         print(step)
    #     print('*' * 40 + '\n\n')
