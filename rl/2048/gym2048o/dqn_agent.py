#!/usr/bin/python3 

import random 
import collections 
import numpy as np 
import datetime
import time

import torch 
from torch import nn 

from torch.utils.tensorboard import SummaryWriter

from env_2048 import Env2048

GAMMA = 0.9 
BATCH_SIZE = 100 
LR = 1e-3
EPSILON_DECAY=1e-4
MIN_EPSILON = 0.2 
DISPLAY_PERIOD = 100 
EPISODES_FOR_EVAL = 300 

Step = collections.namedtuple('Step', field_names=['s', 'a', 'r', 's1', 'is_final'])

class Net(nn.Module):
    def __init__(self, n, n_actions, n_blocks=2, internal_size_f=3):
        super().__init__()
        n2 = n * n
        internal_size = n2 * internal_size_f  
        layers = collections.OrderedDict()
        in_f, out_f = n2, internal_size
        for i in range(n_blocks):
            if i == n_blocks-1:
                # layers[f'Conv-{i}-1'] = nn.Conv1d(1, 10, 3)
                # layers[f'Conv-{i}-2'] = nn.Conv1d(10, 1, 3)
                out_f = n_actions
            layers[f'Linear-{i}'] = nn.Linear(in_f, out_f)
            in_f = out_f        
            layers[f'ReLU-{i}'] = nn.ReLU()

        self.net = nn.Sequential(layers)


    def forward(self, x): 
        return self.net(x)


class Dqn_Agent: 
    def __init__(self, env, start_model=None): 
        self.env = env 
        self.test_env = Env2048(env.n)
        self.epsilon = 1
        self.loss_fn = nn.MSELoss()
        self.writer = SummaryWriter(comment=f'_dqn{env.n}X{env.n}_')
        self.device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu' # on my machine cuda caused it to run slower :(
        if not start_model:
            self.net = Net(env.n, env.action_space.n).to(self.device)
        else:
            self.net = torch.load(start_model).to(self.device)
            self.net.eval()
            print(f'Starting from {start_model}')
        print(f'Using net:\n{self.net}')
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=LR, weight_decay=1e-4) 
        print(f'Running on {self.device}.')


    def choose_action(self, s): 
        if random.random() < self.epsilon:  # explore
            a = self.env.action_space.sample()
        else:                               # exploit   
            s = torch.tensor(s, dtype=torch.float32).to(self.device)
            action_vals = self.net(s) 
            a = torch.argmax(action_vals).item()
        if self.epsilon > MIN_EPSILON: 
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


    def wrap_as_tensors(self, batch):
        states, actions, rewards, next_states, finals = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        
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


    def eval_reward(self, episodes=3):        
        total = 0 
        for _ in range(episodes):
            g = self.play_episode(self.test_env)
            total += g 
        return total / episodes


    def train(self, n_batches=100):
        loss_sum = 0  
        for t, batch in enumerate(self.gain_experience()): 
            if t == n_batches:
                break
            states, actions, rewards, next_states, finals = self.wrap_as_tensors(batch)
            
            self.optimizer.zero_grad()    
            loss = self.calc_loss(states, actions, rewards, next_states, finals)
            loss.backward() 
            self.optimizer.step()

            loss_sum += loss.item()
            if t % DISPLAY_PERIOD == 0:
                avg_loss = loss_sum / DISPLAY_PERIOD
                loss_sum = 0 
                avg_rwd = self.eval_reward(3)
                print(t, f'loss={avg_loss:.2f}  reward={avg_rwd:.2f}')
                self.writer.add_scalar('loss', avg_loss, t)
                self.writer.add_scalar('reward', avg_rwd, t)

        model_filename = f'dqn-net-{datetime.datetime.now()}.pt'
        # torch.save(self.net.state_dict(), model_filename)
        torch.save(self.net, model_filename)


    def play_episode(self, env): 
        total = 0 
        s = env.reset().flatten()
        for _ in range(EPISODES_FOR_EVAL): 
            s = torch.tensor(s, dtype=torch.float32).to(self.device)    
            action_vals = self.net(s) 
            a = torch.argmax(action_vals).item()
            s1, r, d, _ = env.step(a)
            total += r 
            if d: 
                break 
            s = s1.flatten() 
        return total 


def show_model_performance(env, filename): 
    # net = Net(env.n, env.action_space.n)
    # net.load_state_dict(torch.load(filename))
    net = torch.load(filename)
    net.eval()

    s = env.reset().flatten()
    while True: 
        env.render() 
        s = torch.tensor(s, dtype= torch.float32)
        a = torch.argmax(net(s)).item() 
        s1, r, d, _ = env.step(a)
        if d:
            break 
        s = s1.flatten() 
        time.sleep(1)


if __name__ == '__main__': 
    env = Env2048(4)
    agent = Dqn_Agent(env)
    # agent = Dqn_Agent(env, start_model='dqn-net-2022-11-27 00:30:52.012667.pt')
    # agent.train(500)
    agent.train(60_000)
    
    # show_model_performance(env, 'dqn-net-2022-11-26 23:29:18.840492.pt')

    # net = Net(4,4)
    # print(net)
