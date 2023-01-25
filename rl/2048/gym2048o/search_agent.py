import time 
import collections 

import torch 
import torch.nn as nn 

from env_2048 import Env2048

class Net(nn.Module):
    def __init__(self, n, n_actions=4, n_blocks=1, internal_size_f=1):
        super().__init__()
        n2 = n * n
        internal_size = n2 * internal_size_f  
        layers = collections.OrderedDict()
        in_f, out_f = n2, internal_size
        for i in range(n_blocks):
            if i == n_blocks-1:
                out_f = n_actions
            layers[f'Linear-{i}'] = nn.Linear(in_f, out_f)
            in_f = out_f        
            layers[f'ReLU-{i}'] = nn.ReLU()

        self.net = nn.Sequential(layers)


    def forward(self, x): 
        return self.net(x)


def train(its = int(1e3)): 
    env = Env2048(4)
    o = env.reset()
    net = Net(4)

    LR = 1e-3

    optimizer = torch.optim.Adam(params=net.parameters(), lr=LR) 
    loss_fn = nn.MSELoss()

    for i in range(1, its+1):
        _, a = search(env, 4)
        _,r, d, _ = env.step(a)

        optimizer.zero_grad()
        # loss = loss_fn(preds, as).... change to batch programing. gather experiance... 
        # loss.backward() 
        optimizer.step()




def search(env, n):
    if n==0:
        return 0, 0 
    max_score = -100
    best_action = None 
    for action in env.get_valid_actions():
        env1 = env.clone()
        o, r, d, _ = env1.step(action)
        if d:
            score = r
        else:
            score, _ = search(env1, n-1)
            score += r 
        if score > max_score: 
            max_score = score
            best_action = action
    
    return max_score, best_action


def print_valid_actions(env):
    s = 'valid: '
    for a in env.get_valid_actions():
        s += env.action2str[a] + ' '
    print(s) 


def simulate(depth):
    env = Env2048(4)
    o = env.reset()
    total_reward = 0
    d = False
    while not d: 
        _, a = search(env, depth)
        o,r, d, _ = env.step(a)
        total_reward += r 
    return total_reward


def graphic_simulate(depth=4):
    env = Env2048(4)
    o = env.reset()
    total_reward = 0
    d = False
    while not d: 
        env.render()
        # time.sleep(0.05)
        # env.render_text()
        _, a = search(env, depth)
        # print(env.action2str[a])
        o,r, d, _ = env.step(a)
        total_reward += r 
    print(f'{total_reward=}')
    env.render()
    input('press any key')


if __name__ == '__main__':
    # graphic_simulate(1)

    N = 20 
    for depth in range(1, 8):
        total = 0 
        for i in range(1, N+1): 
            score = simulate(depth)
            print(f'{i=} {score=}')
            total += score
        print(f'** avarage {depth}  = {total/N} \n\n')
