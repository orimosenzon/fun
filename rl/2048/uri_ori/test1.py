#!/usr/bin/python3 

from env_2048 import Env2048
from random_agent import RandomAgent

def obs2str(obs):
    n = len(obs)
    bar = '+----' * n + '+\n'
    ret = ''
    for i in range(n):
        ret += bar
        for j in range(n):
            ret += f'|{obs[i, j]:4}'
        ret += '|\n'
    ret += bar
    return ret


def experiment():
    agent = RandomAgent()
    
    env = Env2048(4)
    obs = env.reset() 
    c = 0 
    print(c, obs2str(obs))

    while not env.is_done():
        agent.step(env)
    
    return agent.total_reward, env.score


if __name__ == '__main__': 
    res = []
    for _ in range(10):
        res.append(experiment())
    print('\n' * 4, res)