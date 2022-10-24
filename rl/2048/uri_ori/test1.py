#!/usr/bin/python3 

from env_2048 import Env2048
from random_agent import RandomAgent

def experiment():
    agent = RandomAgent()
    
    env = Env2048(4)
    env.reset()  
    
    c = 0 

    while not env.is_done():
        print(env.get_valid_actions())
        env.render()
        action = agent.step(env)
        print(f'action: {env.action2str[action]}')

    env.render()
    
    return agent.total_reward, env.score


if __name__ == '__main__': 
    res = []
    for _ in range(10):
        res.append(experiment())
    print('\n' * 4, res)