#!/usr/bin/python3 

from env_2048 import Env2048
from random_agent import RandomAgent

def experiment():
    agent = RandomAgent()
    
    env = Env2048(4)
    env.reset()  
    
    c = 0 

    while not env.is_done():
        print('valid: ', end='')
        for a in env.get_valid_actions(): 
            print(env.action2str[a], end=', ')
        print() 
        env.render()
        action = agent.step(env)
        

    env.render()
    
    return agent.total_reward, env.score


if __name__ == '__main__': 
    res = []
    for _ in range(10):
        res.append(experiment())
    print('\n' * 4, res)