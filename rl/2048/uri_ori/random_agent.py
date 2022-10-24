#!/usr/bin/python3 

import random

from env_2048 import Env2048 

class RandomAgent: 
    def __init__(self):
        self.total_reward = 0 

    def step(self, env): 
        action = env.action_space.sample()
        obs, reward, is_done, _ = env.step(action)        
        self.total_reward += reward
        return action 


if __name__ == '__main__': 
    agent = RandomAgent()
    
    env = Env2048(4)
    obs = env.reset() 

    for i in range(7):
        agent.step(env)
        env.render()
