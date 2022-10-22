#!/usr/bin/python3 

import random

from env_2048 import Env2048 

class RandomAgent: 
    def __init__(self):
        self.total_reward = 0 
        self.counter = 0 


    def _obs2str(self, obs):
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


    def step(self, env): 
        current_obs = env.get_observation() #

        print(f'{self.counter}: total reward={self.total_reward}\n{self._obs2str(current_obs)}') 

        actions = env.get_actions()         #

        action = random.choice(actions)

        print('Playing: ' + action)

        reward = env.action(action)         #

        self.total_reward += reward
        self.counter += 1 


if __name__ == '__main__': 
    agent = RandomAgent()
    env = Env2048(4)
    env.reset() 

    for i in range(7):
        agent.step(env)
