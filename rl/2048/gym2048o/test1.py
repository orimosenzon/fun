#!/usr/bin/python3 

from env_2048 import Env2048
from random_agent import RandomAgent

def execute_one_episode(agent):
    env = Env2048(4)
    env.reset()  
    agent.reset() 

    while True:
        obs, r, done, total_reward = agent.act(env)
        if done: 
            return total_reward
        

def evaluate_total_expectation(agent, N): 
    exp = execute_one_episode(agent) 
    for n in range(2, N+1):
        total = execute_one_episode(agent)
        exp = (exp * (n-1) + total)/ n 
    return exp 

if __name__ == '__main__': 
    ra = RandomAgent()

    print(evaluate_total_expectation(ra, 500))
    
    # for _ in range(4):
    #     print(execute_one_episode(ra))