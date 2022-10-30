import random 

class RandomAgent: 
    def __init__(self):
        self.total_reward = 0 


    def act(self, env): 
        action = env.action_space.sample()
        obs, r, done, _ = env.step(action)
        self.total_reward += r
        return obs, r, done, self.total_reward


    def reset(self): 
        self.total_reward = 0  