from collections import namedtuple

import torch 
from torch import nn

Episode = namedtuple('Episode', field_names=['reward', 'steps'])

EpisodeStep = namedtuple(
    'EpisodeStep',field_names=['observation', 'action']
)


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )


    def forward(self, x):
        return self.net(x)


class CrossEntropyAgent: 
    HIDDEN_SIZE = 128
    BATCH_SIZE = 16
    PERCENTILE = 70
    def __init__(self, env):
        self.env = env
        self.obs_size = env.n ** 2 
        self.net = Net(self.obs_size, self.HIDDEN_SIZE, 4) 
        self.episodes = [] 
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()
        self.sm = nn.Softmax() #dim=1)


    def _avrg_reward(self):
        sum = 0  
        for episode in self.episodes:
            sum += episode.reward 
        return sum / len(self.episodes)


    def _gain_experiance(self):
        self.episodes = [] 
        for episode_n in range(self.BATCH_SIZE):
            obs = self.env.reset() 
            total = 0 
            steps = []  
            while True: 
                inp = torch.Tensor(obs).view(self.obs_size)
                pred = self.net(inp)
                action = int(torch.argmax(pred))
                # a1 = int(torch.argmax(self.sm(pred)))
                
                n_obs, r, done, _ = self.env.step(action)
                if r==0:  # invalid action
                    n_obs, r, done, _ = self.env.step(env.action_space.sample())
                total += r 
                episode_step = EpisodeStep(obs, action)
                steps.append(episode_step)
                if done:
                    break 
                obs = n_obs

            episode = Episode(total, steps)
            self.episodes.append(episode)


    def make_hot(self, action): 
        ret = torch.zeros(4)
        ret[action] = 1 
        return ret


    def _train_net(self): 
        for episode in self.episodes: 
            for e_step in episode.steps: 
                inp = torch.Tensor(e_step.observation).view(self.obs_size)
                pred = self.net(inp)
                episode_act = self.make_hot(e_step.action)
                print(pred, episode_act)
                self.optimizer.zero_grad()
                loss = self.loss_fn(pred, episode_act)
                loss.backward()
                self.optimizer.step()


    def _keep_best_episodes(self): 
        self.episodes.sort(key=(lambda x: x[0]))
        idx = int(len(self.episodes)*(100-self.PERCENTILE)/100)
        self.episodes = self.episodes[idx:]


    def train(self): 
        prev_avrg = 0 
        while True: 
            self._gain_experiance()
            self._keep_best_episodes()
            
            avrg_reward = self._avrg_reward()
            print(f'avrg_reward={avrg_reward}')
            if abs(avrg_reward - prev_avrg) < 1: 
                break 
            prev_avrg = avrg_reward

            self._train_net()


if __name__ == '__main__':
    from env_2048 import Env2048
    env = Env2048(4)
    cea = CrossEntropyAgent(env)
    cea.train() 
