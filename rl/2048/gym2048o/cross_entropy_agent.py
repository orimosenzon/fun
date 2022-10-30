import time
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
    PERCENTILE = 10 #70
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
                if r==-1:  # invalid action
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


    def _create_train_data(self):
        x, y = [], [] 
        for episode in self.episodes: 
            for e_step in episode.steps: 
                x.append(e_step.observation.reshape(self.obs_size))
                y.append(e_step.action)
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)
        return x, y 


    def _train_net(self): 
        x, y = self._create_train_data()
        pred = self.net(x)
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()


    def _keep_best_episodes(self): 
        self.episodes.sort(key=(lambda x: x[0]))
        idx = int(len(self.episodes)*(100-self.PERCENTILE)/100)
        self.episodes = self.episodes[idx:]


    def train(self): 
        prev_avrg = 0 
        c = 0 
        while True: 
            self._gain_experiance()
            self._keep_best_episodes()
            
            avrg_reward = self._avrg_reward()
            print(f'avrg_reward={avrg_reward}')
            # if abs(avrg_reward - prev_avrg) < .001: 
            #     break 
            c += 1 
            if c > 200: 
                break
            prev_avrg = avrg_reward

            self._train_net()


    def demonstrate_policy(self):
        invalids = 0 
        obs = self.env.reset()
        while True:
            self.env.render() 
            time.sleep(0.1)
            inp = torch.Tensor(obs).view(self.obs_size)
            pred = self.net(inp)
            action = int(torch.argmax(pred))
            n_obs, r, done, _ = self.env.step(action)
            if r==-1:  # invalid action
                invalids += 1 
                print(f'invalid action! {invalids}')
                n_obs, r, done, _ = self.env.step(env.action_space.sample())
            if done:
                break 
            obs = n_obs


if __name__ == '__main__':
    from env_2048 import Env2048
    env = Env2048(4)
    cea = CrossEntropyAgent(env)
    cea.train() 
    cea.demonstrate_policy()
