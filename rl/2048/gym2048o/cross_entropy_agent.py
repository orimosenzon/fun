import time
from collections import namedtuple

import numpy as np 
import torch 
from torch import nn 

from torch.utils.tensorboard import SummaryWriter

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
            nn.Linear(hidden_size, 7 * hidden_size),
            nn.ReLU(),
            nn.Linear(7 * hidden_size, n_actions)
        )


    def forward(self, x):
        return self.net(x)


class CrossEntropyAgent: 
    HIDDEN_SIZE = 128
    BATCH_SIZE = 160
    PERCENTILE = 10 #70
    def __init__(self, env):
        self.env = env
        self.obs_size = env.n ** 2 
        self.net = Net(self.obs_size, self.HIDDEN_SIZE, 4) 
        self.episodes = [] 
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()
        self.sm = nn.Softmax() 
        self.writer = SummaryWriter(comment='Cross Entropy Agent')


    def _avrg_reward(self):
        sum = 0  
        for episode in self.episodes:
            sum += episode.reward 
        return sum / len(self.episodes)


    def _gain_experiance(self):
        self.episodes = [] 
        for _ in range(self.BATCH_SIZE):
            obs = self.env.reset() 
            total = 0 
            steps = []  
            while True: 
                inp = torch.Tensor(obs).view(self.obs_size)
                pred = self.net(inp)
                # action = int(torch.argmax(pred))
                act_prob = self.sm(pred).data.numpy()
                action = np.random.choice(len(act_prob), p=act_prob)
                
                n_obs, r, done, _ = self.env.step(action)
                if r==-1:  # invalid action
                    action = env.action_space.sample()
                    n_obs, r, done, _ = self.env.step(action)
                    # r = 0 
                total += r 
                episode_step = EpisodeStep(obs, action)
                steps.append(episode_step)
                if done:
                    break 
                obs = n_obs

            episode = Episode(total, steps)
            self.episodes.append(episode)


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
        self.current_loss = loss.item()
        loss.backward()
        self.optimizer.step()


    def _keep_best_episodes(self): 
        self.episodes.sort(key=(lambda x: x[0]))
        idx = int(len(self.episodes)*(100-self.PERCENTILE)/100)
        self.episodes = self.episodes[idx:]


    def _display_training_values(self, loss, avarage, t):
            print(f'{t}: avrg_reward={avarage}, loss={loss}')
            self.writer.add_scalar("loss", loss, t)
            self.writer.add_scalar("avarage reward", avarage, t)


    def train(self, epochs=30): 
        prev_avrg = 0 
        c = 0 
        while True: 
            self._gain_experiance()
            self._keep_best_episodes()            
            avrg_reward = self._avrg_reward()
            # if abs(avrg_reward - prev_avrg) < .001: 
            #     break 
            c += 1 
            if c > epochs: 
                break
            prev_avrg = avrg_reward

            self._train_net()
            self._display_training_values(self.current_loss, avrg_reward, c)


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


    def demonstrate_eposide(self, episode):  
        meanings = env.get_action_meanings()
        self.env.reset()
        total = 0 
        for obs, act in episode.steps:
            env.brd = obs # hard set board (encapsulatio vaiolation)
            self.env.render() 
            time.sleep(.1)
            print(f'action: {meanings[act]}')
        print(f'episode total = {episode.reward}')


if __name__ == '__main__':
    from env_2048 import Env2048
    env = Env2048(2)
    cea = CrossEntropyAgent(env)
    cea.train(150) 
    cea.demonstrate_policy()
    cea.demonstrate_eposide(cea.episodes[-1])
    
