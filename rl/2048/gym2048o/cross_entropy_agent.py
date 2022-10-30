
from collections import namedtuple

from torch import NN as nn 

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
