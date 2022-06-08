
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# can_device_access_peer

print(torch.cuda.device_count()) 

# import gym
# print(gym.__version__)


