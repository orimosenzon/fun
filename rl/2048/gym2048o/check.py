

  
from collections import namedtuple
  
# Declaring namedtuple()
Student = namedtuple('Student', ['name', 'age', 'DOB'])
s = Student('Nandini', '19', '2541997')
print(s)
print(s[0], s.name)


# == == 

# import gym 

# def run(env):
#   c = 0 
#   env.reset()   
#   while True:
#     env.render()
#     a = env.action_space.sample()
#     o, r, d, _ = env.step(a)
#     if d or c > 10: 
#       break
#     c += 1 

# def print_envs():
#   names = [x.id for x in gym.envs.registry.all()]
#   print(names)

# if __name__ == '__main__': 
#   names = ['CartPole-v0', 'CartPole-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v1', 'Acrobot-v1', 'LunarLander-v2', 'LunarLanderContinuous-v2', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'CarRacing-v1', 'CarRacingDomainRandomize-v1', 'CarRacingDiscrete-v1', 'CarRacingDomainRandomizeDiscrete-v1', 'Blackjack-v1', 'FrozenLake-v1', 'FrozenLake8x8-v1', 'CliffWalking-v0', 'Taxi-v3', 'Reacher-v2', 'Reacher-v4', 'Pusher-v2', 'Pusher-v4', 'InvertedPendulum-v2', 'InvertedPendulum-v4', 'InvertedDoublePendulum-v2', 'InvertedDoublePendulum-v4', 'HalfCheetah-v2', 'HalfCheetah-v3', 'HalfCheetah-v4', 'Hopper-v2', 'Hopper-v3', 'Hopper-v4', 'Swimmer-v2', 'Swimmer-v3', 'Swimmer-v4', 'Walker2d-v2', 'Walker2d-v3', 'Walker2d-v4', 'Ant-v2', 'Ant-v3', 'Ant-v4', 'Humanoid-v2', 'Humanoid-v3', 'Humanoid-v4', 'HumanoidStandup-v2', 'HumanoidStandup-v4']
#   for name in names: 
#     print(name)
#     try:
#       env = gym.make(name)
#       run(env)
#     except Exception as e:
#       print(e)

  