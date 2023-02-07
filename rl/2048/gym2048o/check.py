def pair_sum(nums, target): 
    nums.sort()
    i1, i2 = 0, len(nums)-1
    while i1 < i2: 
        s = nums[i1] + nums[i2]
        if s == target:
            return (nums[i1], nums[i2])
        if s < target:
            i1 +=1 
        else: 
            i2 -=1

    return False 


print(pair_sum([2, 11, 28, -3, 4, 15], -1)) 

# def min_win(s, t):
#     N = len(s)
#     for i in range(N): 
#         for j in range(i+1, N+1):
#             print(t, s[i:j])
#             if t in s[i:j]:
#                 return s[i:j]


# print(min_win("ADOBECODEBANC", "ABC"))


# def flatten(lst):
#     if type(lst) == int:
#         return lst
#     ret = []
#     for l in lst: 
#         ret.extend(flatten(l))
#     return ret 


# def sort_by_length(strings):
#     return sorted(strings, key=lambda x: (len(x), x))





# def pair_sum(nums, t): 
#     nums.sort()
#     N = len(nums)
#     for i in range(N): 
#         for j in range(i+1, N): 
#             s = nums[i] + nums[j] 
#             if s == t:
#                 return [i,j]
#             if s > t:
#                 continue


# def pair_sum(nums, target): 
#     N = len(nums)
#     for i in range(N):
#         for j in range(i+1, N):
#             if nums[i] + nums[j] == target:
#                 print(f'nums[{i}] + nums[{j}] = {nums[i]} + {nums[i]} = {target}')

# def longest_prefix(strings):
#     prefix = '' 
#     i = 0 
#     while True: 
#         for s in strings:
#             if i == len(s) or s[i] != strings[0][i]:
#                 return prefix 
#         prefix += strings[0][i]
#         i+=1

# print(longest_prefix(["flower","flow","flight"]))
# print(longest_prefix(["dog","racecar","car"]))

# class A:

#     # @staticmethod
#     def foo():
#         print('foo')

#     def foo1(self):
#         self.foo()

# a = A()
# a.foo1() 

# class A:
#     a = 15 
#     class B:
#         b = 17 
#         def __init__(self): 
#             print(f'A.B.__init__:{A.a}, {self.b}')

#     def __init__(self):
#         b = self.B()


# a = A()
# # b = A.B()


# import random

# def av(lst): 
#     a = sum(lst)/len(lst)
#     b = lst[0]
#     i = 1 
#     for r in lst[1:]:
#         i += 1 
#         b = 1/i * r + (i-1)/i * b
#     return a, b 

# for i in range(7): 
#     lst = [random.randint(0,100) for _ in range(10)]
#     a, b = av(lst)
#     print(a, b, lst)

# import gym
# env = gym.make("CartPole-v0")
# obs = env.reset()
# print(obs, type(obs))  

# from collections import namedtuple
  
# # Declaring namedtuple()
# Student = namedtuple('Student', ['name', 'age', 'DOB'])
# s = Student('Nandini', '19', '2541997')
# print(s)
# print(s[0], s.name)


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

  