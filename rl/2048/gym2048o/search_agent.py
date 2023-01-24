import time 

from env_2048 import Env2048

def search(env, n):
    if n==0:
        return 0, 0 
    max_score = -100
    best_action = None 
    for action in env.get_valid_actions():
        env1 = env.clone()
        o, r, d, _ = env1.step(action)
        if d:
            score = r
        else:
            score, _ = search(env1, n-1)
            score += r 
        if score > max_score: 
            max_score = score
            best_action = action
    
    return max_score, best_action


def print_valid_actions(env):
    s = 'valid: '
    for a in env.get_valid_actions():
        s += env.action2str[a] + ' '
    print(s) 


if __name__ == '__main__':
    env = Env2048(4)
    o = env.reset()
    total_reward = 0
    d = False
    while not d: 
        env.render()
        # time.sleep(0.05)
        _, a = search(env, 4)
        o,r, d, _ = env.step(a)
        total_reward += r 
        # print_valid_actions(env)
        # print(env.action2str[a])
    print(f'{total_reward=}')
    input('press any key')

