import gym_2048
import gym


if __name__ == '__main__':
  env = gym.make('2048-v0')
  env.seed(42)

  env.reset()
  env.render()

  done = False
  moves = 0
  while not done:
    action = env.np_random.choice(range(4), 1).item()
    next_state, reward, done, info = env.step(action)
    moves += 1

    print('Next Action: "{}"\n\nReward: {}'.format(
      gym_2048.Base2048Env.ACTION_STRING[action], reward))
    env.render()

  print('\nTotal Moves: {}'.format(moves))