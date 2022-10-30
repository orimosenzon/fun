#!/usr/bin/python3

# currently doesn't work. I get the error: 
# pygame.error: Unable to make GL context current

import time 

from pynput import keyboard
from pynput.keyboard import Key

from env_2048 import Env2048

a_key_is_pressed = False 

key2char = {
    Key.up: 3, 
    Key.down: 1, 
    Key.left: 0, 
    Key.right: 2,     
}

def on_press(key):
    global a_key_is_pressed, env
    if a_key_is_pressed:
        return 

    a_key_is_pressed = True
    if key not in key2char.keys():
        return True
    print(key)
    a = key2char[key]    
    actions = env.get_valid_actions()
    if not a in actions: 
        print('Invalid move')
        return True
    _, _, done, _ = env.step(a)
    env.render()
    if done:
        print('Game over')
        return False
    return True            


def on_release(key):
    global a_key_is_pressed
    a_key_is_pressed = False


env = None

def check():
    global env  
    time.sleep(1)
    env.step(0)
    env.step(1)
    env.render() 


def play(): 
    global env 
    env = Env2048(4)
    env.reset()
    env.render()
    check()
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == '__main__':
    play() 