#!/usr/bin/python3

from pynput import keyboard
from pynput.keyboard import Key

from board import Board 

a_key_is_pressed = False 

key2char = {
    Key.up: 'u', 
    Key.down: 'd', 
    Key.left: 'l', 
    Key.right: 'r',     
}

def on_press(key):
    global a_key_is_pressed, board
    try:
        if key.char == 'q':
            print('Quit.')
            return False
    except AttributeError:
        if not a_key_is_pressed:
            a_key_is_pressed = True
            if key not in key2char.keys():
                return True
            a = key2char[key]    
            actions = board.get_actions()
            if not a in actions: 
                print('Invalid move')
                return True
            board.step(a)
            print('\n')
            print(f'Valid actions: {board.get_actions()}')
            board.print()
            print('\n')


def on_release(key):
    global a_key_is_pressed
    a_key_is_pressed = False


board = None


def gui_play(): 
    global board 
    board = Board(4)
    board.reset()
    print(f'Valid actions: {board.get_actions()}')
    board.print()
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == '__main__':
    gui_play() 