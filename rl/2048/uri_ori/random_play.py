#!/usr/bin/python3 

import random

from board import Board 


def random_play(): 
    board = Board(5)
    board.reset()
    board.print()
 
    c = 0 
    actions = board.actions 

    while True:
        for i in range(10): 
            action = random.choice(actions)
            print(f'action={action}')
            is_done = board.step(action)
            if is_done:
                print('Done.')
                return False

            print(f'{c}:')
            c += 1 
            board.print()

        cont = input('Continue?')
        if cont == 'n': 
            break


if __name__ == '__main__':
    random_play()
