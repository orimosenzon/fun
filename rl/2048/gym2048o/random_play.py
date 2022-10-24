#!/usr/bin/python3 

import random

from board import Board 


def random_play(): 
    board = Board(4)
    board.reset()
    board.print()
 
    c = 0 

    while True:
        for i in range(10): 
            actions = board.get_actions()
            if not actions:
                print('Game over.')
                return
            a = random.choice(actions)
            print(f'action={a}')
            board.step(a)

            print(f'{c}:')
            c += 1 
            board.print()

        cont = input('Continue?')
        if cont == 'n': 
            break


if __name__ == '__main__':
    random_play()
