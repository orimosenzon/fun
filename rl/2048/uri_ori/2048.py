import random 

import numpy as np


class Board:
    RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3
    directions = [
        (0, 1),  # RIGHT
        (1, 0),  # DOWN
        (0, -1), # LEFT
        (-1, 0), # UP
    ]


    def __init__(self, n):
        self.n = n
        self.clear()
    

    def clear(self):
        self.brd = np.zeros((self.n, self.n), dtype=np.int32)


    def initialize(self):
        for _ in range(2):
            self.place_new_entry()


    def move(self, dir_num):
        dm, dn = dir = self.directions[dir_num]
        if dm != 0:  #  Running on rows
            dim = 0
        else:
            dim = 1
        other_dim = 1 - dim
        loc = (0, 0)
        for i in range(self.n):
            loc[dim] = i
            delta = dir[other_dim] 
            if delta == 1:
                r = self.n - 1
            else:
                r = 0
            for j in range(2, self.n):
                loc[other_dim] = (-j * delta) % self.n
                next_loc = loc[dim], loc[other_dim] - dir[other_dim]
                self.board[loc] = self.board[next_loc] #.... not done ... 


    # ori's version: 
    char2dir = {
        'r': (0, 1),  
        'l': (0, -1),  
        'd': (1, 0), 
        'u': (-1, 0), 
    }


    def ori_move(self, dir_char):
        dm, dn = self.char2dir[dir_char]

        if dm != 0:  #  Running on rows
            dim = 0
            delta = dn
        else:
            dim = 1
            delta = dm
        o_dim = 1 - dim

        loc = (0, 0)
        for i in range(self.n):                
            loc[dim] = i

            if delta == 1:                # 1
                j1, j2 = 0, self.n-1
            else:                         # -1  
                j1, j2 = self.n-1, 0         
                
            for j in range(j2-delta, j1-delta, -delta):
                loc[o_dim] = j 
                cur = self.brd[loc]
                if cur == 0:
                    continue

                ref = (0, 0)
                ref[dim] = i 
                ref[o_dim] = j2 
                
                if self.brd[ref] == 0:   # slide case 
                    self.brd[ref] = cur
                elif cur == self.brd[ref]: # merge case 
                    self.brd[ref] *= 2
                
                self.brd[loc] = 0 
                j2 -= delta 
                ref[o_dim] = j2
                self.brd[ref] = cur 
    

    def print(self):
        bar = '+---' * self.n + '+'
        for i in range(self.n):
            print(bar)
            for j in range(self.n):
                print(f'|{self.brd[i, j]:2}', end='')
            print('|')
        print(bar)


    def get_random_empty_loc(self): 
        N = self.n * self.n
        indices = list(range(N))
        last_i = N-1
        while last_i > 0: 
            i = random.randint(0, last_i)
            loc = i // self.n, i % self.n
            if self.brd[loc] == 0:
                return loc 
            indices[i] = indices[last_i]
            last_i -= 1 
        return None


    def place_new_entry(self):
        loc = self.get_random_empty_loc()
        
        if not loc:
            return False
        
        if random.random() < .1: 
            self.brd[loc] = 4
        else:
            self.brd[loc] = 2
        
        return True


    def play(self): 
        while True: 
            char = input('dir? (r,l,d,u) or e for exit: ')
            
            if char == 'e':
                print('You have decided to quit')
                break
            
            self.ori_move(char)

            if not self.place_new_entry():
                print('You have lost')
                break
            
            self.print()


if __name__ == '__main__':
    board = Board(4)
    board.initialize()
    board.play() 
