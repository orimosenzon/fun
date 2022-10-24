#!/usr/bin/python3

import random 
import numpy as np

class Board:

    def __init__(self, n):
        assert n > 1, 'a too smaller board'
        self.n = n
    

    def clear(self):
        self.score = 0 
        self.brd = np.zeros((self.n, self.n), dtype=np.int32)


    def reset(self):
        self.clear()
        for _ in range(2):
            self._place_new_entry()


    def is_done(self):
        return not self.get_actions()


    char2dir = {
        'r': (0, 1),  
        'l': (0, -1),  
        'd': (1, 0), 
        'u': (-1, 0), 
    }

    all_actions = list(char2dir.keys())

    def _loc(self, i, j, dim):
        loc = [0, 0]
        loc[dim] = i 
        loc[1-dim] = j 
        try:
            return self.brd.item(*loc)
        except Exception as e:
            print(f'{e}: i={i}, j={j}, dim={dim}')
            raise e


    def _set_loc(self, i, j, dim, val): 
        loc = [0, 0]
        loc[dim] = i 
        loc[1-dim] = j 
        self.brd[tuple(loc)] = val


    def _next_item(self, i, j, dim, delta):
        while True: 
            j += delta 
            if j == self.n or j == -1:
                return -1  
            if self._loc(i, j, dim) != 0: 
                return j 


    def _is_valid_action(self, dir_char):
        dm, dn = self.char2dir[dir_char]
        
        if dn != 0:  #  Running on rows (r or l)
            dim = 0
            delta = dn
        else:        #  Running on columns (u or d)
            dim = 1
            delta = dm

        if delta == 1:
            first = 0
            last =  self.n - 1
        else:
            first = self.n - 1
            last = 0

        for i in range(self.n):
            j = self._next_item(i, first-delta, dim, delta) 
            if j == -1:      # empty case 
                continue
            while j != last:
                val = self._loc(i, j+delta, dim)
                if val == 0 or val == self._loc(i, j, dim):
                    return True
                j += delta 
        return False


    def get_actions(self):
        ret = [] 
        for a in self.all_actions:
            if self._is_valid_action(a):
                ret.append(a)
        return ret 


    def _move(self, dir_char):
        dm, dn = self.char2dir[dir_char]

        if dn != 0:  #  Running on rows (r or l)
            dim = 0
            delta = -dn
        else:        #  Running on columns (u or d)
            dim = 1
            delta = -dm

        first = 0 if delta == 1 else self.n - 1 

        for i in range(self.n):
            s = first
            j1 = self._next_item(i, s-delta, dim, delta) 
            while j1 != -1:
                val = self._loc(i, j1, dim)
                self._set_loc(i, j1, dim, 0)
                
                j2 = self._next_item(i, j1, dim, delta)
                
                if j2 == -1: 
                    self._set_loc(i, s, dim, val) # slide 
                    break
    
                if val == self._loc(i, j2, dim):   # merge case   
                    self._set_loc(i, j2, dim, 0)
                    self._set_loc(i, s, dim, 2 * val)
                    self.score += 2 * val
                    j1 = self._next_item(i, j2, dim, delta)
                else:                             # slide case
                    self._set_loc(i, s, dim, val)
                    j1 = j2

                s += delta

            
    def print(self):
        bar = '+----' * self.n + '+'
        for i in range(self.n):
            print(bar)
            for j in range(self.n):
                print(f'|{self.brd[i, j]:4}', end='')
            print('|')
        print(bar)


    def _get_random_empty_loc(self): 
        N = self.n * self.n
        indices = list(range(N))
        # random.shuffle(indices)
        last_i = N-1
        while last_i >= 0: 
            i = random.randint(0, last_i)
            idx = indices[i]
            loc = (idx // self.n, idx % self.n)
            if self.brd[loc] == 0:
                return loc 
            indices[i] = indices[last_i]
            last_i -= 1 

        raise Exception('problem')


    def _place_new_entry(self):
        loc = self._get_random_empty_loc()
        self.new_entry = loc

        if random.random() < .1: 
            self.brd[loc] = 4
        else:
            self.brd[loc] = 2        


    def step(self, action):
        self._move(action)
        self._place_new_entry()


    def play_ui(self): 
        while True: 
            actions = self.get_actions()
            if not actions:
                print('Game over.')
                break 
            print(f'valid actions: {actions}')
            self.print()
            char = input('? (q to quit) ')
            
            if char == 'q':
                print('You have decided to quit')
                break
            
            if char not in self.char2dir.keys():
                print(f'{char} is not a valid option')
                continue  

            if not char in actions:
                print('Invalid action')
            else:
                self.step(char)



if __name__ == '__main__':
    board = Board(3)
    board.reset()
    board.play_ui()

