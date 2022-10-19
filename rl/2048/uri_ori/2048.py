import random 

import numpy as np


class Board:

    def __init__(self, n):
        self.n = n
        self.clear()
    

    def clear(self):
        self.brd = np.zeros((self.n, self.n), dtype=np.int32)


    def initialize(self):
        for _ in range(2):
            self.place_new_entry()


    char2dir = {
        'r': (0, 1),  
        'l': (0, -1),  
        'd': (1, 0), 
        'u': (-1, 0), 
    }


    def _next_item(self, i, j, dim, delta):
        j += delta 


        return j 
        


    def move(self, dir_char):
        dm, dn = self.char2dir[dir_char]

        if dn != 0:  #  Running on rows (r or l)
            dim = 0
            delta = dn
        else:
            dim = 1
            delta = dm

        if delta == 1: 
            first = 0
        else: 
            first = self.n - 1

        for i in range(self.n):
            s = first
            while True:
                j1 = self._next_item(i, s, dim, delta) 
                j2 = self._next_item(i, j1, dim, delta)
                val = self._loc(i, j1, dim)
                self._set_loc(i, j1, dim, 0)

                if val == self.loc(i, j2, dim):   #merge case                    
                    self._set_loc(i, j2, dim, 0)
                    self._set_loc(i, s, dim, 2 * val)

                    j1 = self._next_item(i, j2, dim, delta)
                    if j1 == -1: 
                        break 
                    j2 = self._next_item(i, j1, dim, delta)
                    if j2 == -1:
                        break
                else: 
                    self._set_loc(i, s, dim, val)
                    j1 = j2 
                    j2 = self._next_item(i, j1, dim, delta)
                    if j2 == -1:
                        break
                s += delta         

    

    def uri_move(self, dir_char):
            dm, dn = self.char2dir[dir_char]

            if dn != 0:  #  Running on rows (r or l)
                dim = 0
                delta = dn
            else:
                dim = 1
                delta = dm
            o_dim = 1 - dim

            loc = (-1, -1)
            ref = (-1, -1)
            for i in range(self.n):                
                loc[dim] = ref[dim] = i

                if delta == 1:                # 1
                    j1, j2 = 0, self.n-1
                else:                         # -1  
                    j1, j2 = self.n-1, 0         

                #find initial place for ref 
                for j in range(j1, j2, delta):
                    loc[o_dim] = j
                    if ref = (0,0) && self.brd[i, j] > 0:
                        ref = (i, j)
                    else:
                        if self.brd[i, j] > 0:
                            loc = (i, j)
                            break
                if loc != (-1,-1) && self.brd[ref] == self.brd[loc]:
                    self.brd[loc] *= 2
                    self.brd[ref] = 0
                if loc != (-1,-1) && self.brd[ref] != self.brd[loc] && np.abs(ref[o_dim] - loc[o_dim]) > 1:
                    loc1 = loc
                    loc1[o_dim] += delta
                    self.brd[loc] = self.brd[ref]
                    self.brd[ref] = 0
                if loc == (-1,-1) && ref != (-1,-1) && ref[o_dim] != j2:
                    self.brd[loc] = self.brd[ref]
                    self.brd[ref] = 0
            
            
    def print(self):
        bar = '+----' * self.n + '+'
        for i in range(self.n):
            print(bar)
            for j in range(self.n):
                print(f'|{self.brd[i, j]:4}', end='')
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
            
            self.move(char)

            if not self.place_new_entry():
                print('You have lost')
                break
            
            self.print()


if __name__ == '__main__':
    board = Board(4)
    board.initialize()
    board.play() 
