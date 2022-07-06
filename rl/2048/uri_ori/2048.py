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
        self.brd = np.zeros((n, n))
        self.n = n
    

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


