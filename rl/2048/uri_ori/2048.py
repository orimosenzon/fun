import numpy as np


class Board:
    RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3
    directions = [
        (1, 0),  # RIGHT
        (0, 1),  # DOWN,
        (-1, 0), # LEFT
        (0, -1), # UP
    ]


    def __init__(self, n):
        brd = np.zeros((n, n))
    

    def move(self, dir_num):
        dx, dy = self.directions[dir_num]
        if dx != 0:
            dim = 0
        else:
            dim = 1
        

