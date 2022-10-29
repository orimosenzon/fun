#!/usr/bin/python3

import random 
import numpy as np
import math 

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt 
import gym

class Env2048(gym.Env):

    action2dir = {
        0: (0, -1), # left
        1: (1, 0),  # down
        2: (0, 1),  # right 
        3: (-1, 0), # up
    }

    action2str = {
        0: 'left',
        1: 'down',
        2: 'right', 
        3: 'up',
    }


    log2rgb = {
        1: (238, 228, 218), 
        2: (237, 224, 200), 
        3: (242, 177, 121), 
        4: (245, 149, 99), 
        5: (246, 124, 95), 
        6: (246, 94, 59), 
        7: (237, 207, 114),  
        8: (237, 204, 97),  
        9: (237, 200, 80),  
        10: (237, 197, 63),  
        11: (237, 194, 46), 
    }


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


    def _is_valid_action(self, action):
        dm, dn = self.action2dir[action]
        
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


    def _move(self, action):
        dm, dn = self.action2dir[action]

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


    def _clear(self):
        self.score = 0 
        self.brd = np.zeros((self.n, self.n), dtype=np.int32)


    # == interface == 

    def __init__(self, n):
        assert n > 1, 'a too smaller board'
        self.n = n
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf,
            shape=(n,n), dtype=np.int32
        )
        self.reset()


    def reset(self): #  more arguments? super().render? 
        self._clear()
        for _ in range(2):
            self._place_new_entry()
        
        return self.brd


    def get_valid_actions(self):
        ret = [] 
        for a in range(4):
            if self._is_valid_action(a):
                ret.append(a)
        return ret 


    def is_done(self):
        return not self.get_valid_actions()


    def step(self, action):
        if not action in self.get_valid_actions():
            return self.brd, 0, self.is_done(), {'error': 'Invalid action'}
        self._move(action)
        self._place_new_entry()
        return self.brd, 1, self.is_done(), {}   # observation, reward, done, info


    def __str__(self): 
        bar = '+----' * self.n + '+\n'
        ret = ''
        for i in range(self.n):
            ret += bar
            for j in range(self.n):
                ret += f'|{self.brd[i, j]:4}'
            ret += '|\n'
        ret += bar
        return ret
    
    
    def render_text(self):
        print(str(self))


    def render(self):
        width, height = 500, 500 
        size = min(width, height) * 0.8
        s = size // self.n
        offset_x = (width - size) // 2 
        offset_y = (height - size) // 2 
        n_i, n_j = self.new_entry

        white = (255, 255, 255)
        black = (0, 0, 0)
        img = Image.new('RGB', (width, height), white)
        canvas = ImageDraw.Draw(img)
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)

        canvas.text(
            (offset_x, offset_y //2), 
            text=f'score: {self.score:,}', 
            fill = black, 
            font = fnt 
        )

        for i in range(self.n):
            for j in range(self.n):
                x, y = offset_x + j * s, offset_y + i *s                     
                val = self.brd[i, j]            
                if val != 0: 
                    lg = int(math.log(val) / math.log(2))  
                    f_color = self.log2rgb.get(lg, black)
                    if (i, j) == (n_i, n_j):
                        l_width = 2
                    else:
                        l_width = 1 
                    canvas.rectangle((x, y, x+s, y+s), fill=f_color, 
                                      outline=True, width=l_width)    
                    canvas.text((x+s//2, y+s//2), text=str(val), fill=black, font=fnt)
                else: 
                    canvas.rectangle((x, y, x+s, y+s), 
                                    fill=white, outline=True, width=1)    
        # img.show() 
        return np.asarray(img).swapaxes(0, 1)

if __name__ == '__main__':
    env = Env2048(4)
    env.reset()

    for _ in range(3): 
        img = env.render()
        plt.imshow(img)
        plt.show()
        a = env.action_space.sample()
        o, r, d, _, = env.step(a)
        if d: 
            break 

