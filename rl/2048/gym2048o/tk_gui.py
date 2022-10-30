#!/usr/bin/python3 

import math 

import tkinter as tk

from env_2048 import Env2048

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

def rgb_color(rgb):
    return f'#%02x%02x%02x' % rgb

def draw_board(brd): 

    size = min(width, height) * SIZE_F
    s = size // board.n
    offset_x = (width - size) // 2 
    offset_y = (height - size) // 2 
    n_i, n_j = board.new_entry

    canvas.create_text(offset_x + size, offset_y //2, text=f'score: {board.score:,}')
    for i in range(board.n):
         for j in range(board.n):
            x, y = offset_x + j * s, offset_y + i *s                     
            val = brd[i, j]            
            if val != 0: 
                lg = int(math.log(val) / math.log(2))  
                rgb = log2rgb.get(lg, (255, 255, 255))              
                f_color = rgb_color(rgb)
                if (i, j) == (n_i, n_j):
                    l_width = 2
                else:
                    l_width = 1 
                canvas.create_rectangle(x, y, x+s, y+s, fill=f_color, width=l_width)    
                canvas.create_text(x+s//2, y+s//2, text=str(val))
            else: 
                canvas.create_rectangle(x, y, x+s, y+s, width=1)    


def key_press(e):
    if e.keysym not in key2char.keys():
        return 
    a = key2char[e.keysym]
    if a not in board.get_valid_actions():
        print('Invalid action')
        return 
    observation, reward, done, _ = board.step(a)
    canvas.delete('all')
    draw_board(observation)
    if done:
        print('Game over')


def resize(e):
    global width, height
    width, height = canvas.winfo_width(), canvas.winfo_height()
    canvas.delete('all')
    draw_board(board.brd)


key2char = {
    'Right': 2, 
    'Left':  0, 
    'Up':    3, 
    'Down':  1, 
}

root = tk.Tk()

WIDTH, HEIGHT = 650, 500 
SIZE_F = 0.8 

root.geometry(f'{WIDTH}x{HEIGHT}') # 

canvas = tk.Canvas(background='white')
canvas.pack(expand=True, fill=tk.BOTH)

canvas.bind('<KeyPress>', key_press)
canvas.bind("<Configure>", resize)
canvas.focus_set()

board = Env2048(4)
board.reset()

root.mainloop()
