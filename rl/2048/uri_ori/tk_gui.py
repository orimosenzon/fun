#!/usr/bin/python3 

import math 

import tkinter as tk

from board import Board

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

def draw_board(): 
    brd = board.brd
    s = SIZE // board.n
    offset = (WIDTH - SIZE) // 2 
    for i in range(board.n):
         for j in range(board.n):
            x, y = offset + j * s, offset + i *s                     
            val = brd[i, j]            
            if val != 0: 
                lg = int(math.log(val) / math.log(2))  
                rgb = log2rgb.get(lg, (255, 255, 255))              
                color = rgb_color(rgb)
                canvas.create_rectangle(x, y, x+s, y+s, fill=color, width=1)    
                canvas.create_text(x+s//2, y+s//2, text=str(val))
            else: 
                canvas.create_rectangle(x, y, x+s, y+s, width=1)    


board = Board(4)
board.reset()

def key_press(e):
    if e.keysym not in key2char.keys():
        return 
    a = key2char[e.keysym]
    if a not in board.get_actions():
        print('Invalid action')
        return 
    board.step(a)
    canvas.delete('all')
    draw_board()
    if board.is_done():
        print('Game over')


key2char = {
    'Right': 'r', 
    'Left':  'l', 
    'Up':    'u', 
    'Down':  'd', 
}

root = tk.Tk()

# WIDTH, HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()
WIDTH, HEIGHT = 500, 500 
SIZE = 400

root.geometry(f'{WIDTH}x{HEIGHT}') # 

canvas = tk.Canvas(background='white')
canvas.pack(expand=True, fill=tk.BOTH)


canvas.bind('<KeyPress>', key_press)
canvas.focus_set()

offset = (WIDTH - SIZE) // 2 

draw_board()

root.mainloop()
