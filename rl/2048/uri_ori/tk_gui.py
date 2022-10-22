#!/usr/bin/python3 

import tkinter as tk

from board import Board


def draw_board(): 
    brd = board.brd
    s = SIZE // board.n
    offset = (WIDTH - SIZE) // 2 
    for i in range(board.n):
         for j in range(board.n):
            x, y = offset + j * s, offset + i *s
            canvas.create_rectangle(x, y, x+s, y+s, fill='yellow', width=1)
            val = brd[i, j]
            if val != 0: 
                canvas.create_text(x+s//2, y+s//2, text=str(val))

board = Board(4)
board.reset()

def key_press(e):
    if e.keysym not in key2char.keys():
        return 
    a = key2char[e.keysym]
    board.step(a)
    canvas.delete('all')
    draw_board()


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
