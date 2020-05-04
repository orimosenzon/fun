#!/usr/bin/python3

from sympy import isprime 

import tkinter

WIDTH = 1366
HEIGHT = 730

tk = tkinter.Tk()
tk.title('Primes')

canvas = tkinter.Canvas(tk, width=WIDTH, height=HEIGHT, background="white")
canvas.focus_set()
g_edges = 62
g_size = 20 


def spiral(size=20, edges=10):
    dirs = [(1,0), (0,1), (-1,0), (0,-1)]

    x, y = WIDTH//2 - size//2 , HEIGHT//2 - size//2   
    ix = 0 
    n = 2
    edge_length = 1 
    for ed in range(edges):
        dx, dy = dirs[ix]
        ix = (ix+1)%4
        for k in range(edge_length):
            color = 'red' if isprime(n) else ''
            outline = 'black' if size > 5 else '' 
            canvas.create_rectangle(x, y, x+size, y+size, outline=outline, fill=color)
            if n < 100 and size > 15:
                canvas.create_text(x + size//2, y + size//2, text=str(n))
            x += dx * size
            y += dy * size 
            n += 1
        if ed % 2 == 1:
            edge_length += 1 




def left_pressed(event):
    resize(0.8)


def right_pressed(event):
    resize(1.25)

    
def resize(factor):
    global g_edges, g_size
    g_edges = int(g_edges * factor)
    if g_size > 2:
        g_size = g_size / factor 
    canvas.delete('all')
    spiral(g_size, g_edges)
    
canvas.bind('<Left>', left_pressed)
canvas.bind('<Right>', right_pressed)

canvas.pack()

if __name__ == '__main__':
    spiral(g_size, g_edges)
    tk.mainloop()
    
