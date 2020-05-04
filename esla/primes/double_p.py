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
    dirs1 = [(1,0), (0,1), (-1,0), (0,-1)]
    dirs2 = [(0,1), (1,0), (0,-1), (-1,0)]

    x1, y1 = x2, y2 = WIDTH//2 - size//2 , HEIGHT//2 - size//2   

    ix = 0
    n1 = 1
    n2 = 3
    edge_length = 1 
    for ed in range(edges):
        dx1, dy1 = dirs1[ix]
        dx2, dy2 = dirs2[ix]
        ix = (ix+1)%4
        for k in range(edge_length):
            color1 = 'red' if isprime(n1) else ''
            color2 = 'red' if isprime(n2) else ''
            outline = 'black' if size > 5 else '' 
            canvas.create_rectangle(x1, y1, x1+size, y1+size, outline=outline, fill=color1)
            canvas.create_rectangle(x2, y2, x2+size, y2+size, outline=outline, fill=color2)
            if n1 < 100 and size > 15:
                canvas.create_text(x1 + size//4, y1 + size//4, text=str(n1))
                canvas.create_text(x2 + 3*size//4, y2 + 3*size//4, text=str(n2))
            x1 += dx1 * size
            y1 += dy1 * size 
            x2 += dx2 * size
            y2 += dy2 * size 
            n1 += 4
            n2 += 4
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
    
