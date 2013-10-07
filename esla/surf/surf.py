#!/usr/bin/python

import Tkinter

from math import * 


## gui init ## 
root = Tkinter.Tk()

canvas = Tkinter.Canvas(root, width=700, height=400, bg='white')

canvas.pack(expand=Tkinter.YES, fill=Tkinter.BOTH)


HEIGHT = canvas.winfo_reqheight()
WIDTH = canvas.winfo_reqwidth()

xc,yc = WIDTH/2, HEIGHT/2  

## domain grid ##  

ymin = -pi 
ymax = pi
ystep = pi/14
xmin = 0
xmax = 2*pi
xstep = pi/14 

## globals ## 
r = 200 

a = 0 

id = [[1,0,0],[0,1,0],[0,0,1]]

rot = id

## event handles ## 
def butt1Pressed(event):
    global rot 
    global a 
    a += 1 
    rot = [[cos(a),0,-sin(a)], [0     ,1,   0], [sin(a),0, cos(a)] ] 
    draw()

def dragHandle(event): 
	print event.x, event.y 

def resize(event):
	draw()
	
## event bindings ## 
canvas.bind('<Button-1>', butt1Pressed)
canvas.bind('<B1-Motion>', dragHandle)
canvas.bind('<Configure>', resize)

## misc ## 	
def line((x1,y1),(x2,y2)):
    canvas.create_line((x1 + xc, y1 + yc),
                       (x2 + xc, y2 + yc), fill = "black")

def mulV(m,v): 
    r = [0,0,0] 
    for i in range(3):
        t = 0
        for j in range(3):
            t += m[i][j]*v[j]
        r[i] = t
    return r 

def mulM(m1,m2):
    r = id
    for i in range(3):
        for j in range(3):
            s = 0
            for k in range(3):
                s += m1[i][k]*m2[k][j]
            r[i][j] = s 
    return r 



def draw():   
    y = ymin+ystep
    while y <= ymax:
        x = xmin+xstep
        while x <= xmax:

            [x0,y0,z0] = g(x,y)
            [x1,y1,z1] = g(x-xstep,y)
            [x2,y2,z2] = g(x,y-ystep)

            line((x0,y0),(x1,y1)) 
            line((x0,y0),(x2,y2)) 

            x += xstep
        y += ystep 

## the surface function ## 
def f(x,y):
        xd = cos(x)*cos(y)*r
        yd = sin(y)*r 
        zd = sin(x)*cos(y)*r 


        return [xd,yd,zd]

# map: surface function and rotate  
def g(x,y): 
    v = f(x,y)
    return mulV(rot,v)


## main ##

draw()

Tkinter.mainloop()


