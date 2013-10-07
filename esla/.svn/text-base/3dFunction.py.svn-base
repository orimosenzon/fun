import tkinter
import math 
canvas = tkinter.Canvas(width=700,height=700,background="white")
canvas.pack()


def sign(x):
    if x < 0:
        return - 1
    return 1 

def matMul(m1,m2):
    m = len(m1)
    n = len(m2[0])
    nn = len(m2)
    M = []
    for i in range(m):
        M += [[]]
        for j in range(n):
            s = 0
            for k in range(nn):
                s += m1[i][k]*m2[k][j]
            M[i] += [s]
    return M

def rotMat(a,ax):
    m = [[math.cos(a),-math.sin(a)],
         [math.sin(a),math.cos(a)]]

    M = [[1,0,0],
         [0,1,0],
         [0,0,1]]
    ii = 0
    for i in range(3):
        if i == ax:
            continue
        jj = 0 
        for j in range(3):
            if j == ax:
                continue
            M[i][j] = m[ii][jj]
            jj += 1
        ii += 1
    return M
        

rot = matMul(rotMat(math.pi/4,0) , rotMat(math.pi/4,2))

def f(x,y):
    return math.sin(x/40)*math.sin(y/50)*70 

def trans(x,y):
    m = matMul(rot,[[x],[y],[f(x,y)]])
    return m[0][0]+cx, m[1][0] + cy

def draw():
    s = 10
    for x in range(-150,150,s):
        for y in range(-150,150,s):
            x1,y1 = trans(x,y)
            x2,y2 = trans(x+s,y)
            x3,y3 = trans(x,y+s)
            canvas.create_line(x1,y1,x2,y2)
            canvas.create_line(x1,y1,x3,y3)

def clearScreen():
    canvas.delete("all")
    

cx = 700/2
cy = 700/2
lastX =0
lastY  =0

def butt1Pressed(event):
    global on
    global lastX
    global lastY
    on = True
    lastX = event.x
    lastY = event.y

def butt1Released(event):
    global on
    on = False
    
def butt1Motion(event):
    global rot
    global lastX, lastY
    if not on:
        return

    xs = event.x - lastX 
    ys = event.y - lastY 
    
    if abs(xs) > abs(ys):
        rot = matMul(rotMat(sign(xs)*-0.1,1),rot)
    else:
        rot = matMul(rotMat(sign(ys)*-0.1,0),rot)
 
    lastX, lastY = event.x, event.y 

    clearScreen()

    draw()
    

canvas.bind("<B1-Motion>", butt1Motion)
canvas.bind('<Button-1>', butt1Pressed)
canvas.bind('<ButtonRelease-1>', butt1Released)
  
draw()


