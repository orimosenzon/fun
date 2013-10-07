
import random 
import tkinter
W = 700
H = 700 

canvas = tkinter.Canvas(width= W, height=H, background="white")
canvas.pack()


def inter(seg1,seg2):
    (s1,s2) = seg1
    (t1,t2) = seg2
    return (max(s1,t1),min(s2,t2))

def rectInt(rect1,rect2):
    (x1,y1,x2,y2) = rect1
    (xx1,yy1,xx2,yy2) = rect2  
    (rx1,rx2) = inter((x1,x2),(xx1,xx2))
    (ry1,ry2) = inter((y1,y2),(yy1,yy2))
    if rx2 < rx1 or ry2 < ry1:
        return None
    return (rx1,ry1,rx2,ry2)

def randRect():
    x1 = int(random.random()*W)
    x2 = x1 + int(random.random()*(W-x1))
    y1 = int(random.random()*H) 
    y2 = y1 + int(random.random()*(H-y1))
    return (x1,y1,x2,y2)

def butt1Pressed(event):
    canvas.delete("all")
    (x1,y1,x2,y2) = rect1 = randRect()
    (xx1,yy1,xx2,yy2) = rect2 = randRect()
    canvas.create_rectangle(x1, y1, x2, y2)
    canvas.create_rectangle(xx1, yy1, xx2, yy2)
    rec = rectInt(rect1,rect2)
    if rec:
        (x1,y1,x2,y2) = rec
        canvas.create_rectangle(x1, y1, x2, y2 , fill = "red")

canvas.bind('<Button-1>', butt1Pressed)
                            
