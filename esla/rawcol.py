
#!/usr/bin/python
""" consider a 2X2 matrix:
        x1 x2
        y1 y2 
the blue vectors are the raw vectors - you can drag them on the screen 
the red vectors are the columns vectors - they move accordingly 
note that whenever the blue vectors align the red ones also align (dim cols = dim raws) 
"""
import Tkinter

from math import * 

## gui init 
print __doc__ 

root = Tkinter.Tk()

canvas = Tkinter.Canvas(root, width=400, height=200, bg='white')

canvas.pack(expand=Tkinter.YES, fill=Tkinter.BOTH)

class Int:
    n = 0 
    def __init__(self,n):
        self.n = n
    
    def __str__(self):
        return str(self.n)   
        
    
x1,x2 = Int(50),Int(30)
y1,y2 = Int(20),Int(20)

vecs = [(x1,y1),(x2,y2),(x1,x2),(y1,y2)]

selected = None 


centerX, centerY =  canvas.winfo_reqwidth()/2, canvas.winfo_reqheight()/2 


def resize(event):
    global centerX, centerY  
    centerX, centerY = event.width/2 ,  event.height/2 

    refresh()

def drawVector((x,y),color):
    x, y = x.n, y.n
    x1,y1 = cord2dis(x,y)
    canvas.create_line(cord2dis(0,0),(x1,y1), fill = color) 
    t = .2 
    lengs = 7
    norm = sqrt(x*x+y*y) 
    a,b = -x/norm * lengs, -y/norm * lengs
    x2,y2 = cos(t)*a - sin(t)*b, sin(t)*a + cos(t)*b 
    x3,y3 = cos(-t)*a - sin(-t)*b, sin(-t)*a + cos(-t)*b 

    canvas.create_line((x1, y1), (x1+x2, y1 - y2), fill = color) 
    canvas.create_line((x1, y1), (x1+x3, y1 - y3), fill = color) 

def showVects():
    drawVector(vecs[0],'blue')
    drawVector(vecs[1],'blue')
    drawVector(vecs[2],'red')
    drawVector(vecs[3],'red')

def butt1Pressed(event):
    global selected
    for i in range(3):
        (x,y) = vecs[i]
        (x,y) = (x.n,y.n)
        if dist((event.x,event.y),cord2dis(x,y)) < 4:
            selected = i
            return 

def butt1Motion(event):
    global vecs
    if selected == None:
        return
    x,y = vecs[selected]
    x.n, y.n = dis2cord(event.x,event.y) 
    refresh()

def butt1Released(event):
    global selected
    selected = None

def dist((x1,y1),(x2,y2)):
    a = x1-x2
    b = y1-y2
    return sqrt (a*a+b*b)

def cord2dis(x,y):
    return centerX + x, centerY - y 

def dis2cord(x,y):
    return -centerX + x, centerY - y  

def refresh():
    canvas.delete("all")
    showVects()

    
## main ##

canvas.bind('<Button-1>', butt1Pressed)
canvas.bind("<B1-Motion>", butt1Motion)
canvas.bind('<ButtonRelease-1>', butt1Released)
canvas.bind('<Configure>', resize)

showVects()

Tkinter.mainloop()

