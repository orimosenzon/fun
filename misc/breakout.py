import tkinter
import time 
import random 

WIDTH = 1100
HEIGHT = 900

canvas = tkinter.Canvas(width=WIDTH, height=HEIGHT, background="white")
canvas.pack()
class Ball:
    BS = 20
    R = BS//2 
    def __init__(self):
        self.dx = 1
        self.dy = 1 
        self.ball = canvas.create_oval(100,100,100+self.BS,100+self.BS,fill="red")
    def move(self):
        x,y,_,_ = canvas.coords(self.ball)
        x1 = x + self.BS
        y1 = y + self.BS
        
        if x <0 or x1 > WIDTH:
            self.dx = - self.dx

        if (y1 >= HEIGHT-user.by and
            x1>=user.place and
            x<=user.place+user.size*user.bx) or (y<0 or y1 > HEIGHT):
            self.dy = - self.dy

        canvas.move(self.ball,self.dx,self.dy)


class User:
    size = 10
    bx,by = 20,20
    place = WIDTH//2 - size*bx//2 
    def __init__(self):
        self.rect = canvas.create_rectangle(self.place,
                                       HEIGHT-self.by,
                                       self.place+self.size*self.bx,
                                       HEIGHT,fill="orange")

    
    def goRight(self):
        if self.place+self.size*self.bx+self.bx > WIDTH:
            return
        self.place += self.bx
        canvas.move(self.rect,self.bx,0)
        
    def goLeft(self):
        if self.place - self.bx < 0:
            return
        self.place -= self.bx
        canvas.move(self.rect,-self.bx,0)

def drawBricks():
    bsx,bsy = 40,20
    bsx2 = bsx // 2 
    yOffset = 50 
    xOffset = 0 
    lines = 7
    colors = ["blue","red","green","yellow","orange"] 
    for y in range(yOffset,yOffset+lines*bsy,bsy):
        xOffset = bsx2-xOffset
        for i in range (WIDTH // bsx):
            x = xOffset + i*bsx
            c = int(random.random()*len(colors))
            color = colors[c]
            canvas.create_rectangle(x,y,x+bsx,y+bsy,fill=color,width=2)
        


def keyPressed(event):
    if event.keycode == 37:   #left arrow 
        user.goLeft()
        
    if event.keycode == 39:   #right arraow  
        user.goRight()

#    print(event.keycode)
    
canvas.bind_all('<Key>',keyPressed)

def mysleep():
    x = 0 
    for i in range(100000):
        x+=i
        
user = User()
b = Ball()

def main():
    drawBricks()

    while True:
        b.move()
        canvas.update()
        #time.sleep(.0005)
        mysleep() 
    
main() 
