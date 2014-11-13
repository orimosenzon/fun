WIDTH = 1600
HEIGHT = 1200
size = 5
W2 = WIDTH // 2
H2 = HEIGHT // 2

import tkinter
canvas = tkinter.Canvas(width=WIDTH, height=HEIGHT, background="white")
canvas.pack()

NUM = 100000
prime = [False]* NUM

def calcPrimes():
    global prime 
    array = [True]*(NUM+1) 
    i = 2
    while i <= NUM:
        if array[i]:
            prime[i]=True
            for j in range(i*i,NUM+1,i):
                array[j] = False
        i+=1

seg = 2
def butt1Pressed(e):
    global seg 
    canvas.delete("all")
    drawNum(seg)
    seg+=1

canvas.bind('<Button-1>', butt1Pressed)

dirs =[(1,0),(0,1),(-1,0),(0,-1)]
def drawNum(n):
    x,y = W2-size//2,H2-size//2 
    dirsX=0
    dx,dy = dirs[dirsX]
    ns = [n-1,1]
    nsX = 0 
    i = 2
    c = 0 
    maxI = (min(WIDTH*.9,HEIGHT*.9)//size) ** 2 
    while True:
        if i == maxI:
            return
        
        if c == ns[nsX]:
            ns[nsX] +=1
            nsX = 1-nsX
            c=0
            dirsX = (dirsX+1)%4
            dx,dy = dirs[dirsX]

        x += size*dx
        y += size*dy
        col = "" 
        if prime[i]:
            col = "red"
        canvas.create_rectangle(x,y,x+size,y+size,fill=col)
        #canvas.create_text(x+size//2,y+size//2,text=str(i))
            
        i+=1
        c+=1
        #print(i)

calcPrimes()
