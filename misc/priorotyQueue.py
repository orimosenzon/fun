WIDTH = 1000
HEIGHT = 800
import time
import random 
import tkinter
canvas = tkinter.Canvas(width=WIDTH, height=HEIGHT, background="white")
canvas.pack()

size = 40 

heap = []

def heapDown():
    global heap
    i = 0
    v = heap[0]
    N = len(heap)-1
    while True:
        i2 = i*2+1
        if i2 > N:
            break
        if i2+1 <= N and heap[i2+1] < heap[i2]:
            i2 += 1
        if v[0] > heap[i2][0]:
            heap[i] = heap[i2]
            i = i2
        else:
            break
        
    heap[i] = v

def heapUp():
    global heap
    i = len(heap)-1
    v = heap[i]
    while i>0:
        i2 = (i-1) // 2
        if heap[i2][0] < v[0]:
            break
        heap[i] = heap[i2]
        i = i2
    heap[i] = v 
    
def heapRF():
    global heap
    r = heap[0]
    heap[0] = heap[-1]
    del heap[-1]
    if heap != []:
        heapDown()
    return r

def heapINS(n):
    global heap
    heap += [n]
    heapUp()

##for i in range(100):
##    heapINS( (random.random()*100,'a'))
##for i in range(100):
##    print(heapRF())
    
    

def init_balls():
    global dirs,balls
    for i in range(100):
        x,y = int(random.random()*(WIDTH-size)), int(random.random()*(HEIGHT-size))
        ball = canvas.create_oval(x,y,x+size,y+size,fill="red")
        dr = [int(random.random()*5)-2, int(random.random()*2)*2-1]
        speed = int(random.random()*100)+1
        heapINS( (0,(ball,dr,speed)) )

def butt1Pressed(e):
    global dirs
    while True:
        (t,(ball,dr,speed)) = heapRF()
        [x,y,_,_]= canvas.coords(ball)
        if x>WIDTH-size or x < 0:
            dr[0] = -dr[0]
        if y<0 or y>HEIGHT-size:
            dr[1] = -dr[1]
        canvas.move(ball,dr[0],dr[1])
        canvas.update()
        t += 1/speed
        heapINS( (t,(ball,dr,speed)) )
        #time.sleep(0.005)

    
canvas.bind('<Button-1>', butt1Pressed)

init_balls()

