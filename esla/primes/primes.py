from math import sqrt
import math
import random 
WIDTH = 1900
HEIGHT = 1000

N = 1000000 # number of primes to find 
import tkinter
canvas = tkinter.Canvas(width=WIDTH, height=HEIGHT, background="white")
canvas.pack()


idxes = [] 
primes = [] 
    
def calcPrimes(n):
    global primes 
    array = [True]*(n+1) 
    i = 2
    while i <= n:
        if array[i]:
            primes += [i] 
            for j in range(i*i,n+1,i):
                array[j] = False
        i+=1
    
    
def calIdxes():
    global idxes
    idxes = [-1]*(primes[-1]+1)   
    for i in range(len(primes)):
        idxes[primes[i]] = i+1
        
def init():    
    calcPrimes(N)
    calIdxes()
    
def primesSmaller100():
    i = 0
    while primes[i] < 100:
        print(i+1," - ",primes[i])
        i += 1
    

def factorize(n):
    facs = [] 
    i = 0
    while True:
        p = primes[i]
        if p > n:
            break
        while n % p == 0:
            facs += [p]
            n /= p
        i += 1
    facs.sort()
    return facs



def draw(func):
    start = 2
    while True:
        for x in range(start,start+WIDTH):
            y = func(factorize(x))
            cx = x -start
            canvas.create_line(cx,HEIGHT,cx,HEIGHT-y)

        a = input("continue? (n or e to exit)")
        if a == 'n' or a == 'e':
            return
        canvas.delete("all")
        start += WIDTH
        
def mul(fcs):
    m = 1
    for p in fcs:
        m *= p
    return m

def norm(fcs):
    s = 0
    for p in fcs:
        s += p*p
    return sqrt(s)

def idxesMul(fcs):
    m = 1 
    for p in fcs:
        m *= idxes[p]
    return m 

def idxesSum(fcs):
    s = 0 
    for p in fcs:
        s += idxes[p]
    return s 

def smallerThan(fcs):
    m = 1
    for p in fcs:
        m *= p

    idx = idxes[m]
    if idx == -1:
        return 0
    else:
        return idx

def spiral():
    x0 = WIDTH/2
    y0 = HEIGHT/2
    pi2 = 2*math.pi 
    n = 70 
    for c in range(1,100):
        r = 1
        alp = 0
        i = 2
        while alp/pi2 < n:
            x = x0+math.cos(alp)*r
            y = y0+math.sin(alp)*r
            if idxes[i] != -1: # i is prime 
                canvas.create_rectangle(x,y,x,y) 
            dalp = 1/r* c  
            alp += dalp 
            r+= dalp/pi2 * 5 
            i += 1
        input("press enter to continue")
        canvas.delete("all")

def spiral1():
    x0 = WIDTH/2
    y0 = HEIGHT/2
    pi2 = 2*math.pi 
    n = 100 
    r = 1
    alp = 0
    i = 2
    while alp/pi2 < n:
        x = x0+math.cos(alp)*r
        y = y0+math.sin(alp)*r       
        d = ((len(factorize(i))-1 ) * 20 ) % 256 
        color = '#%02x%02x%02x' % (d, d, d) 
        canvas.create_rectangle(x,y,x,y,outline = color) 
        dalp = 1/r  
        alp += dalp 
        r+= dalp/pi2 
        i += 1
    print(i) 

def spiral2():
    x0 = WIDTH/2
    y0 = HEIGHT/2
    pi2 = 2*math.pi 
    n = 100 
    r = 1
    alp = 0
    i = 2
    colors = ["black","blue","red","green","orange","yellow","white"]
    lc = len(colors) 
    while alp/pi2 < n:
        x = x0+math.cos(alp)*r
        y = y0+math.sin(alp)*r       
        d = len(factorize(i))-1 
        if d>= lc:
            d = lc-1
        color = colors[d] # color by number's dimention (primes in black) 
        #color = colors[int(alp/pi2)% lc]  # each circle has its on color 
        #color = colors[int(random.random()*lc)] # random color 
        #color = colors[i%2]  % flipflop color 
        canvas.create_rectangle(x,y,x+5,y+5,outline = color,fill = color) 
        dalp = 1/r*5  
        alp += dalp 
        r+= dalp/pi2 * 5 
        i += 1
    print(i) 

def spiral3():
    x0 = WIDTH/2
    y0 = HEIGHT/2
    pi2 = 2*math.pi 
    n = 100 
    r = 1
    alp = 0
    i = 2
    while alp/pi2 < n:
        x = x0+math.cos(alp)*r
        y = y0+math.sin(alp)*r       
        if idxes[i] != -1: 
            canvas.create_rectangle(x,y,x+5,y+5,fill="black") 
        dalp = 1/r*5  
        alp += dalp 
        r+= dalp/pi2 * 5 
        i += 1
    print(i) 

X0 = WIDTH/2
Y0 = HEIGHT/2

def squareSpir():
    x = X0
    y = Y0 
    ds = [0,1,0,-1]
    idx = 1
    idy = 0
    r = 1
    i = 2
    ic = 0 
    colors = ["blue","red","green","orange","yellow"]
    flipflop = False
    while r < 700:
        color = colors[ic]
        ic = (ic +1) % len(colors) 
        for c in range(r):
            x += ds[idx]
            y += ds[idy] 
            #if random.random()<0.1:
            if idxes[i] != -1:
                canvas.create_rectangle(x,y,x,y,outline = "black")
            #canvas.create_rectangle(x,y,x,y,outline = color)
            i+=1 
        idx = (idx+1)% 4
        idy = (idy+1)% 4
        if flipflop:
            r += 1
        flipflop = not flipflop
        #input("cont?")

def pauseAndDelete():
        input("Press enter")
        canvas.delete("all")

class Editor:
    SIZE = 20
    curI = 2        
    stack = [] 
    def __init__(self):
        canvas.bind('<Button-1>', self.butt1Pressed)
        canvas.bind('<Button-3>', self.butt3Pressed)


    def butt1Pressed(self,e):
        x = (e.x // self.SIZE) * self.SIZE 
        y = (e.y // self.SIZE) * self.SIZE
        self.stack += [(x,y)] 
        if idxes[self.curI] == -1:
            color = "yellow"
        else:
            color = "red"
        canvas.create_rectangle(x,y,x+SIZE,y+SIZE,fill = color) 
        canvas.create_text(x+SIZE//2,y+SIZE//2,text = str(self.curI))
        self.curI += 1

    def butt3Pressed(self,e):
        x,y = self.stack[-1]
        canvas.create_rectangle(x,y,x+SIZE,y+SIZE,fill = "white") 
        del(self.stack[-1])
        self.curI -= 1 
 
        
    def drawGrid(self):
        for x in range(0,WIDTH,self.SIZE):
            canvas.create_line(x,0,x,HEIGHT)
        for y in range(0,HEIGHT,self.SIZE):
            canvas.create_line(0,y,WIDTH,y)
        
def squareSpir1():
    size = 2 
    x = 20*size
    y = 20*size
    ds = [0,1,0,-1]
    idx = 1
    idy = 0
    r = 1000//size
    i = 2
    ic = 0 
    colors = ["blue","red","green","orange","yellow"]
    flipflop = False
    while r > 0:
        color = colors[ic]
        ic = (ic +1) % len(colors) 
        for c in range(r):
            x += ds[idx]*size
            y += ds[idy]*size 
            #if random.random()<0.1:
            if idxes[i] != -1:
                canvas.create_rectangle(x,y,x+size,y+size,outline = "black",fill = "blue")
            #canvas.create_rectangle(x,y,x,y,outline = color)
            i+=1 
        idx = (idx+1)% 4
        idy = (idy+1)% 4
        if flipflop:
            r -= 1
        flipflop = not flipflop
        #input("cont?")


def diagonals():
    size = 20 
    i = 2
    d = 1
    Y0 = X0 = 20 
    while d<200:
            x = size*d + X0
            y = Y0 
            for k in range(d):
                color = "white"
                if idxes[i] != -1:
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline="black",fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))

                x -= size
                y += size
                i += 1 
            d += 1 
def nuns():
    size = 20 
    i = 2
    d = 1
    Y0 = X0 = 20 
    while d<70:
            x = size*d + X0
            y = Y0 
            for k in range(d):
                color = "white"
                if idxes[i] != -1:
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline="black",fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))
                y += size
                i += 1
            y -= size
            x -= size 
            for k in range(d-1):
                color = "white"
                if idxes[i] != -1:
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline="black",fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))
                x -= size
                i += 1 
            d += 1 


def triangle():
    X0 = WIDTH // 2
    Y0 = 20
    size = 2
    i = 2 
    for d in range(HEIGHT // size):
        y = Y0+d*size
        for k in range(2*d+1):
            x = X0+k*size
            color = "white"
            if idxes[i] != -1:
                color = "red"
            canvas.create_rectangle(x,y,x+size,y+size,outline="black",fill = color)
            if i < 100: 
                canvas.create_text(x+size//2,y+size//2,text = str(i))

            i += 1 
        X0 -= size 

order = [] 
curI=2 
size = 5 
def recTessellation(x,y,level):
    global curI 
    if level ==0:
        color = "white"
        if idxes[curI] != -1:
            color = "red"
            canvas.create_rectangle(x,y,x+size,y+size,outline="black",fill = color)
        if curI < 100: 
            canvas.create_text(x+size//2,y+size//2,text = str(curI))
        curI += 1
        return
    recSize = int(math.pow(3,level-1)) * size 
    for k in range(len(order)):
        dx,dy = order[k]
        recTessellation(x+dx*recSize,y+dy*recSize,level-1)

def tessellation():
    recTessellation(10,10,5)

        
def allPerms(lst):
    if len(lst) == 1:
        return [lst] 
    ans = [] 
    for i in range(len(lst)):
        rec = allPerms(lst[0:i]+lst[i+1:])
        for perm in rec:
            ans += [perm+[lst[i]]]
    return ans

def allOrders():
    global order
    global curI 
    perms = allPerms([(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0),(1,1)])
    for perm in perms:
        order = perm
        tessellation()
        input("press enter")
        canvas.delete("all")
        curI=2
    

def run():
    draw(sum)
    pauseAndDelete()
    draw(norm)
    pauseAndDelete()
    draw(idxesSum)
    pauseAndDelete()
    draw(smallerThan)
    pauseAndDelete()
    spiral() 
    pauseAndDelete()
    spiral1() 
    pauseAndDelete()
    spiral2() 
    pauseAndDelete()
    spiral3() 
    pauseAndDelete()
    squareSpir() 
    pauseAndDelete()

            

# main
init()

#run() 

##ed = Editor()
##ed.drawGrid() 

#squareSpir1() 

#diagonals() 
#nuns()

#triangle()

#tessellation() 

allOrders()

##def isPrime(n):
##    for i in range(2,int(sqrt(n))+1):
##        if n % i == 0:
##            return False
##    return True
##
##def primes1(n):
##    print('start primes1')
##    for i in range(2,n+1):
##        if isPrime(i):
##            pass
##            #print(i, end =", ")
##    print('end primes1')
