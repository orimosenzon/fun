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
    SIZE = 30
    curI = 1        
    stack = []
    X0 = 50
    Y0 = 20 
    def __init__(self):
        canvas.bind('<Button-1>', self.butt1Pressed)
        canvas.bind('<Button-3>', self.butt3Pressed)


    def butt1Pressed(self,e):
        x = ((e.x-self.X0) // self.SIZE) * self.SIZE + self.X0
        y = ((e.y-self.Y0) // self.SIZE) * self.SIZE + self.Y0
        self.stack += [(x,y)] 
        if idxes[self.curI] == -1:
            color = "white"
        else:
            color = "red"
        canvas.create_rectangle(x,y,x+self.SIZE,y+self.SIZE,fill = color) 
        canvas.create_text(x+self.SIZE//2,y+self.SIZE//2,text = str(self.curI),font=("Purisa",17,"bold"))
        self.curI += 1

    def butt3Pressed(self,e):
        x,y = self.stack[-1]
        canvas.create_rectangle(x,y,x+self.SIZE,y+self.SIZE,fill = "white") 
        del(self.stack[-1])
        self.curI -= 1 
 
        
    def drawGrid(self):
        for x in range(self.X0,WIDTH,self.SIZE):
            for y in range(self.Y0,HEIGHT,self.SIZE):
                canvas.create_rectangle(x, y, x+self.SIZE, y+self.SIZE)
        
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
    i = 1
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
    i = 1
    d = 1
    Y0 = X0 = 20 
    while d<WIDTH//size:
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

order = [(0,0),(0,2),(1,2),(1,0),(1,1),(2,0),(0,1),(2,1),(2,2)]
curI=1 
size = 20 
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
    recTessellation(10,10,4)

        
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

            

def allKvaras():
    colors = ["black","blue","red","green","orange","yellow","pink"]
    size = 20
    for k in range(2,WIDTH // size):
        i = 1
        x0 = 0 #k*size*10
        for yi in range(HEIGHT // size):
            for xi in range(k):
                x = x0+size*xi
                y = size*yi 
                color = ""
                if idxes[i] != -1:
                    #color = colors[idxes[i] % len(colors)]
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline = "black", fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))
                i += 1
        pauseAndDelete()

def allKvaras1():
    colors = ["black","blue","red","green","orange","yellow","pink"]
    size = 20
    for k in range(2,HEIGHT // size):
        i = 1
        x0 = 0 #k*size*10
        for xi in range(WIDTH//size):
            for yi in range(k):
                x = x0+size*xi
                y = size*yi 
                color = ""
                if idxes[i] != -1:
                    #color = colors[idxes[i] % len(colors)]
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline = "black", fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))
                i += 1
        pauseAndDelete()

def allKvaras2():
    colors = ["black","blue","red","green","orange","yellow","pink"]
    size = 20
    flipflop = True
    for k in range(2,WIDTH // size):
        i = 1
        x0 = 0 #k*size*10
        for yi in range(HEIGHT // size):
            for xi in range(k):
                x = x0+size*xi
                y = size*yi 
                color = ""
                if idxes[i] != -1:
                    #color = colors[idxes[i] % len(colors)]
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline = "black", fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))
                if flipflop: 
                    i += 4
                else:
                    i += 2
                flipflop = not flipflop
        pauseAndDelete()


def fracRemain():
    m = 1
    c = 1
    for i in range(len(primes)):
        m *= (primes[i]-1)/primes[i]
        c += 1
        if c< 100 or c%100==0:
            print(1/m)


def diagonalsSerp():
    size = 20 
    i = 1
    Y0 = X0 = 20 
    for d in range((WIDTH+HEIGHT)//size):
            odd = d%2   # 1 - odd, 0 - even 
            odd1 = odd*2-1 # 1 - odd, -1 - even 
            dx = -size*odd1
            dy = size*odd1 

            x = X0 + odd*d*size
            y = Y0 + (1-odd)*d*size
     
            for k in range(d+1):
                color = "white"
                if idxes[i] != -1:
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline="black",fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))

                x += dx
                y += dy 
                i += 1 

def nunsSerp():
    size = 20 
    i = 1
    Y0 = X0 = 20 
    for d in range(WIDTH//size):
            odd = d%2   # 1 - odd, 0 - even 

            x = X0 + odd*d*size
            y = Y0 + (1-odd)*d*size

            dx = size*(1-odd)
            dy = size*odd 

            for k in range(d):
                color = "white"
                if idxes[i] != -1:
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline="black",fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))
                x += dx  
                y += dy
                i += 1

            dx = -size*odd
            dy = -size*(1-odd) 

            for k in range(d+1):
                color = "white"
                if idxes[i] != -1:
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline="black",fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))
                x += dx  
                y += dy
                i += 1 
            
    
COLORS  =['snow', 'ghost white', 'white smoke', 'gainsboro', 'floral white', 'old lace',
    'linen', 'antique white', 'papaya whip', 'blanched almond', 'bisque', 'peach puff',
    'navajo white', 'lemon chiffon', 'mint cream', 'azure', 'alice blue', 'lavender',
    'lavender blush', 'misty rose', 'dark slate gray', 'dim gray', 'slate gray',
    'light slate gray', 'gray', 'light grey', 'midnight blue', 'navy', 'cornflower blue', 'dark slate blue',
    'slate blue', 'medium slate blue', 'light slate blue', 'medium blue', 'royal blue',  'blue',
    'dodger blue', 'deep sky blue', 'sky blue', 'light sky blue', 'steel blue', 'light steel blue',
    'light blue', 'powder blue', 'pale turquoise', 'dark turquoise', 'medium turquoise', 'turquoise',
    'cyan', 'light cyan', 'cadet blue', 'medium aquamarine', 'aquamarine', 'dark green', 'dark olive green',
    'dark sea green', 'sea green', 'medium sea green', 'light sea green', 'pale green', 'spring green',
    'lawn green', 'medium spring green', 'green yellow', 'lime green', 'yellow green',
    'forest green', 'olive drab', 'dark khaki', 'khaki', 'pale goldenrod', 'light goldenrod yellow',
    'light yellow', 'yellow', 'gold', 'light goldenrod', 'goldenrod', 'dark goldenrod', 'rosy brown',
    'indian red', 'saddle brown', 'sandy brown',
    'dark salmon', 'salmon', 'light salmon', 'orange', 'dark orange',
    'coral', 'light coral', 'tomato', 'orange red', 'red', 'hot pink', 'deep pink', 'pink', 'light pink',
    'pale violet red', 'maroon', 'medium violet red', 'violet red',
    'medium orchid', 'dark orchid', 'dark violet', 'blue violet', 'purple', 'medium purple',
    'thistle', 'snow2', 'snow3',
    'snow4', 'seashell2', 'seashell3', 'seashell4', 'AntiqueWhite1', 'AntiqueWhite2',
    'AntiqueWhite3', 'AntiqueWhite4', 'bisque2', 'bisque3', 'bisque4', 'PeachPuff2',
    'PeachPuff3', 'PeachPuff4', 'NavajoWhite2', 'NavajoWhite3', 'NavajoWhite4',
    'LemonChiffon2', 'LemonChiffon3', 'LemonChiffon4', 'cornsilk2', 'cornsilk3',
    'cornsilk4', 'ivory2', 'ivory3', 'ivory4', 'honeydew2', 'honeydew3', 'honeydew4',
    'LavenderBlush2', 'LavenderBlush3', 'LavenderBlush4', 'MistyRose2', 'MistyRose3',
    'MistyRose4', 'azure2', 'azure3', 'azure4', 'SlateBlue1', 'SlateBlue2', 'SlateBlue3',
    'SlateBlue4', 'RoyalBlue1', 'RoyalBlue2', 'RoyalBlue3', 'RoyalBlue4', 'blue2', 'blue4',
    'DodgerBlue2', 'DodgerBlue3', 'DodgerBlue4', 'SteelBlue1', 'SteelBlue2',
    'SteelBlue3', 'SteelBlue4', 'DeepSkyBlue2', 'DeepSkyBlue3', 'DeepSkyBlue4',
    'SkyBlue1', 'SkyBlue2', 'SkyBlue3', 'SkyBlue4', 'LightSkyBlue1', 'LightSkyBlue2',
    'LightSkyBlue3', 'LightSkyBlue4', 'SlateGray1', 'SlateGray2', 'SlateGray3',
    'SlateGray4', 'LightSteelBlue1', 'LightSteelBlue2', 'LightSteelBlue3',
    'LightSteelBlue4', 'LightBlue1', 'LightBlue2', 'LightBlue3', 'LightBlue4',
    'LightCyan2', 'LightCyan3', 'LightCyan4', 'PaleTurquoise1', 'PaleTurquoise2',
    'PaleTurquoise3', 'PaleTurquoise4', 'CadetBlue1', 'CadetBlue2', 'CadetBlue3',
    'CadetBlue4', 'turquoise1', 'turquoise2', 'turquoise3', 'turquoise4', 'cyan2', 'cyan3',
    'cyan4', 'DarkSlateGray1', 'DarkSlateGray2', 'DarkSlateGray3', 'DarkSlateGray4',
    'aquamarine2', 'aquamarine4', 'DarkSeaGreen1', 'DarkSeaGreen2', 'DarkSeaGreen3',
    'DarkSeaGreen4', 'SeaGreen1', 'SeaGreen2', 'SeaGreen3', 'PaleGreen1', 'PaleGreen2',
    'PaleGreen3', 'PaleGreen4', 'SpringGreen2', 'SpringGreen3', 'SpringGreen4',
    'green2', 'green3', 'green4', 'chartreuse2', 'chartreuse3', 'chartreuse4',
    'OliveDrab1', 'OliveDrab2', 'OliveDrab4', 'DarkOliveGreen1', 'DarkOliveGreen2',
    'DarkOliveGreen3', 'DarkOliveGreen4', 'khaki1', 'khaki2', 'khaki3', 'khaki4',
    'LightGoldenrod1', 'LightGoldenrod2', 'LightGoldenrod3', 'LightGoldenrod4',
    'LightYellow2', 'LightYellow3', 'LightYellow4', 'yellow2', 'yellow3', 'yellow4',
    'gold2', 'gold3', 'gold4', 'goldenrod1', 'goldenrod2', 'goldenrod3', 'goldenrod4',
    'DarkGoldenrod1', 'DarkGoldenrod2', 'DarkGoldenrod3', 'DarkGoldenrod4',
    'RosyBrown1', 'RosyBrown2', 'RosyBrown3', 'RosyBrown4', 'IndianRed1', 'IndianRed2',
    'IndianRed3', 'IndianRed4', 'sienna1', 'sienna2', 'sienna3', 'sienna4', 'burlywood1',
    'burlywood2', 'burlywood3', 'burlywood4', 'wheat1', 'wheat2', 'wheat3', 'wheat4', 'tan1',
    'tan2', 'tan4', 'chocolate1', 'chocolate2', 'chocolate3', 'firebrick1', 'firebrick2',
    'firebrick3', 'firebrick4', 'brown1', 'brown2', 'brown3', 'brown4', 'salmon1', 'salmon2',
    'salmon3', 'salmon4', 'LightSalmon2', 'LightSalmon3', 'LightSalmon4', 'orange2',
    'orange3', 'orange4', 'DarkOrange1', 'DarkOrange2', 'DarkOrange3', 'DarkOrange4',
    'coral1', 'coral2', 'coral3', 'coral4', 'tomato2', 'tomato3', 'tomato4', 'OrangeRed2',
    'OrangeRed3', 'OrangeRed4', 'red2', 'red3', 'red4', 'DeepPink2', 'DeepPink3', 'DeepPink4',
    'HotPink1', 'HotPink2', 'HotPink3', 'HotPink4', 'pink1', 'pink2', 'pink3', 'pink4',
    'LightPink1', 'LightPink2', 'LightPink3', 'LightPink4', 'PaleVioletRed1',
    'PaleVioletRed2', 'PaleVioletRed3', 'PaleVioletRed4', 'maroon1', 'maroon2',
    'maroon3', 'maroon4', 'VioletRed1', 'VioletRed2', 'VioletRed3', 'VioletRed4',
    'magenta2', 'magenta3', 'magenta4', 'orchid1', 'orchid2', 'orchid3', 'orchid4', 'plum1',
    'plum2', 'plum3', 'plum4', 'MediumOrchid1', 'MediumOrchid2', 'MediumOrchid3',
    'MediumOrchid4', 'DarkOrchid1', 'DarkOrchid2', 'DarkOrchid3', 'DarkOrchid4',
    'purple1', 'purple2', 'purple3', 'purple4', 'MediumPurple1', 'MediumPurple2',
    'MediumPurple3', 'MediumPurple4', 'thistle1', 'thistle2', 'thistle3', 'thistle4',
    'gray1', 'gray2', 'gray3', 'gray4', 'gray5', 'gray6', 'gray7', 'gray8', 'gray9', 'gray10',
    'gray11', 'gray12', 'gray13', 'gray14', 'gray15', 'gray16', 'gray17', 'gray18', 'gray19',
    'gray20', 'gray21', 'gray22', 'gray23', 'gray24', 'gray25', 'gray26', 'gray27', 'gray28',
    'gray29', 'gray30', 'gray31', 'gray32', 'gray33', 'gray34', 'gray35', 'gray36', 'gray37',
    'gray38', 'gray39', 'gray40', 'gray42', 'gray43', 'gray44', 'gray45', 'gray46', 'gray47',
    'gray48', 'gray49', 'gray50', 'gray51', 'gray52', 'gray53', 'gray54', 'gray55', 'gray56',
    'gray57', 'gray58', 'gray59', 'gray60', 'gray61', 'gray62', 'gray63', 'gray64', 'gray65',
    'gray66', 'gray67', 'gray68', 'gray69', 'gray70', 'gray71', 'gray72', 'gray73', 'gray74',
    'gray75', 'gray76', 'gray77', 'gray78', 'gray79', 'gray80', 'gray81', 'gray82', 'gray83',
    'gray84', 'gray85', 'gray86', 'gray87', 'gray88', 'gray89', 'gray90', 'gray91', 'gray92',
    'gray93', 'gray94', 'gray95', 'gray97', 'gray98', 'gray99']


def showColors(cols):
    size = 40
    w = WIDTH - size
    for i in range(len(cols)):
        b = size*i
        y = b // w * size 
        x = b % w
        canvas.create_rectangle(x,y,x+size,y+size,fill = cols[i])
        canvas.create_text(x+size//2,y+size//2,text = str(i))

def allKvarasWithColors():
    colors = ["red"]
    for ci in range(419,len(COLORS),5):
        colors += [COLORS[ci]]

    size = 20
    for k in range(2,WIDTH // size):
        i = 1
        x0 = 0 #k*size*10
        for yi in range(HEIGHT // size):
            for xi in range(k):
                x = x0+size*xi
                y = size*yi 
                n = len(factorize(i))-1
                if n >= len(colors):
                    n = -1 
                color = colors[n]
                canvas.create_rectangle(x,y,x+size,y+size,outline = "black", fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))
                else:
                    canvas.create_text(x+size//2,y+size//2,text = str(n+1))
                i += 1
        pauseAndDelete()


def allKvarasHV():
    colors = ["black","blue","red","green","orange","yellow","pink"]
    size = 5
    for k in range(2,WIDTH // size):
        i = 1
        x0 = 0 #k*size*10
        for yi in range(HEIGHT // size):
            for xi in range(k):
                x = x0+size*xi
                y = size*yi 
                color = ""
                if idxes[i] != -1:
                    #color = colors[idxes[i] % len(colors)]
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline = "black", fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))
                canvas.create_rectangle(y,x,y+size,x+size,outline = "black", fill = color)
                if i < 100: 
                    canvas.create_text(y+size//2,x+size//2,text = str(i))
                i += 1
        pauseAndDelete()

def allKvarasNoDel():
    size = 20
    for yi in range(HEIGHT // size):
        for xi in range(WIDTH // size):
            if gcd(yi,xi+1) > 1:
                x = size*xi
                y = size*yi 
                canvas.create_rectangle(x,y,x+size,y+size,outline = "black", fill = "")
                canvas.create_text(x+size//2,y+size//2,text = str("N"))
                
    for k in range(2,WIDTH // size):
        i = 1
        for yi in range(HEIGHT // size):
            for xi in range(k):
                x = size*xi
                y = size*yi 
                color = ""
                if idxes[i] != -1:
                    #color = colors[idxes[i] % len(colors)]
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline = "black", fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))
                i += 1
        input("Press enter")


def gcd(x,y):
    if y > x:
        x,y = y,x
    while y>0:
        x,y = y,x%y
    return x 

def gcd1(x,y):
    if y==0:
        return (x,1,0)
    (g,a,b) = gcd1(y,x%y)
    return (g,b,a-b*(x//y))

def gcdComb(x,y):
    if y>x:
        x,y = y,x
    (g,a,b) = gcd1(x,y)
    print(a,"*",x,"+",b,"*",y,"=",a*x+b*y)
    print("gcd =",g)


def allKvarasWithBlackSpots():
    #two types of spots that cannot contain prime numbers:
    # if n = k*y+x, when gcd(k,x) > 1 and when gcd (y,x) > 1
    # the first type depends on the k but the other doesn't, it depens only on the location 
    size = 20
    for k in range(2,WIDTH // size):
        for yi in range(HEIGHT // size):
            for xi in range(WIDTH // size):
                x = size*xi
                y = size*yi
                s3 = size // 3 
                if xi <= k and gcd(xi+1,k) > 1:
                    canvas.create_rectangle(x,y+s3,x+size,y+2*s3,outline = "blue", fill = "blue") 
                if yi > 0 and gcd(yi,xi+1) > 1:
                    canvas.create_rectangle(x,y,x+size,y+size,outline = "black", fill = "")
                    canvas.create_rectangle(x+s3,y,x+2*s3,y+size,outline = "blue", fill = "blue")
        i = 1
        for yi in range(HEIGHT // size):
            for xi in range(k):
                x = size*xi
                y = size*yi 
                color = ""
                if idxes[i] != -1:
                    color = "red"
                canvas.create_rectangle(x,y,x+size,y+size,outline = "black", fill = color)
                if i < 100: 
                    canvas.create_text(x+size//2,y+size//2,text = str(i))
                i += 1
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

#allOrders()

#allKvaras()
#allKvaras1()
#allKvaras2()
#fracRemain()

#diagonalsSerp()

#nunsSerp() 

#showColors(colors) 

#allKvarasWithColors()
#allKvarasNoDel()

allKvarasWithBlackSpots()













# left overs 

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
