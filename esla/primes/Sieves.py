import math
from math import sqrt

import tkinter

class Sieves:
    WIDTH = 1900
    HEIGHT = 900

    N = 1000000 # number of primes to find 
    idxes = [] 
    primes = [] 

    k = 2
    size = 20
    def __init__(self):
        self.calcPrimes(self.N)
        self.calIdxes()
        
        self.root = tkinter.Tk() 
        self.root.title("Sieves")


## menue 
        self.menubar = tkinter.Menu(self.root)
        self.viewmenu = tkinter.Menu(self.menubar, tearoff=False)

        self.spotsXYVisiable = tkinter.BooleanVar()
        self.spotsXYVisiable.set(False)
        self.viewmenu.add_checkbutton(label="Show SpotsXY", variable=self.spotsXYVisiable, command=self.draw)

        self.spotsKVisiable = tkinter.BooleanVar()
        self.spotsKVisiable.set(False)
        self.viewmenu.add_checkbutton(label="Show SpotsK", variable=self.spotsKVisiable, command=self.draw)
        
        self.futuresVisiable = tkinter.BooleanVar()
        self.futuresVisiable.set(False)
        self.viewmenu.add_checkbutton(label="Show Lookahead", variable=self.futuresVisiable, command=self.draw)
        self.menubar.add_cascade(label="View", menu=self.viewmenu)

        self.futuresMenu = tkinter.Menu(self.viewmenu)
        self.futuresNum = tkinter.IntVar()
        self.futuresNum.set(1)
        self.futuresMenu.add_radiobutton(label="Current", variable=self.futuresNum, value=0, command = self.draw)
        self.futuresMenu.add_radiobutton(label="One ahead", variable=self.futuresNum, value=1, command = self.draw)
        self.futuresMenu.add_radiobutton(label="two ahead", variable=self.futuresNum, value=2, command = self.draw)

        self.viewmenu.add_cascade(label="Lookahead Num", menu=self.futuresMenu)
        
        self.root.config(menu=self.menubar)


## -- canvas         
        self.canvas = tkinter.Canvas(self.root,width=self.WIDTH, height=self.HEIGHT, background="white")
        self.canvas.pack()


## -- slider 
        self.slider = tkinter.Scale(self.root, orient=tkinter.HORIZONTAL,
                                    length = 2*self.WIDTH // 3,command = self.slideChanged)
        self.slider.set(20)
        
        self.slider.pack(anchor="w")

 ## -- mouse         
        self.canvas.bind('<Button-1>', self.butt1Pressed)
        self.canvas.bind('<Button-3>', self.butt3Pressed)
       
        self.draw()

    def slideChanged(self,val):
        self.size = int(val) 
        self.draw()

    def calcPrimes(self,n):
        array = [True]*(n+1) 
        i = 2
        while i <= n:
            if array[i]:
                self.primes += [i]
                for j in range(i*i,n+1,i):
                    array[j] = False
            i+=1

    def calIdxes(self):
        self.idxes = [-1]*(self.primes[-1]+1)   
        for i in range(len(self.primes)):
            self.idxes[self.primes[i]] = i+1

    def isPrime(self,n):
        return self.idxes[n] != -1 

    def gcd(self,x,y):
        if y > x:
            x,y = y,x
        while y>0:
            x,y = y,x%y
        return x 

    def markWhiteSpotsXY(self):
        for yi in range(self.HEIGHT // self.size):
            for xi in range(self.WIDTH // self.size):
                x = self.size*xi
                y = self.size*yi
                if yi > 0 and self.gcd(yi,xi+1) > 1:
                    self.canvas.create_rectangle(x, y, x+self.size, y+self.size, outline = "black", fill = "yellow")
                    if xi < self.k:
                        n = self.xy2n(xi,yi)
                        self.canvas.create_text(x + self.size//2,y + self.size//2,text = str(n),font=("Helvetica",7))

    def markWhiteSpotsK(self):
        for yi in range(self.HEIGHT // self.size):
            for xi in range(self.k):
                x = self.size*xi
                y = self.size*yi
                if yi>0 and self.gcd(xi+1,self.k) > 1:
                    self.canvas.create_rectangle(x, y, x+self.size, y+self.size, outline = "black", fill = "yellow")
                    n = self.xy2n(xi,yi)
                    self.canvas.create_text(x + self.size//2,y + self.size//2,text = str(n),font=("Helvetica",7))
      

        
    def butt1Pressed(self,e):
##        x = e.x - e.x % self.size
##        y = e.y - e.y % self.size
##        self.canvas.create_rectangle(x,y,x+self.size,y+self.size,fill="yellow")
        self.k +=1
        self.draw()

    def butt3Pressed(self,e):
        if self.k == 2:
            return
        self.k -=1
        self.draw()

    def xy2n(self,x,y):
        x1 = x+1
        y1 = y+1 
        if x1 == self.k:
            return self.k*y1
        return self.k*y+x1
    
    def futureComp(self,w):
        for d in range(0,w+1):
            n = self.k+d 
            while(True):
                n += self.k+d
                xi = (n % self.k - 1)%self.k
                yi = n // self.k  
                if n% self.k == 0:
                    yi -= 1 
                x = xi*self.size
                y = yi*self.size
                if y > self.HEIGHT:
                    break
                self.canvas.create_rectangle(x,y,x+size,y+size,fill = "yellow")
                self.canvas.create_text(x+size//2,y+size//2,text = str(n),font=("Helvetica",7))

        
    def draw(self):
        self.canvas.delete("all")
        i = 1
        for yi in range(self.HEIGHT // self.size):
            for xi in range(self.k):
                x = self.size*xi
                y = self.size*yi
                color = "" 
                if self.isPrime(i):
                    color = "red"
                self.canvas.create_rectangle(x,y,x + self.size,y + self.size,outline = "black", fill = color)
                if i < 100: 
                    self.canvas.create_text(x + self.size//2,y + self.size//2,text = str(i))
                i += 1
        if self.futuresVisiable.get():
            self.futureComp(self.futuresNum.get())
        if self.spotsXYVisiable.get():
            self.markWhiteSpotsXY()
        if self.spotsKVisiable.get():
            self.markWhiteSpotsK()



Sieves()            

