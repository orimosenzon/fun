import math
from math import sqrt

import tkinter

class Sieves:
    WIDTH = 1900
    HEIGHT = 1000

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

        self.canvas = tkinter.Canvas(self.root,width=self.WIDTH, height=self.HEIGHT, background="white")
        self.canvas.pack()

## menue 
        self.menubar = tkinter.Menu(self.root)
        self.viewmenu = tkinter.Menu(self.menubar, tearoff=False)

        self.futuresVisiable = tkinter.BooleanVar()
        self.futuresVisiable.set(False)
        self.viewmenu.add_checkbutton(label="Show Futures", variable=self.futuresVisiable, command=self.draw)
        self.menubar.add_cascade(label="View", menu=self.viewmenu)
        
        self.root.config(menu=self.menubar)

## --         
        self.canvas.bind('<Button-1>', self.butt1Pressed)
        self.canvas.bind('<Button-3>', self.butt3Pressed)
        
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

        
    def butt1Pressed(self,e):
        x = e.x - e.x % self.size
        y = e.y - e.y % self.size
        self.canvas.create_rectangle(x,y,x+self.size,y+self.size,fill="yellow")

    def butt3Pressed(self,e):
        self.k +=1
        self.draw()
    
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
            self.futureComp(1)



Sieves()            

