#!/usr/bin/python

# Copyright (C) 2009 Ori Mosenzon
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# http://www.gnu.org/copyleft/gpl.html

import Tkinter,tkFileDialog
from math import *

## App ##  

class App:

    def __init__(self):
        self.gui_init()

        self.HEIGHT = self.canvas.winfo_reqheight()
        self.WIDTH = self.canvas.winfo_reqwidth()

    def gui_init(self):
        self.root = Tkinter.Tk()
        self.canvas = Tkinter.Canvas(self.root, width=400, height=200, bg='white')
        self.canvas.pack(expand=Tkinter.YES, fill=Tkinter.BOTH)

        panel = Tkinter.Frame(self.root)
        panel.pack(side=Tkinter.BOTTOM)

        complexPanel = Tkinter.Frame(panel)
        complexPanel.pack(side=Tkinter.RIGHT)

        self.real1Entry = Tkinter.Entry(complexPanel)
        self.real1Entry.pack(side = Tkinter.RIGHT)

        self.image1Entry = Tkinter.Entry(complexPanel)
        self.image1Entry.pack(side = Tkinter.RIGHT)

#        self.real2Entry = Tkinter.Entry(complexPanel)
#        self.real2Entry.grid(row=1)

#        self.image2Entry = Tkinter.Entry(complexPanel)
#        self.image2Entry.pack()



        self.canvas.bind('<Button-1>', self.butt1Pressed)
        self.canvas.bind("<B1-Motion>", self.butt1Motion)
        self.canvas.bind('<ButtonRelease-1>', self.butt1Released)
        self.canvas.bind('<Configure>', self.resize)

        self.menubar = Tkinter.Menu(self.root)

        # ** FILE menu **  
        self.filemenu = Tkinter.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

#--
        self.root.config(menu=self.menubar)


    def butt1Pressed(self,event):
#rewrite.. 
        pass

    def butt1Motion(self,event):
#rewrite.. 
        pass

    def butt1Released(self,event):
#rewrite.. 
        pass

    def clear(self):
#rewrite.. 
        pass
    
    def setSize(self, width, height): 
        self.WIDTH = width
        self.HEIGHT = height
        
    def refresh(self):
        self.canvas.delete("all")
        
        self.CX = self.WIDTH/2
        self.CY = self.HEIGHT/2

    def resize(self,event):
        app.setSize(event.width,event.height)
        app.refresh()

    def drawAxes(self):
        size = self.CY*3/4
        self.drawEdge((0,size),(0,-size),'black')
        self.drawEdge((size,0),(-size,0),'black')
        gup = 25 
        shnt = 2
        y = 0 
        while y < size:
            self.drawEdge((-shnt,y),(shnt,y),'black')
            self.drawEdge((-shnt,-y),(shnt,-y),'black')
            y +=  gup 

        x = 0 
        while x < size:
            self.drawEdge((x,-shnt),(x,shnt),'black')
            self.drawEdge((-x,-shnt),(-x,shnt),'black')
            x +=  gup 
            

    def foo(self):
        print "foo"
            
## private members 
    def drawDot(self,v):
            self.drawOval(v)
    
    def drawArrow(self,(x,y)):

        norm = sqrt(x*x+y*y)
        k = min(self.k, self.maxArrowSide/norm) 
        v = (-x*k, -y*k) # i.e the length will be .2*norm or 10 (whichever bigger)

        (tx,ty) = self.matrixMul(self.T1,v)
        (xa1,ya1) = self.cord2disp((x+tx,y+ty))
        (tx,ty) = self.matrixMul(self.T2,v)
        (xa2,ya2) = self.cord2disp((x+tx,y+ty))

        (x0,y0) = self.cord2disp((0,0))
        (x,y) = self.cord2disp((x,y))
        self.canvas.create_line(x0, y0, x, y, fill = 'blue')    

        self.canvas.create_line(x, y, xa1, ya1, fill = 'blue')    
        self.canvas.create_line(x, y, xa2, ya2, fill = 'blue')    

    def drawOval(self,(x,y),outlineColor = 'green', fillColor = 'black'):
        OVR = self.OVALRAD
        (x,y) = self.cord2disp((x,y))
        self.canvas.create_oval(x-OVR, y-OVR, x+OVR, y+OVR, outline = outlineColor, fill = fillColor)
        
    def drawDashedEdge(self,(x1,y1),(x2,y2), color = 'green'):
        (x1,y1) = self.cord2disp((x1, y1))
        (x2,y2) = self.cord2disp((x2, y2))
        self.canvas.create_line(x1, y1, x2, y2, fill = color, dash =(2,2))    

    def drawEdge(self,(x1,y1),(x2,y2), color = 'green'):
        (x1,y1) = self.cord2disp((x1, y1))
        (x2,y2) = self.cord2disp((x2, y2))
        self.canvas.create_line(x1, y1, x2, y2, fill = color)    
    
    def isInVertexOval(self,event,(x,y)):
        (x,y) = self.cord2disp((x,y))
        return sqrt(pow((event.x-x),2)+pow((event.y-y),2)) <= self.OVALRAD 
    
    def cord2disp(self,(x,y)): # translate inner cooredinates to display (canvas) coordinates 
        return (self.CX+x, self.CY-y)
    
    def disp2cord(self,(x,y)):# translate display (canvas) coordinates to inner cooredinates 
        return (x-self.CX, self.CY-y)
    
    def findVertex(self,event):
        for i in range(len(self.vertices)): 
            if self.isInVertexOval(event,self.vertices[i]):
                return i 
        return None  


    def matrixMul(self,T,(x,y)):
        return (T[0][0]*x+T[0][1]*y,T[1][0]*x+T[1][1]*y)    

    
    def mul(self,k,(x,y)):
        return (k*x,k*y)
    
    def add(self,(x1,y1),(x2,y2)):
        return (x1+x2,y1+y2)    
             
## end of App 


## Main ## 

app = App()

Tkinter.mainloop()
