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
import numpy 
from math import *

## App ##  

class App:

    def __init__(self):
        self.gui_init()

        self.HEIGHT = self.canvas.winfo_reqheight()
        self.WIDTH = self.canvas.winfo_reqwidth()
        self.init()

    def gui_init(self):
        self.root = Tkinter.Tk()
        self.canvas = Tkinter.Canvas(self.root, width=400, height=200, bg='white')
        self.canvas.pack(expand=Tkinter.YES, fill=Tkinter.BOTH)

        panel = Tkinter.Frame(self.root)
        panel.pack(side=Tkinter.BOTTOM)

        self.canvas.bind('<Button-1>', self.butt1Pressed)
        self.canvas.bind("<B1-Motion>", self.butt1Motion)
        self.canvas.bind('<ButtonRelease-1>', self.butt1Released)
        self.canvas.bind('<Configure>', self.resize)

        self.menubar = Tkinter.Menu(self.root)

        # ** FILE menu **  
        self.filemenu = Tkinter.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Open", command=self.load)
        self.filemenu.add_command(label="Save", command=self.save)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Clear", command=self.clear)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        # ** VIEW menu **  
        self.viewmenu = Tkinter.Menu(self.menubar, tearoff=0)

        self.areEdgesVisable = Tkinter.BooleanVar()
        self.areEdgesVisable.set(False)
        self.viewmenu.add_checkbutton(label="Show Edges", variable=self.areEdgesVisable, command=self.refresh)

        self.areAxesVisibale = Tkinter.BooleanVar()
        self.areAxesVisibale.set(True)
        self.viewmenu.add_checkbutton(label="Show Axes", variable=self.areAxesVisibale, command=self.refresh)

        self.isDeterminantVisable = Tkinter.BooleanVar()
        self.isDeterminantVisable.set(False)
        self.viewmenu.add_checkbutton(label="Show Determinant", variable=self.isDeterminantVisable, command=self.refresh)

        self.areFuturesVisable = Tkinter.BooleanVar()
        self.areFuturesVisable.set(False)
        self.viewmenu.add_checkbutton(label="Show Futures", variable=self.areFuturesVisable, command=self.refresh)

        self.arePastVisable = Tkinter.BooleanVar()
        self.arePastVisable.set(False)
        self.viewmenu.add_checkbutton(label="Show Pasts", variable=self.arePastVisable, command=self.refresh)

        self.areEigenVisable = Tkinter.BooleanVar()
        self.areEigenVisable.set(False)
        self.viewmenu.add_checkbutton(label="Show Eigenvectors", variable=self.areEigenVisable, command=self.refresh)

        # how to draw each kind of visualization
        self.vectorVisDict = {'dot':self.drawDot,'arrow':self.drawArrow}
        
        isDot = Tkinter.BooleanVar()
        isDot.set(False)
        isArrow = Tkinter.BooleanVar()
        isArrow.set(True)
        # which visualizations are being used 
        self.vectorVisSelectDict = {'dot':isDot, 'arrow':isArrow}

        self.vectorRepresentationMenu \
            = Tkinter.Menu(self.viewmenu,tearoff=0)

        self.vectorRepresentationMenu.add_checkbutton(label="Arrow",variable=self.vectorVisSelectDict['arrow'], command=self.refresh)

        self.vectorRepresentationMenu.add_checkbutton(label="Dot",variable=self.vectorVisSelectDict['dot'], command = self.refresh)

        self.viewmenu.add_cascade(label="Vector Representation",menu=self.vectorRepresentationMenu)


        self.menubar.add_cascade(label="View", menu=self.viewmenu)

        # ** HELP menu ** 
        self.helpmenu = Tkinter.Menu(self.menubar, tearoff=0)
        self.helpmenu.add_command(label="About", command=self.foo)
        self.helpmenu.add_command(label="Help Contents", command=self.foo)

        self.menubar.add_cascade(label="Help", menu=self.helpmenu)

        self.root.config(menu=self.menubar)

        # ** predifined matrices ** 
        self.predefinedMatrices = {"Rotate (CCW)":("cos(.1)","-sin(.1)","sin(.1)","cos(.1)"),
                              "Rotate (CW)":("cos(.1)","sin(.1)","-sin(.1)","cos(.1)"),
                              "Rotate and enlarge":("cos(.1)*1.05","-sin(.1)*1.05","sin(.1)*1.05","cos(.1)*1.05"),
                              "Rotate and shrink":("cos(.1)*.95","-sin(.1)*.95","sin(.1)*.95","cos(.1)*.95"),
                              "ID":(1,0,0,1),
                              "Shrink":(0.95,0,0,0.95),               
                              "Enlarge":(1.05,0,0,1.05),
                              "Y pull":(1,0,0,1.05),
                              "X pull":(1.05,0,0,1),
                              "X revert":(-1,0,0,1),
                              "Y revert":(1,0,0,-1),                    
                             }

        self.matrixStr = Tkinter.StringVar(panel)
#        self.matrixStr.set(self.predefinedMatrices.keys()[4])
        self.matrixStr.set("Enlarge")
        matrixOptionMenu = apply(Tkinter.OptionMenu, (panel, self.matrixStr) + tuple(self.predefinedMatrices.keys()))
        matrixOptionMenu.pack(side=Tkinter.LEFT)

        buttonApplyTransformation = Tkinter.Button(panel,text="Apply transformation",command=self.applyTransformation)
        buttonApplyTransformation.pack(side=Tkinter.LEFT)

        matrixPanel = Tkinter.Frame(panel)
        matrixPanel.pack(side=Tkinter.RIGHT)

        self.entry00Str = Tkinter.StringVar()
        self.entry00 = Tkinter.Entry(matrixPanel, textvariable = self.entry00Str)
        self.entry00.grid(row=0,column=0)

        self.entry10Str = Tkinter.StringVar()
        self.entry10 = Tkinter.Entry(matrixPanel, textvariable = self.entry10Str)
        self.entry10.grid(row=1,column=0)

        self.entry01Str = Tkinter.StringVar()
        self.entry01 = Tkinter.Entry(matrixPanel, textvariable = self.entry01Str)
        self.entry01.grid(row=0,column=1)

        self.entry11Str = Tkinter.StringVar()
        self.entry11 = Tkinter.Entry(matrixPanel, textvariable = self.entry11Str)
        self.entry11.grid(row=1,column=1)

        
    def init(self):
        self.vertices = []     # ((x_1,y_1),...,(x_n,y_n)]   // places in the plane 
        self.pastVertices = [] # [ [(x1,y1),(x2,y2),..], [...], [..]]
        self.edges = []        # [(i_1,j_1),...,(i_m,j_m)]      // indices of vertices 

        (self.e1,self.e2) = ((1,0),(0,1))
        self.selectedVertex = None  
        self.OVALRAD = 5 
        self.lastPlace = None 
        
        # constants for arrow drawing 
        self.maxArrowSide = 10 
        self.k = .2
        a = pi/7 # the angle of (half) arrow head 
        self.T1 = [[cos(a),-sin(a)],[sin(a),cos(a)]] 
        self.T2 = [[cos(a),sin(a)],[-sin(a),cos(a)]] 

        # The following command connects a change of matrixStr to the matrix's values change 
        # The three parametrs to the callback function (onChangeMatrixStr) are required 
        # by the trace_variable mechanism but are not in use in this code. 
        self.matrixStr.trace_variable('w',self.onChangeMatrixStr) 
        self.onChangeMatrixStr(0,0,0)

        self.entry00Str.trace_variable('w',self.onEntryChange) 
        self.entry01Str.trace_variable('w',self.onEntryChange) 
        self.entry10Str.trace_variable('w',self.onEntryChange) 
        self.entry11Str.trace_variable('w',self.onEntryChange) 

    
    def butt1Pressed(self,event):
        self.selectedVertex = self.findVertex(event)
        if self.selectedVertex != None: 
            return 

        v = self.disp2cord((event.x, event.y))
        self.vertices += [v]
        self.pastVertices += [[]] 
        self.selectedVertex = len(self.vertices)-1
        self.refresh()

    def butt1Motion(self,event):
        if self.selectedVertex == None: 
            return 
        
        v = self.vertices[self.selectedVertex] 
        if self.lastPlace != None:
            # wipe out last movement
            v1 = self.disp2cord((self.lastPlace.x, self.lastPlace.y)) 
            self.drawEdge(v,v1,'white')
            self.drawOval(v1, 'white', 'white')
            
        self.lastPlace = event

        v1 = self.disp2cord((event.x,event.y))
        self.drawEdge(v,v1)
        self.drawOval(v1)

    def butt1Released(self,event):
        if self.selectedVertex == None: 
            return 

        self.lastPlace = None
        i = self.findVertex(event)
        if i == None: # create new connected vertex 
            self.vertices += [self.disp2cord((event.x,event.y))]
            self.pastVertices += [[]]
            self.edges += [(self.selectedVertex,len(self.vertices)-1)] 
        else:
            if i == self.selectedVertex:
                return
            else:     # connect two existing vertices 
                self.edges += [(self.selectedVertex,i)] 
        self.refresh()

    def applyTransformationT(self,T):
         for i in range(len(self.vertices)):
             (x,y) = self.vertices[i]
             self.pastVertices[i] += [(x,y)]
             self.vertices[i] = self.matrixMul(T,(x,y))
         self.refresh()
         self.e1 = self.matrixMul(T,self.e1)
         self.e2 = self.matrixMul(T,self.e2)
    
    def clear(self):
        self.init()
    
    def setSize(self, width, height): 
        self.WIDTH = width
        self.HEIGHT = height
        
    def refresh(self):
        
        self.canvas.delete("all")
        
        self.CX = self.WIDTH/2
        self.CY = self.HEIGHT/2

        if self.areAxesVisibale.get():
            self.drawAxes()

        # draw edges 
        if self.areEdgesVisable.get():
            for (i,j) in self.edges:
                self.drawEdge(self.vertices[i],self.vertices[j])    

        T = self.getCurrentMatrix()

        #Eigen 
        if self.areEigenVisable.get():
            self.drawEigen()
        
        # for each vector: draw all the visualizations 
        for i in range(len(self.vertices)):
            v = self.vertices[i] 
            if self.areFuturesVisable.get():
                f =  self.matrixMul(T,v) 
                self.drawEdge(v,f,'gray')
            if self.arePastVisable.get():
                history = self.pastVertices[i]
                for j in range(len(history)-1):
                    self.drawDashedEdge(history[j],history[j+1],'gray')
                if len(history) > 0: 
                    self.drawDashedEdge(history[-1],v,'gray')     

            for vis in self.vectorVisDict.keys():
                if self.vectorVisSelectDict[vis].get():
                   self.vectorVisDict[vis](v)                                         
        if self.isDeterminantVisable.get():
            self.drawStandardBase()    
            self.drawDeterminant()
        
    def resize(self,event):
        app.setSize(event.width,event.height)
        app.refresh()

    def drawAxes(self):
        acolor = 'gray'
        size = self.CY*3/4
        self.drawEdge((0,size),(0,-size),acolor)
        self.drawEdge((size,0),(-size,0),acolor)
        gup = 25 
        shnt = 2
        y = 0 
        while y < size:
            self.drawEdge((-shnt,y),(shnt,y),acolor)
            self.drawEdge((-shnt,-y),(shnt,-y),acolor)
            y +=  gup 

        x = 0 
        while x < size:
            self.drawEdge((x,-shnt),(x,shnt),acolor)
            self.drawEdge((-x,-shnt),(-x,shnt),acolor)
            x +=  gup 
            
    def onEntryChange(self,dmy1,dmy2,dmy3):
        self.refresh()
        

    def onChangeMatrixStr(self,dmy1,dmy2,dmy3):
        matrixName = self.matrixStr.get()
        matrixTuple = self.predefinedMatrices[matrixName]

        self.entry00.delete(0, Tkinter.END)
        self.entry00.insert(0,matrixTuple[0])
        
        self.entry01.delete(0, Tkinter.END)
        self.entry01.insert(0,matrixTuple[1])
        
        self.entry10.delete(0, Tkinter.END)
        self.entry10.insert(0,matrixTuple[2])
        
        self.entry11.delete(0, Tkinter.END)
        self.entry11.insert(0,matrixTuple[3])


    def save(self):
        myFormats = [
                     ('Vectors Graph','*.vgr'),
                     ]

        fileName = tkFileDialog.asksaveasfilename(parent=self.root,filetypes=myFormats ,title="Save as...")
        if len(fileName ) == 0:
            return
        file = open(fileName,'w')
        file.write("#Vertices\n")
        file.write("V")
        file.write(pairs2str(self.vertices))
        file.write("\n#Edges\n")
        file.write("E")
        file.write(pairs2str(self.edges))
        file.write("\n")
        file.close()
        
    def load(self):
        myFormats = [
                     ('Vectors Graph','*.vgr'),
                     ]

        file = tkFileDialog.askopenfile(parent=self.root,mode='rb',filetypes=myFormats, title='Choose a file')      
        if file == None:
            return
        data = file.read()
        file.close()
        self.vertices = [] 
        self.pastVertices = [] 
        self.edges = [] 
        lines = data.split()
        for line in lines:
            c = line[0]
            if c == '#':
                continue
            line = line[1:]
            pairs = line.split('|')
            if c == 'E':
                for pair in pairs: 
                    self.edges.append(str2intPair(pair))
            if c == 'V':
                for pair in pairs: 
                    self.vertices.append(str2floatPair(pair))
                    self.pastVertices += [[]]
            self.refresh()

    def getCurrentMatrix(self):        
        T = [[0,0],[0,0]]  
        T[0][0] = eval(self.strFix(self.entry00.get()))
        T[0][1] = eval(self.strFix(self.entry01.get()))
        T[1][0] = eval(self.strFix(self.entry10.get()))
        T[1][1] = eval(self.strFix(self.entry11.get()))
        return T

    def applyTransformation(self):  
        self.applyTransformationT(self.getCurrentMatrix())

    def foo(self):
        print "foo"
            
## private members
    def strFix(self,s):
        if not s or s == "." or s == "-":
            return "0"
        return s 

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

    def drawEdge(self,(x1,y1),(x2,y2), color = 'black'):
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

    def drawEigen(self):
        step = .1
        norm = 70
        alpha = 0 
        tpi = 2*pi
        T = self.getCurrentMatrix()
        while alpha < tpi:
            v = cos(alpha)*norm, sin(alpha)*norm
            self.drawEdge((0,0),v,"gray")
            v1 = self.matrixMul(T ,v)
            self.drawEdge(v,v1,"green")
            alpha += step
        M = numpy.array(T)
        values, eigens = numpy.linalg.eig(M)
        for i in range(2):
            if numpy.iscomplex(values[i]):
                continue
            v = eigens[:,i]
            vv = (float(v[0])*norm,float(v[1])*norm)
            self.drawEdge((0,0), vv , "red")

    def drawStandardBase(self):
        factor =  25 
        (x,y) = self.e1
        self.drawEdge((0,0),(factor*x,factor*y),'red')
        (x,y) = self.e2
        self.drawEdge((0,0),(factor*x,factor*y),'red')

    def matrixMul(self,T,(x,y)):
        return (T[0][0]*x+T[0][1]*y,T[1][0]*x+T[1][1]*y)    

    def drawDeterminant(self):
        step = .1
        t = 0.1
        while t < 1:
            (x,y) = self.mul(t,self.e1)
            (x1,y1) = self.add((x,y),self.e2)
            (x,y) = self.mul(25,(x,y))
            (x1,y1) = self.mul(25,(x1,y1))
            self.drawEdge((x, y), (x1, y1), 'blue')           
            t += step 
    
    def mul(self,k,(x,y)):
        return (k*x,k*y)
    
    def add(self,(x1,y1),(x2,y2)):
        return (x1+x2,y1+y2)    
             
## end of App 

## global functions ## 
def str2intPair(s):
    [x,y] = s.split(',')
    return (int(x[1:]),int(y[:-1]))

def str2floatPair(s):
    [x,y] = s.split(',')
    return (float(x[1:]),float(y[:-1]))

def pairs2str(pairs):
    ret = "" 
    for (x,y) in pairs[:-1]:
        ret += '('+str(x)+','+str(y)+')|' 
    (x,y) = pairs[-1]
    ret += '('+str(x)+','+str(y)+')'
    return ret 

## Main ## 

app = App()

Tkinter.mainloop()
