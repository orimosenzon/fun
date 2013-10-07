import tkinter
import math 
root = tkinter.Tk()

PW = 1000
PH = 700
canvas = tkinter.Canvas(root,width = PW, height = PH, bg = "white")
canvas.pack()


S = 30 # min number of pixels per coordinate (10S is the maximum) 

cx = cy = 0
pcx = PW/2
pcy = PH/2


#mouse
xm = ym = 0

ppu = 40 # pixels per unit # translate units to pixels  

def compute_acu():
    global acu,texp
    texp = math.ceil(math.log10(S)-math.log10(ppu))
    acu = 10 ** texp  # axes coordinate unit
                                                          # acu is calculated so S <= acu * ppu  <=10S 

def formatCor(x):
    if texp >= 0:
        return str(int(x))
    return ('{:.'+str(-texp)+'f}').format(x)
    
compute_acu()

def x2px(x):  # translate x in unit to px in pixels  
    return pcx + (x-cx)*ppu

def y2py(y):  # translate y in unit to py in pixels
    return pcy - (y-cy)*ppu 


def px2x(px):
    return (px-pcx)/ppu + cx

def py2y(py):
    return (pcy-py)/ppu + cy 

    
def drawYaxes():
    px = x2px(0)
    if px<0:
        px = 25
    elif px>PW:
        px = PW
    else:
        canvas.create_line(px, 0, px, PH,activewidth=2)
     
    y = cy - pcy/ppu
    y -= y%acu
    
    py = y2py(y)
    pacu = acu *ppu

    while py > 0:
        if y != 0:
            canvas.create_line(0, py, PW, py, fill = "GRAY",activewidth=2,dash=3)
            canvas.create_line(px,py,px-5,py)
            canvas.create_text(px-15,py,text = formatCor(y))
        y += acu
        py -= pacu 
    
def drawXaxes():
    py = y2py(0)
    if py<0:
        py = 0
    elif py > PH:
        py = PH-15
    else:    
        canvas.create_line(0,py,PW,py,activewidth=2) 

    x = cx - pcx/ppu
    x = x - x%acu
    
    px = x2px(x) 
    pacu = acu * ppu # number of pixes per coordinate
    
    while px < PW:
        if x!=0: 
            canvas.create_line(px, 0, px, PH, fill = "GRAY",activewidth=2,dash=3)
        canvas.create_line(px, py, px, py+5)
        canvas.create_text(px, py+10, text = formatCor(x))

        x += acu
        px += pacu 

def f(x):
    return math.tan(x)

def drawFunc():
    lastPx = None
    px = 0  
    while px<=PW:
        py = y2py( f(px2x(px)) ) 
        if lastPx!=None:
            canvas.create_line(lastPx,lastPy,px,py,fill="BLUE",width=2,smooth=True)
        lastPx = px
        lastPy = py
        px += 1    

def redraw():
    canvas.delete("all")
    
    drawYaxes()
    drawXaxes()

    drawFunc()
    
def butt1Pressed(event):
    global xm,ym
    xm = event.x
    ym = event.y

def butt1Motion(event):
    global cx,cy,xm,ym 
    cx = cx - (event.x-xm)/ppu 
    cy = cy + (event.y-ym)/ppu
    xm = event.x
    ym = event.y

    redraw()
    
def butt1Released(event):
    pass

def butt2Pressed(event):
    scale(1.1,event.x,event.y)
    
def butt3Pressed(event):
    scale(.9,event.x,event.y)
    
def scale(sc,px,py):
    global ppu,acu,cx,cy
    x = px2x(px)
    y = py2y(py)
    cx = x - (x-cx)/sc
    cy = y - (y-cy)/sc
    ppu *= sc
    compute_acu()
    redraw()

canvas.bind('<Button-1>', butt1Pressed)
canvas.bind("<B1-Motion>", butt1Motion)
canvas.bind('<ButtonRelease-1>', butt1Released)

canvas.bind("<Button-2>", butt2Pressed)
canvas.bind("<Button-3>", butt3Pressed)

redraw()
