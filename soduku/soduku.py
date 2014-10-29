WIDTH = 500
HEIGHT = 600

import tkinter
canvas = tkinter.Canvas(width=WIDTH, height=HEIGHT, background="white")
canvas.pack()

# == globals == 
board = []
raw_board = []

xOffset = 20
yOffset = 120 
size = 50

inEdit = False 
xc = 0
yc = 0

# == functions ==
def butt1Pressed(e):
    global inEdit,xc,yc
    x = (e.x - xOffset)//size

    if inEdit:
        y = (e.y - 10)//size
        if y ==0 and x < 9:
            set_value(yc,xc,x+1)
            draw_board()
            inEdit = False 
    else:
        y = (e.y - yOffset)//size
        if 0 <= x<9 and 0 <= y < 9:
            yc = y
            xc = x
            inEdit = True
            canvas.create_rectangle(xOffset+x*size+1,yOffset+y*size+1,xOffset+(x+1)*size,yOffset+(y+1)*size,fill="yellow",outline="")

def keyPressed(event):
    global inEdit,raw_board
    if inEdit and ('1' <= event.char <= '9' or event.char==" "):
        if event.char==" ":
            raw_board[yc][xc]=0
            recalc()
        else:
            v = int(event.char)
            if raw_board[yc][xc] > 0:
                raw_board[yc][xc] = v
                recalc()
            else:
                set_value(yc,xc,v)

        draw_board()
        inEdit = False 
    
canvas.bind('<Button-1>', butt1Pressed)
canvas.bind_all('<Key>',keyPressed)

 
def draw_board():
    canvas.delete("all")
    for k in range(9):
        canvas.create_rectangle(xOffset+k*size,10,xOffset+(k+1)*size,10+size,outline="black",width=3,fill="yellow")
        canvas.create_text(xOffset+k*size+size//2,10+size//2,text = str(k+1))
                           
    canvas.create_rectangle(xOffset,yOffset,xOffset+size*9,yOffset+size*9,outline="black",width=3)
    for k in range(9):
        if k%3 == 0:
            w = 3
        else:
            w = 1
        canvas.create_line(xOffset+size*k,yOffset,xOffset+size*k,yOffset+size*9,width=w)
        canvas.create_line(xOffset,yOffset+size*k,xOffset+size*9,yOffset+size*k,width=w)

    for i in range(9):
        for j in range(9):
            if board[i][j][0] > 0:
                if raw_board[i][j] >0:
                    c = "red"
                else:
                    c = "black"
                canvas.create_text(xOffset+j*size+size//2,yOffset+i*size+size//2,text=str(board[i][j][0]),fill=c,font=("Purisa",17,"bold"))

def init_raw_board():
    global raw_board
    for i in range(9):
        raw_board += [[0]*9]

def init_board():
    global board
    board = [] 
    for i in range(9):
        line = []
        for j in range(9):
            line += [[0]+[True]*9+[9]]
        board += [line]

def init():
    init_raw_board()
    init_board()
    
def print_board():
    for i in range(9):
        for j in range(9):
            if board[i][j][0] >0:
                s = str(board[i][j][0])
            else:
                s = ""
                for k in range(1,10):
                    if board[i][j][k]:
                        s+=str(k)
            print(s,end="\t")
        print("")



def set_value(i,j,v):
    global raw_board
    raw_board[i][j] = v  
    set_value_inter(i,j,v)

def set_value_inter(i,j,v):
    global board
    board[i][j][0] = v
    board[i][j][-1] = 1
    prop_value(i,j,v)

def delete_value(i,j,v):
    global board
    
    if not board[i][j][v]:
        return
    
    board[i][j][v] = False
    board[i][j][-1] -=1

    if board[i][j][-1] == 1:
        for k in range(1,10):
            if board[i][j][k]:
                v = k
        board[i][j][0] = v
        prop_value(i,j,v)

    prop_line(i)
    prop_raw(j)
    prop_square(i,j)
        
def prop_value(i,j,v):
    for k in range(9):
        if k != i:
            delete_value(k,j,v)
        if k != j:
            delete_value(i,k,v)
    i1 = i - i%3  
    j1 = j - j%3 
    for y in range(3):
        for x in range(3):
            pi = i1 + y
            pj = j1 + x
            if pi != i or pj !=j:
                delete_value(pi,pj,v)

def prop_line(i):
    for k in range(1,10):
        s = 0
        for j in range(9):
            if board[i][j][k]:
                s +=1
                if s>1:
                    break
                j1 = j
        if s == 1:
            set_value_inter(i,j1,k)

def prop_raw(j):
    for k in range(1,10):
        s = 0
        for i in range(9):
            if board[i][j][k]:
                s +=1
                if s>1:
                    break
                i1 = i
        if s == 1:
            set_value_inter(i1,j,k)

def prop_square(i,j):
    ii = i - i%3
    jj = j - j%3 
    for k in range(1,10):
        s = 0
        for y in range(3):
            if s>1:
                break
            for x in range(3):
                pi = ii + y
                pj = jj + x
                if board[pi][pj][k]:
                    s+=1
                    i1 = pi
                    j1 = pj 
        if s == 1:
            set_value_inter(i1,j1,k)
            
def recalc():
    init_board()
    for i in range(9):
        for j in range(9):
            v = raw_board[i][j]
            if v > 0:
                set_value(i,j,v)
            
init()

#recalc... using raw_board.. delete an entry..
# prop using needed one..


#print_board()
draw_board()
