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

import sympy 
import Tkinter


root = Tkinter.Tk()

scrollbar = Tkinter.Scrollbar(root)
scrollbar.pack(side=Tkinter.RIGHT, fill=Tkinter.Y)

canvas = Tkinter.Canvas(root, yscrollcommand=scrollbar.set, width=400, height=200, bg='white')
canvas.pack(expand=Tkinter.YES, fill=Tkinter.BOTH)


def tkInit():
    global root, canvas


num_size = 20 
x_offset = 30 
y_offset = 20 

scroll = 0 

def printMat():
    global scroll
    print '\n#########'
    for i in range(m):
        for j in range(n):
            canvas.create_text((x_offset+j*num_size,y_offset+scroll*4*num_size+i*num_size), text = str(mat[i][j]))
            #canvas.insert(Tkinter.END, "hi")
            print mat[i][j],
        print 
    print '#########'
    scroll += 1

def str1(i):
    return str(i+1)

def findLineSwitch(i):
    # not complete.. a complete solution seems to involve a search 
    # over the possible alternatives. 
    # There should be an example that will cause problems.. 
    for ii in range(i+1,m):
        if mat[ii][i] != 0:
            print '\n --> Swap S'+str1(i)+' S'+str1(ii)
            for j in range(i,n):
                tmp = mat[ii][j]
                mat[ii][j] = mat[i][j]
                mat[i][j] = tmp


def pivot2one(i):
    k = mat[i][i]

    print '\n --> S'+str1(i)+'*'+str(1/k)  

    for j in range(i,n):
        mat[i][j] /= k 


def zeroPivot(i,ii):
    global mat
    k = mat[ii][i]
    
    st = '\n --> S'+str1(ii)+'-'
    
    if k != 1:
        st += str(k)+'*'

    st += 'S'+ str1(i)
    
    print st 

    for j in range(i,n):
        mat[ii][j] -= k*mat[i][j] 

f=open('m1.matrix', 'r')
lines = f.read().split('\n')
f.close()

mat = [] 
for line in lines:
    nums =  line.split()
    mat += [ map(lambda x:sympy.Rational(x),nums)]

m = len(mat)
n = len(mat[0])

printMat()

for i in range(m):
    k = mat[i][i]
    if k == 0:
        findLineSwitch(i)
        printMat()
    if k != 1:
        pivot2one(i)
        printMat()
    for ii in range(m):
        if ii == i or mat[ii][i] == 0: 
            continue 
        zeroPivot(i,ii)
        printMat()

canvas.pack(side=Tkinter.LEFT, fill=Tkinter.BOTH)


scrollbar.config(command=canvas.yview)

tkInit()
Tkinter.mainloop()
    



