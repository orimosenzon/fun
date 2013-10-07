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

import visual
import random
import sympy 

epsilone = 1e-7 
bigEpsilone = 1e-1
showSpan = True 

def find_index(obj,objs):
    i = 0 
    for o in objs:
        if o == obj:
            return i
        i +=1
    return None  


is_span_positive = True

current_span = visual.curve()

span_frame = visual.frame() 

span_color = visual.color.yellow
span_fact = 1.2


def update_convex():
    if not showSpan:
        return 
    global current_span
    current_span.visible = 0 

    poss = [orig] 
    sum = visual.vector(orig)

    vecs = map(lambda i:vectors[i].axis * span_fact, selected)

    poss = all_conv_comb(vecs)

    current_span = visual.convex(pos=poss, color = span_color)

def all_conv_comb(vecs):
    v0 = visual.vector(epsilone,epsilone,epsilone) # epsilone due to a bug in visual.convex

    if len(vecs) == 0:
        return [v0] 

    rec = all_conv_comb(vecs[:-1])

    v1 = vecs[-1] 

    ret = [] 

    if is_span_positive:
        vv = v0
    else:
        vv = -v1

    for s in rec:
        ret += [s+v1,s+vv]

    return ret 


def delete_current_grid(): 
    global span_frame
    for o in span_frame.objects:
        o.visible = 0 

def update_grid():
    delete_current_grid()

    if not showSpan:
        return 

    global span_frame

    span_frame = visual.frame() 

    vecs = map(lambda i: visual.vector(vectors[i].axis)*span_fact,selected)
    for v in vecs:
        rest = vecs[0:len(vecs)] # deep copy  
        rest.remove(v)
        all_sums = all_sums_comb(rest)
        for aSum in all_sums:
            visual.curve(frame = span_frame, pos = [aSum,aSum+v], color = span_color)
            if not is_span_positive:
                visual.curve(frame = span_frame, pos = [aSum,aSum-v], color = span_color)

def all_sums_comb(vecs):
    vecs_segs = [] 
    res = 2 
    for v in vecs:
        norm = visual.mag(v)
        v1 = v*(res/norm)
        segs = [] 
        nn = int(norm/res)
        start = 0 
        if not is_span_positive:
            start = -nn
        for i in range(start,nn):
            segs += [v1*i]
        vecs_segs += [segs]
    cartecians = ordered_cartesian(vecs_segs)
    sums = [] 
    for c in cartecians:
        sum = visual.vector(orig)
        for v in c:
            sum += v 
        sums += [sum] 
    return sums     

def ordered_cartesian(list_of_lists):
    if not list_of_lists:
        return [[]] 
    aList = list_of_lists.pop()
    rec = ordered_cartesian(list_of_lists)
    ret = [] 
    for aCart in rec:
        for element in aList:
            ret += [aCart + [element]]
    return ret 


def is_selected(i):
    return vectors[i].color == select_color

def arrow_fact(v):
    return (v.length - v.headlength )/v.length

def compute_decomposition(fixed,v):
    v = sympy.Matrix(list(v.axis))
    v = decomposition_matrix * v 
    c = 0 
    for j in fixed:
        tmp = vectors[j].axis * float(v[c])
        extensions[j].axis = tmp 
        c += 1 

    if len(fixed) == 1:
        return

    showOnlyTwo = False 

    i0 = fixed[0]    
    i1 = fixed[1]    

    if len(fixed) == 2: 
        if abs(v[2]) >= bigEpsilone:
            pillars[i0].visible = 0 
            pillars[i1].visible = 0 
            return
        pillars[i0].visible = 1 
        pillars[i1].visible = 1 
        showOnlyTwo = True
    else:    
        i2 = fixed[2]    


    exst0 = extensions[i0].axis
    exst1 = extensions[i1].axis

    pillars[i0].pos =  [exst1,exst0 + exst1]
    pillars[i1].pos =  [exst0,exst0 + exst1]

    if not showOnlyTwo:
        exst2 = extensions[i2].axis
        pillars[i2].pos =  [exst0 + exst1, exst0 + exst1 + exst2] 

def compute_decomposition_matrix(fixed):
    global decomposition_matrix

    m = sympy.zeros((3,3)) # create matrix for the extnesions 
    c = 0 
    for i in fixed:
        m[:,c] = list(vectors[i].axis)
        c += 1 
    if len(fixed) < 3:   
        if len(fixed) == 1:
            v1,v2 = complete_1_to_base(m[:,0])
            m[:,1] = v1
            m[:,2] = v2
        else:
            v2 = complete_2_to_base(m[:,0],m[:,1])
            m[:,2] = v2

    decomposition_matrix = m.inv() 

def complete_1_to_base(v):
    if v[0] != 0 or v[1] !=0:
        v1 = sympy.Matrix([-v[1],v[0],0])
    else:
        if v[2] == 0:
            return None
        v1 = sympy.Matrix([0,v[2],0])

    v2 = complete_2_to_base(v,v1)
    v1 = v1/v1.norm()
    return (v1,v2)

def complete_2_to_base(v0,v1):
    v2 = (v0.cross(v1)).T
    return v2/v2.norm()

def zero_pillars():
    for i in range(N): 
        pillars[i].pos = [orig,orig] 


def reset():
    global selected,fixed,decomposition_matrix,decomposedVector

    for i in range(N):
        vectors[i].axis = initial_places_list[i]
        v = vectors[i]
        cylinders[i].axis = v.axis * arrow_fact(v)
        vectors[i].color = colors[i % colors_num]
        vectors[i].opacity = 1 
        pillars[i].visible = 0 
        extensions[i].visible = 0 

    fixed = []
    selected = [] 
    
    decomposition_matrix = None
    decomposedVector = None
    current_span.visible = 0 
    visual.scene.forward = (0,0,-1)  

## main ##

print """ 
Change view angle - Right drag 

Zoom in out       - Middle drag 

Select            - Left click on vector 

Move vctor        - Ledt drag a vector 

Grid span view    - 'g' on keyboard 

Convex span view  - 'c' on keyboard  

Full span         - Up arrow on keyboard 

Only positive span- Down arrow on keyboard

reset scene       - 'r' on keyboard

toggle span view  - 's' on keyboard
"""

span_vis = update_grid


orig = (0,0,0)
x_axis = (10,0,0)
y_axis = (0,10,0)
z_axis = (0,0,10)

rad = .1

select_color = visual.color.cyan
axes_color = visual.color.green
fixed_color = axes_color
extension_color = visual.color.white

visual.cylinder(pos=orig, axis=x_axis, radius=rad, color = axes_color)
visual.cylinder(pos=orig, axis=y_axis, radius=rad, color = axes_color)
visual.cylinder(pos=orig, axis=z_axis, radius=rad, color = axes_color)

visual.label(pos=x_axis, text='X')
visual.label(pos=y_axis, text='Y')
visual.label(pos=z_axis, text='Z')

colors = [visual.color.red,visual.color.blue,visual.color.yellow,visual.color.orange,visual.color.white]
colors_num = len(colors)


vectors = []
extensions = [] 
cylinders = [] 
pillars = [] 

selected = [] # indexes of vectors
fixed = [] 


N = 4 
sw = .6
cyn_fact = .9


initial_places_list = [(7,0,0),(0,7,0),(0,0,7),(9,3,10.5),(-7,-7,-7)]

for i in range(N):
    ax = initial_places_list[i]
    # ax = (random.randint(-10,10), random.randint(-10,10), random.randint(-10,10))

    v = visual.arrow(pos = orig, axis = ax, shaftwidth = sw, fixedwidth = 1, color = colors[i % colors_num])
    
    ex = visual.arrow(pos = orig, axis = visual.vector(ax)*0, shaftwidth = .2, fixedwidth = 1, color = visual.color.white, visible = 0)

    
    v_cyn = visual.cylinder(pos = orig, axis = v.axis * arrow_fact(v) , radius = sw/2, color = colors[i % colors_num]) 
    
    pillar = visual.curve(color = visual.color.white, visible = 0)

    vectors += [v]
    extensions += [ex]
    cylinders += [v_cyn]
    pillars += [pillar] 

dragged = None 

decomposition_matrix = None
decomposedVector = None

while True:

    if dragged != None:
        v = vectors[dragged]
        v.axis = visual.scene.mouse.pos
        cylinders[dragged].axis = visual.vector(v.axis)*arrow_fact(v)

        if decomposedVector: 
            if dragged in fixed:
                compute_decomposition_matrix(fixed) 
            compute_decomposition(fixed,decomposedVector)

    if visual.scene.kb.keys:

        k = visual.scene.kb.getkey() 

        current_span.visible = 0

        if k == 'g':
            span_vis = update_grid
        elif k == 'c':
            delete_current_grid()
            span_vis = update_convex
        elif k=='down': 
            delete_current_grid()
            is_span_positive = True
        elif k=='up': 
            delete_current_grid()
            is_span_positive = False 
        elif k=='x': 
            if decomposition_matrix:
                for i in fixed:
                    vectors[i].color = cylinders[i].color 
                    pillars[i].visible = 0 
                    extensions[i].visible = 0 
                fixed = []     
                decomposition_matrix = None
                decomposedVector = None
            else:
                if not selected or len(selected) > 3:
                    continue 
                fixed = selected
                selected = []
                zero_pillars()

                for i in fixed:
                    vectors[i].color = fixed_color
                    extensions[i].axis = vectors[i].axis
                    extensions[i].visible = 1 
                    pillars[i].visible = 1 

                compute_decomposition_matrix(fixed)  
        elif k=='r':
            reset()
        elif k=='s':
            showSpan = not showSpan

        span_vis()


    if not visual.scene.mouse.events:
        continue

    e = visual.scene.mouse.getevent()



    if e.drag:
        if dragged == None:
            dragged = find_index(e.pick,cylinders)
            if dragged == None:
                continue

    elif e.drop:
        if dragged != None:
            if is_selected(dragged):
                span_vis()
            dragged = None

    elif e.click:
        i = find_index(e.pick,cylinders)
        if i != None:
            if is_selected(i):
                vectors[i].color = colors[i % colors_num]
                selected.remove(i)
            else:
                selected += [i]
                vectors[i].color = select_color
                if decomposition_matrix:
                    decomposedVector = vectors[i] 
                    compute_decomposition(fixed,decomposedVector)

            span_vis()
