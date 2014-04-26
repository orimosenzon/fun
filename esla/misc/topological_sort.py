from random import random as rnd 

marks = []  #this is a global variable
            # should be a member variable of an object in larger projects 

M_UNMARKED, M_TMP, M_MARKED = 0,1,2 

def visit(lst, n, dag, marks):
    global marks;  
    lst += [n]
    marks[n] = M_TMP
    for m in dag[n]:
        if marks[m] == M_TMP:
            print("error, your graph contains a circle")
            continue
        if marks[m] == M_UNMARKED:
            lst += visit(lst, m,dag,marks)
    marks[n] = M_MARKED
    return lst


def topological_sort(dag):
    global marks;
    N = len(dag)
    marks = [M_UNMARKED]*N
    lst = [] 
    for i in range(N):
        if marks[i] == M_UNMARKED:
            lst += visit([],i,dag,marks)
            break
    print(lst)

topological_sort([[],[0],[0],[1,2]])



## verify code ##
df = 0.5 # dencity factor, a number in [0,1]. 0 - sparce, 1 - dence
N = 10   # number of nodes in dag
LOOPS = 10 # number of random graph to check 
def verify():
    for c in range(LOOPS):
        #produce a random DAG
        dag = []
        for i in range(N):
            ngbrs = [] 
            for j in range(i):
                if rnd() < df:
                    ngbrs += [j]
            dag += [ngbrs]
        # sort it 
        lst  = topological_sort(dag)

        # check for consistency of the sort with the DAG (exhaustive) 
        idxs = [0]*N
        for i in range(N):
            idxs[lst[i]] = i 
        for i in range(N):
            for j in range(i):
                if idxs[i] < idxs[j]:
                    if j in dag[i]: #maybe the other way around.. 
                        print("found a bug: in dag: %s \n sort: %s \n indexes:%d,%d" %(dag,lst,i,j))
                        return
                

    


