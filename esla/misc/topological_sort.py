from random import random as rnd 

marks = []  # colors of nodes of the DAG. this is a global variable
            # should be a member variable of an object in larger projects 

M_UNMARKED, M_TMP, M_MARKED = 0,1,2  # possible colors  

# That's the heart of the algorithm, a version of DFS with colors 
def visit(n, dag):
    global marks;  
    marks[n] = M_TMP
    lst = [] 
    for m in dag[n]:
        if marks[m] == M_TMP:
            print("Error: your graph contains a circle (%d is part of it)" % m)
            continue
        if marks[m] == M_UNMARKED:
            lst += visit(m,dag) # * order freedom degree
    marks[n] = M_MARKED
    return lst+[n]


def topological_sort(dag):
    global marks;
    N = len(dag)
    marks = [M_UNMARKED]*N
    lst = [] 
    for i in range(N-1,-1,-1):  # ** order freedom degree
        if marks[i] == M_UNMARKED:
            lst += visit(i,dag) 
    return lst

## specific checks 
#print( topological_sort([[3,2],[3],[3],[]]) )
#print( topological_sort([[],[0],[1],[2]]) )
#print( topological_sort([[],[0],[0],[1,2]]) )
#print( topological_sort([[],[0],[],[1,2],[0,3]]) )

## verify code ##
df = 0.5 # dencity factor, a number in [0,1]. 0 - sparce, 1 - dence
N = 10   # number of nodes in dag
LOOPS = 70 # number of random graph to check 
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

        print("Check",c+1,":") 
        printDag(dag)
        print("Sort: %s \n *******\n\n" % lst)

        # check for consistency of the sort with the DAG (exhaustive) 
        idxs = [0]*N
        for i in range(N):
            idxs[lst[i]] = i 
        for i in range(N):
            for j in range(i):
                if idxs[i] < idxs[j]:
                    if j in dag[i]: #maybe the other way around..
                        print("Sort bug:")
                        printDag(dag)
                        print("Sort: %s \n indexes:%d,%d \n %s" %(lst,i,j,idxs))
                        return
                
def printDag(dag):
    print("Dag - ",end="")
    for i in range(len(dag)):
        print("%d:%s"% (i,dag[i]), end=", ")
    print() 

verify() 
    


