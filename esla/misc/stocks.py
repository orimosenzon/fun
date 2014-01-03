import random

def bestBuy(array):
    isell = imin = ibuy = 0
    for i in range(len(array)):
        if array[i] < array[imin]:
            imin = i         

        if array[i] - array[imin] > array[isell] - array[ibuy]:
            isell = i
            ibuy = imin
    return (ibuy,isell)


def bestBuy1(array):
    n = len(array)-1
    isell = imax = ibuy = n
    for i in range(n,-1,-1):
        if array[i] > array[imax]:
            imax = i
        if array[imax]-array[i] > array[isell]-array[ibuy]:
            isell = imax
            ibuy = i
    
    return (ibuy,isell)

def randomList(n,m):
    l = []
    for i in range(n):
        l += [int(random.random()*m)]
    return l

def printBuySell(buysell,lst):
    ibuy,isell = buysell
    for i in range(len(lst)):
        if i == ibuy or i == isell:
            print("*"+str(lst[i])+"*",end=",")
        else: 
            print(" "+str(lst[i])+" ",end=",")

    print(" value:", lst[isell]-lst[ibuy])

for i in range(20):
    lst = randomList(20,100)
    a1 = bestBuy(lst)
    printBuySell(a1,lst)
    a2 = bestBuy1(lst)
    printBuySell(a2,lst)


