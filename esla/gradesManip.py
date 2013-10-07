## configuration section ##

# set the weight of each exercise (also determine number of exercises)  
weights = [1, 1, 1, 1, 1, 1]

#path to read csv file (after exporting it from moodle to open office and
# saving it as .csv by open office)  
readFileName = '/home/ori/tmp/tmp.csv'

# output file 
writeFileName = '/home/ori/tmp/tmp1.csv'

# how many grades to ignore 
ignorNum = 1 

### end of configuration section ## 

def produceFormulaList(n):
    ret = [] 
    for i in range(N):
        ret += [str(weights[i])+"*"+letters[i]+str(n)]
    return ret 


N = len(weights)

letters = ['G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

f = open(readFileName)
s = f.read()
lines = s.split('\n')

s = lines[0]+'\n' 
for i in range(1,len(lines)):
    line = lines[i]
    if line == '':
        continue 
    words = line.split(',')
    pref = words[:6]
    grades = words[6:] 
    grades = list(map(lambda x: x=='"-"' and '0' or x, grades))

    nums = list(map(float,grades))   

    formula = produceFormulaList(i+1)

    nums_idx = list(zip(nums,range(N)))
    nums_idx.sort() #sorting by nums (cause tuples comparison is lexicographic) 
    _,idx = zip(*nums_idx)
    idx = list(idx)
    idx = idx[:ignorNum] # now idx contain 'ignorNum' indexes of the worst grades 

    weights1 = list(weights)
    for j in idx:
        weights1[j] = formula[j] = 'del'
                 
    fil = lambda x: x != 'del'
    weights1 = list(filter(fil,weights1))
    formula  = list(filter(fil,formula))

    formula = '+'.join(formula)
    sumW = sum(weights1)
    formula = '=('+formula+')'+'/'+str(sumW)
    newLine = pref + grades + [formula]

    newLine = ','.join(newLine)
    s+=newLine+'\n'

f = open(writeFileName,'w')
f.write(s)
f.close()


######################



