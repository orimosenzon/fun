def isr2inter(s):
    i = s.find('-')
    if i!= -1:
        s = s[:i]+s[i+1:]
    return '972'+s[1:]


f = open("parents-mobile.csv")
s = f.read()
f.close()
lines = s.split('\n')
olines = ""
for line in lines:
    words = line.split(",")
    owords = ""
    for num in words:
        if(num):
            owords += isr2inter(num)+','
    olines += owords+'\n'

of = open("parents-mobile-international.csv","w")
of.write(olines)
of.close()
