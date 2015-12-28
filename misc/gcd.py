def gcd(a,b):
    if a>b:
        return gcd_rec(a,b)
    else:
        (d,x,y) = gcd_rec(b,a) 
        return (d,y,x)

def gcd_rec(a,b):
    if b==0:
        return (a,1,0)
    q = a//b
    r = a%b 
    (d,x,y) = gcd_rec(b,r)
    return (d,y,x-q*y)

def print_gcd(a,b):
    (d,x,y) = gcd(a,b)
    print(d,"=",x,"*",a,"+",y,"*",b)
    
