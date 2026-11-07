import json,glob,os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from units import units,V,W
DASH={"jEhM7AYHXj0","Jlwmd-pTtoI"}
def turns(v):
    U=units(v); ch=[]
    for f in sorted(glob.glob(f"{W}/lab/{v}_*.json"),key=lambda p:int(p.rsplit('_',1)[1][:-5])): ch+=json.load(open(f))
    ch=dict(ch); cur=None; T=[]; dash=v in DASH
    for i,u in enumerate(U):
        if i in ch: sp=ch[i]
        elif dash and u["text"].startswith("-"): sp="R" if cur=="M" else "M"  # subtitle-style dash = speaker change
        else: sp=cur
        if sp!=cur:
            cur=sp; T.append([cur,u["t"],[]])
        T[-1][2].append(u["text"])
    return T
if __name__=="__main__":
    v,a,b=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
    T=turns(v); print(len(T),"turns")
    for s,t,x in T:
        if a*60<=t<=b*60: print(f"{int(t//60)}:{int(t%60):02d} {'אורי' if s=='M' else 'מרואיין'}: {' '.join(x)[:330]}")
