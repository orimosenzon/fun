"""Fuse the strong pass (B) with the first pass (A): B is the base text, speakers come from A
through word alignment, and stretches only A heard are kept."""
import difflib,json,os,re,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from units import V,W
from show import turns
v=sys.argv[1] if len(sys.argv)>1 else "ft5Ke0L1pMc"
norm=lambda w:re.sub(r"[^\wא-ת]","",w)
A=[]
for sp,t,us in turns(v):
    for w in " ".join(us).split(): A.append((w,sp,t))
B=[]
for s in json.load(open(f"{W}/tr2/{v}.json",encoding="utf8"))["segments"]:
    for w in s["text"].replace("‫","").split(): B.append((w,s["start"]))
sm=difflib.SequenceMatcher(None,[norm(a[0]) for a in A],[norm(b[0]) for b in B],autojunk=False)
M=[]  # (word, speaker, time, alt) ; alt marks disagreement for the reader
sp=A[0][1]
for op,i1,i2,j1,j2 in sm.get_opcodes():
    if op=="equal":
        for i,j in zip(range(i1,i2),range(j1,j2)):
            sp=A[i][1]; M.append([B[j][0],sp,B[j][1],None])
    elif op=="replace":
        n=j2-j1
        for k,j in enumerate(range(j1,j2)):
            sp=A[i1+k*(i2-i1)//n][1]
            M.append([B[j][0],sp,B[j][1]," ".join(a[0] for a in A[i1:i2]) if k==0 else ""])
    elif op=="insert":
        for j in range(j1,j2): M.append([B[j][0],sp,B[j][1],""])
    else:
        if i2-i1>=3:
            for k,i in enumerate(range(i1,i2)):
                sp=A[i][1]; M.append([A[i][0],sp,A[i][2],"+A" if k==0 else "+"])
        else:
            M.append(["",A[i1][1],A[i1][2],"-"+" ".join(a[0] for a in A[i1:i2])])
T=[]
for w,s,t,alt in M:
    if not T or (T[-1]["sp"]!=s and w): T.append({"sp":s,"t":t,"w":[]})
    T[-1]["w"].append([w,alt])
for k,x in enumerate(T): x["n"]=k
json.dump(T,open(f"{W}/merged_{v}.json","w",encoding="utf8"),ensure_ascii=False)
# reading copy: {B‖A} where the models differ, ⟪..⟫ only A heard, ⟦-..⟧ short drop
L=[]
for x in T:
    o=[];i=0;ws=x["w"]
    while i<len(ws):
        w,alt=ws[i]
        if alt is None: o.append(w); i+=1
        elif alt.startswith("-"): o.append("⟦"+alt+"⟧"); i+=1
        elif alt=="+A" or alt=="+":
            r=[]
            while i<len(ws) and ws[i][1] in("+A","+"): r.append(ws[i][0]); i+=1
            o.append("⟪"+" ".join(r)+"⟫")
        else:
            r=[w];a=alt;i+=1
            while i<len(ws) and ws[i][1]=="": r.append(ws[i][0]); i+=1
            o.append("{"+" ".join(r)+("‖"+a if a else "")+"}")
    m=int(x["t"]//60);L.append(f'#{x["n"]} {m}:{int(x["t"]%60):02d} {"אורי" if x["sp"]=="M" else "כגן"}: '+" ".join(o))
open(f"{W}/read_{v}.txt","w",encoding="utf8").write("\n".join(L))
print(len(T),"turns",sum(len(l.split()) for l in L),"words in reading copy",sum(len(l) for l in L),"chars",len(L),"lines")
