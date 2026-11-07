"""Apply the reviewed correction lists to the fused transcript and build the page."""
import json,os,re,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from units import V,W
from build import clean,ts,DATES
v="ft5Ke0L1pMc"; name=V[v][0]
T=json.load(open(f"{W}/merged_{v}.json",encoding="utf8"))
text={x["n"]:" ".join(w for w,_ in x["w"] if w) for x in T}
GLOBAL=[("המיתאר","המתאר"),("קאגן","כגן"),("מיתאר","מתאר"),("פרדיס חנא קלקוב","פרדס חנה-כרכור")]
ops=miss=0
for f in("edits1.txt","edits2.txt","edits3.txt"):
    for ln in open(f"{W}/{f}",encoding="utf8"):
        ln=ln.rstrip("\n")
        if not ln: continue
        n,old,new=ln.split("|",2); n=int(n); new=new.rstrip("|")
        ops+=1
        if old=="^": text[n]=new+text[n]; continue
        if old not in text[n]:
            miss+=1; print("MISS",n,old[:70]); continue
        text[n]=text[n].replace(old,new)
out=[]
for i,x in enumerate(T):
    s=text[x["n"]]
    for a,b in GLOBAL: s=s.replace(a,b)
    nxt=T[i+1]["t"] if i+1<len(T) else x["t"]+5
    parts=re.split(r"¶([MR]) ?",s); sp=x["sp"]; seq=[(sp,parts[0])]+[(parts[j],parts[j+1]) for j in range(1,len(parts),2)]
    total=sum(len(p) for _,p in seq) or 1; pos=0
    for spk,p in seq:
        p=p.strip()
        if p:
            t=x["t"]+(nxt-x["t"])*pos/total if nxt>x["t"] else x["t"]
            if out and out[-1][0]==spk: out[-1][2]+=" "+p
            else: out.append([spk,t,p])
        pos+=len(p)
print("ops",ops,"miss",miss,"turns",len(out),"unclear",sum(p.count("[לא ברור]") for _,_,p in out))
bad=re.compile(r"פרלס|פרדיס|כרכוף|כרכוב|חנא|קרקור|המיתאר|קאגן|המוטשבה|נרפאת|(?<![א-ת])אגר(?![א-ת])")
for s,t,p in out:
    for m in bad.finditer(p): print("LEFT",int(t//60),p[max(0,m.start()-25):m.end()+25])
paras=[]
for sp,t,p in out:
    who="אורי" if sp=="M" else name.split()[0]
    words=p.split(); chunks=[]; cur=[]
    for w in words:
        cur.append(w)
        if len(cur)>=130 and re.search(r"[.?!]$",w): chunks.append(" ".join(cur)); cur=[]
    if cur: chunks.append(" ".join(cur))
    for i,c in enumerate(chunks):
        c=clean(c)
        if not c: continue
        paras.append((f"'''{who}:''' " if i==0 else "")+c+(f" <small>({ts(v,t)})</small>" if i==0 and len(c.split())>=10 else ""))
page=open(f"{W}/out/{v}.wiki",encoding="utf8").read()
a=page.index("== תמלול ==\n")+len("== תמלול ==\n"); b=page.index("\n== ראו גם ==")
page=page[:a]+"\n\n".join(paras)+"\n"+page[b:]
note=re.search(r"''התמלול נוצר אוטומטית.*?''\n",page,re.S).group(0)
page=page.replace(note,"''התמלול נוצר אוטומטית בשני מודלים עבריים של ivrit.ai, ושיוך הדוברים נעשה בעזרת מודל שפה. אחר כך הוגה בקריאה ותוקן לפי ההקשר, בלי האזנה לסרטון. לכן עדיין עלולות להיות בו טעויות, ולעתים משפט משויך לדובר הלא נכון. '''[לא ברור]''' מסמן מקום שבו לא ניתן היה להבין את הנאמר. הסרטון הוא המקור המחייב, וחותמות הזמן מובילות לרגע המתאים בו. הדברים הם דברי הדוברים כפי שנאמרו בעת הראיון, והם לא אומתו בידי פרדספדיה.''\n")
open(f"{W}/out/{v}.v2.wiki","w",encoding="utf8").write(page)
print(len(page),"chars",len(paras),"paragraphs")
