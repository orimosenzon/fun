import os,re,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from units import V,W
from show import turns
DATES={"oNHE5PFHxgY":"4 במאי 2023","jEhM7AYHXj0":"25 במאי 2023","BIOpgT2fqI8":"26 ביולי 2023",
       "Jlwmd-pTtoI":"5 ביולי 2023","ft5Ke0L1pMc":"11 בספטמבר 2023"}
# names only; every other word stays exactly as the model heard it
FIX=[(r"(?<![א-ת])([שבוול]?)אגר(?![א-ת])",r"\1הגר"),(r"(?<![א-ת])האגר(?![א-ת])","הגר"),
     (r"(?<![א-ת])קגן(?![א-ת])","כגן"),(r"(?<![א-ת])הכגן(?![א-ת])","כגן"),
     (r"(?<![א-ת])גש(?![א-ת])","געש"),(r"במשלה אחת","במושבה אחת"),
     (r"שכונת אומרים","שכונת עומרים"),(r"אותם באומרים","אותם בעומרים"),
     (r"בישי חי","בשי חי"),(r"אייל קבלן","אייל כגן"),
     (r"פרלס-חנה-כרכור","פרדס-חנה-כרכור"),(r"פרדיס חנא קלקוב","פרדס חנה-כרכור")]
def clean(t):
    for a,b in FIX: t=re.sub(a,b,t)
    t=t.replace("~~~","~").replace("''","\"").replace("[[","[").replace("]]","]").replace("{{","{").replace("}}","}").replace("<","").replace(">","")
    return re.sub(r"(^|\s)-(?=[א-ת\d])",r"\1",t).strip()
def ts(v,t):
    h,m,s=int(t//3600),int(t%3600//60),int(t%60)
    lab=f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    return f"[https://www.youtube.com/watch?v={v}&t={int(t)}s {lab}]"
def body(v,name):
    first=name.split()[0]; out=[]
    for sp,t,units in turns(v):
        who="אורי" if sp=="M" else first
        paras=[]; cur=[]; n=0
        for u in units:
            cur.append(u); n+=len(u.split())
            if n>=130 and re.search(r"[.?!]$",u): paras.append(" ".join(cur)); cur=[]; n=0
        if cur: paras.append(" ".join(cur))
        for i,p in enumerate(paras):
            p=clean(p)
            if not p: continue
            head=f"'''{who}:''' " if i==0 else ""
            link=f" <small>({ts(v,t)})</small>" if i==0 and len(p.split())>=10 else ""
            out.append(head+p+link)
    return "\n\n".join(out)
def page(v):
    name=V[v][0]
    return f"""'''תמלול הראיון עם [[{name}]]''', שערך [[משתמש:אורי מוסנזון|אורי מוסנזון]] עם המועמדים לראשות המועצה המקומית פרדס חנה-כרכור לקראת הבחירות המקומיות, ושפורסם ביוטיוב ב-{DATES[v]}.

<youtube width="300" height="180">{v}</youtube>

''התמלול נוצר אוטומטית (תמלול במודל העברי של ivrit.ai, ושיוך הדוברים בעזרת מודל שפה) ולא נבדק מילה במילה מול הסרטון. לכן עלולות להיות בו טעויות, ולעתים משפט משויך לדובר הלא נכון. תוקנו רק שמות שנשמעו לא נכון. הסרטון הוא המקור המחייב, וחותמות הזמן מובילות לרגע המתאים בו. הדברים הם דברי הדוברים כפי שנאמרו בעת הראיון, והם לא אומתו בידי פרדספדיה.''

== תמלול ==
{body(v,name)}

== ראו גם ==
* [[{name}]]
* [[פוליטיקה בפרדס חנה-כרכור]]

== קישורים חיצוניים ==
* [https://www.youtube.com/watch?v={v} הראיון ביוטיוב]

[[קטגוריה:תמלולי ראיונות]]
[[קטגוריה:מועצה מקומית פרדס חנה כרכור]]
[[קטגוריה:תיעוד מקומי]]
"""
if __name__=="__main__":
    os.makedirs(f"{W}/out",exist_ok=True)
    for v in V:
        t=page(v); open(f"{W}/out/{v}.wiki","w",encoding="utf8").write(t)
        print(V[v][0],len(t),"chars",t.count("'''אורי:'''"),"אורי turns",t.count("youtube.com/watch")-1,"links")
