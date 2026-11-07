"""Speaker marking only: the model returns unit numbers, never text."""
import json,os,sys,time
sys.path.insert(0,'/home/ori/.config/ai'); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import ai_azure
from units import units,V,W
MODEL=os.environ.get("LABEL_MODEL","gpt-5.4-mini")
CH=200; CTX=20
SYS="""אתה מסמן מי מדבר בתמלול אוטומטי של ראיון וידאו בעברית. שני דוברים בלבד:
M = המראיין, אורי, תושב שמראיין מועמדים לראשות המועצה. הוא שואל, מגיב קצרות ("כן", "הבנתי", "אוקיי"), מתקן ומאתגר.
R = המרואיין, {name}. הוא מספר על עצמו, עונה ומסביר בהרחבה.

הטקסט מחולק ליחידות ממוספרות. לצד כל יחידה מופיע רמז מזיהוי קולי אוטומטי, שהוא רועש: הוא טועה לעתים קרובות, בעיקר בחילופי דברים מהירים ובתחילת תור. סמוך קודם כל על התוכן וההיגיון של השיחה: שאלה ואחריה תשובה, פנייה בשם ("שלום, אורי" נאמר בידי R), גוף ראשון על הביוגרפיה של המרואיין. מקף בתחילת יחידה כמעט תמיד מסמן החלפת דובר. בדוק במיוחד: מי שמבקש "תספר", "מה הרקע שלך", "השאלה היא" הוא M, גם אם הרמז אומר R, ותשובה שמתחילה אחרי שאלה היא של הדובר השני. משפט אחד לא מתחלק בין שני דוברים, אלא אם אחד קוטע את השני בבירור.
אל תכתוב שום טקסט מהתמלול. החזר JSON בלבד: {{"changes": [[מספר_יחידה, "M" או "R"], ...]}}
כל רשומה אומרת שמהיחידה הזאת ואילך מדבר הדובר הזה, עד הרשומה הבאה. הרשומה הראשונה חייבת להיות היחידה הראשונה שסומנה לתיוג. יחידות ההקשר שלפניה (מסומנות בהקשר) לא מתויגות."""
def main():
    only=set(sys.argv[1:]); cl=ai_azure.client(); tin=tout=0
    os.makedirs(f"{W}/lab",exist_ok=True)
    for v in V:
        if only and v not in only: continue
        U=units(v); name=V[v][0]
        for s in range(0,len(U),CH):
            p=f"{W}/lab/{v}_{s}.json"
            if os.path.exists(p): continue
            e=min(s+CH,len(U)); lines=[]
            prev=json.load(open(f"{W}/lab/{v}_{s-CH}.json")) if s else None
            for i in range(max(0,s-CTX),e):
                lines.append(f"{i} [{U[i]['hint']}]{' (הקשר)' if i<s else ''} {U[i]['text']}")
            for attempt in range(3):
                try:
                    r=cl.chat.completions.create(model=MODEL,messages=[{"role":"system","content":SYS.format(name=name)},
                      {"role":"user","content":f"יחידות לתיוג: {s} עד {e-1}.\n\n"+"\n".join(lines)}],
                      response_format={"type":"json_object"},max_completion_tokens=32000,reasoning_effort="high")
                    ch=json.loads(r.choices[0].message.content)["changes"]
                    ch=[[int(a),b] for a,b in ch if s<=int(a)<e and b in("M","R")]
                    assert ch and ch[0][0]==s
                    tin+=r.usage.prompt_tokens; tout+=r.usage.completion_tokens
                    json.dump(ch,open(p,"w")); break
                except Exception as ex:
                    print("retry",v,s,repr(ex)[:200],flush=True); time.sleep(5)
            else:
                print("FAILED",v,s,flush=True); continue
            print("ok",v,s,len(ch),tin,tout,flush=True)
    print("LABEL DONE",MODEL,tin,tout,flush=True)
if __name__=="__main__": main()
