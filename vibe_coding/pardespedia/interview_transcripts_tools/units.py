import json,os,re
W=os.path.expanduser("~/.cache/pardespedia_interviews")
# video -> (interviewee, diarization label of Ori)
V={"oNHE5PFHxgY":("אביעד סיני","SPEAKER_01"),"jEhM7AYHXj0":("יוני חכימי","SPEAKER_01"),
   "BIOpgT2fqI8":("מוחמד כהן","SPEAKER_00"),"Jlwmd-pTtoI":("אלדד בר כוכבא","SPEAKER_00"),
   "ft5Ke0L1pMc":("אייל כגן","SPEAKER_01")}
def units(v):
    tr=json.load(open(f"{W}/tr/{v}.json",encoding="utf8"))["segments"]
    sp=sorted(json.load(open(f"{W}/spk/{v}.json",encoding="utf8"))["segments"],key=lambda r:r["start"])
    ori=V[v][1]
    words=[]
    for s in tr:
        for a,b,w in s["words"]:
            w=w.replace("\u202b","").replace("\u202c","")
            if w.strip():
                words.append([a,b,w if not words or w[0]==" " or words[-1][1]<a-0.01 and False else w])
        if words and not words[-1][2].endswith(" "): words[-1][2]+=" "  # segment boundary
    out,cur=[],[]
    def flush():
        nonlocal cur
        if cur: out.append(cur); cur=[]
    for wd in words:
        raw=wd[2]
        if re.match(r"\s*-\S",raw) and (not cur or cur[-1][2].endswith(" ") or raw[0]==" "): flush()
        cur.append(wd)
        if re.search(r"[.?!:]\s*$",raw) or (re.search(r",\s*$",raw) and len(cur)>=4) or len(cur)>=14: flush()
    flush()
    res=[]; j=0
    for u in out:
        m=0.0;r=0.0
        for a,b,_ in u:
            while j<len(sp)-1 and sp[j]["end"]<a: j+=1
            for x in sp[max(0,j-1):j+3]:
                ov=min(x["end"],b)-max(x["start"],a)
                if ov>0:
                    if x["speaker"]==ori: m+=ov
                    else: r+=ov
        res.append({"t":u[0][0],"text":re.sub(r"\s+"," ","".join(w[2] for w in u)).strip(),"hint":"M" if m>=r else "R"})
    return res
if __name__=="__main__":
    for v in V:
        u=units(v); print(V[v][0],len(u),sum(len(x["text"].split()) for x in u))
