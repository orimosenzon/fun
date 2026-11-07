"""Second, stronger ASR pass with local vocabulary, used to cross-check the first one."""
import json,os,sys,time
sys.path.insert(0,'/home/ori/fun/vibe_coding/transparency/tools')
from transcribe_meetings import cut_points, SR
from faster_whisper import WhisperModel, decode_audio
W=os.path.expanduser("~/.cache/pardespedia_interviews")
MODEL="ivrit-ai/whisper-large-v3-ct2"
PROMPT=("ראיון עם אייל כגן, מועמד לראשות מועצת פרדס חנה-כרכור. מדברים על הגר פרי יגור, יוני חכימי, "
        "אביעד סיני, אלדד בר כוכבא, חיים געש, שכונת עומרים, שי חי, תוכנית מתאר, ועדה מחוזית, "
        "ועדה מקומית לתכנון ובנייה, פינוי-בינוי, קדנציה, קואליציה, אופוזיציה, מנכ\"ל המועצה.")
model=WhisperModel(MODEL,device="cuda",compute_type="int8")
print("ASR2 model loaded",flush=True)
for vid in sys.argv[1:]:
    out=f"{W}/tr2/{vid}.json"
    if os.path.exists(out): continue
    t=time.time()
    audio=decode_audio(f"{W}/audio/{vid}.m4a",sampling_rate=SR)
    pts=cut_points(audio); segs=[]
    for a,b in zip(pts,pts[1:]):
        off=a/SR
        ss,_=model.transcribe(audio[a:b],language="he",beam_size=5,vad_filter=True,
                              word_timestamps=True,initial_prompt=PROMPT,condition_on_previous_text=False)
        for s in ss:
            segs.append({"start":round(s.start+off,2),"end":round(s.end+off,2),"text":s.text.strip(),
                         "words":[[round(w.start+off,2),round(w.end+off,2),w.word] for w in (s.words or [])]})
        print("ASR2 chunk",round(b/SR/60),"min",flush=True)
    json.dump({"model":MODEL,"segments":segs},open(out+".tmp","w",encoding="utf-8"),ensure_ascii=False)
    os.replace(out+".tmp",out)
    print("ASR2 done",vid,len(segs),"segs",round((time.time()-t)/60,1),"min",flush=True)
print("ASR2 END",flush=True)
