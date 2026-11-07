import json,os,sys,time
sys.path.insert(0,'/home/ori/fun/vibe_coding/transparency/tools')
from transcribe_meetings import transcribe_chunked, MODEL
from faster_whisper import WhisperModel
W=os.path.expanduser("~/.cache/pardespedia_interviews")
model=WhisperModel(MODEL,device="cuda",compute_type="int8")
for vid in sys.argv[1:]:
    out=f"{W}/tr/{vid}.json"
    if os.path.exists(out): continue
    t=time.time()
    segs,dur=transcribe_chunked(model,f"{W}/audio/{vid}.m4a")
    json.dump({"meeting":{"id":vid},"model":MODEL,"duration":dur,"segments":segs},open(out+".tmp","w",encoding="utf-8"),ensure_ascii=False)
    os.replace(out+".tmp",out)
    print("TR",vid,round(dur/60),"min in",round((time.time()-t)/60,1),"min",len(segs),"segs",flush=True)
print("TR all done",flush=True)
