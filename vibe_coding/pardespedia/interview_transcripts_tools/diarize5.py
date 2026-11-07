import json,os,sys,time
from collections import Counter
sys.path.insert(0,'/home/ori/fun/vibe_coding/transparency/tools')
import torch
from diarize_meetings import load_audio, merge, SR
from pyannote.audio import Pipeline
W=os.path.expanduser("~/.cache/pardespedia_interviews")
pipe=Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",token=True)
pipe.to(torch.device("cuda"))
for vid in sys.argv[1:]:
    out=f"{W}/spk/{vid}.json"
    if os.path.exists(out): continue
    t=time.time()
    wav=load_audio(f"{W}/audio/{vid}.m4a")
    res=pipe({"waveform":wav,"sample_rate":SR})
    ann=getattr(res,"exclusive_speaker_diarization",res)
    turns=[[round(s.start,2),round(s.end,2),spk] for s,_,spk in ann.itertracks(yield_label=True)]
    tr=json.load(open(f"{W}/tr/{vid}.json",encoding="utf-8"))
    segs=merge(tr,turns)
    json.dump({"duration":tr["duration"],"segments":segs},open(out+".tmp","w",encoding="utf-8"),ensure_ascii=False)
    os.replace(out+".tmp",out)
    c=Counter()
    for s in segs: c[s["speaker"]]+=len(s["text"].split())
    print("DZ",vid,len(segs),"turns",dict(c),round((time.time()-t)/60,1),"min",flush=True)
    del wav; torch.cuda.empty_cache()
print("DZ all done",flush=True)
