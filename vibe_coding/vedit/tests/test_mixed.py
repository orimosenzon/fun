import json
from playwright.sync_api import sync_playwright
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = HERE                      # קובצי המדיה יושבים ב-tests/testmedia
URL = os.environ.get("VEDIT_URL", "http://127.0.0.1:8777/index.html")

errors=[]
with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
    pg = br.new_context(viewport={"width":1600,"height":950}).new_page()
    pg.on("pageerror", lambda e: errors.append(f"PAGEERROR {e}"))
    pg.on("console", lambda m: errors.append(f"CONSOLE {m.text}") if m.type=="error" else None)
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("indexedDB.deleteDatabase('vedit')"); pg.reload(); pg.wait_for_timeout(500)
    pg.set_input_files("#fileInput", [f"{SCRATCH}/testmedia/clip_a.mp4",
                                      f"{SCRATCH}/testmedia/photo.png",
                                      f"{SCRATCH}/testmedia/music.mp3"])
    pg.wait_for_timeout(4000)
    print("media:", json.dumps(pg.evaluate("""() => window.__vedit.state.proj.media.map(m=>
        ({n:m.name,t:m.type,d:+m.duration.toFixed(2),w:m.width,h:m.height,a:m.hasAudio}))"""), ensure_ascii=False))
    # להוסיף את התמונה ואת המוזיקה
    pg.locator(".mitem").nth(1).dblclick(); pg.wait_for_timeout(700)
    pg.locator(".mitem").nth(2).dblclick(); pg.wait_for_timeout(700)
    print("tracks:", json.dumps(pg.evaluate("""() => window.__vedit.state.proj.tracks.map(t=>
        ({n:t.name,k:t.kind,c:t.clips.map(c=>`${c.kind}:${c.name}@${c.start.toFixed(1)}+${c.duration.toFixed(1)}`)}))"""), ensure_ascii=False))
    # רינדור על התמונה
    pg.evaluate("""() => {const s=window.__vedit.state;
       const c=s.proj.tracks.flatMap(t=>t.clips).find(c=>c.kind==='image');
       s.playhead=c.start+1; window.__vedit.engine.seek(s.playhead,true);}""")
    pg.wait_for_timeout(900)
    print("frame on image:", pg.evaluate("""() => {const cv=document.getElementById('stage');
       const c=document.createElement('canvas');c.width=32;c.height=18;
       c.getContext('2d').drawImage(cv,0,0,32,18);
       const d=c.getContext('2d').getImageData(0,0,32,18).data; let s=0;
       for(let i=0;i<d.length;i+=4)s+=(d[i]+d[i+1]+d[i+2])/3; return +(s/(d.length/4)).toFixed(1);}"""))
    # ניגון על האודיו
    pg.evaluate("""() => {const s=window.__vedit.state;
       const c=s.proj.tracks.find(t=>t.kind==='audio').clips[0];
       s.playhead=c.start+0.5; window.__vedit.engine.seek(s.playhead,true);}""")
    pg.click("#btnPlay"); pg.wait_for_timeout(1500)
    print("audio players:", pg.evaluate("""() => [...window.__vedit.engine.players.values()].map(p=>
        ({k:p.kind, gain:p.gain?+p.gain.gain.value.toFixed(2):null, paused:p.el.paused}))"""))
    pg.click("#btnPlay"); pg.wait_for_timeout(200)
    pg.screenshot(path=f"{SCRATCH}/shot_08_mixed.png")
    print("ERRORS:", errors)
    br.close()
