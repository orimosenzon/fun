import base64, os
from playwright.sync_api import sync_playwright
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = HERE                      # קובצי המדיה יושבים ב-tests/testmedia
URL = os.environ.get("VEDIT_URL", "http://127.0.0.1:8777/index.html")

os.makedirs(f"{SCRATCH}/trans", exist_ok=True)
errors=[]
with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
    pg = br.new_context(viewport={"width":1400,"height":900}).new_page()
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("indexedDB.deleteDatabase('vedit')"); pg.reload(); pg.wait_for_timeout(500)
    pg.set_input_files("#fileInput", [f"{SCRATCH}/testmedia/clip_a.mp4", f"{SCRATCH}/testmedia/clip_b.mp4"])
    pg.wait_for_timeout(3500)
    pg.locator(".mitem").nth(1).dblclick(); pg.wait_for_timeout(1000)

    types = pg.evaluate("window.__vedit ? null : null") # placeholder
    types = ['dissolve','fadeblack','fadewhite','wipeleft','wiperight','wipeup',
             'slideleft','slideup','zoomin','circle','blinds','push']
    for ty in types:
        pg.evaluate(f"""() => {{
          const s=window.__vedit.state, tr=s.proj.tracks.find(t=>t.kind==='video');
          const c=tr.clips[1];
          c.tin = {{type:{ty!r}, dur:1.2}};
        }}""")
        t = pg.evaluate("() => {const c=window.__vedit.state.proj.tracks.find(t=>t.kind==='video').clips[1]; return c.start + c.tin.dur*0.5;}")
        pg.evaluate(f"window.__vedit.state.playhead={t}; window.__vedit.engine.seek({t},true)")
        pg.wait_for_timeout(700)
        data = pg.evaluate("""() => {
           const cv=document.getElementById('stage');
           const o=document.createElement('canvas'); o.width=320;o.height=180;
           o.getContext('2d').drawImage(cv,0,0,320,180);
           return o.toDataURL('image/png').split(',')[1];
        }""")
        open(f"{SCRATCH}/trans/{ty}.png","wb").write(base64.b64decode(data))
        print("ok", ty)
    print("ERRORS:", errors)
    br.close()
