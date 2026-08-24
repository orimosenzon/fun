"""בודק שהתמונה במסך באמת משתנה בזמן ניגון, לא רק שהסמן רץ."""
from playwright.sync_api import sync_playwright
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = HERE                      # קובצי המדיה יושבים ב-tests/testmedia
URL = os.environ.get("VEDIT_URL", "http://127.0.0.1:8777/index.html")


SIG = """() => {
  const cv=document.getElementById('stage');
  const c=document.createElement('canvas');c.width=24;c.height=14;
  const x=c.getContext('2d');x.drawImage(cv,0,0,24,14);
  const d=x.getImageData(0,0,24,14).data;
  let h=0; for(let i=0;i<d.length;i+=4) h=(h*31 + d[i]+d[i+1]*3+d[i+2]*7)|0;
  return h;
}"""

def run(headless_flag):
    with sync_playwright() as p:
        args = ["--autoplay-policy=no-user-gesture-required"] if headless_flag else []
        br=p.chromium.launch(channel="chrome", headless=True, args=args)
        pg=br.new_context(viewport={"width":1500,"height":900}).new_page()
        errs=[]
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(300)
        pg.evaluate("indexedDB.deleteDatabase('vedit')"); pg.reload(); pg.wait_for_timeout(600)
        pg.set_input_files("#fileInput", [f"{SCRATCH}/testmedia/real720.mp4"])
        pg.wait_for_timeout(4000)
        pg.click("#btnPlay")
        sigs=[]
        for _ in range(10):
            pg.wait_for_timeout(220)
            sigs.append(pg.evaluate(SIG))
        st = pg.evaluate("""() => {const e=window.__vedit.engine;
            const p=[...e.players.values()][0];
            return {playhead:+window.__vedit.state.playhead.toFixed(2),
                    ct:+p.el.currentTime.toFixed(2), paused:p.el.paused,
                    presented:p.presented, drawn:p.framesDrawn,
                    decoded:p.el.getVideoPlaybackQuality?p.el.getVideoPlaybackQuality().totalVideoFrames:null};}""")
        uniq = len(set(sigs))
        print(f"  autoplay-flag={headless_flag}: unique frames {uniq}/10  state={st}")
        stuck = pg.evaluate("""() => window.__vedit.log.entries().filter(e=>
             e.msg.includes('not advancing')||e.msg.includes('stale')||e.msg.includes('rejected')).map(e=>e.msg)""")
        print("  warnings:", stuck or "none")
        print("  page errors:", errs)
        br.close()
        return uniq

print("motion test:")
a = run(True)
b = run(False)
print("VERDICT:", "OK" if a>4 and b>4 else "PROBLEM")
