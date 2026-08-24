import sys, json
from playwright.sync_api import sync_playwright
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = HERE                      # קובצי המדיה יושבים ב-tests/testmedia
URL = os.environ.get("VEDIT_URL", "http://127.0.0.1:8777/index.html")


errors = []

def summary(page):
    return page.evaluate("""() => {
      const s = window.__vedit.state;
      return {
        playhead:+s.playhead.toFixed(3),
        tracks: s.proj.tracks.map(t=>({n:t.name,clips:t.clips.map(c=>
          `${c.name}@${c.start.toFixed(2)}+${c.duration.toFixed(2)} in=${c.inPoint.toFixed(2)}${c.tin?' TR:'+c.tin.type+'/'+c.tin.dur.toFixed(2):''}`)}))
      };
    }""")

def frame_stats(page):
    """ממוצע בהירות של הקנבס — כדי לוודא שמשהו מצויר"""
    return page.evaluate("""() => {
      const cv = document.getElementById('stage');
      const c = document.createElement('canvas'); c.width=64; c.height=36;
      const x = c.getContext('2d'); x.drawImage(cv,0,0,64,36);
      const d = x.getImageData(0,0,64,36).data;
      let sum=0, mx=0; for(let i=0;i<d.length;i+=4){const v=(d[i]+d[i+1]+d[i+2])/3; sum+=v; if(v>mx)mx=v;}
      return {avg:+(sum/(d.length/4)).toFixed(1), max:mx};
    }""")

with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True,
                           args=["--autoplay-policy=no-user-gesture-required"])
    pg = br.new_page(viewport={"width":1600,"height":950})
    pg.on("pageerror", lambda e: errors.append(f"PAGEERROR {e}"))
    pg.on("console", lambda m: errors.append(f"CONSOLE[{m.type}] {m.text}") if m.type=="error" else None)

    pg.goto(URL); pg.wait_for_timeout(500)
    pg.evaluate("indexedDB.deleteDatabase('vedit')")
    pg.reload(); pg.wait_for_timeout(600)

    pg.set_input_files("#fileInput", [f"{SCRATCH}/testmedia/clip_a.mp4", f"{SCRATCH}/testmedia/clip_b.mp4"])
    pg.wait_for_timeout(3500)
    print("1. after import:", json.dumps(summary(pg), ensure_ascii=False))
    print("   frame at 0:", frame_stats(pg))

    # הוספת הקליפ השני לסוף
    pg.locator(".mitem").nth(1).dblclick()
    pg.wait_for_timeout(1200)
    print("2. after append b:", json.dumps(summary(pg), ensure_ascii=False))

    # הזזת הסמן לשנייה 3 וחיתוך
    pg.evaluate("window.__vedit.state.playhead=3; window.__vedit.engine.seek(3,true)")
    pg.wait_for_timeout(900)
    print("   frame at 3s:", frame_stats(pg))
    pg.click("#btnSplit"); pg.wait_for_timeout(400)
    print("3. after split at 3s:", json.dumps(summary(pg), ensure_ascii=False))

    # מחיקת הקטע האמצעי (הקליפ שמתחיל ב-3)
    pg.evaluate("""() => {
      const s=window.__vedit.state; const tr=s.proj.tracks[0];
      const c=tr.clips.find(c=>Math.abs(c.start-3)<0.01);
      s.selection.clear(); s.selection.add(c.id);
    }""")
    pg.click("#btnRipple"); pg.wait_for_timeout(500)
    print("4. after ripple delete:", json.dumps(summary(pg), ensure_ascii=False))

    # החלת מעבר על הקליפ השני
    res = pg.evaluate("""() => {
      const s=window.__vedit.state; const tr=s.proj.tracks[0];
      if (tr.clips.length<2) return 'not enough clips';
      const c=tr.clips[1];
      s.selection.clear(); s.selection.add(c.id);
      return c.name;
    }""")
    pg.locator(".tab[data-tab='transitions']").click()
    pg.wait_for_timeout(300)
    pg.locator(".tcard").first.dblclick()
    pg.wait_for_timeout(400)
    print("5. after transition on", res, ":", json.dumps(summary(pg), ensure_ascii=False))
    pg.screenshot(path=f"{SCRATCH}/shot_02_transitions.png")

    # בדיקת רינדור באמצע המעבר
    st = pg.evaluate("""() => {const t=window.__vedit.state.proj.tracks[0].clips[1]; return t.start + (t.tin?t.tin.dur/2:0);}""")
    pg.evaluate(f"window.__vedit.state.playhead={st}; window.__vedit.engine.seek({st},true)")
    pg.wait_for_timeout(1200)
    print("6. frame mid-transition (t=%.2f):" % st, frame_stats(pg))
    pg.screenshot(path=f"{SCRATCH}/shot_03_mid_transition.png")

    # ניגון
    pg.click("#btnStart"); pg.wait_for_timeout(300)
    pg.click("#btnPlay"); pg.wait_for_timeout(2500)
    ph = pg.evaluate("window.__vedit.state.playhead")
    print("7. playhead after 2.5s of playback:", round(ph,2), "playing:", pg.evaluate("window.__vedit.engine.playing"))
    print("   frame during play:", frame_stats(pg))
    pg.click("#btnPlay"); pg.wait_for_timeout(300)

    # undo פעמיים
    pg.click("#btnUndo"); pg.wait_for_timeout(300)
    print("8. after undo:", json.dumps(summary(pg), ensure_ascii=False))

    print("ERRORS:", errors)
    br.close()
