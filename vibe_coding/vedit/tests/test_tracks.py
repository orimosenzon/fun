"""בודק שכפתורי הערוץ באמת עושים משהו, לא רק שהם לא זורקים שגיאה."""
from playwright.sync_api import sync_playwright
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = HERE                      # קובצי המדיה יושבים ב-tests/testmedia
URL = os.environ.get("VEDIT_URL", "http://127.0.0.1:8777/index.html")

fails=[]
def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond: fails.append(msg)

with sync_playwright() as p:
    br=p.chromium.launch(channel="chrome", headless=True)
    pg=br.new_context(viewport={"width":1600,"height":950}).new_page()
    errs=[]
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("indexedDB.deleteDatabase('vedit')"); pg.reload(); pg.wait_for_timeout(600)
    pg.set_input_files("#fileInput", [f"{SCRATCH}/testmedia/clip_a.mp4", f"{SCRATCH}/testmedia/real720.mp4"])
    pg.wait_for_timeout(5500)

    n_clips = pg.locator(".clip").count()
    check(n_clips == 2, f"importing 2 videos puts 2 clips on the timeline (got {n_clips})")

    T = lambda i=0: pg.evaluate(f"window.__vedit.state.proj.tracks[{i}]")
    bright = """() => {const cv=document.getElementById('stage');
      const c=document.createElement('canvas');c.width=24;c.height=14;
      const x=c.getContext('2d');x.drawImage(cv,0,0,24,14);
      const d=x.getImageData(0,0,24,14).data;let s=0;
      for(let i=0;i<d.length;i+=4)s+=(d[i]+d[i+1]+d[i+2])/3;return s/(d.length/4);}"""

    pg.evaluate("() => {const V=window.__vedit; V.state.playhead=1; V.engine.seek(1,true);}")
    pg.wait_for_timeout(800)
    before = pg.evaluate(bright)

    # ── עין ──
    pg.locator(".thead button[data-act='hide']").first.click(); pg.wait_for_timeout(400)
    check(T()["hidden"] is True, "eye click hides the track")
    check(pg.evaluate(bright) < 3, f"hidden track disappears from the monitor (brightness {pg.evaluate(bright):.0f})")
    check(pg.locator(".track.is-hidden").count() == 1, "hidden track is marked in the timeline")
    check(pg.locator(".clip").count() == 2, "clips stay on the timeline when the track is hidden")

    pg.locator(".thead button[data-act='hide']").first.click(); pg.wait_for_timeout(400)
    check(T()["hidden"] is False, "second eye click un-hides")
    check(abs(pg.evaluate(bright) - before) < 2, "picture comes back")
    check(pg.locator(".track.is-hidden").count() == 0, "hidden marking is gone")

    # ── השתקה ──
    pg.locator(".thead button[data-act='mute']").first.click(); pg.wait_for_timeout(300)
    check(T()["muted"] is True, "mute click mutes")
    pg.locator(".thead button[data-act='mute']").first.click(); pg.wait_for_timeout(300)
    check(T()["muted"] is False, "second mute click un-mutes")

    # ── נעילה ──
    pg.locator(".thead button[data-act='lock']").first.click(); pg.wait_for_timeout(300)
    check(T()["locked"] is True, "lock click locks")
    pg.locator(".thead button[data-act='lock']").first.click(); pg.wait_for_timeout(300)
    check(T()["locked"] is False, "second lock click unlocks")

    # ── בטל מחזיר מצב עין ──
    pg.locator(".thead button[data-act='hide']").first.click(); pg.wait_for_timeout(300)
    pg.click("#btnUndo"); pg.wait_for_timeout(400)
    check(T()["hidden"] is False, "undo restores visibility")

    # ── דריסה מזהירה ──
    pg.evaluate("""() => {const V=window.__vedit, s=V.state, tr=s.proj.tracks[0];
        s.selection.clear(); s.selection.add(tr.clips[1].id);}""")
    n_before = pg.evaluate("window.__vedit.state.proj.tracks[0].clips.length")
    pg.evaluate("""() => {const V=window.__vedit; V.state.playhead=0.2;}""")
    print("  (overwrite check) clips before:", n_before)

    print("page errors:", errs)
    br.close()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
