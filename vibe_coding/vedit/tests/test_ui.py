"""לוחץ על כל כפתור ומפעיל כל קיצור מקלדת, ומחפש שגיאות ריצה."""
from playwright.sync_api import sync_playwright
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = HERE                      # קובצי המדיה יושבים ב-tests/testmedia
URL = os.environ.get("VEDIT_URL", "http://127.0.0.1:8777/index.html")


errors = []

def check(pg, label):
    if errors:
        print(f"  !! errors after {label}: {errors}")
        errors.clear()

with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
    pg = br.new_context(viewport={"width":1600,"height":950}).new_page()
    pg.on("pageerror", lambda e: errors.append(f"PAGEERROR {e}"))
    pg.on("console", lambda m: errors.append(f"CONSOLE {m.text}") if m.type == "error" else None)
    pg.on("dialog", lambda d: d.accept())

    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("indexedDB.deleteDatabase('vedit')"); pg.reload(); pg.wait_for_timeout(500)
    pg.set_input_files("#fileInput", [f"{SCRATCH}/testmedia/clip_a.mp4", f"{SCRATCH}/testmedia/clip_b.mp4"])
    pg.wait_for_timeout(3500)
    pg.locator(".mitem").nth(1).dblclick(); pg.wait_for_timeout(900)
    check(pg, "setup")

    # ── כל כפתורי הסרגלים ──
    ids = ["btnUndo","btnRedo","btnStart","btnNextFrame","btnNextFrame","btnPrevFrame",
           "btnMarkIn","btnEnd","btnMarkOut","btnClearMarks","btnSplitT","btnSnapshot",
           "btnSplit","btnDup","btnDelete","btnRipple","btnCutSel",
           "btnAddVideoTrack","btnAddAudioTrack","btnZoomIn","btnZoomOut","btnZoomFitTl"]
    for i in ids:
        pg.click(f"#{i}"); pg.wait_for_timeout(180)
        check(pg, i)
    print("1. toolbar buttons ok")

    # ── כלים ──
    for t in ["razor","hand","select"]:
        pg.click(f".tool[data-tool='{t}']"); pg.wait_for_timeout(120); check(pg, t)
    print("2. tools ok")

    # ── טאבים ──
    for t in ["transitions","titles","media"]:
        pg.click(f".tab[data-tab='{t}']"); pg.wait_for_timeout(200); check(pg, t)
    print("3. tabs ok")

    # ── אינספקטור: לגעת בכל שדה (קודם מוודאים שיש קליפים) ──
    pg.click(".tab[data-tab='media']"); pg.wait_for_timeout(200)
    if pg.evaluate("window.__vedit.state.proj.tracks.flatMap(t=>t.clips).length") < 2:
        pg.locator(".mitem").nth(0).dblclick(); pg.wait_for_timeout(700)
        pg.locator(".mitem").nth(1).dblclick(); pg.wait_for_timeout(700)
    pg.locator(".clip").first.click(position={"x": 40, "y": 20})
    pg.wait_for_timeout(400)
    n_open = pg.evaluate("document.querySelectorAll('#inspBody details').length")
    pg.evaluate("document.querySelectorAll('#inspBody details').forEach(d=>d.open=true)")
    pg.wait_for_timeout(200)
    nsl = pg.evaluate("""() => {
      const sl = [...document.querySelectorAll('#inspBody input[type=range]')];
      for (const s of sl) {
        const mn=+s.min, mx=+s.max, st=+s.step||0.01;
        const v = mn + Math.round(((mx-mn)*0.6)/st)*st;
        s.value = v;
        s.dispatchEvent(new Event('input', {bubbles:true}));
      }
      return sl.length;
    }""")
    pg.wait_for_timeout(600)
    print("   sliders exercised:", nsl)
    check(pg, "inspector sliders")
    selects = pg.locator("#inspBody select")
    for i in range(selects.count()):
        opts = selects.nth(i).locator("option")
        if opts.count() > 1:
            selects.nth(i).select_option(index=opts.count()-1)
            pg.wait_for_timeout(150)
    check(pg, "inspector selects")
    print(f"4. inspector ok ({n_open} groups)")

    # ── תפריט הקשר על קליפ ──
    pg.locator(".clip").first.click(button="right"); pg.wait_for_timeout(250)
    nitems = pg.locator("#ctxMenu button").count()
    pg.locator("#ctxMenu button").first.click(); pg.wait_for_timeout(250)
    check(pg, "context menu")
    print(f"5. context menu ok ({nitems} items)")

    # ── תפריט הקשר על כותרת ערוץ ──
    pg.locator(".thead").first.click(button="right"); pg.wait_for_timeout(200)
    pg.keyboard.press("Escape")
    pg.mouse.click(700, 400); pg.wait_for_timeout(150)
    check(pg, "track menu")

    # ── כפתורי הערוץ ──
    for act in ["hide","mute","lock","lock","mute","hide"]:
        b = pg.locator(f".thead button[data-act='{act}']").first
        if b.count(): b.click(); pg.wait_for_timeout(120)
    check(pg, "track buttons")
    print("6. track header ok")

    # ── קיצורי מקלדת ──
    pg.mouse.click(700, 780)  # פוקוס על הטיימליין
    keys = ["v","c","h","v","i","o","s","Control+a","Control+d","Control+c","Control+v",
            "ArrowRight","ArrowLeft","Shift+ArrowRight","Home","End","+","-","Shift+Z",
            "m","Delete","Control+z","Control+Shift+z","Escape","?"]
    for k in keys:
        pg.keyboard.press(k); pg.wait_for_timeout(130); check(pg, f"key {k}")
    pg.keyboard.press("Escape")
    pg.click("#helpClose"); pg.wait_for_timeout(200)
    print("7. keyboard ok")

    # ── ניגון עד הסוף (בדיקת עצירה טבעית) ──
    pg.evaluate("() => {const d=window.__vedit.state; d.playhead=Math.max(0,window.__vedit.engine.canvas?0:0);}")
    pg.evaluate("""() => {
      const s=window.__vedit.state;
      const dur = s.proj.tracks.flatMap(t=>t.clips).reduce((a,c)=>Math.max(a,c.start+c.duration),0);
      s.playhead = Math.max(0, dur-1.2); window.__vedit.engine.seek(s.playhead,true);
    }""")
    pg.click("#btnPlay"); pg.wait_for_timeout(2600)
    print("8. after natural end — playing:", pg.evaluate("window.__vedit.engine.playing"),
          "button:", pg.evaluate("document.getElementById('btnPlay').textContent"))
    check(pg, "playback end")

    # ── פרויקט חדש ──
    pg.click("#btnNewProj"); pg.wait_for_timeout(600)
    print("9. after new project — clips:", pg.evaluate("window.__vedit.state.proj.tracks.flatMap(t=>t.clips).length"))
    pg.screenshot(path=f"{SCRATCH}/shot_06_empty.png")
    check(pg, "new project")

    print("FINAL ERRORS:", errors)
    br.close()
