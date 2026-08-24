"""עריכה מלאה מקצה לקצה: 3 סרטונים → חיתוכים → מעברים → כותרת → ייצוא."""
import os, json
from playwright.sync_api import sync_playwright
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = HERE                      # קובצי המדיה יושבים ב-tests/testmedia
URL = os.environ.get("VEDIT_URL", "http://127.0.0.1:8777/index.html")


errors = []

with sync_playwright() as p:
    br = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
    ctx = br.new_context(viewport={"width":1600,"height":950}, accept_downloads=True)
    pg = ctx.new_page()
    pg.on("pageerror", lambda e: errors.append(f"PAGEERROR {e}"))
    pg.on("console", lambda m: errors.append(f"CONSOLE {m.text}") if m.type=="error" else None)

    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("indexedDB.deleteDatabase('vedit')"); pg.reload(); pg.wait_for_timeout(500)
    pg.set_input_files("#fileInput", [f"{SCRATCH}/testmedia/clip_a.mp4",
                                      f"{SCRATCH}/testmedia/clip_b.mp4",
                                      f"{SCRATCH}/testmedia/clip_c.mp4"])
    pg.wait_for_timeout(4500)

    # מרכיבים סרט: A (0-3), B (3-6), C (6-9) — כל אחד קטע של 3 שניות
    pg.evaluate("""() => {
      const V = window.__vedit;
      const s = V.state, tr = s.proj.tracks.find(t=>t.kind==='video');
      tr.clips.length = 0;
      const mk = (i, start) => {
        const m = s.proj.media[i];
        return {id:'c'+i, kind:'video', mediaId:m.id, name:m.name, trackId:tr.id,
                start, inPoint:0.5, duration:3, speed:1, volume:1, mute:false,
                aFadeIn:0, aFadeOut:0, opacity:1, scale:1, posX:0, posY:0, rotation:0,
                flipH:false, vFadeIn:0, vFadeOut:0,
                filters:{brightness:100,contrast:100,saturate:100,blur:0}, tin:null};
      };
      tr.clips.push(mk(0,0), mk(1,3), mk(2,6));
      tr.clips[0].vFadeIn = 0.8;                       // פתיחה משחור
      tr.clips[1].tin = {type:'dissolve', dur:1.0};    // מעבר הדרגתי
      tr.clips[2].tin = {type:'wipeleft', dur:1.0};    // מחיקה
      tr.clips[2].vFadeOut = 0.8;                      // סיום לשחור
    }""")
    # כותרת
    pg.locator(".tab[data-tab='titles']").click(); pg.wait_for_timeout(200)
    pg.fill("#titleText", "הסרט שלי")
    pg.evaluate("window.__vedit.state.playhead=0.5")
    pg.click("#btnAddTitle"); pg.wait_for_timeout(500)
    pg.evaluate("""() => {const V=window.__vedit; const t=V.state.proj.tracks.find(t=>t.clips.some(c=>c.kind==='title'));
        const c=t.clips.find(c=>c.kind==='title'); c.start=0.5; c.duration=2.5;}""")

    print("timeline:", json.dumps(pg.evaluate("""() => window.__vedit.state.proj.tracks.map(t=>({n:t.name,
        c:t.clips.map(c=>`${c.name}@${c.start}+${c.duration}${c.tin?'/'+c.tin.type:''}`)}))"""), ensure_ascii=False))

    pg.evaluate("window.__vedit.engine.seek(0,true)"); pg.wait_for_timeout(600)
    pg.screenshot(path=f"{SCRATCH}/shot_07_full_edit.png")

    # ── ייצוא מלא ──
    pg.click("#btnExport"); pg.wait_for_timeout(300)
    pg.select_option("#expRes", "0.5")
    with pg.expect_download(timeout=120000) as dl:
        pg.click("#expStart")
    d = dl.value
    out = f"{SCRATCH}/final_movie.{d.suggested_filename.split('.')[-1]}"
    d.save_as(out)
    print("exported:", out, os.path.getsize(out), "bytes")
    print("ERRORS:", errors)
    br.close()
