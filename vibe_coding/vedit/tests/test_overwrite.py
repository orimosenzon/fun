"""גרירת קליפ מעל קליפ אחר: לוודא שהמשתמש מקבל התראה ושאפשר לבטל."""
from playwright.sync_api import sync_playwright
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = HERE                      # קובצי המדיה יושבים ב-tests/testmedia
URL = os.environ.get("VEDIT_URL", "http://127.0.0.1:8777/index.html")

with sync_playwright() as p:
    br=p.chromium.launch(channel="chrome", headless=True)
    pg=br.new_context(viewport={"width":1600,"height":950}).new_page()
    errs=[]; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("indexedDB.deleteDatabase('vedit')"); pg.reload(); pg.wait_for_timeout(600)
    pg.set_input_files("#fileInput", [f"{SCRATCH}/testmedia/clip_b.mp4", f"{SCRATCH}/testmedia/clip_a.mp4"])
    pg.wait_for_timeout(5000)
    st = lambda: pg.evaluate("""() => window.__vedit.state.proj.tracks[0].clips.map(c=>
        `${c.name}@${c.start.toFixed(1)}+${c.duration.toFixed(1)}`)""")
    print("before:", st())

    # גוררים את הקליפ השני (6 שניות) שמאלה, מעל הראשון (5 שניות) — דריסה מלאה
    c2 = pg.locator(".clip").nth(1).bounding_box()
    pg.mouse.move(c2["x"]+c2["width"]/2, c2["y"]+c2["height"]/2)
    pg.mouse.down()
    pg.mouse.move(c2["x"]+c2["width"]/2-760, c2["y"]+c2["height"]/2, steps=14)
    pg.mouse.up(); pg.wait_for_timeout(600)
    print("after drag:", st())
    toasts = pg.evaluate("[...document.querySelectorAll('.toast')].map(t=>t.textContent)")
    print("toast shown:", toasts)
    logged = pg.evaluate("""() => window.__vedit.log.entries()
        .filter(e=>e.msg==='clips removed').map(e=>e.data)""")
    print("logged removals:", logged)
    pg.click("#btnUndo"); pg.wait_for_timeout(500)
    print("after undo:", st())
    print("errors:", errs)
    br.close()
