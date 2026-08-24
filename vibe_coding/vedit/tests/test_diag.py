from playwright.sync_api import sync_playwright
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = HERE                      # קובצי המדיה יושבים ב-tests/testmedia
URL = os.environ.get("VEDIT_URL", "http://127.0.0.1:8777/index.html")

errs=[]
with sync_playwright() as p:
    br=p.chromium.launch(channel="chrome", headless=True)   # בלי דגל autoplay, כמו דפדפן אמיתי
    pg=br.new_context(viewport={"width":1600,"height":950}).new_page()
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("indexedDB.deleteDatabase('vedit')"); pg.reload(); pg.wait_for_timeout(600)
    pg.set_input_files("#fileInput", [f"{SCRATCH}/testmedia/real720.mp4"])
    pg.wait_for_timeout(4500)
    pg.evaluate("() => {const V=window.__vedit; V.state.playhead=2.3; V.engine.seek(2.3,true);}")
    pg.wait_for_timeout(1500)
    pg.click("#btnDiag"); pg.wait_for_timeout(600)
    print("VERDICT:", pg.inner_text("#diagVerdict"))
    rep = pg.input_value("#diagText")
    print("report length:", len(rep), "chars")
    print("="*70)
    # מדפיס את החלקים המעניינים
    import re
    for block in ["── סביבה ──", "── מצב נוכחי ──"]:
        i = rep.find(block)
        print(rep[i:i+1500] if i>=0 else f"MISSING {block}")
        print("-"*70)
    print("LOG TAIL:")
    print("\n".join(rep.splitlines()[-28:]))
    pg.screenshot(path=f"{SCRATCH}/shot_10_diag.png")
    print("page errors:", errs)
    br.close()
