"""מדגמן גרירה של הסמן ובודק שהמסך אף פעם לא נהיה שחור."""
from playwright.sync_api import sync_playwright
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = HERE                      # קובצי המדיה יושבים ב-tests/testmedia
URL = os.environ.get("VEDIT_URL", "http://127.0.0.1:8777/index.html")

errs=[]
with sync_playwright() as p:
    br=p.chromium.launch(channel="chrome", headless=True)
    pg=br.new_context(viewport={"width":1600,"height":950}).new_page()
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_timeout(300)
    pg.evaluate("indexedDB.deleteDatabase('vedit')"); pg.reload(); pg.wait_for_timeout(600)
    pg.set_input_files("#fileInput", [f"{SCRATCH}/testmedia/real720.mp4"])
    pg.wait_for_timeout(4000)

    brightness = """() => {
      const cv=document.getElementById('stage');
      const c=document.createElement('canvas');c.width=32;c.height=18;
      const x=c.getContext('2d'); x.drawImage(cv,0,0,32,18);
      const d=x.getImageData(0,0,32,18).data; let s=0;
      for(let i=0;i<d.length;i+=4)s+=(d[i]+d[i+1]+d[i+2])/3;
      return +(s/(d.length/4)).toFixed(1);
    }"""

    # גרירה רציפה על הסרגל, כמו משתמש שמחפש נקודה
    ruler = pg.locator("#ruler").bounding_box()
    pg.mouse.move(ruler["x"]+30, ruler["y"]+13)
    pg.mouse.down()
    vals=[]
    for x in range(40, 620, 22):
        pg.mouse.move(ruler["x"]+x, ruler["y"]+13)
        pg.wait_for_timeout(70)
        vals.append(pg.evaluate(brightness))
    pg.mouse.up(); pg.wait_for_timeout(500)
    vals.append(pg.evaluate(brightness))
    blacks = [v for v in vals if v < 3]
    print("brightness samples:", vals)
    print(f"black frames during scrub: {len(blacks)}/{len(vals)}")

    # קפיצות אקראיות
    import random
    bad=0
    for t in [0.4, 7.9, 1.1, 9.5, 3.3, 0.05, 6.6]:
        pg.evaluate(f"() => {{const V=window.__vedit; V.state.playhead={t}; V.engine.seek({t},true);}}")
        pg.wait_for_timeout(120)
        b = pg.evaluate(brightness)
        if b < 3: bad += 1
        print(f"  t={t}: brightness={b}")
    print("black after random seeks:", bad)

    print("blank warnings in log:", pg.evaluate("""() =>
        window.__vedit.log.entries().filter(e=>e.msg.includes('blank')).length"""))
    print("errors in log:", pg.evaluate("""() =>
        window.__vedit.log.entries().filter(e=>e.lvl==='error').map(e=>e.msg)"""))
    print("page errors:", errs)
    br.close()
