"""שני סיומים: לאורך לפלט גבוה, לרוחב לשאר (יורם שלח סיום 9:16 ב-6/10/2026).

הסיומים נטענים מתיקיית brand/, ולכן האפליקציה מוגשת כאן מ-http. מייצאים קטע קצר פעם
ב-9:16 ופעם ב-16:9, ומשווים את הפריים האחרון בכל קובץ לפריים האחרון בסיום המתאים.

הרצה:  ./make_media.sh && python3 test_outro.py
"""
import base64, functools, http.server, os, subprocess, sys, threading
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
PORT = 8768

fails = []
def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        fails.append(msg)

def frame(path, at_end=0.25, size=(54, 96)):
    """פריים מוקטן (גווני אפור) מעט לפני סוף הקובץ"""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-sseof", f"-{at_end}", "-i", str(path), "-frames:v", "1",
                          "-vf", f"scale={size[0]}:{size[1]},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(size[1], size[0]).astype(float)

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
handler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

files = {}
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=False, args=["--autoplay-policy=no-user-gesture-required"])
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"http://127.0.0.1:{PORT}/index.html")
    page.wait_for_function("document.body.classList.contains('ready')")
    page.evaluate("C.setLang('he')")
    page.click("[data-tab='brand']")
    check("outro_tall.mp4" in page.inner_text("#statOutroTall"), "הסיום לאורך נטען מתיקיית brand/")
    check("outro.mp4" in page.inner_text("#statOutro"), "הסיום לרוחב נטען מתיקיית brand/")
    page.set_input_files("#fileInput", str(HERE / "media" / "source.mp4"))
    page.wait_for_function("window.__clipper.dur > 0")
    page.click("[data-tab='clips']")
    page.fill("#inIn", "0:02"); page.press("#inIn", "Enter")
    page.fill("#inOut", "0:03"); page.press("#inOut", "Enter")
    page.click("[data-tab='format']")
    for fmt in ["9:16", "16:9"]:
        page.click(f"#segFormat button[data-v='{fmt}']")
        page.click("[data-frame='whole']")
        page.click("#btnExport")
        page.wait_for_function("!document.body.classList.contains('exporting') && !document.getElementById('result').hidden",
                               timeout=60000)
        b64 = page.evaluate("""async () => {
            const buf = new Uint8Array(await __clipper.lastExport.blob.arrayBuffer());
            let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
            return btoa(s);
        }""")
        page.click("#btnResultClose")
        f = OUT / f"outro_{fmt.replace(':', 'x')}.mp4"
        f.write_bytes(base64.b64decode(b64))
        files[fmt] = f
    browser.close()
srv.shutdown()

tall, wide = ROOT / "brand" / "outro_tall.mp4", ROOT / "brand" / "outro.mp4"
for fmt, want, other, size in [("9:16", tall, wide, (54, 96)), ("16:9", wide, tall, (96, 54))]:
    got = frame(files[fmt], size=size)
    d_want = np.abs(got - frame(want, size=size)).mean()
    d_other = np.abs(got - frame(other, size=size)).mean()
    check(d_want < 12 and d_want < d_other,
          f"{fmt}: הסוף הוא {want.name} (הפרש {d_want:.1f}, מול {other.name} {d_other:.1f})")
check(not errors, f"אין שגיאות JS ({errors[:3]})")
print("\nנכשל:" if fails else "\nהכול עבר", *fails, sep="\n  ")
sys.exit(1 if fails else 0)
