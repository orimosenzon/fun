"""מסגור נפרד לכל סרטון ברשימה (בקשה של יורם, 10/2026).

סרטון 1 מקבל מסגור משלו (ממלא את המסגרת), סרטון 2 נשאר עם המסגור הכללי (כל התמונה).
השטח הריק נצבע במג'נטה, כך שבקובץ רואים מיד איזה מסגור יצא: במסגור "כל התמונה" יש
פסים מג'נטה למעלה ולמטה, ובמסגור שממלא את המסגרת אין.

הרצה:  ./make_media.sh && python3 test_frames.py
"""
import base64, os, subprocess, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
URL = (HERE.parent / "index.html").as_uri()

fails = []
def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        fails.append(msg)

def top_pixel(path):
    """צבע הפיקסל במרכז הרצועה העליונה, בשנייה הראשונה"""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "0.5", "-i", str(path), "-frames:v", "1",
                          "-vf", "crop=2:2:iw/2:ih*0.05", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return tuple(raw[:3])

is_magenta = lambda c: c[0] > 200 and c[1] < 60 and c[2] > 200

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=os.environ.get("HEADLESS") == "1",
                                args=["--autoplay-policy=no-user-gesture-required"])
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.goto(URL)
    page.wait_for_function("document.body.classList.contains('ready')")
    page.evaluate("C.setLang('he'); window.showDirectoryPicker = undefined")
    page.click("#btnBatchClear")
    page.click("#platforms button[data-id='ig_reels']")
    page.select_option("#selFill", "color")
    page.evaluate("""(() => { const c = document.getElementById('inFillColor'); c.value = '#ff00ff';
                      c.dispatchEvent(new Event('input')); })()""")
    page.click("[data-frame='whole']")
    page.set_input_files("#fileInput", str(MEDIA / "source.mp4"))
    page.wait_for_function("window.__clipper.dur > 0")
    page.fill("#inBatch", "0.02 - 0.04\n0.10 - 0.12\n")
    page.wait_for_timeout(300)

    print("בלי סרטון מסומן: המסגור כללי")
    check("כל הסרטונים" in page.inner_text("#frameScope"), "השורה מסבירה שהמסגור חל על כולם")

    print("סרטון 1: מסגור משלו")
    page.locator(".bitem").nth(0).locator(".br").click()
    check("סרטון 1 משתמש עכשיו במסגור הכללי" in page.inner_text("#frameScope"), "לפני שינוי: עדיין המסגור הכללי")
    page.click("[data-frame='fill']")
    frames = page.evaluate("__clipper.set.frames")
    check(list(frames) == ["2-4"], f"נוצר מסגור לסרטון 1 בלבד (יצא {frames})")
    check(page.evaluate("__clipper.set.zoom") == 1, "המסגור הכללי לא השתנה")
    check("מסגור משלו לסרטון 1" in page.inner_text("#frameScope"), "השורה מראה שיש לו מסגור משלו")
    check(page.locator(".bitem").nth(0).locator(".bl").inner_text().startswith("🔍"), "סימן 🔍 בשורה של סרטון 1")
    # גרירה בתצוגה מזיזה רק את סרטון 1
    box = page.locator("#preview").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(cx, cy); page.mouse.down(); page.mouse.move(cx + 40, cy, steps=5); page.mouse.up()
    f1 = page.evaluate("__clipper.set.frames['2-4']")
    check(f1["px"] != 0.5, f"גרירה הזיזה את סרטון 1 (px={f1['px']:.2f})")
    check(page.evaluate("__clipper.set.px") == 0.5, "והכללי נשאר במרכז")

    print("סרטון 2: נשאר עם המסגור הכללי")
    page.locator(".bitem").nth(1).locator(".br").click()
    check(float(page.input_value("#rngFrameZoom")) == 1, "הזום בפאנל חוזר ל-100%")
    check(not page.locator(".bitem").nth(1).locator(".bl").inner_text().startswith("🔍"), "בלי 🔍 בשורה של סרטון 2")

    print("ייצוא")
    page.click("#btnBatchRun")
    page.wait_for_function("!document.body.classList.contains('exporting') && (__clipper.lastBatch||[]).length === 2",
                           timeout=60000)
    files = []
    for i in range(2):
        b64 = page.evaluate(f"""async () => {{
            const buf = new Uint8Array(await __clipper.lastBatch[{i}].blob.arrayBuffer());
            let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
            return btoa(s);
        }}""")
        f = OUT / f"frames_{i + 1}.mp4"
        f.write_bytes(base64.b64decode(b64))
        files.append(f)

    print("חזרה למסגור הכללי ונשמר אחרי טעינה מחדש")
    page.locator(".bitem").nth(0).locator(".br").click()
    page.wait_for_timeout(600)   # השמירה נדחית ב-400ms
    page.reload()
    page.wait_for_function("document.body.classList.contains('ready')")
    check(list(page.evaluate("__clipper.set.frames")) == ["2-4"], "המסגור של סרטון 1 נשמר בדפדפן")
    page.set_input_files("#fileInput", str(MEDIA / "source.mp4"))
    page.wait_for_function("window.__clipper.dur > 0")
    page.locator(".bitem").nth(0).locator(".br").click()
    page.click("#btnFrameShared")
    check(page.evaluate("__clipper.set.frames") == {}, "'חזור למסגור הכללי' מוחק את המסגור הנפרד")
    page.click("#btnFrameDone")
    check(page.evaluate("__batch.cur") is None, "'סיום' מבטל את הסימון")
    browser.close()

c1, c2 = top_pixel(files[0]), top_pixel(files[1])
check(not is_magenta(c1), f"סרטון 1 ממלא את המסגרת, בלי פס מג'נטה (פיקסל {c1})")
check(is_magenta(c2), f"סרטון 2 עם פס מג'נטה, כל התמונה (פיקסל {c2})")
print()
check(not errors, f"אין שגיאות JS ({errors[:3]})")
print("\nנכשל:" if fails else "\nהכול עבר", *fails, sep="\n  ")
sys.exit(1 if fails else 0)
