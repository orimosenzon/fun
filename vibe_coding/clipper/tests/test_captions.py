"""כיתובים ודיוק בחיתוך.

כיתובים: הוספה בנקודה בהקלטה, עיצוב, גרירה על התצוגה, ובקובץ המיוצא הם מופיעים
רק בזמן שלהם. דיוק: חלקי שנייה ברשימה, כפתורי ‎±0.1, וכתיבה חזרה לשורה המסומנת.

הרצה:  ./make_media.sh && python3 test_captions.py   (Chrome עם חלון, כמו test_e2e.py)
"""
import base64, json, os, subprocess
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

def pixel(video, t, x, y, W, H):
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    i = (y * W + x) * 3
    return tuple(raw[i:i + 3])

near = lambda c, ref, tol=45: all(abs(a - b) <= tol for a, b in zip(c, ref))

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=os.environ.get("HEADLESS") == "1",
                                args=["--autoplay-policy=no-user-gesture-required"])
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.goto(URL)
    page.wait_for_function("document.body.classList.contains('ready')")
    page.evaluate("C.setLang('he')")
    page.set_input_files("#fileInput", str(MEDIA / "source.mp4"))
    page.wait_for_function("window.__clipper.dur > 0")
    # בלי פתיחה, סיום ולוגו, ורקע שחור מלא מעל ומתחת לתמונה: קל לבדוק צבע של כיתוב
    page.evaluate("Object.assign(__clipper.set.use, { intro: false, outro: false, logo: false })")
    page.click("[data-tab='format']")
    page.click('#platforms button[data-id="ig_reels"]')
    page.click("#segFill button[data-fill='color']")
    page.click('[data-frame="whole"]')
    page.click("[data-tab='clips']")

    print("דיוק: רשימה עם חלקי שנייה")
    page.fill("#inBatch", "Session 1:\n00.05 - 00.09\n00.12,5 - 00.15")
    page.wait_for_timeout(400)
    rng = page.evaluate("__batch.items.filter(i => i.n).map(i => i.ranges)")
    check(rng == [[[5, 9]], [[12.5, 15]]], f"00.12,5 נקרא כ-12.5 שנ׳ ({rng})")
    page.click("#btnBatchDone")
    page.locator(".bitem").nth(0).click()
    s = page.evaluate("[__clipper.selIn, __clipper.selOut]")
    check(s == [5, 9], f"לחיצה על שורה 1 בוחרת 5–9 ({s})")

    print("דיוק: כפתורי הזזה כותבים לשורה")
    page.click('[data-nudge="in:0.1"]')
    page.click('[data-nudge="out:-1"]')
    page.click('[data-nudge="out:0.1"]')
    page.wait_for_timeout(200)
    txt = page.evaluate("__clipper.set.batchText")
    check(txt.split("\n")[1] == "00.05,1 - 00.08,1", f"השורה התעדכנה ל-00.05,1 - 00.08,1 ({txt.splitlines()[1]!r})")
    check(txt.split("\n")[0] == "Session 1:" and txt.split("\n")[2] == "00.12,5 - 00.15", "שאר השורות לא השתנו")
    t = page.evaluate("document.getElementById('srcVideo').currentTime")
    check(abs(t - 8.1) < 0.01, f"הסרטון עומד על נקודת החיתוך ({t:.2f})")
    check(page.evaluate("__batch.cur") == "00.05,1 - 00.08,1", "השורה נשארת מסומנת")

    print("דיוק: גרירת קצה בטיימליין מעדכנת, גרירה חדשה לא")
    page.click("#btnZoomFit")   # לחיצה על שורה עושה זום אליה; כאן מחשבים מיקום לפי כל הסרטון
    box = page.locator("#timeline").bounding_box()
    dur = 30
    y = box["y"] + 24 + 18 + 30
    x_at = lambda tt: box["x"] + box["width"] * tt / dur
    page.mouse.move(x_at(8.1), y); page.mouse.down(); page.mouse.move(x_at(10), y, steps=6); page.mouse.up()
    line = page.evaluate("__clipper.set.batchText.split('\\n')[1]")
    check(line.startswith("00.05,1 - 00.") and line != "00.05,1 - 00.08,1", f"גרירת הסוף נכתבה לשורה ({line})")
    page.mouse.move(x_at(20), y); page.mouse.down(); page.mouse.move(x_at(24), y, steps=6); page.mouse.up()
    check(page.evaluate("__batch.cur") is None, "גרירה של קטע חדש מבטלת את סימון השורה")
    check(page.evaluate("__clipper.set.batchText.split('\\n')[1]") == line, "והשורה לא השתנתה")
    page.click("#btnBatchClear")

    print("כיתובים")
    page.fill("#inIn", "5"); page.press("#inIn", "Enter")
    page.fill("#inOut", "9"); page.press("#inOut", "Enter")
    page.evaluate("document.getElementById('srcVideo').currentTime = 6")
    page.wait_for_timeout(300)
    page.click("[data-tab='text']")
    page.click("#btnCapAdd")
    cap = page.evaluate("__clipper.set.captions[Object.keys(__clipper.set.captions)[0]][0]")
    check(abs(cap["a"] - 6) < 0.01 and abs(cap["b"] - 9) < 0.01, f"כיתוב חדש מהנקודה הנוכחית, ל-3 שניות ({cap['a']:.2f}–{cap['b']:.2f})")
    page.fill("#inCapText", "Hello שלום")
    page.select_option("#selCapFont", "montserrat")
    page.click("#segCapStyle button[data-v='box']")
    page.fill("#inCapBg", "#ff0000")   # input type=color מקבל fill
    page.click("#segCapPlace button[data-place='top']")
    page.fill("#inCapB", "0:08"); page.press("#inCapB", "Enter")
    cap = page.evaluate("__clipper.set.captions[Object.keys(__clipper.set.captions)[0]][0]")
    check(cap["font"] == "montserrat" and cap["style"] == "box" and cap["bg"] == "#ff0000" and abs(cap["y"] - 0.17) < 1e-6, f"העיצוב נשמר (למעלה = 5% מתחת לאזור המכוסה ברילס) ({cap})")
    check(abs(cap["b"] - 8) < 0.01, "שעת ההיעלמות מהשדה")
    check(page.locator(".cap-item").count() == 1, "הכיתוב ברשימה")

    print("גרירת הכיתוב על התצוגה")
    pv = page.locator("#preview").bounding_box()
    # הקנבס מוצג ב-contain: מחשבים איפה הפריים 9:16 יושב בתוך האלמנט
    s = min(pv["width"] / 540, pv["height"] / 960)
    fx, fy = pv["x"] + (pv["width"] - 540 * s) / 2, pv["y"] + (pv["height"] - 960 * s) / 2
    cx, cy = fx + 270 * s, fy + 0.17 * 960 * s
    page.mouse.move(cx, cy); page.mouse.down(); page.mouse.move(cx, cy + 0.1 * 960 * s, steps=6); page.mouse.up()
    cap = page.evaluate("__clipper.set.captions[Object.keys(__clipper.set.captions)[0]][0]")
    check(abs(cap["y"] - 0.27) < 0.02, f"הכיתוב זז למטה ל-0.27 ({cap['y']:.3f})")
    check(abs(page.evaluate("__clipper.set.px") - 0.5) < 1e-6, "התמונה עצמה לא זזה")
    page.click("#segCapPlace button[data-place='top']")
    page.wait_for_timeout(1200)
    page.screenshot(path=OUT / "c1_caption.png")

    print("ייצוא")
    page.click("#btnExport")
    page.wait_for_function("!!window.__clipper.lastExport", timeout=60000)
    name = page.evaluate("__clipper.lastExport.name")
    b64 = page.evaluate("""async () => {
        const buf = new Uint8Array(await window.__clipper.lastExport.blob.arrayBuffer());
        let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
        return btoa(s);
    }""")
    check(not errors, f"אין שגיאות JS ({errors})")
    browser.close()

out = OUT / ("cap_" + name)
out.write_bytes(base64.b64decode(b64))
W, H = 1080, 1920
# פס עליון של התיבה, מעל הטקסט: מרכז ב-0.17×1920=326, fs=65, lh=81, pad=23 → y ≈ 326 - 40 - 11
X, Y = W // 2, 274
before, during, after = pixel(out, 0.5, X, Y, W, H), pixel(out, 1.5, X, Y, W, H), pixel(out, 3.5, X, Y, W, H)
print("  פיקסלים:", before, during, after)
check(near(before, (0, 0, 0)), f"שנייה 0.5 (מקור 5.5): אין כיתוב עדיין ({before})")
check(near(during, (178, 0, 0)), f"שנייה 1.5 (מקור 6.5): תיבה אדומה ב-70% ({during})")
check(near(after, (0, 0, 0)), f"שנייה 3.5 (מקור 8.5): הכיתוב נעלם ({after})")

print("נכשלו:" if fails else "הכול עבר", *fails, sep="\n  ")
raise SystemExit(1 if fails else 0)
