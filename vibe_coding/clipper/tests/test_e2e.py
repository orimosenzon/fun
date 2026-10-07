"""בדיקה מקצה לקצה בכרום אמיתי: טעינה, מיתוג, בחירת קטע בגרירה, ייצוא, ובדיקת הקובץ.

הרצה:  ./make_media.sh && python3 test_e2e.py
צריך Chrome מותקן (channel="chrome"), כי Chromium של Playwright לא מקודד H.264.
רץ עם חלון אמיתי כברירת מחדל: בלי GPU (HEADLESS=1) הקנבס מצויר בתוכנה, הלולאה
מגיעה רק ל-~12 פריימים בשנייה, ובדיקת האורך נכשלת למרות שהאפליקציה תקינה.
צילומי מסך ופריימים נשמרים ב-tests/out/.
"""
import base64, json, subprocess, sys
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
    """צבע RGB של פיקסל בזמן t בקובץ המיוצא"""
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    i = (y * W + x) * 3
    return tuple(raw[i:i + 3])

def near(c, ref, tol=60):
    return all(abs(a - b) <= tol for a, b in zip(c, ref))

with sync_playwright() as p:
    import os
    browser = p.chromium.launch(channel="chrome", headless=os.environ.get("HEADLESS") == "1",
                                args=["--autoplay-policy=no-user-gesture-required"])
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.goto(URL)
    page.wait_for_function("document.body.classList.contains('ready')")
    page.evaluate("C.setLang('he')")
    page.screenshot(path=OUT / "0_empty.png")

    print("טעינת סרטון")
    page.set_input_files("#fileInput", str(MEDIA / "source.mp4"))
    page.wait_for_function("window.__clipper.dur > 0")
    dur = page.evaluate("window.__clipper.dur")
    check(abs(dur - 30) < 0.2, f"משך המקור 30 שנ׳ (יצא {dur:.2f})")

    print("מיתוג")
    page.click("[data-tab='brand']")
    for kind, f in [("logo", "logo.png"), ("intro", "intro.mp4"), ("outro", "outro.mp4")]:
        with page.expect_file_chooser() as fc:
            page.click(f'[data-pick="{kind}"]')
        fc.value.set_files(str(MEDIA / f))
        page.wait_for_function(f"!!window.__clipper.brand.{kind}")
    page.wait_for_timeout(500)
    check("2.0" in page.inner_text("#statIntro"), "סטטוס הפתיחה מציג 2 שניות")
    check(page.is_checked("#useIntro") and page.is_checked("#useOutro") and page.is_checked("#useLogo"),
          "פתיחה, סיום ולוגו מסומנים במבנה הסרטון")
    page.uncheck("#useIntro")
    check("פתיחה" not in page.inner_text("#sumBrand"), f"בלי סימון, הפתיחה יוצאת מהסיכום ({page.inner_text('#sumBrand')})")
    page.check("#useIntro")
    page.click("[data-tab='clips']")

    print("בחירת קטע בגרירה על הטיימליין")
    box = page.locator("#timeline").bounding_box()
    y = box["y"] + 24 + 30          # בתוך שורת התמונות
    x_at = lambda t: box["x"] + box["width"] * t / dur
    page.mouse.move(x_at(5), y)
    page.mouse.down()
    page.mouse.move(x_at(7), y, steps=5)
    page.mouse.move(x_at(9), y, steps=5)
    page.mouse.up()
    s = page.evaluate("[window.__clipper.selIn, window.__clipper.selOut]")
    check(abs(s[0] - 5) < 0.2 and abs(s[1] - 9) < 0.2, f"גרירה יצרה קטע 5–9 (יצא {s[0]:.2f}–{s[1]:.2f})")

    print("גרירת קצה הסוף")
    page.mouse.move(x_at(9), y)
    page.mouse.down()
    page.mouse.move(x_at(11), y, steps=6)
    page.mouse.up()
    s = page.evaluate("[window.__clipper.selIn, window.__clipper.selOut]")
    check(abs(s[0] - 5) < 0.2 and abs(s[1] - 11) < 0.2, f"הקצה זז ל-11 (יצא {s[0]:.2f}–{s[1]:.2f})")

    print("הקלדת זמנים מדויקים")
    page.fill("#inIn", "0:10")
    page.press("#inIn", "Enter")
    page.fill("#inOut", "14")
    page.press("#inOut", "Enter")
    s = page.evaluate("[window.__clipper.selIn, window.__clipper.selOut]")
    check(abs(s[0] - 10) < 1e-6 and abs(s[1] - 14) < 1e-6, f"קטע 10–14 מהשדות (יצא {s})")
    check(page.inner_text("#totalLen").strip() == "0:08.00", f"אורך התוצאה 8 שנ׳ (יצא {page.inner_text('#totalLen')})")

    print("פלטפורמות")
    page.click("[data-tab='format']")
    page.click('#platforms button[data-id="youtube"]')
    check(page.evaluate("__clipper.set.format") == "16:9", "יוטיוב בוחר 16:9")
    page.evaluate("C.platform('wa_status').maxSec = 5")
    page.click('#platforms button[data-id="wa_status"]')
    check(page.is_visible("#lenWarn"), "אזהרת אורך מופיעה כשהתוצאה ארוכה מהמקסימום")
    page.evaluate("C.platform('wa_status').maxSec = 60")
    page.click('#platforms button[data-id="ig_reels"]')
    check(not page.is_visible("#lenWarn"), "אין אזהרת אורך ברילס (8 שנ׳)")
    page.click('#segFormat button[data-v="9:16"]')
    check(page.evaluate("__clipper.set.platform") == "ig_reels", "לחיצה על אותו יחס משאירה את הפלטפורמה")
    page.click("#segFill button[data-fill='blur']")
    page.wait_for_timeout(300)
    page.screenshot(path=OUT / "1_loaded.png")

    print("מסגור")
    page.click('[data-frame="fill"]')
    cz = page.evaluate("C.compose.coverZoom(document.getElementById('srcVideo'), 1080, 1920)")
    check(abs(page.evaluate("__clipper.set.zoom") - cz) < 0.01, f"'מלא את המסגרת' = זום {cz:.2f}")
    pb = page.locator("#preview").bounding_box()
    cx, cy = pb["x"] + pb["width"] / 2, pb["y"] + pb["height"] / 2
    page.mouse.move(cx, cy); page.mouse.down(); page.mouse.move(cx + 60, cy, steps=6); page.mouse.up()
    px = page.evaluate("__clipper.set.px")
    check(px < 0.45, f"גרירה ימינה מזיזה את התמונה ימינה (px={px:.2f})")
    page.mouse.move(cx, cy); page.mouse.wheel(0, 300)
    z = page.evaluate("__clipper.set.zoom")
    check(z < cz - 0.1, f"גלגלת למטה מקטינה (זום {z:.2f})")
    page.check("#chkText")
    page.fill("#inTextTop", "מפגש 9: נשימה")
    page.fill("#inTextBottom", "יורם · סדנת תנועה")
    page.wait_for_timeout(200)
    page.screenshot(path=OUT / "1b_framing.png")
    page.click('[data-frame="whole"]')
    page.uncheck("#chkText")
    check(page.evaluate("__clipper.set.zoom") == 1, "'כל התמונה' מחזיר לזום 1")

    print("ייצוא")
    page.click("#btnExport")
    page.wait_for_timeout(2500)
    page.screenshot(path=OUT / "2_exporting.png")
    page.wait_for_function("!!window.__clipper.lastExport", timeout=60000)
    info = page.evaluate("({stats: __clipper.lastExport.stats, name: __clipper.lastExport.name, mime: __clipper.lastExport.mime, W: __clipper.lastExport.W, H: __clipper.lastExport.H, sec: __clipper.lastExport.seconds})")
    print("  ", info)
    b64 = page.evaluate("""async () => {
        const b = window.__clipper.lastExport.blob;
        const buf = new Uint8Array(await b.arrayBuffer());
        let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
        return btoa(s);
    }""")
    page.screenshot(path=OUT / "3_done.png")
    browser.close()

out = OUT / info["name"]
out.write_bytes(base64.b64decode(b64))
probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(out)],
                                  capture_output=True, text=True).stdout)
vs = [s for s in probe["streams"] if s["codec_type"] == "video"][0]
aus = [s for s in probe["streams"] if s["codec_type"] == "audio"]
fdur = float(probe["format"].get("duration", 0))
print("קובץ:", out.name, vs["codec_name"], f'{vs["width"]}x{vs["height"]}', [a["codec_name"] for a in aus], f"{fdur:.2f}s")
check(vs["width"] == 1080 and vs["height"] == 1920, "רזולוציה 1080×1920")
check(vs["codec_name"] == "h264", f"וידאו H.264 (יצא {vs['codec_name']})")
check(len(aus) == 1, "יש פס קול")
check(abs(fdur - 8) < 0.6, f"אורך הקובץ כ-8 שנ׳ (יצא {fdur:.2f})")

W, H = 1080, 1920
# לוגו: פינה ימנית עליונה, 18% מ-1080 = 194 רוחב, בתוך האזור הבטוח של רילס
# (ימין 14% = 151, למעלה 12% = 230)
lx, ly = W - 151 - 97, 230 + 48
c = pixel(out, 1.0, W // 2, H // 2, W, H)
check(near(c, (208, 32, 32)), f"שנייה 1: פתיחה אדומה (יצא {c})")
c = pixel(out, 1.0, lx, ly, W, H)
check(not near(c, (0, 255, 0)), f"שנייה 1: אין לוגו על הפתיחה (יצא {c})")
c = pixel(out, 4.0, lx, ly, W, H)
check(near(c, (0, 255, 0), 70), f"שנייה 4: הלוגו הירוק על הקטע (יצא {c})")
c = pixel(out, 7.0, W // 2, H // 2, W, H)
check(near(c, (32, 64, 208)), f"שנייה 7: סיום כחול (יצא {c})")
c = pixel(out, 7.0, W // 2, 100, W, H)
check(not near(c, (32, 64, 208), 30), f"שנייה 7: מעל הסיום רקע מטושטש כהה ולא אותו כחול מלא (יצא {c})")
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(out), "-vf", "fps=2,scale=180:-1,tile=8x2", "-frames:v", "1", str(OUT / "contact.png")])

print()
check(not errors, f"אין שגיאות JS ({errors[:3]})")
print("\nנכשל:" if fails else "\nהכול עבר", *fails, sep="\n  ")
sys.exit(1 if fails else 0)
