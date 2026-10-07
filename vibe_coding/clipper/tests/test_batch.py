"""בדיקת רשימת קטעים: טעינה מקובץ, הדבקה, פענוח, אזהרות, וייצוא של כמה סרטונים בבת אחת.

הרצה:  ./make_media.sh && python3 test_batch.py   (כמו test_e2e.py: Chrome אמיתי, חלון גלוי)
"""
import base64, json, os, subprocess, sys
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

# הפורמט של יורם: כותרת, נקודות בזמנים, And, ומפריד שורות של Pages (U+2028)
LIST = "Session 1:\n\n0.02 - 0.04\n0.10-0.12 And 0.20 - 0.21 0.40 - 0.50\n0.11 - 0.12\n"

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=os.environ.get("HEADLESS") == "1",
                                args=["--autoplay-policy=no-user-gesture-required"])
    page = browser.new_page(viewport={"width": 1400, "height": 900}, accept_downloads=True)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.goto(URL)
    page.wait_for_function("document.body.classList.contains('ready')")
    page.evaluate("C.setLang('he')")
    # בלי בורר תיקייה: הבדיקה עוברת במסלול ההורדות, שעובד בכל דפדפן
    page.evaluate("delete window.showDirectoryPicker; window.showDirectoryPicker = undefined")
    page.evaluate("document.getElementById('btnBatchClear').click()")

    print("טעינה מקובץ טקסט")
    lst = OUT / "list.txt"
    lst.write_text(LIST, encoding="utf-8")
    page.set_input_files("#batchInput", str(lst))
    page.wait_for_function("window.__batch.items.length === 4")
    check(page.input_value("#inBatch") == LIST.replace("\u2028", "\n"), "הטקסט מהקובץ נכנס לתיבה, כל טווח בשורה משלו")

    print("טעינת סרטון ופענוח")
    page.set_input_files("#fileInput", str(MEDIA / "source.mp4"))
    page.wait_for_function("window.__clipper.dur > 0")
    page.wait_for_timeout(300)
    items = page.evaluate("__batch.items.map(it => ({n: it.n, ranges: it.ranges, errors: it.errors}))")
    check([it["n"] for it in items] == [1, 2, 3, 4], f"ארבעה סרטונים (יצא {[it['n'] for it in items]})")
    check(items[1]["ranges"] == [[10, 12], [20, 21]], f"And מחבר שני טווחים (יצא {items[1]['ranges']})")
    rows = page.locator(".bitem")
    check("אחרי סוף הסרטון" in rows.nth(2).inner_text(), "קטע 40–50 מסומן כאחרי סוף הסרטון")
    check(rows.nth(2).locator("input").is_disabled(), "ואי אפשר לבחור אותו")
    check("דומה מאוד לקטע 2" in rows.nth(3).inner_text(), "קטע 11–12 מסומן כדומה לקטע 2")

    print("הדבקה ועריכה")
    check(not page.is_visible("#inBatch"), "אחרי טעינת קובץ תיבת הטקסט מקופלת")
    page.click("#btnBatchEdit")
    page.fill("#inBatch", LIST + "1.05 -\n")
    page.wait_for_timeout(300)
    check("לא מצאתי כאן טווח" in page.locator(".bitem").last.inner_text(), "שורה עם זמן בלי טווח מקבלת הסבר")
    page.fill("#inBatch", LIST)
    page.wait_for_timeout(300)
    page.locator(".bitem").nth(3).locator("input").uncheck()
    check("2 סרטונים" in page.inner_text("#dockSum"), f"סיכום: 2 סרטונים (יצא {page.inner_text('#dockSum')})")

    print("לחיצה על שורה מציגה אותה בטיימליין")
    page.locator(".bitem").nth(1).locator(".br").click()
    s = page.evaluate("[__clipper.selIn, __clipper.selOut]")
    check(s == [10, 12], f"הבחירה 10–12 (יצא {s})")

    print("לחיצה על פס של קטע בטיימליין")
    page.locator(".bitem").nth(0).locator(".br").click()
    page.click("#btnZoomFit")
    tb = page.locator("#timeline").bounding_box()
    dur = page.evaluate("__clipper.dur")
    page.mouse.click(tb["x"] + tb["width"] * 11 / dur, tb["y"] + 24 + 9)
    cur = page.evaluate("__batch.cur")
    check(cur and cur.startswith("0.10-0.12"), f"הפס של קטע 2 בוחר אותו (יצא {cur!r})")

    print("ייצוא של כולם")
    page.click("#btnBatchRun")
    page.wait_for_timeout(1500)
    page.screenshot(path=OUT / "batch_running.png")
    page.wait_for_function("!document.body.classList.contains('exporting') && (__clipper.lastBatch||[]).length === 2", timeout=60000)
    page.wait_for_timeout(300)
    page.screenshot(path=OUT / "batch_done.png")
    got = page.evaluate("__clipper.lastBatch.map(b => ({name: b.name, size: b.size}))")
    print("  ", got)
    blobs = []
    for i in range(2):
        b64 = page.evaluate(f"""async () => {{
            const b = __clipper.lastBatch[{i}].blob;
            const buf = new Uint8Array(await b.arrayBuffer());
            let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
            return btoa(s);
        }}""")
        blobs.append(base64.b64decode(b64))
    browser.close()

check(got[0]["name"].startswith("source_01_0-00-02_"), f"שם הקובץ הראשון (יצא {got[0]['name']})")
check(got[1]["name"].startswith("source_02_0-00-10_"), f"שם הקובץ השני (יצא {got[1]['name']})")
for (g, data), want in zip(zip(got, blobs), [2, 3]):
    f = OUT / g["name"]
    f.write_bytes(data)
    pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_format", "-of", "json", str(f)],
                                   capture_output=True, text=True).stdout)
    d = float(pr["format"].get("duration", 0))
    check(abs(d - want) < 0.5, f"{g['name']}: אורך כ-{want} שנ׳ (יצא {d:.2f})")

print()
check(not errors, f"אין שגיאות JS ({errors[:3]})")
print("\nנכשל:" if fails else "\nהכול עבר", *fails, sep="\n  ")
sys.exit(1 if fails else 0)
