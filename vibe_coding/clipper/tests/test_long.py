"""סרטון ארוך מהמקסימום של הרשת: אזהרה על המסך, בלי חלון אישור, והייצוא באורך מלא.

הרצה:  python3 test_long.py   (לוקח כ-4 דקות, כי הייצוא בזמן אמת)
יוצר בעצמו מקור של 200 שנ׳ ב-media/long.mp4 אם אין.
"""
import base64, json, os, subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
URL = (HERE.parent / "index.html").as_uri()
SEC = 200
LONG = MEDIA / "long.mp4"
if not LONG.exists():
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={SEC}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={SEC}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(LONG)], check=True)

fails = []
def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        fails.append(msg)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=os.environ.get("HEADLESS") == "1",
                                args=["--autoplay-policy=no-user-gesture-required"])
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    dialogs = []
    page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    page.goto(URL)
    page.wait_for_function("document.body.classList.contains('ready')")
    page.evaluate("C.setLang('he')")
    page.set_input_files("#fileInput", str(LONG))
    page.wait_for_function("window.__clipper.dur > 0")
    page.click("[data-tab='format']")
    page.click('#platforms button[data-id="ig_reels"]')
    page.click("[data-tab='clips']")
    page.fill("#inIn", "0")
    page.press("#inIn", "Enter")
    page.fill("#inOut", f"{SEC // 60}:{SEC % 60:02d}")
    page.press("#inOut", "Enter")

    print("אזהרה")
    check(page.is_visible("#lenWarn"), "אזהרת אורך מופיעה ברילס")
    warn = page.inner_text("#lenWarn")
    print("  ", warn)
    check("תייצא את כל האורך" in warn, "האזהרה אומרת שהייצוא יהיה באורך מלא")

    print("ייצוא")
    page.click("#btnExport")
    page.wait_for_function("!!window.__clipper.lastExport", timeout=(SEC + 120) * 1000)
    check(not dialogs, f"לא קפץ חלון אישור ({dialogs})")
    name = page.evaluate("__clipper.lastExport.name")
    b64 = page.evaluate("""async () => {
        const buf = new Uint8Array(await window.__clipper.lastExport.blob.arrayBuffer());
        let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
        return btoa(s);
    }""")
    browser.close()

out = OUT / name
out.write_bytes(base64.b64decode(b64))
probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_format", "-of", "json", str(out)],
                                  capture_output=True, text=True).stdout)
fdur = float(probe["format"].get("duration", 0))
print(f"קובץ: {out.name} {fdur:.2f}s {out.stat().st_size / 1e6:.0f}MB")
check(abs(fdur - SEC) < 2, f"אורך הקובץ כ-{SEC} שנ׳ (יצא {fdur:.2f})")

print("נכשלו:" if fails else "הכול עבר", *fails, sep="\n  ")
raise SystemExit(1 if fails else 0)
