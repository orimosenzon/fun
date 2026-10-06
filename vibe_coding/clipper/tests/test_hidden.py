"""ייצוא כשהלשונית ברקע: הקובץ חייב לצאת באורך הנכון, בלי פריימים קפואים ובלי שקט בסוף.

יורם דיווח (10/2026) שבייצוא של רשימה הסרטונים יצאו ארוכים מהמתוכנן, עם שקט ארוך בסוף,
והתמונה קפאה עוד לפני כן. הסיבה: בזמן שהייצוא רץ הוא עבר ללשונית אחרת, וכרום מאט שם את
הטיימרים לפעם בשנייה (ואחרי חמש דקות לפעם בדקה).

בלי Playwright: כל חיבור של Playwright מכריח את הדף להישאר "גלוי" (document.hidden נשאר
false גם בלשונית ברקע), ואז הבדיקה לא רואה את מה שקורה אצל יורם. לכן מפעילים כרום עצמאי,
מדברים איתו ב-CDP גולמי (רק Runtime.evaluate), ופותחים לשונית חדשה ב-Ctrl+T אמיתי (xdotool).

הרצה:  ./make_media.sh && python3 test_hidden.py      [MUTED=0 כדי לבדוק עם קול]
"""
import base64, functools, http.server, itertools, json, os, re, shutil, subprocess, sys, tempfile, threading, time, urllib.request
import websocket

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
MUTED = os.environ.get("MUTED", "1") == "1"
A, B = 3, 23          # קטע של 20 שניות
PORT, DBG = 8767, 9333

fails = []
def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        fails.append(msg)

# האפליקציה מוגשת מ-http כדי שהדף יוכל להביא את סרטון הבדיקה ב-fetch
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.dirname(HERE))
handler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

prof = tempfile.mkdtemp(prefix="clipper-hidden-")
chrome = subprocess.Popen(["google-chrome", f"--remote-debugging-port={DBG}", f"--user-data-dir={prof}",
                           "--no-first-run", "--no-default-browser-check", "--window-size=1400,900",
                           "--autoplay-policy=no-user-gesture-required",
                           f"http://127.0.0.1:{PORT}/index.html"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{DBG}/json"))
            tab = next(t for t in tabs if t["type"] == "page" and "index.html" in t["url"])
            break
        except Exception:
            time.sleep(0.2)
    ws = websocket.create_connection(tab["webSocketDebuggerUrl"], suppress_origin=True)
    ids = itertools.count(1)

    def js(expr, wait=True):
        i = next(ids)
        ws.send(json.dumps({"id": i, "method": "Runtime.evaluate",
                            "params": {"expression": expr, "awaitPromise": wait, "returnByValue": True}}))
        while True:
            m = json.loads(ws.recv())
            if m.get("id") == i:
                r = m["result"]
                if "exceptionDetails" in r:
                    raise RuntimeError(r["exceptionDetails"])
                return r["result"].get("value")

    def until(expr, timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            if js(expr):
                return
            time.sleep(0.5)
        raise TimeoutError(expr)

    until("document.body && document.body.classList.contains('ready')")
    js("window.__errs = []; addEventListener('error', (e) => __errs.push(String(e.message)))")
    # בלי בורר תיקייה ובלי הורדה אמיתית: הקובץ נשאר ב-__clipper.lastBatch
    js("window.showDirectoryPicker = undefined; HTMLAnchorElement.prototype.click = function () {}")
    js("document.getElementById('btnBatchClear').click()")
    js("""(async () => {
        const b = await (await fetch('tests/media/source.mp4')).blob();
        const dt = new DataTransfer();
        dt.items.add(new File([b], 'source.mp4', { type: 'video/mp4' }));
        const inp = document.getElementById('fileInput');
        inp.files = dt.files;
        inp.dispatchEvent(new Event('change'));
    })()""")
    until("window.__clipper.dur > 0")
    js(f"""(() => {{ const t = document.getElementById('inBatch'); t.value = '0.{A:02d} - 0.{B:02d}\\n';
              t.dispatchEvent(new Event('input')); }})()""")
    time.sleep(0.5)
    if MUTED and not js("__clipper.set.muted"):
        js("document.getElementById('btnMute').click()")
    js("window.__vis = []; document.addEventListener('visibilitychange', () => __vis.push(document.visibilityState))")
    print(f"ייצוא {B - A} שניות, לשונית ברקע, {'מושתק' if MUTED else 'עם קול'}")
    js("document.getElementById('btnBatchRun').click()")
    time.sleep(0.8)

    # לפי הכותרת ולא לפי pid: חיפוש לפי pid מוצא גם חלונות של הכרום הרגיל של המשתמש
    title = js("document.title")
    wid = subprocess.run(["xdotool", "search", "--onlyvisible", "--name", f"^{re.escape(title)} - Google Chrome$"],
                         capture_output=True, text=True).stdout.split()
    assert len(wid) == 1, f"חלון הבדיקה לא נמצא בבירור ({wid})"
    subprocess.run(["xdotool", "windowactivate", "--sync", wid[0], "key", "ctrl+t"])
    time.sleep(0.5)
    print("   חלונות:", title, wid, subprocess.run(["xdotool", "getwindowname", wid[0]], capture_output=True, text=True).stdout.strip())
    hidden = js("document.hidden")
    print("   מוסתר:", hidden)
    check(hidden, "הדף באמת מוסתר בזמן הייצוא")

    until("!document.body.classList.contains('exporting') && (__clipper.lastBatch||[]).length === 1", 180)
    print("   stats:", json.dumps(js("window.__lastStats")))
    print("   visibility:", js("__vis"))
    errs = js("__errs")
    want = js("__clipper.lastBatch[0].seconds")   # הקטע ועוד הפתיחה והסיום, אם יש
    b64 = js("""(async () => {
        const buf = new Uint8Array(await __clipper.lastBatch[0].blob.arrayBuffer());
        let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
        return btoa(s);
    })()""")
    ws.close()
finally:
    chrome.terminate()
    chrome.wait()
    shutil.rmtree(prof, ignore_errors=True)
    srv.shutdown()

f = os.path.join(OUT, f"hidden_{'muted' if MUTED else 'sound'}.mp4")
open(f, "wb").write(base64.b64decode(b64))
pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_format", "-of", "json", f],
                               capture_output=True, text=True).stdout)
d = float(pr["format"].get("duration", 0))
check(abs(d - want) < 0.25, f"אורך {want:.2f} שנ׳ (יצא {d:.2f})")
log = subprocess.run(["ffmpeg", "-hide_banner", "-i", f, "-vf", "freezedetect=n=0.001:d=0.7",
                      "-af", "silencedetect=n=-50dB:d=0.7", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
freezes = re.findall(r"freeze_start: ([\d.]+).*?freeze_duration: ([\d.]+)", log, re.S)
# שקט מותר רק בסיום (הסיום של יורם שקט במקור): לא לפני סוף הקטע ולא הרבה אחרי סוף הקובץ
silences = [x for x in re.findall(r"silence_start: ([\d.]+)", log) if float(x) < B - A - 0.3]
print("   קפיאות:", freezes, " שקט מ:", silences)
check(not freezes, "אין קפיאות של התמונה")
check(not silences, "אין שקט בתוך הקטע")
check(not errs, f"אין שגיאות JS ({errs[:3]})")
print("\nנכשל:" if fails else "\nהכול עבר", *fails, sep="\n  ")
sys.exit(1 if fails else 0)
