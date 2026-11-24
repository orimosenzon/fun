"""מוריד את הקבצים שב-src/manifest.json ל-src/<שנה>/ (דרך כרום, האתר חוסם סקריפטים).

מדלג על קובצי "תקציב ודברי הסבר" (m), שהם רק איחוד של t ו-d.
שם הקובץ המקומי: <מזהה המדיה>_<שם>, כי אותו שם חוזר בשנים ובמדורים שונים.
"""
import json, sys, re, base64
from pathlib import Path
from urllib.parse import unquote
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent

# fetch מתוך הדף: בקשה צדדית של Playwright נחסמת (403), fetch של הדף עצמו עובר.
FETCH_JS = """async (u) => {
  const r = await fetch(u, {credentials: 'include'});
  if (!r.ok) return {status: r.status, b64: null};
  const buf = new Uint8Array(await r.arrayBuffer());
  let s = ''; const CH = 0x8000;
  for (let i = 0; i < buf.length; i += CH) s += String.fromCharCode.apply(null, buf.subarray(i, i + CH));
  return {status: r.status, b64: btoa(s)};
}"""

def local_name(url):
    url = url.split("#")[0]
    parts = url.split("/media/")[1].split("/")
    return unquote("_".join(parts))

def wanted(man, years):
    for y in years:
        d = man[y]
        urls = [x["href"] for x in d["loose"]]
        for t in d["tables"]:
            for r in t["rows"]:
                for ci, c in enumerate(r):
                    col = t["head"][ci] if ci < len(t["head"]) else ""
                    if "ודברי" in col or "+" in col:
                        continue
                    urls += c["links"]
        seen = set()
        for u in urls:
            u = u.split("#")[0]
            if u not in seen:
                seen.add(u)
                yield y, u

def main():
    man = json.loads((ROOT / "src/manifest.json").read_text())
    years = sys.argv[1:] or sorted(man)
    jobs = [(y, u) for y, u in wanted(man, years)
            if not (ROOT / "src" / y / local_name(u)).exists()]
    print("to download:", len(jobs))
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=False, args=["--window-position=2000,2000"])
        ctx = b.new_context()
        pg = ctx.new_page()
        pg.goto("https://www.jerusalem.muni.il/he/city/budget/", wait_until="networkidle")
        bad = 0
        for i, (y, u) in enumerate(jobs):
            out = ROOT / "src" / y / local_name(u)
            out.parent.mkdir(parents=True, exist_ok=True)
            res = pg.evaluate(FETCH_JS, u)
            body = base64.b64decode(res["b64"]) if res["b64"] else b""
            if res["status"] != 200 or body[:5].startswith(b"<"):
                bad += 1
                print("FAIL", res["status"], u)
                continue
            out.write_bytes(body)
            if i % 50 == 0:
                print(i, y, out.name, len(body))
        print("failed:", bad)
        b.close()

if __name__ == "__main__":
    main()
