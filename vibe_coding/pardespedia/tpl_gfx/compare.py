#!/usr/bin/env python3
"""לפני/אחרי על ערך אמיתי, בלי לשמור בו דבר: מחליף בזיכרון את הקריאות לתבניות
בעותקי ארגז החול, מרנדר דרך action=parse, ומזריק את התוצאה לדף החי בדפדפן.
  python3 tpl_gfx/compare.py <כותרת> <תיקיית פלט> [desk|phone]
"""
import sys, os, re, urllib.parse, requests
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build

API = "https://pardespedia.info/api.php"
title, out = sys.argv[1], sys.argv[2]
which = sys.argv[3:] or ["desk", "phone"]
VPS = {"desk": {"width": 1280, "height": 900}, "phone": {"width": 390, "height": 800}}
s = requests.Session(); s.headers["User-Agent"] = "wiki-template-reader/1.0"
d = s.get(API, params=dict(action="query", titles=title, prop="revisions", rvprop="content", rvslots="main", format="json", formatversion=2)).json()
text = d["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
for name in build.TEMPLATES:
    text = re.sub(r"\{\{\s*" + re.escape(name) + r"\s*(?=[|}\n])", "{{" + build.SANDBOX + name, text)
def period(m):
    src = s.get(API, params=dict(action="query", titles="תבנית:תקופה " + m.group(1), prop="revisions", rvprop="content", rvslots="main", format="json", formatversion=2)).json()
    body = src["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
    call = re.search(r"\{\{תקופה היסטורית(.*?)\}\}", body, re.S).group(1)
    return "{{" + build.SANDBOX + "תקופה היסטורית" + call + "}}"
text = re.sub(r"\{\{תקופה (\d)\}\}", period, text)
html = s.post(API, data=dict(action="parse", text=text, title=title, contentmodel="wikitext", prop="text", format="json", formatversion=2, disablelimitreport=1)).json()["parse"]["text"]
os.makedirs(out, exist_ok=True)
url = "https://pardespedia.info/index.php?title=" + urllib.parse.quote(title.replace(" ", "_"))
with sync_playwright() as p:
    for w in which:
        br = p.chromium.launch(args=["--no-zygote", "--single-process", "--disable-gpu"], timeout=90000)
        pg = br.new_page(viewport=VPS[w])
        pg.goto(url, wait_until="networkidle", timeout=90000)
        pg.locator(".mw-parser-output").first.screenshot(path=os.path.join(out, f"{w}_before.png"))
        pg.evaluate("h => { const o = document.querySelector('.mw-parser-output'); const t = document.createElement('div'); t.innerHTML = h; o.replaceWith(t.firstElementChild); }", html)
        pg.wait_for_timeout(500)
        sw = pg.evaluate("document.documentElement.scrollWidth")
        print(w, "scrollWidth", sw, "OVERFLOW" if sw > VPS[w]["width"] + 1 else "ok")
        pg.locator(".mw-parser-output").first.screenshot(path=os.path.join(out, f"{w}_after.png"))
        br.close()
