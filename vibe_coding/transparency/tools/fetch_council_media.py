#!/usr/bin/env python3
"""מסמכים מאתר המועצה, דרך ה-API של WordPress (פתוח, בלי קפצ'ה), ל"מה אמרו במועצה".

באתר כ-3,500 קובצי PDF, ורובם מודעות דרושים ומסמכי מכרז. נכנס רק מה שעונה לשאלה "מה המועצה
החליטה, כתבה או התחייבה לו", לפי רשימה מותרת של סוגים (CATS): הסכמים קואליציוניים, חוקי עזר,
דוחות ביקורת, דוחות כספיים, מדיניות ונהלים, תוכניות עבודה ומתאר, סקרים, מצגות למפגשי תושבים,
הודעות. ועוד דבר אחד: ב-3/2025 האתר העלה מחדש את הארכיון הישן של המליאה, ובו תמלילים וסדרי
יום של 2011–2017 ש-goinfo לא מחזיק. אלה נכנסים כשאין ל-goinfo מסמך מאותו סוג באותו תאריך.

עמוד בלי שכבת טקסט עובר OCR (אותו Tesseract של fetch_council_protocols.py), עד MAX_OCR עמודים
למסמך, כי יש באתר סקרים של מאות עמודים סרוקים.

    python3 tools/fetch_council_media.py   ->   private/council_media/docs.json
"""
import hashlib
import html
import json
import os
import re
import subprocess
import time
from urllib.parse import unquote

import requests

from fetch_council_protocols import ocr_page, text_pages, norm_url, ID_RE, encode_url, UA, TESS
from fetch_goinfo import DATE_RE

HERE = os.path.dirname(os.path.abspath(__file__))
PRIV = os.path.join(HERE, "..", "private")
OUT = os.path.join(PRIV, "council_media")
API = "https://www.pardes-hanna-karkur.muni.il/wp-json/wp/v2/media"
MAX_OCR = 60
EMPTY = 40

# (גוף, סוג, ביטוי בכותרת). הסדר קובע: הראשון שמתאים.
CATS = [
    ("מליאת המועצה", "תמלול", r"תמלי|תמלו"),
    ("מליאת המועצה", "סדר יום", r"סדר.?יום|זימון"),
    ("המועצה", "הסכם קואליציוני", r"קואליצ"),
    ("מבקר המועצה", "דוח ביקורת", r"מבקר|ביקורת|תלונות.?הציבור"),
    ("המועצה", "חוק עזר", r"חוק.?ה?עזר"),
    ("גזברות", "דוח כספי", r"דוח.?(ות)?.?כספי|דו\"ח.?כספי|רבעוני"),
    ("המועצה", "מדיניות ונהלים", r"מדיניות|נוהל|נהלי|אמנת"),
    ("המועצה", "תוכנית", r"(תוכנית|תכנית).?(אב|אסטרטג|עבודה|מתאר)|חזון"),
    ("המועצה", "סקר", r"סקר"),
    ("המועצה", "מצגת למפגש תושבים", r"מצגת|מפגש.?תושבים|שיתוף.?ציבור|מידעון"),
    ("המועצה", "הקצאת מקרקעין", r"הקצאת.?(מקרקעין|קרקע)"),
    ("המועצה", "הודעה", r"הודעה.?(על|בדבר|לגופים|בנושא)|הודעת.?פרסום"),
]
SKIP = r"דרושים|מכרז|הבהר|קול.?קורא|הצעת.?מחיר|תפקיד|השכרת|חניה.?שמורה|יועלה.?לאחר|הגשת.?תלונה|טופס"


def listing():
    out, page = [], 1
    while True:
        r = requests.get(API, params={"per_page": 100, "page": page, "mime_type": "application/pdf",
                                      "_fields": "id,date,title,source_url"}, headers=UA, timeout=60)
        if r.status_code != 200 or not r.json():
            return out
        out += [{"id": x["id"], "uploaded": x["date"][:10], "title": html.unescape(x["title"]["rendered"]),
                 "url": x["source_url"]} for x in r.json()]
        page += 1
        time.sleep(0.3)


def title_date(title, uploaded):
    """התאריך שבכותרת ("תמליל 22.6.17"), ואם אין, תאריך ההעלאה."""
    for d, m, y in DATE_RE.findall(re.sub(r"(\.\d)\s+(\d)", r"\1\2", title)):
        y = int(y) + (2000 if len(y) == 2 else 0)
        if 1 <= int(m) <= 12 and 1 <= int(d) <= 31 and 2005 <= y and f"{y}" <= uploaded[:4]:
            return f"{y:04d}-{int(m):02d}-{int(d):02d}"
    return uploaded


def main():
    if not os.path.exists(TESS):
        raise SystemExit(f"אין Tesseract ב-{TESS}")
    os.makedirs(os.path.join(OUT, "pdf"), exist_ok=True)
    goinfo = json.load(open(os.path.join(PRIV, "goinfo", "docs.json"), encoding="utf-8"))
    g_urls = {norm_url(d["url"]) for d in goinfo}
    g_dates = {(d["date"], d["kind"]) for d in goinfo if d["body"] == "מליאת המועצה"}
    site = os.path.join(PRIV, "council_site", "docs.json")
    s_urls = {norm_url(d["url"]) for d in json.load(open(site, encoding="utf-8"))} if os.path.exists(site) else set()

    items = listing()
    json.dump(items, open(os.path.join(OUT, "list.json"), "w", encoding="utf-8"), ensure_ascii=False)
    docs, seen = [], set()
    for it in sorted(items, key=lambda x: x["uploaded"]):
        t = re.sub(r"[-_]", " ", it["title"])
        if re.search(SKIP, t) or norm_url(it["url"]) in g_urls | s_urls:
            continue
        cat = next(((b, k) for b, k, rx in CATS if re.search(rx, t)), None)
        if not cat:
            continue
        date = title_date(t, it["uploaded"])
        # מהארכיון הישן נכנס רק מה ש-goinfo לא מחזיק; תמלול ופרוטוקול מאותו יום הם אותה ישיבה
        if cat[0] == "מליאת המועצה" and ((date, cat[1]) in g_dates or (date, "פרוטוקול") in g_dates
                                         and cat[1] == "תמלול"):
            continue
        key = f"{it['id']}"
        pdf = os.path.join(OUT, "pdf", key + ".pdf")
        if not os.path.exists(pdf):
            r = requests.get(encode_url(it["url"]), headers=UA, timeout=180)
            if not r.ok or b"%PDF" not in r.content[:1024]:
                print("נכשל:", it["title"][:60])
                continue
            open(pdf, "wb").write(r.content)
            time.sleep(0.3)
        digest = hashlib.md5(open(pdf, "rb").read()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        try:
            pages = text_pages(pdf)
        except subprocess.CalledProcessError:
            print("PDF פגום:", it["title"][:60])
            continue
        out_pages, n_ocr = [], 0
        for i, p in enumerate(pages, 1):
            if len(p.strip()) < EMPTY:
                if n_ocr >= MAX_OCR:
                    continue
                p = ocr_page(pdf, i, "m" + key)
                n_ocr += 1
            p = ID_RE.sub("ת.ז. (הושמט)", p)
            if len(p.strip()) >= EMPTY:
                out_pages.append({"page": i, "ocr": n_ocr > 0 and len(pages[i - 1].strip()) < EMPTY, "text": p})
        if not out_pages:
            continue
        title = re.sub(r"^\d_|[-_]?[0-9a-f]{20,}$", "", it["title"]).replace("-", " ").strip()
        docs.append({"date": date, "body": cat[0], "kind": cat[1], "title": title, "url": it["url"],
                     "pages": out_pages})
        print(f"{date} {cat[1]}: {title[:60]} ({len(out_pages)} עמ', {n_ocr} OCR)", flush=True)
    json.dump(docs, open(os.path.join(OUT, "docs.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(docs), "מסמכים")


if __name__ == "__main__":
    main()
