#!/usr/bin/env python3
"""דוחות הפעילות השנתיים של הוועדה המקומית לתכנון ובנייה, ל"מה אמרו במועצה".

דף "דוחות שנתיים" באתר הוועדה (בלי קפצ'ה): לכל שנה, או חציון, PDF של היתרים שהוצאו, רישוי
וביצוע, תכניות, תשריטים, שימוש חורג, ישיבות ומחשוב, ועוד מדיניות האכיפה של הוועדה. זה המקום
היחיד שמרכז את ההיתרים שהוצאו בפועל, ולא רק את ההחלטות שבפרוטוקולים.

שנה וחציון נלקחים משם הקובץ, או מהכותרת האחרונה עם שנה לפני הקישור. עמוד בלי טקסט עובר OCR.

    python3 tools/fetch_vaada_reports.py   ->   private/vaada_reports/docs.json
"""
import html
import json
import os
import re
from urllib.parse import quote, unquote

import requests

from fetch_council_protocols import ocr_page, text_pages, TESS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "private", "vaada_reports")
BASE = "https://vaada.phk.org.il"
PAGE = BASE + "/" + quote("מידע-כללי/דוחות-שנתיים/")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120 phk-transparency"}
EMPTY = 40
MAX_OCR = 40


def main():
    if not os.path.exists(TESS):
        raise SystemExit(f"אין Tesseract ב-{TESS}")
    os.makedirs(os.path.join(OUT, "pdf"), exist_ok=True)
    t = requests.get(PAGE, headers=UA, timeout=60).text
    t = re.sub(r"<(script|style).*?</\1>", "", t, flags=re.S)
    docs, seen = [], set()
    for m in re.finditer(r'<a [^>]*href="([^"]+\.pdf)"[^>]*>(.*?)</a>', t, re.S | re.I):
        href, label = m.group(1), html.unescape(re.sub(r"<[^>]+>|\(נפתח בחלון חדש\)", "", m.group(2))).strip()
        url = href if href.startswith("http") else BASE + quote(href)
        url = url.replace("http://", "https://")
        if url in seen:
            continue
        seen.add(url)
        name = unquote(url.rsplit("/", 1)[-1])
        before = re.sub(r"<[^>]+>", " ", t[max(0, m.start() - 4000):m.start()])
        years = re.findall(r"20[12]\d", name) or re.findall(r"20[12]\d", before)
        year = years[-1] if years else ""
        if "policy" in name:  # למדיניות אין שנה בשם או בדף; היא כתובה בגוף המסמך
            year = ""
        half = "חציון שני" if re.search(r"h2|חציון.?שני|אפריל|10-12", name, re.I) else \
            "חציון ראשון" if re.search(r"h1|חציון.?1|חציון.?ראשון", name, re.I) else ""
        pdf = os.path.join(OUT, "pdf", re.sub(r"[^\w.-]", "_", url.rsplit("/", 2)[-2]) + ".pdf")
        if not os.path.exists(pdf):
            r = requests.get(url, headers=UA, timeout=120)
            if not r.ok or b"%PDF" not in r.content[:1024]:
                print("נכשל:", name)
                continue
            open(pdf, "wb").write(r.content)
        pages, out, n_ocr = text_pages(pdf), [], 0
        for i, p in enumerate(pages, 1):
            if len(p.strip()) < EMPTY and n_ocr < MAX_OCR:
                p = ocr_page(pdf, i, "v" + os.path.basename(pdf)[:-4])
                n_ocr += 1
            if len(p.strip()) >= EMPTY:
                out.append({"page": i, "text": p})
        if not out:
            continue
        if not year:
            ys = re.findall(r"20[12]\d", out[0]["text"])
            year = max(ys) if ys else ""
        what = label or name[:-4].replace("-", " ").replace("_", " ")
        policy = "policy" in name
        title = what if policy else re.sub(r"\s+:", ":", f"דוח שנתי {year} {half}: {what}")
        docs.append({"date": f"{year}-12-31" if year else "", "body": "הוועדה המקומית לתכנון ובנייה",
                     "kind": "מדיניות" if policy else "דוח פעילות שנתי", "title": title.strip(),
                     "url": url, "pages": out})
        print(f"{year} {title[:70]} ({len(out)} עמ', {n_ocr} OCR)", flush=True)
    json.dump(docs, open(os.path.join(OUT, "docs.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(docs), "דוחות")


if __name__ == "__main__":
    main()
