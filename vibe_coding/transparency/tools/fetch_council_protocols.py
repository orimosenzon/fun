#!/usr/bin/env python3
"""פרוטוקולי המליאה מהארכיון של המועצה עצמה, עם OCR לעמודים הסרוקים, ל"מה אמרו במועצה".

goinfo מכסה את רוב הפרוטוקולים, אבל לא את כולם, ובשני מקרים חסרות בו בדיוק ההצבעות:
- **פרוטוקול שכולו סריקה** (12/2024 עד 11/2025 באתר המועצה) ש-goinfo לא העלה: 12/2024,
  1/2025, 2/2025, אחד משני הפרוטוקולים של 3/2025. אין להם שום טקסט בשום מקום.
- **העמוד האחרון של פרוטוקול דיגיטלי** (מ-12/2025): עמוד החתימות נסרק בחזרה כתמונה, ועליו
  נופלת לא פעם ההצבעה האחרונה של הישיבה, עם שמות המצביעים בעד ונגד. goinfo מחזיק רק את שכבת
  הטקסט, ולכן העמוד הזה חסר גם אצלו.

מה נכנס: פרוטוקול ש-goinfo לא מחזיק נכנס כולו (שכבת טקסט, ו-OCR לכל עמוד בלי טקסט).
פרוטוקול ש-goinfo מחזיק תורם רק את העמודים הסרוקים שלו, כמסמך "עמוד סרוק" עם מספר העמוד.

ה-OCR הוא Tesseract עם המודל המדויק לעברית (tessdata_best). נבדק מול מודלי תמונה של Azure:
אלה המציאו שמות של חברי מועצה (שם משפחה אחר לגמרי, שנשמע סביר), וזה פוסל אותם לעמוד הצבעות.
Tesseract מותקן בלי sudo, מחבילות דביאן שחולצו ל-~/.local/opt/tesseract (ראו TESS למטה).

    python3 tools/fetch_council_protocols.py   ->   private/council_site/docs.json
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PARDES = os.path.join(HERE, "..", "..", "pardespedia")
sys.path.insert(0, PARDES)
# אותו קורא של דף הארכיון שמזין את [[ישיבות מועצה]] בפרדספדיה, ואותו מטמון PDF
from council_protocols import (SOURCE, UA, CACHE, ID_RE, parse_cards, card_date,  # noqa: E402
                               encode_url, protocol_date, meeting_number)

OUT = os.path.join(HERE, "..", "private", "council_site")
GOINFO = os.path.join(HERE, "..", "private", "goinfo", "docs.json")

TESS_ROOT = os.path.expanduser("~/.local/opt/tesseract")
TESS = os.path.join(TESS_ROOT, "root", "usr", "bin", "tesseract")
TESS_ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.join(TESS_ROOT, "root", "usr", "lib", "x86_64-linux-gnu"))
TESSDATA = os.path.join(TESS_ROOT, "tessdata_best")
EMPTY = 40  # עמוד עם פחות תווים מזה הוא סריקה (pdftotext מחזיר ממנו לכל היותר מספר עמוד)


def norm_url(u):
    return unquote(u).strip().rsplit("/", 1)[-1]


def download(url, key):
    path = os.path.join(CACHE, key + ".pdf")
    if not os.path.exists(path):
        r = requests.get(encode_url(url), headers=UA, timeout=120)
        r.raise_for_status()
        if b"%PDF" not in r.content[:1024]:
            raise ValueError("לא PDF")
        os.makedirs(CACHE, exist_ok=True)
        open(path, "wb").write(r.content)
    return path


def text_pages(pdf):
    out = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, check=True).stdout
    return out.decode("utf-8", "replace").split("\f")[:-1]


def ocr_page(pdf, pno, key):
    cache = os.path.join(OUT, "ocr", f"{key}-p{pno:03d}.txt")
    if os.path.exists(cache):
        return open(cache, encoding="utf-8").read()
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "p")
        subprocess.run(["pdftoppm", "-r", "300", "-f", str(pno), "-l", str(pno), "-png", "-singlefile",
                        pdf, base], check=True, capture_output=True)
        txt = subprocess.run([TESS, base + ".png", "-", "-l", "heb", "--tessdata-dir", TESSDATA,
                              "--psm", "6"], capture_output=True, env=TESS_ENV).stdout.decode("utf-8", "replace")
    txt = re.sub(r"[ \t]+", " ", txt).strip()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    open(cache, "w", encoding="utf-8").write(txt)
    return txt


def main():
    if not os.path.exists(TESS):
        sys.exit(f"אין Tesseract ב-{TESS}. ההתקנה בראש tools/fetch_council_protocols.py")
    goinfo = {norm_url(d["url"]) for d in json.load(open(GOINFO, encoding="utf-8"))} \
        if os.path.exists(GOINFO) else set()
    page = requests.get(SOURCE, headers=UA, timeout=60)
    page.raise_for_status()
    cards = [(card_date(c), c) for c in parse_cards(page.text)]
    docs, seen, n_ocr = [], set(), 0
    for d, card in sorted([x for x in cards if x[0]], key=lambda x: x[0]):
        url = (card.get("protocol") or {}).get("url", "")
        if not url:
            continue
        key = "%s-%s" % (d.isoformat(), hashlib.md5(url.encode()).hexdigest()[:8])
        try:
            pdf = download(url, key)
        except Exception as e:
            print("נכשל:", key, e)
            continue
        digest = hashlib.md5(open(pdf, "rb").read()).hexdigest()
        if digest in seen:  # אותו קובץ הועלה פעמיים (5/1/2025)
            continue
        seen.add(digest)
        pages = text_pages(pdf)
        scanned = [i + 1 for i, p in enumerate(pages) if len(p.strip()) < EMPTY]
        in_goinfo = norm_url(url) in goinfo
        digital = len(scanned) < len(pages)
        if in_goinfo and not digital:
            continue  # goinfo כבר עשה OCR לכל הסריקה
        texts = []
        for i, p in enumerate(pages, 1):
            if i in scanned:
                p = ocr_page(pdf, i, key)
                n_ocr += 1
            elif in_goinfo:
                continue
            texts.append(ID_RE.sub("ת.ז. (הושמט)", p))
        full = "\n".join(texts)
        date = (protocol_date(full, d) or d).isoformat()
        title = (card.get("title") or {}).get("text", "") or "ישיבת מליאה"
        num = meeting_number(card, full)
        kept = [i for i in range(1, len(pages) + 1) if not in_goinfo or i in scanned]
        docs.append({"date": date, "body": "מליאת המועצה",
                     "kind": "עמוד סרוק" if in_goinfo else "פרוטוקול",
                     "title": title + (f" (ישיבה {num})" if num and num not in title else ""),
                     "url": url, "pages": [{"page": i, "ocr": i in scanned, "text": t}
                                           for i, t in zip(kept, texts)]})
        print(f"{date} {'עמוד סרוק' if in_goinfo else 'פרוטוקול מלא'}: {len(kept)} עמודים, "
              f"{sum(1 for i in kept if i in scanned)} ב-OCR", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(docs, open(os.path.join(OUT, "docs.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(docs)} מסמכים, {n_ocr} עמודים ב-OCR")


if __name__ == "__main__":
    main()
