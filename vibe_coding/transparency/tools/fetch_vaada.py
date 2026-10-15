#!/usr/bin/env python3
"""פרוטוקולים של הוועדה המקומית לתכנון ובנייה, מאתר הוועדה, לחיפוש ב"מה אמרו במועצה".

החיפוש באתר (SearchMeetings) מוגן ב-reCAPTCHA, אבל דף הארכיון של כל ישיבה
(MeetingsArchive) פתוח, וכך גם הקבצים עצמם (DocumentViewer). לכן לא מחפשים: עוברים
על מספרי הישיבות של כל שנה (YYYY001, YYYY002...) ועוצרים אחרי רצף של מספרים ריקים.
המספר בכתובת הוא מספר פנימי, ולפעמים הוא גדול באחד מהמספר האמיתי (2026008 הוא ישיבה
2026007). המספר האמיתי נלקח מכותרת דף הארכיון.

בכל ישיבה לוקחים את הקובץ הטוב ביותר: "פרוטוקול" (דיון והחלטות, כולל מה שנאמר),
ואם אין, "פרוטוקול החלטות" האחרון, ואם גם אין, "סדר יום" (ישיבה שעוד לא התקיימה).

    python3 tools/fetch_vaada.py [--from 2015] [--types 4,2,1]
        -> private/vaada/meetings.json, private/vaada/pdf/, private/vaada/text/

בקובצי ה-PDF האות נ מקודדת לתו פרטי (U+F8FF), ו-pdftotext מוסיף סימני כיווניות סביבו.
clean() מחזיר את האות ומסיר את הסימנים. מספרי תעודת זהות מושמטים (כמו ב-council_protocols).
"""
import argparse
import html
import json
import os
import re
import subprocess
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "private", "vaada")
BASE = "https://vaada.phk.org.il"
TYPES = {4: "ועדת המשנה לתכנון ובנייה", 2: "מליאת הוועדה לתכנון ובנייה",
         1: "רשות הרישוי המקומית", 40: "ועדת השימור", 21: "ועדת ערר"}
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120 phk-transparency"}
MISSES = 8          # רצף מספרים ריקים שאחריו השנה נגמרה
BIDI = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩"), None)
ID_RE = re.compile(r"(תעודת זהות|ת\.?\s*ז\.?)\s*[:\-]?\s*\d{6,9}")


def clean(s):
    s = s.translate(BIDI).replace("", "נ").replace("", "•")
    s = ID_RE.sub(r"\1 (הושמט)", s)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()


def get(url, **kw):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            if r.ok:
                return r
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    return None


def archive(ctype, number):
    """הקבצים של ישיבה אחת: [(תיאור, סוג, תאריך, מזהה קובץ)], והמספר האמיתי."""
    r = get(f"{BASE}/MeetingsArchive", params={"Committee_Type_Code": ctype, "Meeting_Number": number})
    if not r or "Name_File=" not in r.text:
        return None, []
    t = r.text
    docs = []
    for row in re.findall(r"<tr>(.*?)</tr>", t, re.S):
        fid = re.search(r"Name_File=(\d+)", row)
        if not fid:
            continue
        cells = [html.unescape(re.sub(r"<[^>]+>", " ", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        cells = [re.sub(r"\s+", " ", c) for c in cells]
        docs.append({"desc": cells[0] if cells else "", "kind": cells[1] if len(cells) > 1 else "",
                     "date": cells[2] if len(cells) > 2 else "", "file": fid.group(1)})
    # הכותרת מציגה את המספר שבכתובת; המספר האמיתי כתוב בתיאור הקבצים ("...מספר 2026007")
    real = [m for d in docs for m in re.findall(r"מספר\s*(\d{5,})", d["desc"])]
    return (max(set(real), key=real.count) if real else str(number)), docs


def best(docs):
    def iso(d):
        m = re.match(r"(\d\d)/(\d\d)/(\d{4})", d["date"])
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else ""
    for kind in ("פרוטוקול", "פרוטוקול החלטות", "סדר יום"):
        c = [d for d in docs if d["kind"].strip() == kind]
        if c:
            return max(c, key=iso), kind
    return None, None


def pdf_pages(fid):
    """טקסט נקי לכל עמוד. ה-PDF וה-txt נשמרים, והרצה חוזרת לא מורידה שוב."""
    pdf = os.path.join(OUT, "pdf", fid + ".pdf")
    txt = os.path.join(OUT, "text", fid + ".json")
    if os.path.exists(txt):
        return json.load(open(txt, encoding="utf-8"))
    if not os.path.exists(pdf):
        r = get(f"{BASE}/DocumentViewer", params={"pageId": 14365, "Name_File": fid})
        if not r or not r.content.startswith(b"%PDF"):
            return None
        open(pdf, "wb").write(r.content)
    raw = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
    pages = [clean(p) for p in raw.split("\f")]
    while pages and not pages[-1]:
        pages.pop()
    json.dump(pages, open(txt, "w", encoding="utf-8"), ensure_ascii=False)
    return pages


def meeting_date(pages, docs):
    """תאריך הישיבה מהכותרת ("בתאריך: 26/08/2026"), ואם אין, מתאריך הקובץ המוקדם."""
    head = " ".join(pages[:2]) if pages else ""
    m = re.search(r"בתאריך\s*:?\s*(\d\d)/(\d\d)/(\d{4})", head) or \
        re.search(r"(\d\d)/(\d\d)/(\d{4})\s*:?\s*בתאריך", head)
    if not m:
        ds = sorted(re.findall(r"(\d\d)/(\d\d)/(\d{4})", " ".join(d["date"] for d in docs)),
                    key=lambda x: (x[2], x[1], x[0]))
        if not ds:
            return ""
        return f"{ds[0][2]}-{ds[0][1]}-{ds[0][0]}"
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2015)
    ap.add_argument("--to", dest="end", type=int, default=2026)
    ap.add_argument("--types", default="4,2,1,40,21")
    a = ap.parse_args()
    for d in ("pdf", "text"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    path = os.path.join(OUT, "meetings.json")
    known = {(m["type"], m["internal"]): m for m in
             (json.load(open(path, encoding="utf-8")) if os.path.exists(path) else [])}
    for ctype in map(int, a.types.split(",")):
        for year in range(a.start, a.end + 1):
            miss, n, found = 0, 0, 0
            while miss < MISSES and n < 200:
                n += 1
                internal = year * 1000 + n
                prev = known.get((ctype, internal))
                # ישיבה שכבר יש לה פרוטוקול מלא לא משתנה; ישיבה עם סדר יום בלבד נבדקת שוב
                if prev and prev["kind"] == "פרוטוקול":
                    miss, found = 0, found + 1
                    continue
                real, docs = archive(ctype, internal)
                time.sleep(0.3)
                if not docs:
                    miss += 1
                    continue
                miss, found = 0, found + 1
                doc, kind = best(docs)
                if not doc:
                    continue
                pages = pdf_pages(doc["file"]) or []
                known[(ctype, internal)] = {
                    "type": ctype, "body": TYPES.get(ctype, str(ctype)), "internal": internal,
                    "number": real, "date": meeting_date(pages, docs), "kind": kind,
                    "file": doc["file"], "pages": len(pages), "chars": sum(map(len, pages)),
                    "url": f"{BASE}/DocumentViewer?pageId=14365&Name_File={doc['file']}",
                    "archive": f"{BASE}/MeetingsArchive?Committee_Type_Code={ctype}&Meeting_Number={internal}",
                    "docs": docs}
            print(f"{TYPES.get(ctype)} {year}: {found}", flush=True)
            json.dump(sorted(known.values(), key=lambda m: (m["date"], m["type"])),
                      open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
