#!/usr/bin/env python3
"""מסמכי המועצה מארכיון goinfo, לחיפוש ב"מה אמרו במועצה": פרוטוקולים ותמלולים של
המליאה, סדרי יום, ופרוטוקולים וסיכומים של ועדות המועצה (תנועה, תמיכות, מכרזים, איכות
הסביבה ועוד).

goinfo מחזיק דף HTML סטטי לכל מסמך שהמועצה פרסמה, עם הטקסט המלא (OCR) וקישור ל-PDF
המקורי באתר המועצה. זה המקור היחיד לטקסט של הפרוטוקולים עד סוף 2025, שבאתר המועצה הם
סריקות בלי שכבת טקסט. תקציבים, מכרזים וקולות קוראים לא נכנסים: לתקציב יש אפליקציה
משלו, והשאר הם מסמכי רכש ולא דברים שנאמרו או הוחלטו.

    python3 tools/fetch_goinfo.py   ->   private/goinfo/pages/, private/goinfo/docs.json
"""
import hashlib
import html
import json
import os
import re
import time
from urllib.parse import quote

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "private", "goinfo")
BASE = "https://goinfo.co.il/pro/protocols/pardes-hanna-karkur/"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120 phk-transparency"}
ID_RE = re.compile(r"(תעודת זהות|ת\.?\s*ז\.?)\s*[:\-]?\s*\d{6,9}")


def classify(name):
    """(גוף, סוג) לפי שם הקובץ, או None למסמך שלא נכנס. השמות מגיעים עם רווחים ומקפים
    באמצע מילים ("פרו-טקול", "תמ-לול"), ולכן בודקים על השם בלי רווחים ומקפים."""
    n = re.sub(r"[-\s_.]", "", name)
    if re.search(r"תקציב|מכרז|קולקורא|הצעתמחיר|מסמךשינויים|הבהרות|נגישות|התחייבות|קישורלטופס|"
                 r"תבחינים|רשימתחברי|מכירתכלי|תכניתמפורטת|תוכניתפיתוח|מבקר|הודעהבנושא", n) \
            and not re.search(r"פרוטו?קול|סיכום", n):
        return None
    if re.search(r"ועדתמכרזים|וועדתמכרזים", n):
        return "ועדת המכרזים", "פרוטוקול"
    for key, body in (("תנועה", "ועדת התנועה"), ("תמיכות", "ועדת התמיכות"),
                      ("איכותהסביבה", "ועדת איכות הסביבה"), ("גביה", "ועדת הגבייה"),
                      ("ביטחון", "ועדת הביטחון"), ("שלושה", "ועדת שלושה")):
        if key in n:
            return body, "סיכום" if "סיכום" in n else "פרוטוקול"
    if re.search(r"תמלול|תמליל", n):
        return "מליאת המועצה", "תמלול"
    if re.search(r"סדריום|זימון", n):
        return "מליאת המועצה", "סדר יום"
    if re.search(r"פרוטו?קול", n) or re.search(r"ישיבתמועצה", n):
        return "מליאת המועצה", "פרוטוקול"
    return None


def page_text(t):
    m = re.search(r'<div class="protocol-text">(.*?)</article>', t, re.S)
    body = m.group(1) if m else ""
    body = re.sub(r"<br\s*/?>|</(h\d|p|div|li|tr)>", "\n", body)
    body = html.unescape(re.sub(r"<[^>]+>", "", body))
    body = ID_RE.sub(r"\1 (הושמט)", body)
    body = re.sub(r"[ \t]+", " ", body)
    return re.sub(r"\n\s*\n+", "\n", body).strip()


DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](20\d\d|\d\d)(?!\d)")


def sane_date(date, title, text, pdf_url):
    """התאריך ש-goinfo נותן בשם הקובץ, אלא אם הוא חסר או סותר את תיקיית ההעלאה של ה-PDF באתר
    המועצה (uploads/YYYY/MM). goinfo קרא את "מועצה 27 11.12.25" כ-27/11/2012, ומסמכים אחרים
    הגיעו בלי תאריך בכלל. אז לוקחים את התאריך הראשון שבכותרת או בפתיחת המסמך, ובלבד שהוא לא
    אחרי ההעלאה ולא יותר משנתיים לפניה."""
    up = re.search(r"/uploads/(20\d\d)/(\d\d)/", pdf_url)
    if not up:
        return date
    upload = f"{up.group(1)}-{up.group(2)}-31"
    if date and date <= upload and int(date[:4]) >= int(up.group(1)) - 1:
        return date
    for src in (title, text[:1500]):
        # בכותרות יש רווח באמצע תאריך ("02.0 8.26"), אבל "מועצה 27 11.12.25" חייב להישאר מופרד
        for d, m, y in DATE_RE.findall(re.sub(r"(\.\d)\s+(\d)", r"\1\2", src)):
            y = int(y) + (2000 if len(y) == 2 else 0)
            if not (1 <= int(m) <= 12 and 1 <= int(d) <= 31):
                continue
            cand = f"{y:04d}-{int(m):02d}-{int(d):02d}"
            if cand <= upload and y >= int(up.group(1)) - 2:
                return cand
    return date if date and date <= upload else ""


def main():
    os.makedirs(os.path.join(OUT, "pages"), exist_ok=True)
    index = requests.get(BASE + "index.html", headers=UA, timeout=60).text
    files = sorted(set(re.findall(r'href="([^"/]+\.html)"', index)) - {"index.html"})
    docs, seen = [], set()
    for f in files:
        cls = classify(f.split("_", 2)[-1])
        if not cls:
            continue
        path = os.path.join(OUT, "pages", f)
        if not os.path.exists(path):
            r = requests.get(BASE + quote(f), headers=UA, timeout=60)
            if not r.ok:
                print("נכשל:", f)
                continue
            open(path, "w", encoding="utf-8").write(r.text)
            time.sleep(0.2)
        t = open(path, encoding="utf-8").read()
        text = page_text(t)
        h = hashlib.md5(text.encode()).hexdigest()
        if len(text) < 200 or h in seen:   # מסמך ריק או כפול (אותו סדר יום פורסם פעמיים)
            continue
        seen.add(h)
        title = html.unescape(re.search(r"<h1>(.*?)</h1>", t, re.S).group(1))
        title = re.sub(r"\s+[0-9a-f]{20,}$", "", title).strip()   # בלי ה-hash שבסוף שם הקובץ
        title = re.sub(r"^\d+\s+", "", title)
        pdf = re.search(r'href="(https?://[^"]+\.pdf)"', t, re.I)
        date = f[:10] if re.match(r"\d{4}-\d\d-\d\d", f) else ""
        date = sane_date(date, title, text, pdf.group(1) if pdf else "")
        docs.append({"date": date, "body": cls[0], "kind": cls[1], "title": title,
                     "url": pdf.group(1) if pdf else BASE + quote(f), "goinfo": BASE + quote(f),
                     "text": text})
    json.dump(docs, open(os.path.join(OUT, "docs.json"), "w", encoding="utf-8"), ensure_ascii=False)
    from collections import Counter
    for k, v in sorted(Counter((d["body"], d["kind"]) for d in docs).items()):
        print(k, v)
    print(len(docs), "מסמכים,", sum(len(d["text"]) for d in docs) // 1000, "אלף תווים")


if __name__ == "__main__":
    main()
