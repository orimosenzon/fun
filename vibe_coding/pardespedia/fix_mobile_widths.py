#!/usr/bin/env python3
"""תיקון גלישה אופקית בנייד: תמונות רחבות וטבלאות ללא מעטפת גלילה.

הרקע: הוויקי מוגש לטלפונים בעיצוב הדסקטופ (אין MobileFrontend). כל אלמנט
ברוחב קבוע גדול ממסך הטלפון דוחף את כל הדף הצידה — ובדף RTL זה מסתיר את
הטקסט מחוץ למסך, והמבקר חושב שהאתר ריק.

מה נעשה כאן:
  * תמונה ברוחב 331–450px  -> מוקטנת ל-320px (שינוי זניח בדסקטופ)
  * תמונה ברוחב 451px ומעלה -> שני גדלים במחלקות Bootstrap:
      d-none d-md-block (הגודל המקורי, דסקטופ) + d-md-none (300px, טלפון)
  * wikitable ללא מעטפת   -> נעטפת ב-div עם overflow-x:auto

הרצה:  python3 fix_mobile_widths.py [--write]
"""
import argparse
import re
import sys

from wiki_client import WikiClient, API_URL

SMALL = 320          # רוחב יעד לתמונות בינוניות
MOBILE = 300         # רוחב הגרסה לטלפון בתמונות גדולות
SPLIT_ABOVE = 450    # מעליו עוברים לשני גדלים
THRESHOLD = 330      # רוחב שמעליו יש גלישה במסך 360px

FILE_RE = re.compile(r'\[\[(?:קובץ|תמונה|File|Image):[^\[\]]*?\]\]')
WIDTH_RE = re.compile(r'\|\s*(\d{3,4})\s*(?:px|פיקסלים)')


def fix_images(wt):
    """מחזיר (wikitext מעודכן, מספר תמונות שטופלו)."""
    out, count, pos = [], 0, 0
    for m in FILE_RE.finditer(wt):
        out.append(wt[pos:m.start()])
        tag = m.group(0)
        pos = m.end()
        wm = WIDTH_RE.search(tag)
        if not wm or int(wm.group(1)) <= THRESHOLD:
            out.append(tag)
            continue
        # דילוג אם התמונה כבר בתוך מעטפת רספונסיבית
        before = wt[max(0, m.start() - 90):m.start()]
        if 'd-none d-md-block' in before or 'd-md-none' in before:
            out.append(tag)
            continue
        width = int(wm.group(1))
        if width <= SPLIT_ABOVE:
            out.append(WIDTH_RE.sub(f'|{SMALL}px', tag, count=1))
        else:
            mobile_tag = WIDTH_RE.sub(f'|{MOBILE}px', tag, count=1)
            out.append(f'<div class="d-none d-md-block">{tag}</div>'
                       f'<div class="d-md-none">{mobile_tag}</div>')
        count += 1
    out.append(wt[pos:])
    return "".join(out), count


def fix_tables(wt):
    """עוטף כל wikitable שאינה עטופה ב-div עם גלילה אופקית."""
    if 'overflow-x:auto' in wt:
        return wt, 0
    lines = wt.split("\n")
    out, count, depth = [], 0, 0
    for line in lines:
        if depth == 0 and re.match(r'\{\|\s*class="[^"]*wikitable', line):
            out.append('<div style="overflow-x:auto;">')
            out.append(line)
            depth = 1
            count += 1
            continue
        if depth:
            if line.startswith('{|'):
                depth += 1
            elif line.startswith('|}'):
                depth -= 1
                out.append(line)
                if depth == 0:
                    out.append('</div>')
                continue
        out.append(line)
    if depth:                       # טבלה לא נסגרה — לא נוגעים
        return wt, 0
    return "\n".join(out), count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="לשמור בוויקי (בלעדיו: תצוגה בלבד)")
    args = ap.parse_args()

    c = WikiClient()
    if args.write:
        c.login()
    titles = [p if isinstance(p, str) else p.get("title")
              for p in c.list_pages(limit=0)]
    print(f"סורק {len(titles)} דפים...")

    changed = 0
    for i in range(0, len(titles), 40):
        r = c.session.get(API_URL, params={
            "action": "query", "titles": "|".join(titles[i:i + 40]),
            "prop": "revisions", "rvprop": "content", "rvslots": "main",
            "format": "json"}).json()
        for p in r["query"]["pages"].values():
            try:
                wt = p["revisions"][0]["slots"]["main"]["*"]
            except (KeyError, IndexError):
                continue
            new, n_img = fix_images(wt)
            new, n_tbl = fix_tables(new)
            if new == wt:
                continue
            parts = []
            if n_img:
                parts.append(f"{n_img} תמונות")
            if n_tbl:
                parts.append(f"{n_tbl} טבלאות")
            what = " ו".join(parts)
            print(f"  {'✓' if args.write else '·'} {p['title']}: {what}")
            if args.write:
                c.edit_page(p["title"], new,
                            summary=f"תיקון תצוגה בנייד: {what}")
            changed += 1
    print(f"\n{'עודכנו' if args.write else 'יעודכנו'} {changed} דפים.")
    if not args.write:
        print("להרצה בפועל: python3 fix_mobile_widths.py --write")


if __name__ == "__main__":
    sys.exit(main())
