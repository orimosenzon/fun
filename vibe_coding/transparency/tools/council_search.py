#!/usr/bin/env python3
"""חיפוש בכתוביות של ישיבות מליאת המועצה ביוטיוב.

הכתוביות נמשכות עם yt-dlp לתיקייה private/council_subs (מחוץ ל-git, כי יש בהן
שמות של דוברים). כל תוצאה מודפסת עם קישור שפותח את הסרטון בשנייה המתאימה.

    python3 tools/council_search.py "רחוב הנשיא" "269"
"""
import glob
import json
import os
import re
import sys

SUBS = os.path.join(os.path.dirname(__file__), "..", "private", "council_subs")
WINDOW = 25  # שניות של הקשר לכל צד


def load(vid_path):
    info_path = vid_path.replace(".iw-orig.json3", ".info.json")
    info = json.load(open(info_path, encoding="utf-8"))
    segs = []
    for e in json.load(open(vid_path, encoding="utf-8")).get("events", []):
        txt = "".join(s.get("utf8", "") for s in e.get("segs", [])).strip()
        if txt:
            segs.append((e["tStartMs"] / 1000, txt))
    return info, segs


def meeting_date(title):
    m = re.search(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", title)
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else "????"


def main(terms):
    videos = []
    for p in glob.glob(os.path.join(SUBS, "*.iw-orig.json3")):
        info, segs = load(p)
        videos.append((meeting_date(info["title"]), info["id"], segs))
    total = 0
    for date, vid, segs in sorted(videos):
        for i, (t, txt) in enumerate(segs):
            if not any(term in txt for term in terms):
                continue
            ctx = " ".join(s for tt, s in segs if t - WINDOW <= tt <= t + WINDOW)
            print(f"\n{date}  https://youtu.be/{vid}?t={int(t)}\n  {ctx}")
            total += 1
    print(f"\n{total} תוצאות ב-{len(videos)} ישיבות")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
