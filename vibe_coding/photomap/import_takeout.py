#!/usr/bin/env python3
"""מייבא Google Takeout של גוגל פוטוס אל מפת התמונות, בלי לחלץ את ה-zip-ים.

שימוש:
    python3 import_takeout.py ~/Pictures/takeout-*.zip

- מיקום וזמן מגיעים מקובץ ה-JSON שליד כל פריט (geoData, ואם אין אז geoDataExif).
- את היסט אזור הזמן מחשבים מה-EXIF של התמונה (שעון מקומי) מול photoTakenTime (UTC).
- אותה תמונה מופיעה גם ב-"Photos from YYYY" וגם בתיקיית האלבום. מאחדים לפי המזהה שב-url.
- data/takeout_members.json אומר ל-serve.py איפה כל פריט יושב בתוך איזה zip.
- zip כפול (אותו שם עם " (1)" ואותו גודל) נקרא פעם אחת.
"""
import bisect
import io
import json
import os
import re
import sys
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import trips

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LIBRARY = DATA / "library.json"
MEMBERS = DATA / "takeout_members.json"
SOURCE = "takeout"
VIDEO_EXT = {".mp4", ".3gp", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
MEDIA_EXT = VIDEO_EXT | {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".bmp"}
JSON_RE = re.compile(r"^(.*?)\.su?p?p?[^/]*?(\(\d+\))?\.json$")

nfc = lambda s: unicodedata.normalize("NFC", s)


def unique_zips(paths):
    seen, out = {}, []
    for p in sorted(map(Path, paths)):
        key = (re.sub(r" \(\d+\)(?=\.zip$)", "", p.name), p.stat().st_size)
        if key in seen:
            print(f"מדלג על {p.name}: עותק של {seen[key].name}")
            continue
        seen[key] = p
        out.append(p)
    return out


def media_for(json_name, media):
    """מוצא את קובץ המדיה שה-JSON מתאר, כולל שמות מקוצרים וכפילויות (1)."""
    d, b = json_name.rsplit("/", 1)
    m = JSON_RE.match(b)
    stem, dup = (m.group(1), m.group(2) or "") if m else (b[:-5], "")
    stem = re.sub(r"\.su?p?p?[a-z-]*$", "", stem)
    root, ext = os.path.splitext(stem)
    for c in (f"{d}/{root}{dup}{ext}", f"{d}/{stem}"):
        if c in media:
            return c
    prefix = [k for k in media.by_dir.get(d, ()) if k.startswith(f"{d}/{root}")]
    return prefix[0] if len(prefix) == 1 else None


class MediaIndex(dict):
    def __init__(self):
        super().__init__()
        self.by_dir = {}

    def add(self, name, where):
        self[name] = where
        self.by_dir.setdefault(name.rsplit("/", 1)[0], []).append(name)


def scan(zips):
    media, records = MediaIndex(), {}
    jsons = []
    for zp in zips:
        z = zipfile.ZipFile(zp)
        for info in z.infolist():
            n = nfc(info.filename)
            if n.endswith("/"):
                continue
            if n.endswith(".json"):
                jsons.append((zp, info.filename, n))
            elif os.path.splitext(n)[1].lower() in MEDIA_EXT:
                media.add(n, (str(zp), info.filename, info.header_offset))
    opened = {zp: zipfile.ZipFile(zp) for zp in zips}
    for zp, raw, n in jsons:
        try:
            j = json.loads(opened[zp].read(raw))
        except Exception:
            continue
        if not isinstance(j, dict) or "url" not in j or "photoTakenTime" not in j:
            continue
        iid = j["url"].rstrip("/").rsplit("/", 1)[-1]
        r = records.setdefault(iid, {"id": iid, "title": j.get("title", ""), "taken": int(j["photoTakenTime"]["timestamp"]) * 1000,
                                     "lat": None, "lon": None, "folders": set(), "member": None,
                                     "description": j.get("description", "")})
        for key in ("geoData", "geoDataExif"):
            g = j.get(key) or {}
            if r["lat"] is None and (g.get("latitude") or g.get("longitude")):
                r["lat"], r["lon"] = round(g["latitude"], 6), round(g["longitude"], 6)
        folder = n.split("/")[2] if n.count("/") >= 3 else ""
        if not folder.startswith("Photos from"):
            r["folders"].add(folder)
        if r["member"] is None:
            hit = media_for(n, media)
            if hit:
                r["member"] = media[hit]
    return records


def exif_tz(z, member, taken):
    """היסט אזור הזמן במילישניות, מתוך EXIF: שעון מקומי פחות UTC. None אם אין."""
    try:
        with z.open(member) as f:
            head = f.read(65536)
        ex = Image.open(io.BytesIO(head)).getexif().get_ifd(0x8769)
    except Exception:
        return None
    off = ex.get(0x9011)                      # OffsetTimeOriginal, למשל "+03:00"
    if isinstance(off, str) and re.match(r"^[+-]\d\d:\d\d$", off):
        sign = -1 if off[0] == "-" else 1
        return sign * (int(off[1:3]) * 60 + int(off[4:6])) * 60000
    dto = ex.get(0x9003)                      # DateTimeOriginal, שעון מקומי בלי אזור
    try:
        local = datetime.strptime(dto.strip("\x00 "), "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None
    diff = local.timestamp() * 1000 - taken
    q = round(diff / 900000) * 900000       # רבע שעה
    return q if abs(q) <= 14 * 3600000 and abs(diff - q) < 120000 else None


def fill_tz(recs):
    # הדיסק מכני: קוראים EXIF רק לתמונות שעל המפה, ולפי הסדר הפיזי בתוך ה-zip
    todo = sorted((r for r in recs if r["type"] == "photo" and r["lat"] is not None), key=lambda r: r["member"][::2])
    zs = {}
    for n, r in enumerate(todo, 1):
        z = zs.setdefault(r["member"][0], zipfile.ZipFile(r["member"][0]))
        r["tz"] = exif_tz(z, r["member"][1], r["taken"])
        if n % 500 == 0:
            print(f"  {n}/{len(todo)}", flush=True)
    # לכל השאר: ההיסט של התמונה הקרובה בזמן
    known = sorted((r["taken"], r["tz"]) for r in recs if r.get("tz") is not None)
    times = [k[0] for k in known]
    for r in recs:
        if r.get("tz") is None:
            i = bisect.bisect_left(times, r["taken"])
            near = [known[j] for j in (i - 1, i) if 0 <= j < len(known)]
            r["tz"] = min(near, key=lambda k: abs(k[0] - r["taken"]))[1] if near else 0


def main(paths):
    if not paths:
        sys.exit(__doc__)
    zips = unique_zips(paths)
    print("סורק", ", ".join(z.name for z in zips), flush=True)
    records = scan(zips)
    no_media = [r for r in records.values() if r["member"] is None]
    recs = [r for r in records.values() if r["member"] is not None]
    for r in recs:
        r["type"] = "video" if os.path.splitext(r["member"][1])[1].lower() in VIDEO_EXT else "photo"
    print(f"{len(records)} פריטים ייחודיים, {len(no_media)} בלי קובץ מדיה (מדלג)")
    print("קורא אזורי זמן מה-EXIF...", flush=True)
    fill_tz(recs)

    lib = json.loads(LIBRARY.read_text()) if LIBRARY.exists() else {"albums": {}, "items": {}}
    # פריט שכבר הגיע מאלבום משותף (אותה שנייה ואותו מקום בערך) לא נכנס פעמיים
    existing = {(i["taken"] // 1000, round(i["lat"] or 0, 2), round(i["lon"] or 0, 2))
                for i in lib["items"].values() if i["album"] != SOURCE}
    lib["items"] = {k: v for k, v in lib["items"].items() if v["album"] != SOURCE}
    members, ids, dupes = {}, [], 0
    for r in sorted(recs, key=lambda r: r["taken"]):
        if (r["taken"] // 1000, round(r["lat"] or 0, 2), round(r["lon"] or 0, 2)) in existing:
            dupes += 1
            continue
        iid = r["id"]
        video = r["type"] == "video"
        lib["items"][iid] = {
            "id": iid, "type": r["type"], "w": None, "h": None,
            "taken": r["taken"], "tz": r["tz"], "duration": None,
            "album": SOURCE, "lat": r["lat"], "lon": r["lon"],
            "thumb": f"media/thumb/{iid}.jpg",
            "large": f"media/poster/{iid}.jpg" if video else f"media/view/{iid}",
            "video": f"media/view/{iid}" if video else None,
            "title": r["title"], "folders": sorted(r["folders"]),
        }
        members[iid] = [r["member"][0], r["member"][1], r["type"], r["member"][2]]
        ids.append(iid)
    lib["albums"][SOURCE] = {"id": SOURCE, "title": "גוגל פוטוס (Takeout)", "url": "", "items": ids}
    LIBRARY.write_text(json.dumps(lib, ensure_ascii=False))
    MEMBERS.write_text(json.dumps(members, ensure_ascii=False))
    located = sum(lib["items"][i]["lat"] is not None for i in ids)
    print(f"✔ {len(ids)} פריטים נכנסו, {located} עם מיקום ({dupes} כבר היו מאלבום משותף)")
    trips.build(DATA)


if __name__ == "__main__":
    main(sys.argv[1:])
