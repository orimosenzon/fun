#!/usr/bin/env python3
"""מושך אלבום משותף מגוגל פוטוס אל מפת התמונות.

שימוש:
    python3 fetch_album.py <קישור-לאלבום> [<קישור> ...]

לכל פריט באלבום: מיקום, זמן צילום, תמונה ממוזערת ותמונה גדולה.
הכל נשמר ב-data/ (מחוץ ל-git), והספרייה המאוחדת ב-data/library.json.
הרצה חוזרת על אותו אלבום מדלגת על פריטים שכבר עובדו.

מאיפה המיקום:
- תמונה: EXIF GPS. מספיק להוריד את 128KB הראשונים של המקור.
- סרטון: תגית ©xyz (או ISO6709 של אפל) ב-moov שבסוף הקובץ. השרת לא מכבד
  בקשת טווח לסרטונים, אז מורידים את כולו בזרם וזורקים.
"""
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

import trips

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LIBRARY = DATA / "library.json"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/128 Safari/537.36"
THUMB = "=s300-c"          # ריבוע חתוך, לסמנים על המפה
LARGE = "=w2048-h2048"     # לגלריה
EXIF_BYTES = 131072


def get(url, headers=None, data=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=60)


# ---------- קריאת האלבום ----------

def init_data(html, key):
    m = re.search(r"AF_initDataCallback\(\{key: '%s'.*?data:(.*?), sideChannel" % key, html, re.S)
    return json.loads(m.group(1)) if m else None


def next_page(album_id, auth_key, token):
    """עמוד נוסף של אלבום גדול, דרך ה-RPC שהדף עצמו משתמש בו."""
    inner = json.dumps([album_id, token, None, auth_key])
    body = urllib.parse.urlencode({"f.req": json.dumps([[["snAcKc", inner, None, "generic"]]])}).encode()
    raw = get("https://photos.google.com/_/PhotosUi/data/batchexecute",
              {"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}, body).read().decode()
    for line in raw.splitlines():
        if line.startswith('[["wrb.fr","snAcKc"'):
            return json.loads(json.loads(line)[0][2])
    raise RuntimeError("לא הצלחתי לקרוא עמוד נוסף של האלבום")


def read_album(link):
    resp = get(link)
    final = resp.geturl()
    html = resp.read().decode("utf-8")
    album_id = re.search(r"/share/([A-Za-z0-9_-]+)", final).group(1)
    auth_key = urllib.parse.parse_qs(urllib.parse.urlparse(final).query).get("key", [None])[0]
    d = init_data(html, "ds:1")
    if not d:
        raise RuntimeError("הדף לא נראה כמו אלבום משותף. האם הקישור ציבורי?")
    title = d[3][1] if d[3] and len(d[3]) > 1 else album_id
    raw_items, token = list(d[1] or []), d[2]
    while token:
        page = next_page(album_id, auth_key, token)
        raw_items += page[1] or []
        token = page[2]
        print(f"  ...{len(raw_items)} פריטים")
    items = []
    for it in raw_items:
        extra = it[-1] if isinstance(it[-1], dict) else {}
        video = extra.get("76647426")
        items.append({
            "id": it[0],
            "base": it[1][0],
            "w": it[1][1], "h": it[1][2],
            "taken": it[2],          # מילישניות UTC
            "tz": it[4] or 0,        # היסט אזור הזמן במילישניות
            "type": "video" if video else "photo",
            "duration": video[0] if video else None,
        })
    return {"id": album_id, "title": title, "url": link, "items": items}


# ---------- מיקום ----------

def photo_location(base):
    head = get(base + "=d", {"Range": f"bytes=0-{EXIF_BYTES - 1}"}).read()
    try:
        gps = Image.open(io.BytesIO(head)).getexif().get_ifd(0x8825)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def deg(v, ref):
        x = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        return -x if ref in ("S", "W") else x
    lat, lon = deg(gps[2], gps.get(1, "N")), deg(gps[4], gps.get(3, "E"))
    return None if lat == 0 and lon == 0 else (round(lat, 6), round(lon, 6))


ISO6709 = re.compile(rb"([+-]\d{1,2}\.\d+)([+-]\d{1,3}\.\d+)")


def video_location(base):
    tail = b""
    with get(base + "=dv") as r:
        while chunk := r.read(1 << 20):
            tail = (tail + chunk)[-(4 << 20):]   # ה-moov בסוף, מספיק לשמור את הזנב
    for marker in (b"\xa9xyz", b"ISO6709"):
        i = tail.rfind(marker)
        if i >= 0:
            m = ISO6709.search(tail, i, i + 300)
            if m:
                lat, lon = float(m.group(1)), float(m.group(2))
                if lat or lon:
                    return (round(lat, 6), round(lon, 6))
    return None


# ---------- עיבוד ----------

def save(url, path):
    if not path.exists():
        path.write_bytes(get(url).read())


def process(item, album_id):
    iid = item["id"]
    save(item["base"] + THUMB, DATA / "thumbs" / f"{iid}.jpg")
    save(item["base"] + LARGE, DATA / "large" / f"{iid}.jpg")
    try:
        loc = video_location(item["base"]) if item["type"] == "video" else photo_location(item["base"])
    except Exception as e:
        print(f"  ! {iid[:12]}: {e}")
        loc = None
    out = {k: item[k] for k in ("id", "type", "w", "h", "taken", "tz", "duration")}
    out.update(album=album_id,
               lat=loc[0] if loc else None, lon=loc[1] if loc else None,
               thumb=f"data/thumbs/{iid}.jpg", large=f"data/large/{iid}.jpg",
               video=item["base"] + "=dv" if item["type"] == "video" else None,
               orig=item["base"] + "=d" if item["type"] == "photo" else None)   # רזולוציה מלאה, לזום עמוק
    return out


def main(links):
    if not links:
        sys.exit(__doc__)
    for sub in ("thumbs", "large"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    lib = json.loads(LIBRARY.read_text()) if LIBRARY.exists() else {"albums": {}, "items": {}}

    for link in links:
        print(f"קורא את {link}")
        album = read_album(link)
        todo = [it for it in album["items"] if it["id"] not in lib["items"]]
        for it in album["items"]:            # פריטים ישנים מלפני שהיה orig
            old = lib["items"].get(it["id"])
            if old and old["type"] == "photo" and not old.get("orig"):
                old["orig"] = it["base"] + "=d"
        print(f"«{album['title']}»: {len(album['items'])} פריטים, {len(todo)} חדשים")
        with ThreadPoolExecutor(6) as pool:
            for n, rec in enumerate(pool.map(lambda it: process(it, album["id"]), todo), 1):
                lib["items"][rec["id"]] = rec
                print(f"  {n}/{len(todo)} {'📍' if rec['lat'] is not None else '  '} {rec['type']}")
        lib["albums"][album["id"]] = {"id": album["id"], "title": album["title"], "url": link,
                                      "items": [it["id"] for it in album["items"]]}
        LIBRARY.write_text(json.dumps(lib, ensure_ascii=False, indent=1))
        located = sum(lib["items"][i]["lat"] is not None for i in lib["albums"][album["id"]]["items"])
        print(f"  ✔ {located}/{len(album['items'])} עם מיקום")
    trips.build(DATA)


if __name__ == "__main__":
    main(sys.argv[1:])
