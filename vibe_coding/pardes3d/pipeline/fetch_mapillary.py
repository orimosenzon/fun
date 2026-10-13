"""מושך תמונות רחוב פתוחות מ-Mapillary לאזור המושבה.

למה Mapillary ולא גוגל סטריט ויו: תנאי השימוש של Google Maps Platform אוסרים
ליצור תוכן נגזר מתמונות סטריט ויו (מודלים, טקסטורות, חילוץ מאפיינים),
וגם אוסרים לשמור אותן. התמונות של Mapillary ברישיון CC BY-SA 4.0,
ומגיעות עם מיקום וכיוון מצלמה שחושבו בשחזור מבנה-מתנועה (SfM).

דרוש מפתח חינמי: https://www.mapillary.com/dashboard/developers
(יוצרים אפליקציה, מעתיקים את ה-Client Token) ושמים ב-pardes3d/.env:
    MAPILLARY_TOKEN=MLY|...

שימוש:
    python fetch_mapillary.py            # רק מטא-דאטה: מיקומים, כיוונים, תאריכים
    python fetch_mapillary.py --download # גם תמונות 1024 פיקסל (לתיקייה data/raw/mapillary/img)

פלט: data/raw/mapillary/images.jsonl, ו-game/world/street_images.json (נקודות למפה בגודו).
"""
import argparse
import json
import os
import pathlib
import sys
import time

import requests

from fetch_osm import BBOX, ROOT
from geo import to_local

API = "https://graph.mapillary.com/images"
FIELDS = "id,geometry,computed_geometry,compass_angle,computed_compass_angle,captured_at,camera_type,sequence,thumb_1024_url,creator"
OUT = ROOT / "data" / "raw" / "mapillary"
CELL = 0.005  # ה-API מגביל את גודל ה-bbox ואת מספר התוצאות, אז מחלקים לתאים של כחצי ק"מ


def token():
    t = os.environ.get("MAPILLARY_TOKEN")
    env = ROOT / ".env"
    if not t and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("MAPILLARY_TOKEN="):
                t = line.split("=", 1)[1].strip()
    if not t:
        sys.exit(__doc__ + "\nחסר MAPILLARY_TOKEN.")
    return t


def cells():
    s, w, n, e = BBOX
    lat = s
    while lat < n:
        lon = w
        while lon < e:
            yield (lon, lat, min(lon + CELL, e), min(lat + CELL, n))
            lon += CELL
        lat += CELL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    tok = token()
    OUT.mkdir(parents=True, exist_ok=True)
    seen = {}
    for c in cells():
        r = requests.get(API, params={"access_token": tok, "bbox": ",".join(f"{v:.6f}" for v in c),
                                      "fields": FIELDS, "limit": 2000}, timeout=60)
        if r.status_code != 200:
            print("שגיאה", r.status_code, r.text[:200])
            time.sleep(2)
            continue
        for im in r.json().get("data", []):
            seen[im["id"]] = im
        print(f"\r{len(seen)} תמונות", end="", flush=True)
    print()
    with open(OUT / "images.jsonl", "w") as f:
        for im in seen.values():
            f.write(json.dumps(im) + "\n")
    pts = []
    for im in seen.values():
        g = (im.get("computed_geometry") or im["geometry"])["coordinates"]
        x, y = to_local(g[1], g[0])
        pts.append([round(x, 1), round(y, 1), round(im.get("computed_compass_angle") or im.get("compass_angle") or 0),
                    im["id"], 1 if im.get("camera_type") in ("spherical", "equirectangular") else 0])
    (ROOT / "game" / "world" / "street_images.json").write_text(json.dumps(pts, separators=(",", ":")))
    pano = sum(p[4] for p in pts)
    print(f"נשמרו {len(pts)} תמונות ({pano} פנורמיות 360)")
    if a.download:
        d = OUT / "img"
        d.mkdir(exist_ok=True)
        for i, im in enumerate(seen.values()):
            f = d / f"{im['id']}.jpg"
            if f.exists() or not im.get("thumb_1024_url"):
                continue
            f.write_bytes(requests.get(im["thumb_1024_url"], timeout=60).content)
            if i % 50 == 0:
                print(f"\rהורדו {i}/{len(seen)}", end="", flush=True)
        print()


if __name__ == "__main__":
    main()
