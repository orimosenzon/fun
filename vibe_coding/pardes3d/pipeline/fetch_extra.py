"""מקורות נוספים מעבר ל-OSM ולגבהים. כל אחד נשמר תחת data/raw/.

  ms       בניינים מזוהים מצילומי לוויין, Microsoft Global ML Building Footprints
           (ODbL). כ-15 אלף בשטח, לעומת כ-8,600 ב-OSM. בלי גבהים בישראל.
  ghsl     גובה ממוצע של בניינים במשבצות של כ-100 מ', GHS-BUILT-H 2018
           של מרכז המחקר המשותף של האיחוד האירופי (CC BY 4.0).
  s2       צילום לוויין Sentinel-2 cloudless 2024 של EOX, כ-5 מ' לפיקסל
           (CC BY-NC-SA 4.0: מותר לשימוש לא מסחרי עם ייחוס).
  commons  תמונות עם מיקום מ-Wikimedia Commons (רובן מ-PikiWiki), עם יוצר ורישיון.
  pardes   מקומות מפרדספדיה עם מיקום, תיאור ותמונות (דרך הנתונים של דרך קיצור).

שימוש:  python fetch_extra.py [ms ghsl s2 commons pardes]   (בלי פרמטרים: הכל)
"""
import csv
import gzip
import io
import json
import math
import sys
import zipfile

import numpy as np
import requests
from PIL import Image

from fetch_dem import tile_xy
from fetch_osm import BBOX, ROOT, UA

RAW = ROOT / "data" / "raw"
DK = ROOT.parent / "dereh_kitzur" / "web" / "data"


def get(url, **kw):
    r = requests.get(url, headers=UA, timeout=300, **kw)
    r.raise_for_status()
    return r


def quadkey(lat, lon, z):
    x, y = (int(v) for v in tile_xy(lat, lon, z))
    return "".join(str((1 if x >> (i - 1) & 1 else 0) + (2 if y >> (i - 1) & 1 else 0)) for i in range(z, 0, -1))


def fetch_ms():
    s, w, n, e = BBOX
    keys = {quadkey(a, b, 9) for a in (s, n) for b in (w, e)}
    links = get("https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv").text
    out = []
    for row in csv.DictReader(io.StringIO(links)):
        if row["Location"] != "Israel" or row["QuadKey"] not in keys:
            continue
        print("מוריד", row["QuadKey"], row["Size"])
        with gzip.open(io.BytesIO(get(row["Url"]).content), "rt") as f:
            for line in f:
                o = json.loads(line)
                lon, lat = o["geometry"]["coordinates"][0][0]
                if w <= lon <= e and s <= lat <= n:
                    out.append(o)
    (RAW / "ms").mkdir(parents=True, exist_ok=True)
    (RAW / "ms" / "pardes.json").write_text(json.dumps(out))
    print(f"מיקרוסופט: {len(out)} בניינים")


def fetch_ghsl():
    s, w, n, e = BBOX
    # אריחים של 10 מעלות: שורה מ-90 צפון, עמודה מ-180 מערב
    r, c = int((90 - n) // 10) + 1, int((w + 180) // 10) + 1
    name = f"GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_4326_3ss_V1_0_R{r}_C{c}"
    url = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_H_GLOBE_R2023A/"
           f"GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_4326_3ss/V1-0/tiles/{name}.zip")
    print("מוריד", name)
    import rasterio
    from rasterio.windows import from_bounds
    d = RAW / "ghsl"
    d.mkdir(parents=True, exist_ok=True)
    tif = d / (name + ".tif")
    if not tif.exists():
        zipfile.ZipFile(io.BytesIO(get(url).content)).extract(name + ".tif", d)
    with rasterio.open(tif) as src:
        win = from_bounds(w - 0.002, s - 0.002, e + 0.002, n + 0.002, src.transform).round_offsets().round_lengths()
        crop = src.read(1, window=win)
        t = src.window_transform(win)
    np.savez_compressed(d / "height.npz", h=crop, top=t.f, left=t.c, res=t.a)
    v = crop[crop > 0]
    print(f"GHSL: {crop.shape}, חציון {np.median(v):.1f} מ', מקסימום {v.max():.1f} מ'")


def fetch_s2(z=15):
    s, w, n, e = BBOX
    x0, y0 = (int(v) for v in tile_xy(n, w, z))
    x1, y1 = (int(v) for v in tile_xy(s, e, z))
    img = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            url = f"https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{ty}/{tx}.jpg"
            img.paste(Image.open(io.BytesIO(get(url).content)).convert("RGB"), ((tx - x0) * 256, (ty - y0) * 256))
        print(f"\rשורה {ty - y0 + 1}/{y1 - y0 + 1}", end="", flush=True)
    print()
    img.save(RAW / "s2.png")
    (RAW / "s2.json").write_text(json.dumps({"z": z, "x0": x0, "y0": y0}))
    print(f"Sentinel-2: {img.size}")


def fetch_commons():
    import time
    api = "https://commons.wikimedia.org/w/api.php"
    s, w, n, e = BBOX

    def call(params):
        for wait in (0, 5, 20, 60):
            time.sleep(wait or 1)
            r = requests.get(api, params={"format": "json", **params}, headers=UA, timeout=120)
            if r.status_code == 200 and "query" in r.json():
                return r.json()
        sys.exit(f"Commons נכשל: {r.status_code} {r.text[:300]}")

    # שלב 1: מזהים. geosearch מחזיר עד 500 בכל קריאה, אז עוברים על גריד בצפיפות שמספיקה
    ids = {}
    for lat in np.arange(s + 0.008, n, 0.016):
        for lon in np.arange(w + 0.009, e, 0.019):
            r = call({"action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}",
                      "gsradius": 1500, "gslimit": 500, "gsnamespace": 6})
            for g in r["query"]["geosearch"]:
                ids[g["pageid"]] = (g["lat"], g["lon"])
        print(f"\r{len(ids)} תמונות", end="", flush=True)
    print()
    # שלב 2: פרטים, 50 בכל קריאה
    out = []
    keys = list(ids)
    for i in range(0, len(keys), 50):
        r = call({"action": "query", "pageids": "|".join(map(str, keys[i:i + 50])), "prop": "imageinfo",
                  "iiprop": "url|extmetadata", "iiurlwidth": 1024})
        for p in r["query"]["pages"].values():
            if "imageinfo" not in p:
                continue
            ii = p["imageinfo"][0]
            md = ii.get("extmetadata", {})
            val = lambda k: md.get(k, {}).get("value", "")
            lat, lon = ids[p["pageid"]]
            out.append({
                "title": p["title"].removeprefix("File:").rsplit(".", 1)[0],
                "lat": lat, "lon": lon, "thumb": ii.get("thumburl"), "page": ii.get("descriptionurl"),
                "author": val("Artist"), "license": val("LicenseShortName"),
                "desc": val("ImageDescription"), "date": val("DateTimeOriginal"),
            })
    pics = [p for p in out if s <= p["lat"] <= n and w <= p["lon"] <= e]
    (RAW / "commons.json").write_text(json.dumps(pics, ensure_ascii=False))
    print(f"Commons: {len(pics)} תמונות בשטח")


def fetch_pardes():
    d = json.loads((DK / "places.json").read_text())
    (RAW / "pardespedia_places.json").write_text(json.dumps(d["places"], ensure_ascii=False))
    print(f"פרדספדיה: {len(d['places'])} מקומות")


STEPS = {"ms": fetch_ms, "ghsl": fetch_ghsl, "s2": fetch_s2, "commons": fetch_commons, "pardes": fetch_pardes}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for k in sys.argv[1:] or STEPS:
        STEPS[k]()
