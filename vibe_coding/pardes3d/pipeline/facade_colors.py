"""צבע הטיח האמיתי של בניינים, מתוך תמונות הרחוב של Mapillary.

לכל תמונה שצולמה עד 35 מטר מבניין: מקרינים נקודות על הקיר שפונה למצלמה
לתוך התמונה (לפי הסיבוב שחושב לה ב-SfM של Mapillary), ודוגמים שם פיקסלים.
פוסלים צמחייה, שמיים, צל עמוק, וקרן שבניין אחר חוסם בדרך. לבניין שנאספו לו
מספיק פיקסלים לוקחים חציון, ומרככים אותו לכיוון טיח מואר: התמונה צולמה
באור של רגע אחד, והשיידר מוסיף את התאורה שלו.

דרוש fetch_mapillary.py קודם (images.jsonl). התמונות נשמרות ב-data/raw/mapillary/img.
פלט: data/raw/mapillary/facade_colors.json, {מזהה בניין: [r, g, b, מספר תמונות]}
"""
import colorsys
import io
import json
import math

import numpy as np
import requests
from PIL import Image
from shapely import STRtree
from shapely.geometry import LineString, Point

import build_world as bw
from fetch_mapillary import OUT, token
from geo import to_local

MAX_DIST = 35.0     # מטרים. רחוק יותר הקיר קטן מדי בתמונה של 1024
MIN_DIST = 3.0
CAM_H = 1.6         # גובה המצלמה מעל הקרקע: טלפון ביד או מצלמת רכב
MIN_PIX = 40        # פיקסלים טובים שצריך כדי לסמוך על הצבע


def rodrigues(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-9:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def details(ids, tok):
    """פרטי מצלמה וכתובת תמונה טרייה (הכתובות ב-images.jsonl פגות אחרי זמן)."""
    out = {}
    for i in range(0, len(ids), 50):
        r = requests.get("https://graph.mapillary.com/", timeout=60, params={
            "ids": ",".join(ids[i:i + 50]), "access_token": tok,
            "fields": "id,camera_parameters,computed_rotation,width,height,thumb_1024_url"})
        r.raise_for_status()
        out.update(r.json())
        print(f"\rפרטים {len(out)}/{len(ids)}", end="", flush=True)
    print()
    return out


def image(iid, url):
    f = OUT / "img" / f"{iid}.jpg"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(requests.get(url, timeout=60).content)
    return np.asarray(Image.open(io.BytesIO(f.read_bytes())).convert("RGB"))


def good(px):
    """מסכה של פיקסלים שנראים כמו קיר: לא עלווה, לא שמיים, לא חושך."""
    p = px.astype(int)
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    lum = (r * 3 + g * 6 + b) / 10
    veg = (g > r + 6) & (g > b + 6)
    sky = b > r + 4                     # טיח בארץ חם או ניטרלי; כחלחל זה שמיים, עננים או זכוכית
    sat = (p.max(1) - p.min(1)) > 90   # שלט, מכונית צבעונית
    return ~veg & ~sky & ~sat & (lum > 70) & (lum < 236)


def soften(rgb):
    """מהתמונה לוקחים גוון ורוויה; את הבהירות מקרבים לטיח מואר."""
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    s = min(s, 0.32)
    v = 0.62 + 0.33 * min(1.0, v / 0.85)
    return [round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)]


def main():
    tok = token()
    ims = [json.loads(l) for l in open(OUT / "images.jsonl")]
    els = bw.load_osm()
    items, _ = bw.collect_buildings(els)
    polys = [p for _, p, _ in items]
    tree = STRtree(polys)

    cams = []
    for im in ims:
        lon, lat = (im.get("computed_geometry") or im["geometry"])["coordinates"]
        cams.append(to_local(lat, lon))
    near = {}  # מזהה תמונה -> מספרי בניינים קרובים
    for im, (x, y) in zip(ims, cams):
        idx = tree.query(Point(x, y).buffer(MAX_DIST))
        if len(idx):
            near[im["id"]] = idx.tolist()
    print(f"{len(near)} מתוך {len(ims)} תמונות קרובות לבניין")
    det = details(list(near), tok)

    # שלב א': איפה נופל כל קיר בכל תמונה. בלי להוריד כלום
    pos = {im["id"]: c for im, c in zip(ims, cams)}
    plan = {}  # מזהה תמונה -> [(מספר בניין, [(u, v, מרחק)])]
    for iid, idx in near.items():
        d = det.get(iid) or {}
        if not d.get("computed_rotation") or not d.get("thumb_1024_url"):
            continue
        R = rodrigues(d["computed_rotation"])
        # תמונה שמכוונת לרצפה או לשמיים: מיקום הקיר בה רגיש מדי לשגיאה קטנה בסיבוב
        if abs((R.T @ [0, 0, 1])[2]) > 0.3:
            continue
        W, H = d["width"], d["height"]
        f = (d.get("camera_parameters") or [0.85])[0]
        cx, cy = pos[iid]
        for bi in idx:
            key, poly, t = items[bi]
            if poly.contains(Point(cx, cy)):
                continue
            h = bw.building_height(key, t, poly.area, 0)
            # הקיר שפונה למצלמה: הנקודה הקרובה ביותר בהיקף, ועוד שתיים מכל צד שלה
            ring = poly.exterior
            s0 = ring.project(Point(cx, cy))
            pts = []
            for ds in (-3, -1.5, 0, 1.5, 3):
                p = ring.interpolate((s0 + ds) % ring.length)
                dist = math.hypot(p.x - cx, p.y - cy)
                if not MIN_DIST < dist < MAX_DIST:
                    continue
                # קרן חסומה: בניין אחר בדרך
                ray = LineString([(cx, cy), (cx + (p.x - cx) * 0.97, cy + (p.y - cy) * 0.97)])
                if any(j != bi for j in tree.query(ray, predicate="intersects")):
                    continue
                # נמוך על הקיר: הגובה של רוב הבתים במודל מוערך, ומעל הגג יש שמיים
                for z in (2.0, 2.6, min(h - 1.0, 3.4)):
                    if z < 1.8:  # מתחת לזה מכוניות וגדרות
                        continue
                    pc = R @ np.array([p.x - cx, p.y - cy, z - CAM_H])
                    if pc[2] < 1.0:
                        continue
                    S = max(W, H)
                    u = (W / 2 + f * S * pc[0] / pc[2]) / W
                    v = (H / 2 + f * S * pc[1] / pc[2]) / H
                    if 0.05 < u < 0.95 and 0.05 < v < 0.8:
                        pts.append((u, v, dist))
            if pts:
                plan.setdefault(iid, []).append((bi, pts))
    print(f"{len(plan)} תמונות רואות קיר של {len({b for v in plan.values() for b, _ in v})} בניינים")

    # שלב ב': הורדה במקביל (חלק כבר בדיסק מריצה קודמת)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(12) as ex:
        imgs = dict(zip(plan, ex.map(lambda i: image(i, det[i]["thumb_1024_url"]), plan)))

    # שלב ג': דגימה
    samples = {}  # מספר בניין -> [פיקסלים], ותמונות
    for iid, walls in plan.items():
        img = imgs[iid]
        ih, iw = img.shape[:2]
        for bi, pts in walls:
            got = []
            for u, v, dist in pts:
                r = max(2, int(60 / dist))  # כתם קטן יותר כשהקיר רחוק
                X, Y = int(u * iw), int(v * ih)
                patch = img[max(0, Y - r):Y + r, max(0, X - r):X + r].reshape(-1, 3)
                got.append(patch[good(patch)])
            got = np.concatenate(got)
            if len(got) >= 10:
                s = samples.setdefault(bi, [[], set()])
                s[0].append(got)
                s[1].add(iid)

    out = {}
    for bi, (chunks, iids) in samples.items():
        px = np.concatenate(chunks)
        if len(px) < MIN_PIX:
            continue
        out[str(items[bi][0])] = soften(np.median(px, axis=0)) + [len(iids)]
    (OUT / "facade_colors.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"נשמרו צבעים ל-{len(out)} בניינים")


if __name__ == "__main__":
    main()
