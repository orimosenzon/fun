"""שכבות שבונים מהמקורות הנוספים (ראו fetch_extra.py). כל פונקציה כאן
מקבלת את מה שכבר נבנה מ-OSM ומחזירה תוספת. אם מקור חסר בדיסק, היא מחזירה
תוספת ריקה, כך שהעולם עדיין נבנה גם בלי make extra.
"""
import hashlib
import json
import math

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from shapely import STRtree, contains_xy, points
from shapely.geometry import LineString, MultiLineString, Polygon, shape
from shapely.ops import orient, unary_union

from fetch_osm import BBOX, ROOT
from geo import LAT0, LON0, M_PER_DEG_LAT, M_PER_DEG_LON, to_latlon, to_local

RAW = ROOT / "data" / "raw"
DK = ROOT.parent / "dereh_kitzur" / "web"


def stable_rand(key, salt=""):
    return int(hashlib.md5(f"{key}{salt}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def merc_px(lat, lon, z):
    """פיקסל גלובלי במרקטור בזום z (אריחים של 256)."""
    n = 2 ** z * 256
    lat = np.asarray(lat, dtype=np.float64)
    return ((np.asarray(lon) + 180) / 360 * n,
            (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n)


# ---------------------------------------------------------------- בניינים

def ms_buildings(osm_footprints):
    """בניינים של מיקרוסופט שאין להם מקבילה ב-OSM."""
    f = RAW / "ms" / "pardes.json"
    if not f.exists():
        return []
    tree = STRtree(osm_footprints)
    out = []
    for i, o in enumerate(json.loads(f.read_text())):
        ring = [to_local(lat, lon) for lon, lat in o["geometry"]["coordinates"][0]]
        p = Polygon(ring)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area < 12 or p.geom_type != "Polygon":
            continue
        hit = tree.query(p, predicate="intersects")
        if len(hit) and sum(osm_footprints[j].intersection(p).area for j in hit) > 0.15 * p.area:
            continue
        out.append((f"ms{i}", orient(p, 1.0), o["properties"].get("confidence", 1.0)))
    return out


class Ghsl:
    """גובה ממוצע נטו של בניינים במשבצת של כ-90 מ' (GHS-BUILT-H 2018)."""

    def __init__(self):
        f = RAW / "ghsl" / "height.npz"
        self.ok = f.exists()
        if self.ok:
            d = np.load(f)
            self.h, self.top, self.left, self.res = d["h"], float(d["top"]), float(d["left"]), float(d["res"])

    def at(self, x, y):
        if not self.ok:
            return 0.0
        lat, lon = to_latlon(x, y)
        i, j = int((self.top - lat) / self.res), int((lon - self.left) / self.res)
        if 0 <= i < self.h.shape[0] and 0 <= j < self.h.shape[1]:
            # המקסימום בסביבה הקרובה: בניין גבוה ליד גבול משבצת "נמרח" על שתיים
            return float(self.h[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].max() * 0.5 + self.h[i, j] * 0.5)
        return 0.0


# ---------------------------------------------------------------- קרקע

def s2_ground(terrain, tex):
    """צילום הלוויין דגום לגריד של טקסטורת הקרקע, או None אם לא הורד."""
    f = RAW / "s2.png"
    if not f.exists():
        return None
    meta = json.loads((RAW / "s2.json").read_text())
    img = np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255
    xs = terrain.west + (np.arange(tex) + 0.5) / tex * (terrain.east - terrain.west)
    ys = terrain.north - (np.arange(tex) + 0.5) / tex * (terrain.north - terrain.south)
    X, Y = np.meshgrid(xs, ys)
    lat, lon = LAT0 + Y / M_PER_DEG_LAT, LON0 + X / M_PER_DEG_LON
    px, py = merc_px(lat, lon, meta["z"])
    px -= meta["x0"] * 256
    py -= meta["y0"] * 256
    out = np.stack([map_coordinates(img[..., c], [py - 0.5, px - 0.5], order=1, mode="nearest")
                    for c in range(3)], -1)
    # Sentinel כהה ודהוי ביחס למה שהעין רואה בשטח: מעט בהירות ורוויה
    out = np.clip(out, 0, 1) ** 0.8
    gray = out.mean(-1, keepdims=True)
    out = np.clip(gray + (out - gray) * 1.25, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


# ---------------------------------------------------------------- עצים

class Canopy:
    """מסכת הצמרות הארצית (המרכז למיפוי ישראל 2021, דרך GoInfo ודרך קיצור)."""
    Z = 14

    def __init__(self):
        d = DK / "overlays" / "canopy" / str(self.Z)
        self.ok = d.exists()
        if not self.ok:
            return
        tiles = {(int(x.name), int(y.stem)): y for x in d.iterdir() for y in x.glob("*.png")}
        xs, ys = [k[0] for k in tiles], [k[1] for k in tiles]
        self.x0, self.y0 = min(xs), min(ys)
        self.a = np.zeros(((max(ys) - self.y0 + 1) * 512, (max(xs) - self.x0 + 1) * 512), np.uint8)
        for (x, y), p in tiles.items():
            im = np.asarray(Image.open(p).convert("RGBA"))[..., 3]
            self.a[(y - self.y0) * 512:(y - self.y0 + 1) * 512, (x - self.x0) * 512:(x - self.x0 + 1) * 512] = im
        # גודל פיקסל על הקרקע, במטרים רבועים
        self.px_m = 40075016.7 * math.cos(math.radians(LAT0)) / (2 ** self.Z * 512)

    def at(self, x, y):
        if not self.ok:
            return np.zeros(np.shape(x))
        lat, lon = to_latlon(np.asarray(x), np.asarray(y))
        px, py = merc_px(lat, lon, self.Z + 1)  # אריחי 512 בזום 14 = פיקסלים של זום 15
        i = (py - self.y0 * 512).astype(int)
        j = (px - self.x0 * 512).astype(int)
        ok = (i >= 0) & (i < self.a.shape[0]) & (j >= 0) & (j < self.a.shape[1])
        out = np.zeros(np.shape(x))
        out[ok] = self.a[i[ok], j[ok]] / 255
        return out


def canopy_trees(canopy, footprints, survey, terrain):
    """עץ לכל כ-80 מ"ר של צמרת, במקומות שבהם יש צמרת באמת.
    ממוצע הגודל מכויל כך שמספר העצים יתאים למה שהסקר הארצי ספר (כ-24 אלף)."""
    if not canopy.ok:
        return []
    rng = np.random.default_rng(5)
    m2 = 80.0
    cand = int((terrain.east - terrain.west) * (terrain.north - terrain.south) / m2 * 1.0)
    xs = rng.uniform(terrain.west, terrain.east, cand)
    ys = rng.uniform(terrain.south, terrain.north, cand)
    keep = canopy.at(xs, ys) > 0.5
    xs, ys = xs[keep], ys[keep]
    bt = STRtree(footprints)
    inside = np.zeros(len(xs), bool)
    hits = bt.query(points(xs, ys), predicate="intersects")
    inside[hits[0]] = True
    xs, ys = xs[~inside], ys[~inside]
    if survey:
        sv = STRtree(points([t["x"] for t in survey], [t["y"] for t in survey]))
        near = np.zeros(len(xs), bool)
        near[sv.query(points(xs, ys), predicate="dwithin", distance=3.0)[0]] = True
        xs, ys = xs[~near], ys[~near]
    out = []
    for x, y in zip(xs, ys):
        h = float(rng.uniform(5, 13))
        out.append({"x": round(float(x), 2), "y": round(float(y), 2), "h": round(h, 1),
                    "c": round(h * float(rng.uniform(0.6, 0.95)), 1), "kind": "canopy"})
    return out


# ---------------------------------------------------------------- גדרות וחומות

def parcel_walls(footprints, road_lines, terrain):
    """חומות לאורך גבולות המגרשים הבנויים, מהקדסטר (govmap, דרך דרך קיצור).
    חזית (ליד כביש): חומת טיח נמוכה. צד ואחור: חומת בלוקים או גדר חיה.
    מחזיר רשימה של (קו, סוג, גובה, עובי, צבע)."""
    f = DK / "data" / "parcels.json"
    if not f.exists():
        return []
    parcels = []
    for ft in json.loads(f.read_text())["features"]:
        p = shape(ft["geometry"])
        ring = [to_local(lat, lon) for lon, lat in p.exterior.coords] if p.geom_type == "Polygon" else None
        if ring:
            parcels.append((ft["id"], Polygon(ring)))
    cents = points([b.representative_point().x for b in footprints], [b.representative_point().y for b in footprints])
    ptree = STRtree([p for _, p in parcels])
    built = set()
    for bi, pi in zip(*ptree.query(cents, predicate="within")):
        built.add(pi)
    lots = [parcels[i] for i in built if 120 < parcels[i][1].area < 6000]
    lines = unary_union([p.exterior for _, p in lots])
    # לא חוצים בניינים וכבישים
    blockers = unary_union([b.buffer(0.6) for b in footprints] +
                           [ln.buffer(w / 2 + 0.4, cap_style=2) for ln, w, k in road_lines if k in (0, 1, 3)])
    lines = lines.difference(blockers)
    segs = list(lines.geoms) if isinstance(lines, MultiLineString) else [lines]
    rtree = STRtree([ln for ln, w, k in road_lines])
    widths = [w for ln, w, k in road_lines]
    out = []
    for i, s in enumerate(segs):
        if s.length < 1.0:
            continue
        mid = s.interpolate(0.5, normalized=True)
        j = rtree.nearest(mid)
        d = road_lines[j][0].distance(mid)
        front = d < widths[j] / 2 + 5
        r = stable_rand(i, "w")
        if front:
            out.append((s, 0, 1.0 + 0.5 * r, 0.25))  # טיח
        elif r < 0.45:
            out.append((s, 2, 1.2 + 0.7 * stable_rand(i, "h"), 0.7))  # גדר חיה
        else:
            out.append((s, 1, 1.8, 0.2))  # בלוקים
    return out


# ---------------------------------------------------------------- ריהוט רחוב

def street_furniture(els, road_lines, terrain):
    """אובייקטים נקודתיים: [x, y, כיוון במעלות, סוג]. סוג:
    0 תחנת אוטובוס, 1 רמזור, 2 פנס רחוב, 3 עמוד חשמל, 4 עמוד מתח גבוה, 5 ספסל."""
    rtree = STRtree([ln for ln, w, k in road_lines])
    out = []

    def road_dir(x, y):
        from shapely.geometry import Point
        pt = Point(x, y)
        j = rtree.nearest(pt)
        ln = road_lines[j][0]
        s = ln.project(pt)
        a, b = ln.interpolate(max(s - 1, 0)), ln.interpolate(min(s + 1, ln.length))
        return math.degrees(math.atan2(b.x - a.x, b.y - a.y)), ln, road_lines[j][1]

    kinds = {("highway", "bus_stop"): 0, ("highway", "traffic_signals"): 1, ("highway", "street_lamp"): 2,
             ("power", "pole"): 3, ("power", "tower"): 4, ("amenity", "bench"): 5}
    for el in els:
        if el["type"] != "node":
            continue
        t = el.get("tags", {})
        k = next((v for (kk, vv), v in kinds.items() if t.get(kk) == vv), None)
        if k is None:
            continue
        x, y = to_local(el["lat"], el["lon"])
        ang, ln, w = road_dir(x, y)
        if k == 0:
            # תחנה שממופה על הכביש עצמו: מזיזים למדרכה
            from shapely.geometry import Point
            p = Point(x, y)
            if ln.distance(p) < w / 2 + 0.5:
                s = ln.project(p)
                a, b = ln.interpolate(max(s - 1, 0)), ln.interpolate(min(s + 1, ln.length))
                dx, dy = b.x - a.x, b.y - a.y
                L = math.hypot(dx, dy) or 1
                c = ln.interpolate(s)
                x, y = c.x + dy / L * (w / 2 + 1.8), c.y - dx / L * (w / 2 + 1.8)
        out.append([round(x, 2), round(y, 2), round(ang, 1), k])
    # פנסי רחוב: OSM לא ממפה אותם במושבה, אז מציבים כל 32 מ' לאורך רחובות סלולים
    for i, (ln, w, kind) in enumerate(road_lines):
        if kind != 0 or w < 5 or ln.length < 20:
            continue
        side = 1 if stable_rand(i, "side") < 0.5 else -1
        for s in np.arange(10 + 20 * stable_rand(i, "o"), ln.length - 5, 32):
            a, b = ln.interpolate(max(s - 1, 0)), ln.interpolate(min(s + 1, ln.length))
            dx, dy = b.x - a.x, b.y - a.y
            L = math.hypot(dx, dy) or 1
            c = ln.interpolate(s)
            x, y = c.x + side * dy / L * (w / 2 + 0.7), c.y - side * dx / L * (w / 2 + 0.7)
            # הזרוע של הפנס פונה אל הכביש
            ang = math.degrees(math.atan2(-side * dy, side * dx))
            out.append([round(x, 2), round(y, 2), round(ang, 1), 2])
    return out


def power_lines(els):
    """קווי מתח: לכל קו רשימת [x, y, גובה תלייה]. 22 מ' ליד עמוד מתח גבוה, 9 מ' ליד עמוד רגיל."""
    towers = []
    for el in els:
        t = el.get("tags", {})
        if el["type"] == "node" and t.get("power") in ("tower", "pole"):
            towers.append((*to_local(el["lat"], el["lon"]), 22.0 if t["power"] == "tower" else 9.0))
    from shapely.geometry import Point
    tt = STRtree(points([t[0] for t in towers], [t[1] for t in towers])) if towers else None
    out = []
    for el in els:
        t = el.get("tags", {})
        if el["type"] == "way" and t.get("power") == "line" and "geometry" in el:
            line = []
            for p in el["geometry"]:
                x, y = to_local(p["lat"], p["lon"])
                h = 9.0
                if tt is not None:
                    j = tt.nearest(Point(x, y))
                    if math.hypot(towers[j][0] - x, towers[j][1] - y) < 8:
                        h = towers[j][2]
                line.append([round(x, 1), round(y, 1), h])
            out.append(line)
    return out


def crossings(els, road_lines):
    """מעברי חצייה: [x, y, כיוון הכביש, רוחב הכביש]."""
    from shapely.geometry import Point
    rtree = STRtree([ln for ln, w, k in road_lines])
    out = []
    for el in els:
        if el["type"] == "node" and el.get("tags", {}).get("highway") == "crossing":
            x, y = to_local(el["lat"], el["lon"])
            p = Point(x, y)
            cands = [j for j in rtree.query(p.buffer(3)) if road_lines[j][2] == 0]
            if not cands:
                continue
            j = min(cands, key=lambda j: road_lines[j][0].distance(p))
            ln, w = road_lines[j][0], road_lines[j][1]
            s = ln.project(p)
            a, b = ln.interpolate(max(s - 1, 0)), ln.interpolate(min(s + 1, ln.length))
            c = ln.interpolate(s)
            out.append((c.x, c.y, math.atan2(b.x - a.x, b.y - a.y), w))
    return out


# ---------------------------------------------------------------- נקודות מידע

def pois():
    """מקומות מפרדספדיה ותמונות מ-Commons, לנקודות מידע בעולם."""
    out = []
    f = RAW / "pardespedia_places.json"
    if f.exists():
        for p in json.loads(f.read_text()):
            g = p.get("geo")
            if not g:
                continue
            x, y = to_local(g["lat"], g["lng"])
            ph = (p.get("photos") or [{}])[0]
            out.append({"src": "pp", "x": round(x, 1), "y": round(y, 1), "name": p["name"],
                        "group": p.get("group", ""), "text": (p.get("note") or "")[:600],
                        "address": p.get("address", ""), "url": p.get("url", ""),
                        "img": ph.get("thumb", ""), "credit": "פרדספדיה",
                        "approx": g.get("source") in ("street", "nearby")})
    f = RAW / "commons.json"
    if f.exists():
        import re
        strip = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()
        for p in json.loads(f.read_text()):
            x, y = to_local(p["lat"], p["lon"])
            out.append({"src": "commons", "x": round(x, 1), "y": round(y, 1), "name": p["title"],
                        "group": "תמונה", "text": strip(p.get("desc"))[:400], "url": p.get("page", ""),
                        "img": p.get("thumb", ""), "credit": f"{strip(p.get('author'))[:80]} · {p.get('license', '')} · Wikimedia Commons"})
    return out
