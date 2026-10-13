"""בונה את העולם התלת-ממדי מהנתונים הגולמיים.

קלט:  data/raw/osm.json, data/raw/dem.npz, וסקר העצים של דרך קיצור.
פלט (ל-game/world/, שגודו טוען בזמן ריצה):
  terrain.glb    רשת הקרקע, עם UV לטקסטורת הקרקע
  ground.png     טקסטורת הקרקע: שימושי קרקע וכבישים מצוירים
  buildings.glb  בניינים מוחצנים; UV: u = מטרים לאורך ההיקף, v = גובה במטרים; בגג u = -1000
                 (השיידר בגודו מצייר מזה חלונות וקומות)
  roads.glb      סרטי כביש שמונחים על הקרקע; סוג ורוחב מקודדים ב-UV.x
  heights.bin    גריד הגבהים (float32), לשימוש בגודו להליכה על הקרקע
  trees.json     עצים: מיקום, גובה, קוטר צמרת, מין
  places.json    מקומות עם שם, לתפריט הקפיצה
  meta.json      מקור הקואורדינטות, גבולות, מקורות וייחוס
"""
import hashlib
import json
import math
import pathlib

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import orient, unary_union

import extras
from geo import LAT0, LON0, M_PER_DEG_LAT, M_PER_DEG_LON, to_local
from fetch_dem import tile_xy
from fetch_osm import BBOX, ROOT

OUT = ROOT / "game" / "world"
TREES_SRC = ROOT.parent / "dereh_kitzur" / "web" / "data" / "trees.json"

GRID = 10.0          # מרווח הגריד של הקרקע במטרים
TEX = 4096           # רזולוציית טקסטורת הקרקע
FLOOR = 3.2          # גובה קומה


# ---------------------------------------------------------------- קרקע

class Terrain:
    def __init__(self):
        d = np.load(ROOT / "data" / "raw" / "dem.npz")
        self.dem, self.z, self.x0, self.y0 = d["h"], int(d["z"]), int(d["x0"]), int(d["y0"])
        s, w, n, e = BBOX
        self.west, self.south = to_local(s, w)
        self.east, self.north = to_local(n, e)
        self.nx = int((self.east - self.west) / GRID) + 1
        self.ny = int((self.north - self.south) / GRID) + 1
        xs = self.west + np.arange(self.nx) * GRID
        ys = self.south + np.arange(self.ny) * GRID
        X, Y = np.meshgrid(xs, ys)
        self.h = self._sample_dem(X, Y)
        # החלקה קלה: ה-SRTM רועש ביחס לרזולוציה
        from scipy.ndimage import gaussian_filter
        self.h = gaussian_filter(self.h, 1.5).astype(np.float32)

    def _sample_dem(self, X, Y):
        lat = LAT0 + Y / M_PER_DEG_LAT
        lon = LON0 + X / M_PER_DEG_LON
        n = 2 ** self.z
        px = ((lon + 180) / 360 * n - self.x0) * 256
        py = ((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n - self.y0) * 256
        from scipy.ndimage import map_coordinates
        return map_coordinates(self.dem, [py - 0.5, px - 0.5], order=1, mode="nearest")

    def height(self, x, y):
        """גובה הקרקע (אינטרפולציה בילינארית) בנקודה מקומית או במערך נקודות."""
        fx = np.clip((np.asarray(x) - self.west) / GRID, 0, self.nx - 1.001)
        fy = np.clip((np.asarray(y) - self.south) / GRID, 0, self.ny - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        tx, ty = fx - ix, fy - iy
        h = self.h
        return (h[iy, ix] * (1 - tx) * (1 - ty) + h[iy, ix + 1] * tx * (1 - ty)
                + h[iy + 1, ix] * (1 - tx) * ty + h[iy + 1, ix + 1] * tx * ty)

    def mesh(self):
        xs = self.west + np.arange(self.nx) * GRID
        ys = self.south + np.arange(self.ny) * GRID
        X, Y = np.meshgrid(xs, ys)
        # גודו: x=מזרח, y=גובה, z=-צפון
        v = np.stack([X.ravel(), self.h.ravel(), -Y.ravel()], 1)
        uv = np.stack([(X.ravel() - self.west) / (self.east - self.west),
                       (Y.ravel() - self.south) / (self.north - self.south)], 1)
        i = np.arange(self.nx * self.ny).reshape(self.ny, self.nx)
        a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
        f = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
        m = trimesh.Trimesh(v, f, process=False)
        m.visual = trimesh.visual.TextureVisuals(uv=uv)
        return m


# ---------------------------------------------------------------- OSM

def load_osm():
    return json.loads((ROOT / "data" / "raw" / "osm.json").read_text())["elements"]


def ring(geom):
    return [to_local(p["lat"], p["lon"]) for p in geom]


def polygons(el):
    """פוליגונים מקומיים מ-way סגור או מ-multipolygon relation."""
    if el["type"] == "way":
        g = el.get("geometry")
        if not g or len(g) < 4 or g[0] != g[-1]:
            return []
        p = Polygon(ring(g))
    elif el["type"] == "relation":
        from shapely.ops import polygonize
        lines = {"outer": [], "inner": []}
        for m in el.get("members", []):
            if m.get("type") != "way" or len(m.get("geometry") or []) < 2:
                continue
            lines["inner" if m.get("role") == "inner" else "outer"].append(LineString(ring(m["geometry"])))
        # טבעות שמפוצלות לכמה ways מתחברות כאן חזרה
        outers = list(polygonize(lines["outer"]))
        inners = list(polygonize(lines["inner"]))
        if not outers:
            return []
        p = unary_union(outers)
        for h in inners:
            p = p.difference(h)
    else:
        return []
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty:
        return []
    return list(p.geoms) if isinstance(p, MultiPolygon) else [p]


def stable_rand(key, salt=""):
    """מספר "אקראי" יציב לכל אלמנט, כדי שהעולם ייראה אותו דבר בכל בנייה."""
    return int(hashlib.md5(f"{key}{salt}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


# ---------------------------------------------------------------- בניינים

WALL_COLORS = [  # טיח ישראלי: לבן, שמנת, בז', אבן ירושלמית, מעט אפור
    (236, 233, 224), (240, 236, 222), (226, 216, 196), (221, 205, 172),
    (232, 225, 210), (214, 210, 202), (238, 228, 205), (205, 196, 180),
]


def building_height(key, t, area, g):
    """גובה בניין במטרים. g = הגובה הממוצע של הבניינים במשבצת GHSL (0 אם אין)."""
    for k in ("height", "building:height"):
        try:
            return float(t[k].split()[0])
        except (KeyError, ValueError):
            pass
    try:
        return float(t["building:levels"]) * FLOOR + 0.8
    except (KeyError, ValueError):
        pass
    kind = t.get("building", "yes")
    r = stable_rand(key)
    if kind in ("greenhouse",):
        return 4.0
    if kind in ("industrial", "warehouse", "retail", "cowshed", "farm_auxiliary", "shed", "roof"):
        return 5 + 4 * r
    if area < 25:  # מחסן, חדר שומר
        return 2.6
    if area > 250 and g >= 5.5:
        # משבצת עם בניינים גבוהים: בניין גדול בה הוא כנראה בניין דירות.
        # הממוצע במשבצת כולל גם בתים נמוכים, ולכן מעט מעליו
        lv = int(np.clip(round(g * 1.35 / FLOOR + (r - 0.5) * 1.5), 3, 14))
        return lv * FLOOR + 0.8
    if kind == "apartments" or area > 600:
        return (3 + int(r * 3)) * FLOOR + 0.8
    # בתים צמודי קרקע: ככל שהמשבצת גבוהה יותר, יותר בתים של שתי קומות
    p2 = 0.45 if g <= 0 else float(np.clip(0.2 + (g - 2.5) * 0.25, 0.15, 0.85))
    return (2 if r < p2 else 1) * FLOOR + 0.6


def enu_to_godot(v):
    """(מזרח, צפון, גובה) -> (x, y, z) של גודו. זה סיבוב, לא שיקוף, אז כיוון המשולשים נשמר."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


def collect_buildings(els):
    """(מזהה, פוליגון, תגיות) לכל בניין: OSM, ואחריו מה שחסר בו ונמצא אצל מיקרוסופט."""
    items = []
    for el in els:
        t = el.get("tags", {})
        if "building" not in t or t["building"] == "no":
            continue
        for poly in polygons(el):
            if poly.area < 6:
                continue
            poly = orient(poly.simplify(0.15), 1.0)  # חיצוני נגד כיוון השעון, חורים עם
            if poly.is_empty or not isinstance(poly, Polygon):
                continue
            items.append((el["id"], poly, t))
    n_osm = len(items)
    for key, poly, conf in extras.ms_buildings([p for _, p, _ in items]):
        poly = orient(poly.simplify(0.3), 1.0)
        if isinstance(poly, Polygon) and not poly.is_empty:
            items.append((key, poly, {"building": "yes", "source": "microsoft"}))
    return items, n_osm


def build_buildings(items, terrain, ghsl):
    import mapbox_earcut as earcut
    V, F, UV, C = [], [], [], []
    n = 0
    footprints = []
    for key, poly, t in items:
        if True:
            footprints.append(poly)
            c = poly.centroid
            hgt = building_height(key, t, poly.area, ghsl.at(c.x, c.y))
            ex = np.array(poly.exterior.coords)
            hs = terrain.height(ex[:, 0], ex[:, 1])
            base = float(hs.min()) - 0.3  # הבסיס קבור קצת, כדי שלא ירחף בשיפוע
            top = float(hs.max()) + hgt
            col = WALL_COLORS[int(stable_rand(key, "c") * len(WALL_COLORS))]
            v, f, uv = [], [], []
            # קירות: u = מטרים לאורך ההיקף, v = מטרים מעל הקרקע הגבוהה ביותר בבניין פחות הגובה
            for r in [poly.exterior, *poly.interiors]:
                pts = np.array(r.coords)
                u = 0.0
                for a, b in zip(pts[:-1], pts[1:]):
                    L = float(np.linalg.norm(b - a))
                    if L < 0.05:
                        continue
                    k = len(v)
                    v += [(a[0], a[1], base), (b[0], b[1], base), (b[0], b[1], top), (a[0], a[1], top)]
                    vb = base - (top - hgt)
                    uv += [(u, vb), (u + L, vb), (u + L, hgt), (u, hgt)]
                    f += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
                    u += L
            # גג: earcut על הפוליגון עם החורים
            rings = [np.array(poly.exterior.coords)[:-1]] + [np.array(r.coords)[:-1] for r in poly.interiors]
            flat = np.vstack(rings)
            ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
            tri = earcut.triangulate_float64(flat, ends).reshape(-1, 3)
            k = len(v)
            v += [(x, y, top) for x, y in flat]
            uv += [(-1000.0, 0.0)] * len(flat)  # u שלילי = גג (השיידר מזהה)
            for a, b, c in tri:
                pa, pb, pc = flat[a], flat[b], flat[c]
                cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
                f.append((k + a, k + b, k + c) if cross > 0 else (k + a, k + c, k + b))
            V.append(enu_to_godot(v))
            F.append(np.array(f) + n)
            UV.append(np.array(uv, dtype=np.float64))
            C.append(np.tile(col + (255,), (len(v), 1)))
            n += len(v)
    m = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.vstack(UV))
    return m, np.vstack(C).astype(np.uint8), footprints


# ---------------------------------------------------------------- כבישים

ROAD_W = {
    "motorway": 14, "trunk": 12, "primary": 10, "secondary": 9, "tertiary": 8,
    "motorway_link": 7, "trunk_link": 7, "primary_link": 7, "secondary_link": 7, "tertiary_link": 7,
    "residential": 6.5, "unclassified": 6, "living_street": 5, "service": 4,
    "track": 3.5, "footway": 2, "path": 1.6, "cycleway": 2.2, "pedestrian": 4, "steps": 2,
}
ROAD_KIND = {  # 0 אספלט, 1 שביל עפר, 2 מדרכה/שביל אופניים
    "track": 1, "path": 1, "footway": 2, "cycleway": 2, "pedestrian": 2, "steps": 2,
}


def build_roads(els, terrain):
    V, F, UV = [], [], []
    n = 0
    lines = []
    for el in els:
        t = el.get("tags", {})
        hw = t.get("highway")
        rw = t.get("railway")
        if el["type"] != "way" or "geometry" not in el:
            continue
        if hw in ROAD_W and t.get("area") != "yes" and t.get("tunnel") != "yes":
            w = ROAD_W[hw]
            try:
                w = float(t.get("width", w))
            except ValueError:
                pass
            kind = ROAD_KIND.get(hw, 0)
            if t.get("surface") in ("unpaved", "dirt", "ground", "gravel", "sand", "compacted"):
                kind = 1
        elif rw in ("rail",) and t.get("tunnel") != "yes":
            w, kind = 4.0, 3
        else:
            continue
        pts = np.array(ring(el["geometry"]))
        if len(pts) < 2:
            continue
        line = LineString(pts)
        lines.append((line, w, kind))
        # צפיפות: נקודה כל 8 מטר כדי שהסרט ילך עם הקרקע
        L = line.length
        if L < 1:
            continue
        k = max(2, int(L / 8) + 1)
        p = np.array([line.interpolate(s).coords[0] for s in np.linspace(0, L, k)])
        d = np.gradient(p, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) * (w / 2)
        left, right = p + nrm, p - nrm
        lift = {0: 0.12, 1: 0.08, 2: 0.16, 3: 0.2}[kind]
        if hw in ("primary", "trunk", "secondary"):
            lift += 0.02  # כבישים ראשיים מעל המשניים בצמתים
        hl = terrain.height(left[:, 0], left[:, 1]) + lift
        hr = terrain.height(right[:, 0], right[:, 1]) + lift
        s = np.linspace(0, L, k)
        gl = np.stack([left[:, 0], hl, -left[:, 1]], 1)
        gr = np.stack([right[:, 0], hr, -right[:, 1]], 1)
        v = np.empty((2 * k, 3))
        v[0::2], v[1::2] = gl, gr
        # UV.x = רוחבי (0..1) + 2 * round(רוחב * 4) + 1000 * סוג; UV.y = מטרים לאורך
        # (צבע קודקוד עובר המרת sRGB בדרך, אז מקודדים ב-UV)
        code = 2 * round(w * 4) + 1000 * kind
        uv = np.empty((2 * k, 2))
        uv[0::2, 0], uv[1::2, 0] = code, code + 1
        uv[0::2, 1] = uv[1::2, 1] = s
        i = np.arange(k - 1) * 2 + n
        f = np.concatenate([np.stack([i, i + 1, i + 2], 1), np.stack([i + 1, i + 3, i + 2], 1)])
        V.append(v)
        F.append(f)
        UV.append(uv)
        n += 2 * k
    # מעברי חצייה: מלבן קצר מעל הכביש, סוג 4
    for x, y, ang, w in extras.crossings(els, lines):
        fx, fy = math.sin(ang), math.cos(ang)  # כיוון הכביש
        rx, ry = fy, -fx                        # ימינה
        L = 1.6
        corners = [(x - fx * L - rx * w / 2, y - fy * L - ry * w / 2), (x - fx * L + rx * w / 2, y - fy * L + ry * w / 2),
                   (x + fx * L - rx * w / 2, y + fy * L - ry * w / 2), (x + fx * L + rx * w / 2, y + fy * L + ry * w / 2)]
        hh = terrain.height(np.array([c[0] for c in corners]), np.array([c[1] for c in corners])) + 0.19
        V.append(np.array([(cx, h, -cy) for (cx, cy), h in zip(corners, hh)]))
        code = 2 * round(w * 4) + 4000
        UV.append(np.array([(code, 0), (code + 1, 0), (code, 2 * L), (code + 1, 2 * L)], dtype=float))
        F.append(np.array([(n, n + 1, n + 2), (n + 1, n + 3, n + 2)]))
        n += 4
    m = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.vstack(UV))
    return m, lines


# ---------------------------------------------------------------- חומות וגדרות

def build_walls(walls, terrain):
    """קופסאות דקות לאורך כל קטע. UV.x = מטרים לאורך + 10000 * סוג, UV.y = גובה."""
    V, F, UV, C = [], [], [], []
    n = 0
    for i, (line, kind, h, thick) in enumerate(walls):
        col = WALL_COLORS[int(stable_rand(i, "wc") * len(WALL_COLORS))] if kind == 0 else (190, 188, 182)
        pts = np.array(line.coords)
        u0 = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.linalg.norm(b - a))
            if L < 0.2:
                continue
            k = max(1, int(L / 12) + 1)
            for j in range(k):  # קטעים קצרים כדי לעקוב אחרי הקרקע
                p0 = a + (b - a) * j / k
                p1 = a + (b - a) * (j + 1) / k
                d = (p1 - p0) / np.linalg.norm(p1 - p0)
                nn = np.array([d[1], -d[0]]) * thick / 2
                g0 = float(terrain.height(p0[0], p0[1])) - 0.2
                g1 = float(terrain.height(p1[0], p1[1])) - 0.2
                quad = [p0 + nn, p1 + nn, p1 - nn, p0 - nn]  # ימין-התחלה, ימין-סוף, שמאל-סוף, שמאל-התחלה
                gs = [g0, g1, g1, g0]
                bot = [(q[0], q[1], g) for q, g in zip(quad, gs)]
                top = [(q[0], q[1], g + h + 0.2) for q, g in zip(quad, gs)]
                v = bot + top
                ul = u0 + L * j / k
                ur = u0 + L * (j + 1) / k
                base = 10000 * kind
                uv = [(base + ul, 0), (base + ur, 0), (base + ur, 0), (base + ul, 0),
                      (base + ul, h), (base + ur, h), (base + ur, h), (base + ul, h)]
                # צד ימין (0,1,5,4), צד שמאל (2,3,7,6), קצוות (1,2,6,5),(3,0,4,7), גג (4,5,6,7)
                f = [(0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6), (4, 5, 6), (4, 6, 7)]
                if j == k - 1:
                    f += [(1, 2, 6), (1, 6, 5)]
                if j == 0:
                    f += [(3, 0, 4), (3, 4, 7)]
                V.append(enu_to_godot(v))
                UV.append(np.array(uv, dtype=float))
                F.append(np.array(f) + n)
                C.append(np.tile(col + (255,), (8, 1)))
                n += 8
            u0 += L
    m = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.vstack(UV))
    return m, np.vstack(C).astype(np.uint8)


# ---------------------------------------------------------------- טקסטורת קרקע

LAND = {  # צבע לפי שימוש קרקע, בסדר הציור (הראשון מתחת)
    "farmland": (176, 160, 112), "meadow": (150, 158, 96), "grass": (120, 150, 80),
    "orchard": (92, 122, 60), "vineyard": (120, 130, 70), "residential": (168, 160, 138),
    "industrial": (160, 156, 150), "commercial": (170, 165, 155), "retail": (170, 165, 155),
    "farmyard": (160, 145, 115), "cemetery": (130, 140, 100), "sand": (222, 205, 160),
    "scrub": (120, 130, 80), "forest": (60, 95, 45), "wood": (60, 95, 45),
    "park": (95, 140, 65), "garden": (95, 140, 65), "playground": (190, 170, 120),
    "pitch": (80, 140, 70), "sports_centre": (150, 150, 130), "construction": (190, 170, 135),
    "water": (70, 120, 150), "swimming_pool": (80, 170, 210), "reservoir": (70, 120, 150),
}


def build_ground_texture(els, terrain, road_lines, footprints):
    """בסיס: צילום הלוויין (Sentinel-2) אם הורד, אחרת צבעי שימושי הקרקע מ-OSM.
    מעליו כבישים, מים וצל מגע של בניינים, שהלוויין רזולוציה גסה מכדי להראות."""
    sat = extras.s2_ground(terrain, TEX)
    if sat is not None:
        return _overlay(els, sat, terrain, road_lines, footprints, sat_base=True)
    img = Image.new("RGB", (TEX, TEX), (186, 170, 130))  # אדמת חמרה בהירה
    return _overlay(els, img, terrain, road_lines, footprints, sat_base=False)


def _overlay(els, img, terrain, road_lines, footprints, sat_base):
    dr = ImageDraw.Draw(img)
    sx = TEX / (terrain.east - terrain.west)
    sy = TEX / (terrain.north - terrain.south)

    def px(coords):
        return [((x - terrain.west) * sx, (terrain.north - y) * sy) for x, y in coords]

    order = list(LAND)
    areas = []
    for el in els:
        t = el.get("tags", {})
        k = t.get("landuse") or t.get("leisure") or t.get("natural") or t.get("amenity")
        if k in LAND and "building" not in t:
            for p in polygons(el):
                areas.append((order.index(k), k, p))
    areas.sort(key=lambda a: (a[0], -a[2].area))
    if sat_base:  # בלוויין רואים את השטח עצמו; מ-OSM לוקחים רק מים
        areas = [a for a in areas if a[1] in ("water", "swimming_pool", "reservoir")]
    for _, k, p in areas:
        dr.polygon(px(p.exterior.coords), fill=LAND[k])
        for r in p.interiors:
            dr.polygon(px(r.coords), fill=(186, 170, 130))
    # כתמי ירק אקראיים: גינות פרטיות שלא ממופות
    rng = np.random.default_rng(7)
    for _ in range(0 if sat_base else 25000):
        x, y = rng.uniform(0, TEX, 2)
        r = rng.uniform(1.5, 5)
        g = int(rng.uniform(90, 130))
        dr.ellipse((x - r, y - r, x + r, y + r), fill=(int(g * 0.72), g, int(g * 0.5)))
    # כבישים: גם בטקסטורה, כדי שירחוק ייראו טוב כשהסרטים כבר לא נראים
    for line, w, kind in road_lines:
        col = {0: (70, 70, 72), 1: (170, 150, 115), 2: (150, 145, 140), 3: (110, 100, 90)}[kind]
        dr.line(px(line.coords), fill=col, width=max(1, int(w * sx)))
    # צל מגע מתחת לבניינים
    for p in footprints:
        dr.polygon(px(p.buffer(1.2).exterior.coords), fill=(110, 105, 98))
    return img


# ---------------------------------------------------------------- עצים ומקומות

def build_trees(els):
    trees = []
    if TREES_SRC.exists():
        for p in json.loads(TREES_SRC.read_text())["places"]:
            g, t = p.get("geo"), p.get("tree", {})
            if not g:
                continue
            x, y = to_local(g["lat"], g["lng"])
            trees.append({"x": round(x, 2), "y": round(y, 2), "h": t.get("h") or 7,
                          "c": t.get("c") or 6, "name": p.get("name", ""), "health": t.get("health", "")})
    for el in els:
        if el["type"] == "node" and el.get("tags", {}).get("natural") == "tree":
            x, y = to_local(el["lat"], el["lon"])
            trees.append({"x": round(x, 2), "y": round(y, 2), "h": 7, "c": 6, "name": "", "health": ""})
    return trees


def scatter_trees(els, footprints, road_lines, canopy):
    """עצים "ממולאים" בשטחים ירוקים ובשכונות, כי הסקר ו-OSM מכסים רק חלק קטן מהם.
    מסומנים ב-name ריק ו-health="synthetic" כדי שיהיה ברור שאינם עצים אמיתיים."""
    from shapely.strtree import STRtree
    rng = np.random.default_rng(11)
    # לא שותלים על בניינים ועל כבישים
    blockers = footprints + [ln.buffer(w / 2 + 1.0, cap_style=2) for ln, w, _ in road_lines]
    tree_index = STRtree(blockers)
    # בשכונות ובפארקים העצים באים ממסכת הצמרות. כאן רק פרדסים וחורשות,
    # ורק היכן שהמסכה לא כבר שתלה (היא מכסה בעיקר את השטח הבנוי)
    density = {"orchard": 1 / 40, "forest": 1 / 60, "wood": 1 / 60}
    if not canopy.ok:
        density.update({"park": 1 / 150, "garden": 1 / 150, "residential": 1 / 260, "cemetery": 1 / 120})
    out = []
    for el in els:
        t = el.get("tags", {})
        k = t.get("landuse") or t.get("leisure") or t.get("natural")
        if k not in density:
            continue
        for p in polygons(el):
            cnt = min(int(p.area * density[k]), 8000)
            if cnt == 0:
                continue
            minx, miny, maxx, maxy = p.bounds
            xs = rng.uniform(minx, maxx, cnt * 3)
            ys = rng.uniform(miny, maxy, cnt * 3)
            from shapely import points, contains
            pts = points(xs, ys)
            inside = contains(p, pts) & (canopy.at(xs, ys) < 0.5)
            kept = 0
            for x, y, ok, pt in zip(xs, ys, inside, pts):
                if not ok:
                    continue
                if len(tree_index.query(pt.buffer(2.0), predicate="intersects")) > 0:
                    continue
                h = rng.uniform(4, 9) if k == "orchard" else rng.uniform(5, 14)
                out.append({"x": round(float(x), 2), "y": round(float(y), 2), "h": round(h, 1),
                            "c": round(h * rng.uniform(0.5, 0.8), 1), "name": "", "health": "synthetic",
                            "kind": "citrus" if k == "orchard" else ""})
                kept += 1
                if kept >= cnt:
                    break
    return out


def build_places(els):
    out, seen = [], set()
    for el in els:
        t = el.get("tags", {})
        name = t.get("name:he") or t.get("name")
        if not name or "highway" in t:
            continue
        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
        elif "bounds" in el:
            b = el["bounds"]
            lat, lon = (b["minlat"] + b["maxlat"]) / 2, (b["minlon"] + b["maxlon"]) / 2
        else:
            continue
        kind = (t.get("amenity") or t.get("railway") or t.get("leisure") or t.get("shop")
                or t.get("building") or t.get("place") or t.get("landuse") or "")
        if el.get("id") == 1392849 or kind in ("village",) or "/" in name:
            continue
        x, y = to_local(lat, lon)
        key = (name, round(x / 50), round(y / 50))
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "kind": kind, "x": round(x, 1), "y": round(y, 1)})
    return out


def boundary(els):
    for el in els:
        if el["type"] == "relation" and el["id"] == 1392849:
            return [[round(x, 1), round(y, 1)] for p in polygons(el) for x, y in p.exterior.coords]
    return []


# ---------------------------------------------------------------- main

def export_glb(mesh, path, colors=None):
    if colors is not None:
        mesh.visual.vertex_attributes["color"] = colors
    path.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh), include_normals=True))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    els = load_osm()
    print("קרקע...")
    terrain = Terrain()
    export_glb(terrain.mesh(), OUT / "terrain.glb")
    terrain.h.astype("<f4").tofile(OUT / "heights.bin")
    print("בניינים...")
    items, n_osm = collect_buildings(els)
    bm, bcol, footprints = build_buildings(items, terrain, extras.Ghsl())
    export_glb(bm, OUT / "buildings.glb", bcol)
    print(f"  {len(footprints)} בניינים ({n_osm} מ-OSM, {len(footprints) - n_osm} ממיקרוסופט), {len(bm.faces)} משולשים")
    print("כבישים...")
    rm, lines = build_roads(els, terrain)
    export_glb(rm, OUT / "roads.glb")
    print(f"  {len(lines)} קטעים")
    print("טקסטורת קרקע...")
    build_ground_texture(els, terrain, lines, footprints).save(OUT / "ground.png", optimize=True)
    print("חומות וגדרות...")
    walls = extras.parcel_walls(footprints, lines, terrain)
    if walls:
        wm, wcol = build_walls(walls, terrain)
        export_glb(wm, OUT / "walls.glb", wcol)
        print(f"  {len(walls)} קטעים, {len(wm.faces)} משולשים")
    furniture = extras.street_furniture(els, lines, terrain)
    (OUT / "furniture.json").write_text(json.dumps(furniture, separators=(",", ":")))
    (OUT / "power.json").write_text(json.dumps(extras.power_lines(els), separators=(",", ":")))
    print(f"ריהוט רחוב: {len(furniture)}")
    pois = extras.pois()
    (OUT / "pois.json").write_text(json.dumps(pois, ensure_ascii=False))
    print(f"נקודות מידע: {len(pois)}")
    trees = build_trees(els)
    canopy = extras.Canopy()
    ctrees = extras.canopy_trees(canopy, footprints, trees, terrain)
    synth = scatter_trees(els, footprints, lines, canopy)
    # קומפקטי: [x, y, גובה, קוטר צמרת, סוג]
    # סוג 0 = עץ מהסקר, 1 = ממולא, 2 = הדר ממולא, 3 = ממסכת הצמרות (מיקום אמיתי, גודל משוער)
    rows = [[t["x"], t["y"], t["h"], t["c"], 0] for t in trees]
    rows += [[t["x"], t["y"], t["h"], t["c"], 3] for t in ctrees]
    rows += [[t["x"], t["y"], t["h"], t["c"], 2 if t.get("kind") == "citrus" else 1] for t in synth]
    (OUT / "trees.json").write_text(json.dumps(rows, separators=(",", ":")))
    (OUT / "survey_trees.json").write_text(json.dumps(trees, ensure_ascii=False))
    print(f"עצים: {len(trees)} מהסקר + {len(ctrees)} ממסכת הצמרות + {len(synth)} ממולאים")
    places = build_places(els)
    (OUT / "places.json").write_text(json.dumps(places, ensure_ascii=False, indent=0))
    meta = {
        "lat0": LAT0, "lon0": LON0, "m_per_deg_lat": M_PER_DEG_LAT, "m_per_deg_lon": M_PER_DEG_LON,
        "grid": GRID, "nx": terrain.nx, "ny": terrain.ny, "west": terrain.west, "south": terrain.south,
        "east": terrain.east, "north": terrain.north,
        "boundary": boundary(els),
        "attribution": [
            "© OpenStreetMap contributors (ODbL)",
            "Elevation: Mapzen Terrarium / SRTM (AWS Open Data)",
            "סקר העצים 2018–2019, פרדספדיה, דרך קיצור",
            "Microsoft Global ML Building Footprints (ODbL)",
            "GHS-BUILT-H R2023A, European Commission JRC (CC BY 4.0)",
            "Sentinel-2 cloudless 2024 by EOX IT Services GmbH, contains modified Copernicus Sentinel data 2024 (CC BY-NC-SA 4.0)",
            "מפת הצמרות הארצית 2021, המרכז למיפוי ישראל, דרך GoInfo",
            "גושים וחלקות: GovMap, המרכז למיפוי ישראל",
            "פרדספדיה; Wikimedia Commons (יוצר ורישיון לכל תמונה)",
        ],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"מקומות: {len(places)}. נשמר ל-{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
