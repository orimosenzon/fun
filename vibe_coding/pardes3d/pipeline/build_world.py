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


def building_height(el, area):
    t = el.get("tags", {})
    for k in ("height", "building:height"):
        try:
            return float(t[k].split()[0]), None
        except (KeyError, ValueError):
            pass
    try:
        lv = float(t["building:levels"])
        return lv * FLOOR + 0.8, lv
    except (KeyError, ValueError):
        pass
    kind = t.get("building", "yes")
    r = stable_rand(el["id"])
    if kind in ("greenhouse",):
        return 4.0, None
    if kind in ("industrial", "warehouse", "retail", "cowshed", "farm_auxiliary", "shed", "roof"):
        return 5 + 4 * r, None
    if kind in ("apartments",) or area > 600:
        lv = 3 + int(r * 4)
        return lv * FLOOR + 0.8, lv
    if area < 25:  # מחסן, חדר שומר
        return 2.6, 1
    # רוב פרדס חנה: בתים צמודי קרקע בקומה או שתיים
    lv = 1 if r < 0.45 else 2
    return lv * FLOOR + 0.6, lv


def enu_to_godot(v):
    """(מזרח, צפון, גובה) -> (x, y, z) של גודו. זה סיבוב, לא שיקוף, אז כיוון המשולשים נשמר."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


def build_buildings(els, terrain):
    import mapbox_earcut as earcut
    V, F, UV, C = [], [], [], []
    n = 0
    footprints = []
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
            footprints.append(poly)
            hgt, _ = building_height(el, poly.area)
            ex = np.array(poly.exterior.coords)
            hs = terrain.height(ex[:, 0], ex[:, 1])
            base = float(hs.min()) - 0.3  # הבסיס קבור קצת, כדי שלא ירחף בשיפוע
            top = float(hs.max()) + hgt
            col = WALL_COLORS[int(stable_rand(el["id"], "c") * len(WALL_COLORS))]
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
    m = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.vstack(UV))
    return m, lines


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
    img = Image.new("RGB", (TEX, TEX), (186, 170, 130))  # אדמת חמרה בהירה
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
    for _, k, p in areas:
        dr.polygon(px(p.exterior.coords), fill=LAND[k])
        for r in p.interiors:
            dr.polygon(px(r.coords), fill=(186, 170, 130))
    # כתמי ירק אקראיים: גינות פרטיות שלא ממופות
    rng = np.random.default_rng(7)
    for _ in range(25000):
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
        dr.polygon(px(p.buffer(1.2).exterior.coords), fill=(120, 115, 105))
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


def scatter_trees(els, footprints, road_lines):
    """עצים "ממולאים" בשטחים ירוקים ובשכונות, כי הסקר ו-OSM מכסים רק חלק קטן מהם.
    מסומנים ב-name ריק ו-health="synthetic" כדי שיהיה ברור שאינם עצים אמיתיים."""
    from shapely.strtree import STRtree
    rng = np.random.default_rng(11)
    # לא שותלים על בניינים ועל כבישים
    blockers = footprints + [ln.buffer(w / 2 + 1.0, cap_style=2) for ln, w, _ in road_lines]
    tree_index = STRtree(blockers)
    density = {"orchard": 1 / 40, "forest": 1 / 60, "wood": 1 / 60, "park": 1 / 150,
               "garden": 1 / 150, "residential": 1 / 260, "cemetery": 1 / 120}
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
            inside = contains(p, pts)
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
    bm, bcol, footprints = build_buildings(els, terrain)
    export_glb(bm, OUT / "buildings.glb", bcol)
    print(f"  {len(footprints)} בניינים, {len(bm.faces)} משולשים")
    print("כבישים...")
    rm, lines = build_roads(els, terrain)
    export_glb(rm, OUT / "roads.glb")
    print(f"  {len(lines)} קטעים")
    print("טקסטורת קרקע...")
    build_ground_texture(els, terrain, lines, footprints).save(OUT / "ground.png", optimize=True)
    trees = build_trees(els)
    synth = scatter_trees(els, footprints, lines)
    # קומפקטי: [x, y, גובה, קוטר צמרת, סוג]; סוג 0 = עץ מהסקר, 1 = ממולא, 2 = הדר ממולא
    rows = [[t["x"], t["y"], t["h"], t["c"], 0] for t in trees]
    rows += [[t["x"], t["y"], t["h"], t["c"], 2 if t.get("kind") == "citrus" else 1] for t in synth]
    (OUT / "trees.json").write_text(json.dumps(rows, separators=(",", ":")))
    (OUT / "survey_trees.json").write_text(json.dumps(trees, ensure_ascii=False))
    print(f"עצים: {len(trees)} אמיתיים + {len(synth)} ממולאים")
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
        ],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"מקומות: {len(places)}. נשמר ל-{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
