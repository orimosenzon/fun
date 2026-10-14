"""מייצא את המודל לדרך קיצור, כשכבת תלת-ממד מעל MapLibre.

גודו מקבל רשתות (glb). מפה בדפדפן צריכה משהו אחר: טביעות רגל עם גובה, כי
MapLibre מוחצן אותן בעצמו ומניח אותן על הקרקע שלו (fill-extrusion). לכן לא
מייצאים כאן רשתות אלא את מה שהרשתות נבנו ממנו: אותם בניינים, אותם גבהים
ואותם צבעים של build_world, אותם עצים ואותן חומות.

פלט ל-dereh_kitzur/web/data/model3d/, בפורמט קומפקטי (הקואורדינטות כמספרים
שלמים של מיליונית מעלה, יחסית לנקודת מוצא, ודלתא בין נקודה לנקודה):
  buildings.json  [גובה, צבע קיר, צבע גג, x0, y0, dx, dy, ...]
  trees.json      [x, y, גובה, קוטר צמרת, סוג] (0 = סקר העצים, 3 = מסכת הצמרות)
  walls.json      [סוג, גובה, עובי, x0, y0, dx, dy, ...] (קווים, הדפדפן מעבה אותם)
"""
import json

import numpy as np

import build_world as bw
import extras
from fetch_osm import ROOT
from geo import to_latlon

OUT = ROOT.parent / "dereh_kitzur" / "web" / "data" / "model3d"
Q = 1e6  # מיליונית מעלה: כ-10 ס"מ, מספיק לבניין

# צבעי גגות בארץ: רוב הגגות שטוחים ומכוסים בזפת לבנה או אפורה, חלק רעפים
ROOF_COLORS = [(214, 212, 205), (190, 188, 182), (226, 224, 218), (168, 98, 72), (150, 140, 128)]
ROOF_WEIGHTS = [0.34, 0.26, 0.16, 0.14, 0.10]


def pack(coords, origin):
    """רשימת (lat, lon) -> x0, y0, dx, dy... במספרים שלמים."""
    out, px, py = [], None, None
    for lat, lon in coords:
        x, y = round((lon - origin[0]) * Q), round((lat - origin[1]) * Q)
        if px is None:
            out += [x, y]
        else:
            if x == px and y == py:
                continue
            out += [x - px, y - py]
        px, py = x, y
    return out


def latlon(xy):
    xy = np.asarray(xy)
    lat, lon = to_latlon(xy[:, 0], xy[:, 1])
    return list(zip(lat.tolist(), lon.tolist()))


def roof_color(key, kind):
    r = bw.stable_rand(key, "roof")
    if kind in ("greenhouse",):
        return 5
    acc = 0.0
    for i, w in enumerate(ROOF_WEIGHTS):
        acc += w
        if r < acc:
            return i
    return 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    els = bw.load_osm()
    terrain = bw.Terrain()
    ghsl = extras.Ghsl()
    origin = (bw.LON0, bw.LAT0)

    items, n_osm = bw.collect_buildings(els)
    rows, footprints = [], []
    for key, poly, t in items:
        footprints.append(poly)
        c = poly.centroid
        h = bw.building_height(key, t, poly.area, ghsl.at(c.x, c.y))
        ex = np.array(poly.exterior.coords)
        hs = terrain.height(ex[:, 0], ex[:, 1])
        # MapLibre מניח את כל הבניין בגובה הקרקע שבמרכזו. בשיפוע, הצד הגבוה
        # היה נקבר; מוסיפים את חצי ההפרש כדי שהגג יישאר מעל הקרקע בכל צד
        h += float(hs.max() - hs.min()) / 2
        wall = int(bw.stable_rand(key, "c") * len(bw.WALL_COLORS))
        kind = t.get("building", "yes")
        rows.append([round(h, 1), wall, roof_color(key, kind)] + pack(latlon(ex[:-1]), origin))
    print(f"בניינים: {len(rows)} ({n_osm} מ-OSM)")

    _, lines = bw.build_roads(els, terrain)  # בשביל החומות: הן לא חוצות כבישים

    trees = bw.build_trees(els)
    canopy = extras.Canopy()
    ctrees = extras.canopy_trees(canopy, footprints, trees, terrain)
    # רק עצים במקום אמיתי: הסקר ומסכת הצמרות. 120 אלף העצים שבגודו ממולאים
    # בפרדסים היו פי ארבעה גאומטריה, ובמפה צילום הלוויין כבר מראה את הפרדסים
    # מסכת הצמרות רואה צמרת שנוטה מעל הכביש ושותלת שם עץ. מגודו זה לא בולט,
    # מהמדרכה זה עץ באמצע הכביש. עצי הסקר נמדדו בשטח ונשארים
    from shapely import STRtree, points
    roads = [ln.buffer(w / 2 + 0.5, cap_style=2) for ln, w, k in lines if k in (0, 3)]
    on_road = set(STRtree(roads).query(points([t["x"] for t in ctrees], [t["y"] for t in ctrees]),
                                       predicate="intersects")[0].tolist())
    ctrees = [t for i, t in enumerate(ctrees) if i not in on_road]
    print(f"  {len(on_road)} עצים של המסכה הוסרו מהכביש")
    trows = []
    for group, kind in ((trees, 0), (ctrees, 3)):
        for t in group:
            k = kind
            (lat, lon), = latlon([(t["x"], t["y"])])
            trows.append([round((lon - origin[0]) * Q), round((lat - origin[1]) * Q),
                          round(float(t["h"]), 1), round(float(t["c"]), 1), k])
    print(f"עצים: {len(trees)} מהסקר, {len(ctrees)} ממסכת הצמרות")

    wrows = []
    for line, kind, h, w in extras.parcel_walls(footprints, lines, terrain):
        for part in getattr(line, "geoms", [line]):
            if part.length < 1.0:
                continue
            pts = np.array(part.simplify(0.2).coords)
            wrows.append([kind, round(h, 2), round(w, 2)] + pack(latlon(pts), origin))
    print(f"חומות: {len(wrows)} קטעים")

    meta = {
        "origin": origin, "q": Q,
        "wall_colors": bw.WALL_COLORS, "roof_colors": ROOF_COLORS + [(222, 230, 226)],
        "attribution": "מודל: פרדס חנה תלת-ממד · © OpenStreetMap · Microsoft Building Footprints · "
                       "GHS-BUILT-H (JRC) · מפת הצמרות, המרכז למיפוי ישראל",
    }
    for name, data in (("buildings", rows), ("trees", trows), ("walls", wrows)):
        (OUT / f"{name}.json").write_text(json.dumps({**meta, "rows": data} if name == "buildings"
                                                     else {"origin": origin, "q": Q, "rows": data},
                                                     separators=(",", ":")))
        print(f"  {name}.json: {(OUT / f'{name}.json').stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
