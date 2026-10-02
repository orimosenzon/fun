#!/usr/bin/env python3
"""Build web/data/landuse.json: the designation of every piece of ground in the moshava.

Ori, 1/10/2026: "ייעודי קרקע" for the whole moshava, each place in its colour -
public open space, public buildings, housing, roads and the rest - and in the
colours a planner reads without a key. build_public.py answered the narrower
question (which ground is public) off the national register alone, and had to
leave most of the moshava blank. This file fills it.

Why the national register is not enough
---------------------------------------
Xplan's layer 4 (see build_public.py) holds the cells of every plan submitted
online, roughly 2011 on. Measured on 1/10/2026 against the municipal boundary:

    the comprehensive plan 353-0138586    22,832 dunam in 277 cells
    every other approved plan              3,225 dunam

So the register does cover the whole moshava, but almost all of it through the
comprehensive plan - which cannot issue a permit and paints whole neighbourhoods
"מגורים" without a street or a school in them. The parcel-level designations of
the old core are in the old plans (ש/1, ש/17, ש/139 and four hundred more),
which exist in the register only as scanned sheets.

Where it comes from, then
-------------------------
1. **The local planning committee's own compilation** ("קומפילציה", layer `Q`,
   which the committee's GIS labels "מגרשים פעילים"): 11,437 active lots, each
   with its plan, lot number and designation, digitised off the old plans as
   well as the new. It sits in the committee's public GIS:

       mg1.gis-net.co.il/PardesHanaKarkurGis      (Taldor MapExpert, project 550)

   Anonymous login is part of the public site. Features come off
   `api/map/GetObjectsByGeometry`, which answers for whatever is on in the
   session - and `Q` is on by default. There is no paging and no cap that a
   square kilometre comes near (2,150 lots answered in five seconds), so the
   town is 49 one-kilometre squares. **The site's firewall blocks an address
   for about twelve minutes after a few dozen fast requests** (seen 30/8/2026),
   so the squares go one at a time, five seconds apart, and are cached in
   .cache/landuse/: a rebuild that does not pass --refresh asks nothing.

   The lots come in the Israeli grid (EPSG:2039) and tile the ground: checked,
   1,065 m² of overlap among them in all, three pairs.

2. **Xplan, for what the compilation has not caught up with.** It is current to
   June 2026 but not complete: on 1/10/2026, of the approved plans in the
   register 57 were not in it, among them 308-0707372, 170 dunam of employment
   along תדהר, approved October 2025. Each such plan's cells are laid over the
   compilation where they are newer than what is there - see `overlay()`.

What is left out
----------------
* Plans that cannot issue a permit (`plan_charactor_name` says "לא ניתן"): the
  comprehensive plan, and the national plans' section 77-78 notices, which are
  a freeze rather than a designation.
* The register's non-designations - "יעוד עפ"י תכנית מאושרת אחרת", building
  restrictions, "the plan does not apply here" - which say "look elsewhere".
* The compilation's "שייך לרשות אחרת", ground the committee lists as some other
  authority's.
* Anything outside the municipal boundary (govmap's `muni_il`, the line every
  other layer here is cut to). The committee's planning area reaches about
  2,200 dunam past it.
* Plans still in process. This is the ground as it is designated today; what is
  proposed is the planning layer's.

The colours
-----------
Every designation is mapped to its מבא"ת code, and the code's colour is read off
the register's own renderer (layer 4's `drawingInfo`) on every build, so that
the palette is the planning administration's and not a guess at it. The old
plans predate מבא"ת and name things their own way - "אזור מגורים א",
"בניני ציבור", "אזור משקי עזר" - and OLD maps each to the code it corresponds
to. A name in neither is printed at the end of a run and the build fails: a
new designation is a decision, not something to paint grey.

Three departures, all on purpose:
* **Hatching.** The renderer hatches a mixed use in a single colour on
  nothing. Here, as on a paper תשריט, a mixed use is the colour of its first
  use striped with the colour of its second (MIXED), and a single-use hatch is
  its colour striped with a darker shade of itself.
* **עירוני מעורב** is a grey hatch in the renderer, which reads as commerce;
  it is housing with shops, so here it is housing yellow striped grey.
* **שטח פרטי פתוח** has exactly the colour of a שצ"פ in the renderer. On this
  map the difference between the two is the whole point, so it gets dark green
  stripes. The layer's note says this is ours.

    python3 build_landuse.py              # from the cache where there is one
    python3 build_landuse.py --refresh    # ask the committee's GIS again
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

from pyproj import Transformer
from shapely.geometry import LinearRing, Polygon, mapping
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

import build_public as bp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web", "data", "landuse.json")
CACHE = os.path.join(HERE, ".cache", "landuse")

GIS = "https://mg1.gis-net.co.il/PardesHanaKarkurApi/api"
GIS_SITE = "https://mg1.gis-net.co.il/PardesHanaKarkurGis/"
PROJECT = 550
UA = bp.UA

# The committee's map extent, in the Israeli grid, to the next kilometre.
EXTENT = (194700, 705600, 201700, 712400)
SQUARE = 1000
PAUSE = 5                               # seconds between requests; see above

COMPREHENSIVE = "353-0138586"
OTHER_AUTHORITY = "שייך לרשות אחרת"

# ITM <-> WGS84.
TO_ITM = Transformer.from_crs(4326, 2039, always_xy=True).transform
TO_WGS = Transformer.from_crs(2039, 4326, always_xy=True).transform

# Below anything the map can show at its closest zoom (a pixel is ~30cm there),
# the same tolerance build_parcels.py uses.
SIMPLIFY_M = 0.1


# ---------------------------------------------------------------- designations

# The old plans' names, and a few of the compilation's own spellings of new
# ones, -> the מבא"ת code they correspond to. Exact names after `norm()`.
OLD = {
    "אזור מגורים א": 20, "מגורים א 1": 20, "מגורים א 2": 20, "מגורים א2": 20,
    "מגורים א3": 20,
    "אזור מגורים ב": 60, "מגורים ב1": 60,
    "אזור מגורים ג": 100, "מגורים ג1": 100,
    # Three or four storeys is what מגורים ב' allows under מבא"ת.
    "מגורים 4-3 קומות": 60, "אזור מגורים 3-4 קומות": 60,
    # Residential with no density in its name: the generic מגורים, which is
    # the paler yellow, rather than a guess at a letter.
    "אזור מגורים": 10, "מגורים מיוחד": 10, "מגורים מיוחד א": 10,
    "מגורים מיוחד ב": 10,
    "אזור משקי עזר": 170,
    "שטח חקלאי": 660,
    "דרך קיימת/מאושרת": 820,
    "שביל להולכי רגל": 860, "שביל שרות": 860, "שביל הולכי רגל או כיכר": 860,
    "חניה ציבורית": 870, "חניה": 870, "חניה פרטית": 870, "דרך גישה לחניה": 870,
    "דרכים וחניות": 870,
    "בניני ציבור": 400, "בניני ציבור ומוסדות": 400, "מוסד חינוכי": 410,
    "מוסדות לימוד": 410, "מוסדות חינוך, בריאות וסעד": 400,
    "שטח לבנין ציבורי": 400, "שטח למוסדות ציבור": 400,
    "שטח לשירותי רווחה ומוסדות": 400, "מרכז אזרחי": 400,
    "שטח לשרותים עירוניים": 400, "שטחי ציבורי מיוחד": 400,
    "שטח ציבורי משולב": 400,
    "שצ\"פ נחל": 670, "שטח ירק": 670, "פס נטיעות": 670,
    "שטח ספורט": 690, "אזור ספורט ותרבות": 690,
    "שפ\"פ ושביל": 680,
    "אזור מסחרי": 210, "מרכז מסחרי": 210, "אזור מסחרי מיוחד": 210,
    "אזור תעשיה": 230, "תעשיה ומלאכה": 260, "מלאכה/תעשיה זעירה": 260,
    "אזור מלאכה": 260,
    "אזור תעסוקה": 200,
    "מרכז תעסוקה ומסחר": 1502,
    "אזור מסחר ומגורים": 1000,
    "שטח למתקנים הנדסיים": 280, "שטח למתקני-הנדסה": 280, "מתקן הנדסי": 280,
    "תחנת טרנספורמציה": 280,
    "תחנת דלק": 910,
    "בית עלמין": 980,
    "מסילת ברזל מוצעת": 890, "שטח מסילת ברזל, רכבת": 880,
    "מסילת ברזל ושרותי רכבת - ציבורי": 880,
    "מסילת ברזל ושרותי רכבת - פרטי": 880,
    "אזור לתכנון בעתיד": 950, "אזור לתכנון מיוחד": 950,
    "שטח לבניני משק": 300, "שטח לשירותי חקלאות אזוריים": 300,
    "שרותי מיכון למטע": 300,
    "נחל/תעלה/מאגר מים": 740,
    "מגורים מיוחד ומבנים ומוסדות ציבור": 1300,
}

# Names that are a mix the מבא"ת table has no single code for. Drawn as the
# first code's colour striped with the second's, and listed as the first code.
MIXED_OLD = {
    "מגורים ומלאכה": (10, 260),
    "מגורים א-3 + חזית מסחרית": (20, 210),
    "מגורים ב + חזית מסחרית": (60, 210),
    "בניני ציבור עם שצ\"פ": (400, 670),
    "שבילים ופסי ירק": (860, 670),
}

# A private institution: brown like a public one, since that is what it is
# built as, but striped and filed under "other" - it is not public ground.
PRIVATE_INSTITUTION = {"מוסד פרטי", "שטח למוסד"}

# The codes whose renderer symbol is a mix: first use, second use.
MIXED = {
    # עירוני מעורב is housing over shops, and the renderer's grey hatch drew the
    # apartment blocks of the centre as if they were a commercial zone (Ori,
    # 2/10/2026, about the centre of the moshava). Housing yellow striped grey, like
    # every other housing-and-something.
    290: (10, 290),
    1000: (10, 210), 1001: (150, 210), 1050: (10, 210), 1100: (10, 200),
    1200: (10, 220), 1250: (10, 400), 1300: (150, 400), 1350: (10, 150),
    1410: (10, 210), 1420: (10, 210), 1470: (10, 200), 1480: (10, 200),
    1502: (210, 200), 1500: (210, 230), 1520: (210, 220), 1550: (210, 400),
    1560: (210, 600), 1578: (210, 400), 1576: (210, 200),
    1600: (200, 400), 1602: (400, 280), 1610: (400, 220), 1630: (230, 220),
    1640: (230, 240), 1650: (400, 600), 1670: (650, 400), 1660: (650, 280),
    1680: (660, 280),
}

# What the register says when it is not saying what the ground is for.
NOT_DESIGNATIONS = {994, 995, 996, 997, 998, 999, 3000}

PRIVATE_OPEN = 680
PRIVATE_STRIPE = "#1b5e20"

# The legend: what each code is filed under. Order is the legend's order.
GROUPS = [
    ("res_a", "מגורים א"),
    ("res_b", "מגורים ב"),
    ("res_c", "מגורים ג ו-ד"),
    ("res", "מגורים, בלי צפיפות בשם"),
    ("rural", "משק עזר"),
    ("mixed", "מגורים ושימוש נוסף"),
    ("public", "מבנים ומוסדות ציבור"),
    ("open", "שטח ציבורי פתוח, ספורט ונופש"),
    ("private", "שטח פרטי פתוח"),
    ("trade", "מסחר ותיירות"),
    ("work", "תעסוקה ומשרדים"),
    ("industry", "תעשייה ומלאכה"),
    ("road", "דרך קיימת או מאושרת"),
    ("newroad", "דרך מוצעת"),
    ("path", "שביל וחניון"),
    ("farm", "קרקע חקלאית ומבני משק"),
    ("nature", "שמורה, יער ונחל"),
    ("infra", "מתקנים הנדסיים, מסילה ותשתיות"),
    ("cemetery", "בית עלמין"),
    ("other", "אחר"),
]


def group_of(code):
    if code in MIXED and code != 290:
        first = MIXED[code][0]
        return "mixed" if first in (10, 20, 60, 100, 140, 150) else group_of(first)
    if code == 20:
        return "res_a"
    if code == 60:
        return "res_b"
    if code in (100, 140, 145):
        return "res_c"
    if code in (10, 150, 160, 5690):
        return "res"
    if code == 170:
        return "rural"
    if code == 290:
        return "mixed"
    if 400 <= code <= 460 or code in (461, 463, 805):
        return "public"
    if code in (650, 670, 690, 700, 750, 780):
        return "open"
    if code == PRIVATE_OPEN:
        return "private"
    if code in (210, 600, 610, 620, 630, 910):
        return "trade"
    if code in (200, 220):
        return "work"
    if code in (230, 240, 250, 260, 972):
        return "industry"
    if code in (820, 806):
        return "road"
    # Red on the תשריט is a road the plan proposes, and so are the road-and-
    # landscape strips drawn in the same red; the legend goes by the colour.
    if code in (830, 850, 940, 800, 810, 902, 903):
        return "newroad" if code in (830, 850, 940) else "road"
    if code == 840:
        return "road"
    if code in (860, 861, 870):
        return "path"
    if code in (660, 661, 300, 462):
        return "farm"
    if 710 <= code <= 741 or 5663 <= code <= 5670:
        return "nature"
    if code in (280, 880, 890, 900, 955, 960, 970, 807, 1690, 855, 930):
        return "infra"
    if code == 980:
        return "cemetery"
    return "other"


def norm(name):
    """A designation as the two sources spell it -> one spelling.

    The compilation appends "- מבא"ת" to a name taken from the table, spells
    ציבור without its yod and is loose with spaces and geresh; the register's
    own labels are inconsistent about the same things.
    """
    s = re.sub(r'\s*-\s*מבא"ת\s*$', "", (name or "").strip())
    s = s.replace("'", "").replace("׳", "")
    s = s.replace("צבור", "ציבור").replace("תירות", "תיירות").replace("איזור", "אזור")
    s = re.sub(r"\s*,\s*", ", ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- colours

def hexc(rgba):
    return "#%02x%02x%02x" % tuple(rgba[:3])


def darker(colour, by=0.38):
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(c * (1 - by)) for c in (r, g, b))


def renderer():
    """code -> (מבא"ת name, fill colour, hatched?), off the register itself."""
    url = bp.XPLAN + "/4?f=pjson"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120, context=bp.tls()) as handle:
        doc = json.loads(handle.read().decode("utf-8-sig"))
    out = {}
    for info in doc["drawingInfo"]["renderer"]["uniqueValueInfos"]:
        sym = info["symbol"]
        out[int(info["value"])] = (info["label"].strip(), hexc(sym["color"]),
                                   sym.get("style") != "esriSFSSolid")
    return out


def style_for(name, code, table):
    """(code, fill, stripe or None, group) for one designation."""
    if name in PRIVATE_INSTITUTION:
        fill = table[400][1]
        return 400, fill, darker(fill), "other"
    pair = MIXED_OLD.get(name) or MIXED.get(code)
    if pair:
        first, second = pair
        fill = table[first][1]
        stripe = darker(fill) if first == second else table[second][1]
        return code if code else first, fill, stripe, (
            "mixed" if name in MIXED_OLD and first in (10, 20, 60) else group_of(code or first))
    _, fill, hatched = table[code]
    if code == PRIVATE_OPEN:
        return code, fill, PRIVATE_STRIPE, "private"
    return code, fill, darker(fill) if hatched else None, group_of(code)


# ---------------------------------------------------------------- the committee's GIS

def gis_post(op, body, token=None):
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(f"{GIS}/{op}", data=json.dumps(body).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=120) as handle:
        return json.loads(handle.read().decode("utf-8-sig"))


def gis_session():
    """Log in as the anonymous visitor the site itself logs in as, open a map."""
    token = gis_post("auth/UserLogin",
                     {"userName": "Anonymous", "userPassword": "", "projId": PROJECT})
    first = gis_post("map/FirstLoadingMap",
                     {"mapSession": "", "mapName": "", "mapStateId": 0, "projId": PROJECT},
                     token)
    return token, first["sessionId"], first["mapName"]


def pull(refresh):
    """Every lot of the compilation, square by square, through the cache."""
    os.makedirs(CACHE, exist_ok=True)
    session = None
    x0, y0, x1, y1 = EXTENT
    asked = 0
    lots = {}
    for x in range(x0, x1, SQUARE):
        for y in range(y0, y1, SQUARE):
            path = os.path.join(CACHE, f"{x}_{y}.json")
            if refresh or not os.path.exists(path):
                if session is None:
                    session = gis_session()
                token, sid, name = session
                wkt = (f"POLYGON(({x} {y},{x + SQUARE} {y},{x + SQUARE} {y + SQUARE},"
                       f"{x} {y + SQUARE},{x} {y}))")
                for attempt in range(3):
                    try:
                        doc = gis_post("map/GetObjectsByGeometry", {
                            "mapSession": sid, "mapName": name, "geometry": wkt,
                            "geometryType": "Polygon", "zoomWidth": 50,
                            "projId": PROJECT}, token)
                        break
                    except Exception as err:            # noqa: BLE001
                        if attempt == 2:
                            raise
                        print(f"    ...{err}, מחכה דקה", file=sys.stderr)
                        time.sleep(60)
                rows = [row for layer in doc.get("layers", []) if layer["name"] == "Q"
                        for row in layer["rowItems"]]
                bp.save_json(path, rows)
                asked += 1
                print(f"  ריבוע {x},{y}: {len(rows)} מגרשים")
                time.sleep(PAUSE)
            with open(path, encoding="utf-8") as handle:
                for row in json.load(handle):
                    at = {f["fieldName"]: f["fieldValue"] for f in row["fieldItems"]}
                    lots[at["Lot_id"]] = (at, row["geom"],
                                          [u["url"] for u in row.get("urlItems") or []])
    print(f"  {len(lots)} מגרשים בקומפילציה ({asked} בקשות לשרת)")
    return lots


def lot_geometry(geom):
    """The GIS's polygon -> shapely, in the Israeli grid."""
    def one(g):
        holes = []
        for ring in g.get("rings") or []:
            holes += ring if ring and isinstance(ring[0][0], list) else [ring]
        poly = Polygon(g["points"][0], [h for h in holes if len(h) >= 4])
        return poly if poly.is_valid else poly.buffer(0)
    if geom.get("geometryType") == 3:
        return unary_union([one(g) for g in geom["polygons"]])
    return one(geom)


# ---------------------------------------------------------------- the register

def esri_geometry(rings):
    """An esri polygon in WGS84 -> shapely in the Israeli grid."""
    outers, holes = [], []
    for ring in rings:
        if len(ring) < 4:
            continue
        (holes if LinearRing(ring).is_ccw else outers).append(Polygon(ring))
    g = unary_union(outers)
    if holes:
        g = g.difference(unary_union(holes))
    return transform(TO_ITM, g.buffer(0))


def register_cells(area):
    """Every approved cell of the register in the moshava, with its code."""
    bp.FIELDS = ["mavat_code", "mavat_name", "num", "pl_number", "pl_name", "mp_id"]
    bp.OFFSET = "0.000002"              # ~20cm; the compilation is finer than 1m
    return bp.landuse(area)


def permit_plans(area):
    """pl_number -> plan_charactor_name, for telling the plans that cannot issue
    a permit from the ones that can."""
    out = {}
    offset = 0
    while True:
        doc = bp.post(bp.BLUELINES, {
            "geometry": json.dumps(area), "geometryType": "esriGeometryPolygon",
            "inSR": "4326", "spatialRel": "esriSpatialRelIntersects", "where": "1=1",
            "outFields": "pl_number,plan_charactor_name", "returnGeometry": "false",
            "orderByFields": "pl_number", "resultOffset": str(offset),
            "resultRecordCount": str(bp.PAGE), "f": "json"})
        got = doc.get("features", [])
        for f in got:
            at = f["attributes"]
            out[(at.get("pl_number") or "").strip()] = at.get("plan_charactor_name") or ""
        if len(got) < bp.PAGE:
            return out
        offset += bp.PAGE


# ---------------------------------------------------------------- putting it together

def overlay(lots, cells, meta, permits, table):
    """The register's plans the compilation does not have, laid over it.

    Later wins, by publication in רשומות (`meta[...]['date']`). A register cell
    covers the compilation except where the lot there comes from a plan with a
    later date; a lot from an old plan with no online record (every ש/ plan)
    is older than anything online, by construction. Among the register's own
    plans, the same rule, in date order. A plan with no date is left out -
    nothing to decide its place by - and counted.
    """
    in_q = {at["Taba_Name"] for at, _, _ in lots.values()}
    by_plan = {}
    skipped = {"no_permit": set(), "no_date": set(), "not_designation": 0}
    for feat in cells:
        at = feat["attributes"]
        plan = (at.get("pl_number") or "").strip()
        code = int(at.get("mavat_code") or 0)
        if plan in in_q:
            continue
        if "לא ניתן" in permits.get(plan, ""):
            skipped["no_permit"].add(plan)
            continue
        if code in NOT_DESIGNATIONS:
            skipped["not_designation"] += 1
            continue
        if code not in table:
            raise SystemExit(f"קוד מבא\"ת {code} ({at.get('mavat_name')}) לא מופיע בטבלת "
                             "הצבעים של מנהל התכנון")
        date = (meta.get(plan) or {}).get("date")
        if not date:
            skipped["no_date"].add(plan)
            continue
        geom = esri_geometry(feat["geometry"]["rings"])
        if not geom.is_empty:
            by_plan.setdefault(plan, (date, []))[1].append((at, code, geom))

    dated_lots = [(lid, (meta.get(at["Taba_Name"]) or {}).get("date") or 0, g)
                  for lid, (at, g, _) in lots.items()]
    pieces = []                            # (at, code, plan, geometry), newest last
    for plan, (date, plan_cells) in sorted(by_plan.items(), key=lambda kv: kv[1][0]):
        newer = [g for _, d, g in dated_lots if d > date]
        newer_union = unary_union(newer) if newer else None
        for at, code, geom in plan_cells:
            if newer_union is not None:
                geom = geom.difference(newer_union)
            # A later plan of the register's own wins over this one.
            pieces = [(a, c, p, g.difference(geom)) for a, c, p, g in pieces]
            if not geom.is_empty:
                pieces.append((at, code, plan, geom))
    pieces = [p for p in pieces if p[3].area > 1]
    return pieces, skipped


def build(refresh):
    print("גבול המושבה, ממאגר govmap")
    area_name, area = bp.boundary()
    boundary = esri_geometry(area["rings"])

    print("טבלת הצבעים של מבא\"ת, מהמאגר של מנהל התכנון")
    table = renderer()
    by_name = {norm(label): code for code, (label, _, _) in table.items()}

    print("הקומפילציה של הוועדה המקומית")
    raw = pull(refresh)
    lots = {}
    for lid, (at, geom, urls) in raw.items():
        if at["Land_Use_Name"].strip() == OTHER_AUTHORITY:
            continue
        g = lot_geometry(geom)
        if not g.is_empty:
            lots[lid] = (at, g, urls)

    print("ייעודים מאושרים במאגר המקוון של מנהל התכנון")
    cells = register_cells(area)
    meta = bp.plans(area)
    permits = permit_plans(area)
    pieces, skipped = overlay(lots, cells, meta, permits, table)
    over = unary_union([p[3] for p in pieces]) if pieces else None

    features = []
    covered = []
    unknown = {}
    counts = {}

    def emit(geom, props):
        geom = geom.intersection(boundary)
        if over is not None and props["src"] == "q":
            geom = geom.difference(over)
        geom = geom.simplify(SIMPLIFY_M, preserve_topology=True)
        polys = [g for g in getattr(geom, "geoms", [geom])
                 if g.geom_type == "Polygon" and g.area > 1]
        if not polys:
            return
        geom = unary_union(polys)
        wgs = transform(TO_WGS, geom)
        props["a"] = round(geom.area)
        if props.get("m") == props["u"]:
            del props["m"]                 # the app falls back to the name
        covered.append(geom)
        counts[props["k"]] = counts.get(props["k"], 0) + geom.area
        features.append({
            "type": "Feature",
            "id": len(features) + 1,
            "properties": props,
            "geometry": json.loads(json.dumps(mapping(wgs)),
                                   parse_float=lambda v: round(float(v), 6)),
        })

    for lid, (at, geom, urls) in lots.items():
        name = at["Land_Use_Name"].strip()
        key = norm(name)
        code = by_name.get(key) or OLD.get(key) or 0
        if not code and key not in MIXED_OLD and key not in PRIVATE_INSTITUTION:
            unknown[name] = unknown.get(name, 0) + 1
            continue
        code, fill, stripe, group = style_for(key, code, table)
        # The מבא"ת name of an old mix, or of a private institution, is the
        # name of only its first use ("מגורים" for "מגורים ומלאכה"), and the
        # legend lists designations by it: those keep their own name.
        own = key in MIXED_OLD or key in PRIVATE_INSTITUTION
        props = {"u": name.replace(' - מבא"ת', "").replace('- מבא"ת', "")
                 .replace('-מבא"ת', "").strip(),
                 "m": "" if own else (table[code][0] if code in table else ""),
                 "t": at["Taba_Name"].strip(), "l": (at.get("Lot_No") or "").strip(),
                 "f": fill, "k": group, "src": "q"}
        if stripe:
            props["s"] = stripe
            props["p"] = f"lu-{fill[1:]}-{stripe[1:]}"
        # No link stored: every lot's page on the committee's site is the
        # same address with the plan and the lot in it, and the app builds it
        # (LandUse in app.js). 10,735 copies of it were a fifth of the file.
        emit(geom, props)

    for at, code, plan, geom in pieces:
        code, fill, stripe, group = style_for("", code, table)
        props = {"u": table[code][0], "m": table[code][0], "t": plan,
                 "l": (at.get("num") or "").strip(), "f": fill, "k": group, "src": "x"}
        if stripe:
            props["s"] = stripe
            props["p"] = f"lu-{fill[1:]}-{stripe[1:]}"
        url = (meta.get(plan) or {}).get("url")
        if url:
            props["w"] = url
        emit(geom, props)

    if unknown:
        print("\nייעודים שאין להם מקבילה בטבלה. צריך להחליט עליהם ב-OLD:", file=sys.stderr)
        for name, n in sorted(unknown.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {name}", file=sys.stderr)
        raise SystemExit(1)

    covered = unary_union([g.buffer(0) for g in covered])
    doc = {
        "type": "FeatureCollection",
        "version": 1,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "area": area_name,
        "sources": [GIS_SITE, bp.XPLAN_SITE],
        "features": features,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:     # compact: 3MB, not 9
        json.dump(doc, handle, ensure_ascii=False, separators=(",", ":"))

    print(f"\n{len(features)} פוליגונים ל-{OUT}, "
          f"{os.path.getsize(OUT) / 1e6:.1f}MB")
    print(f"  מהקומפילציה: {sum(1 for f in features if f['properties']['src'] == 'q')}, "
          f"מהמאגר המקוון: {sum(1 for f in features if f['properties']['src'] == 'x')} "
          f"(מ-{len({p[2] for p in pieces})} תכניות)")
    print(f"  מכוסה: {covered.area / 1000:,.0f} מתוך {boundary.area / 1000:,.0f} דונם")
    print(f"  תכניות שאין מכוחן היתר, בחוץ: {', '.join(sorted(skipped['no_permit'])) or 'אין'}")
    if skipped["no_date"]:
        print(f"  תכניות בלי תאריך פרסום, בחוץ: {', '.join(sorted(skipped['no_date']))}")
    print("  דונם לפי קבוצה:")
    for key, name in GROUPS:
        if key in counts:
            print(f"    {counts[key] / 1000:8,.0f}  {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true",
                        help="למשוך מחדש את הקומפילציה מה-GIS של הוועדה")
    build(parser.parse_args().refresh)


if __name__ == "__main__":
    main()
