#!/usr/bin/env python3
"""Build web/data/public.json: the land in the moshava that is designated public.

The question this answers is the one you ask standing in front of a gap in a
hedge: may I walk through here, or is this somebody's garden. A shortcut across
a שצ"פ is a shortcut across land the plan set aside for exactly that; the same
line across "מגורים א'" is a path through a back yard that survives only as long
as its owner tolerates it. Until now the map could not tell the two apart, and
the initiative was mapping both as if they were the same kind of thing.

Designation, not ownership
--------------------------
This layer says what the approved plan **designates** the ground for, which is
not the same as who owns it and is not a permission to enter. The two usually
agree - a שצ"פ is normally municipal, having been handed over when the plan was
realised - and where they disagree the designation is still the stronger fact
for this app: it is what the ground is legally for, it binds whoever owns it,
and it is what any argument about a blocked path is eventually decided on.

Ownership itself is not available. The cadastre that govmap publishes openly
carries parcels and blocks and no owner column, רשות מקרקעי ישראל publishes no
open ownership service, and data.gov.il has no such dataset. A layer claiming to
show ownership would have to be assembled out of guesswork, and guesswork about
whether a piece of land is yours is the one thing this layer must not do.

Where it comes from
-------------------
The planning administration's land-use layer, the same public ArcGIS service
`build_plans.py` reads the blue lines off, one layer along:

    ags.iplan.gov.il/arcgisiplan/rest/services/PlanningPublic/Xplan/MapServer/4

Only `station_desc='אישור'` - approved and in force. The other twelve statuses in
there are plans on their way through the system, and a designation somebody has
proposed is not what the ground is; those belong to the planning layer, which
already shows them as blue lines.

The moshava's share of it is cut server-side against the municipal boundary as
published by govmap (`opendata:muni_il`, מועצה מקומית פרדס חנה - כרכור), rather
than against a bounding box: the box holds three times as many polygons, most of
them Binyamina's and Karkur's neighbours'.

What comes out, and what does not
---------------------------------
Only the public designations, in five groups, plus a sixth that is deliberately
not public: שטח פרטי פתוח, private open space. It is a handful of polygons and a
couple of dunam, and dunam for dunam it is the most useful thing in the file -
open ground, no fence, mown grass, and privately owned, which is exactly the
place somebody walks through assuming it is a park. Anything not in `GROUPS` is
left out:
מגורים, קרקע חקלאית, תעסוקה, תעשיה, מסחר and the rest are what the map already
shows as houses and fields.

An unmapped designation is never quietly dropped. Whatever the register returns
that this file has not heard of is printed at the end of a run, so that a new
one is a line of output to decide about rather than a silent hole in the layer.

The two things it does not know
-------------------------------
**Ground with no online designation.** 17,700 dunam of the moshava is covered by
approved plans whose own designation field reads "יעוד עפ"י תכנית מאושרת אחרת" -
the plan defers to an older one that predates the digital register. Nothing is
drawn there. Blank means unknown, not private, and the layer's note says so.

**Supersession.** Two approved plans may cover the same ground, and the later
one wins - an old שצ"פ painted green under a newer plan that turned it into
housing is the failure this layer would be worst at. Layer 4 carries no date to
sort plans by, but layer 1 does: `pl_date_8`, publication in רשומות. Joining
the two on the plan number is what lets a public cell be dropped when a later
plan re-designated the same spot for something private, and the run prints how
many went that way. Where either side has no date the cell stays - being unable
to prove a שצ"פ was built over is not a reason to erase it.

    python3 build_public.py
"""

import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web", "data", "public.json")

XPLAN = ("https://ags.iplan.gov.il/arcgisiplan/rest/services"
         "/PlanningPublic/Xplan/MapServer")
LANDUSE = XPLAN + "/4/query"
BLUELINES = XPLAN + "/1/query"
XPLAN_SITE = "https://ags.iplan.gov.il/xplan/"

WFS = "https://open.govmap.gov.il/geoserver/opendata/ows"
LOCALITY = "פרדס חנה"

# The same browser user-agent build_plans.py needs: the government WAF answers a
# plain urllib with an HTML error page.
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128.0 Safari/537.36")

APPROVED = "אישור"

# The buckets, their colours, and every designation that falls in each. Exact
# names rather than substrings: "שטחים פתוחים ומבנים ומוסדות ציבור" would match
# two of these buckets under any looser rule, and it belongs to the first.
#
# The colours lean on the ones a תשריט is drawn in - green for open space, red
# for public buildings - so that somebody who has seen a plan on paper reads the
# map without the key.
GROUPS = [
    ("שטח ציבורי פתוח", "#2e7d32", [
        "שטח ציבורי פתוח",
        "שטח ציבורי פתוח מיוחד",
        "שטחים פתוחים",
        "שטחים פתוחים ומבנים ומוסדות ציבור",
        "ככר עירונית",
    ]),
    ("מבנים ומוסדות ציבור", "#c62828", [
        "מבנים ומוסדות ציבור",
        "מבנים ומוסדות ציבור לחינוך",
        "מבנים ומוסדות ציבור ומשרדים",
        "מסחר ומבנים ומוסדות ציבור",
        "תעסוקה ומבנים ומוסדות ציבור",
        "מסחר תיירות ומבנים ומוסדות ציבור",
    ]),
    ("דרכים ושבילים", "#546e7a", [
        "דרך מאושרת",
        "דרך מוצעת",
        "דרך משולבת",
        "דרך ו/או טיפול נופי",
        "דרך נופית",
        "שביל",
        "חניון",
        "מרכז תחבורה",
        "מפגש דרך-מסילה",
    ]),
    ("טבע, יער ומים", "#00838f", [
        "שמורת טבע",
        "גן לאומי",
        "יער",
        "יער טבעי",
        "נחל/ תעלה/מאגר מים",
        "נחל/תעלת נחל",
        "אתר איגום ו/או החדרה",
    ]),
    ("תשתית ומוסדות אחרים", "#6d4c41", [
        "מסילה מאושרת",
        "מסילה מוצעת",
        "מסילה ו/או טיפול נופי",
        "רצועת תשתיות",
        "מתקנים הנדסיים",
        "בית קברות",
        "ספורט ונופש",
    ]),
    # Not public, and here on purpose. See the module docstring.
    ("שטח פרטי פתוח", "#e65100", [
        "שטח פרטי פתוח",
    ]),
]

PRIVATE_GROUP = "שטח פרטי פתוח"

# The designations that mean "look somewhere else" rather than naming a use.
# Counted, reported, never drawn: painting them would cover the moshava in a
# colour that says nothing.
DEFERS = {
    "יעוד עפ\"י תכנית מאושרת אחרת",
    "שטח שהתוכנית אינה חלה עליו",
    # An overlay restriction - a runway cone, a pipeline setback - laid over
    # whatever the designation underneath happens to be, not a designation.
    "מגבלות בניה ופיתוח",
    "מגבלות בניה ופיתוח ב'",
}

FIELDS = ["mavat_name", "num", "pl_number", "pl_name",
          "shape_area", "last_update_date", "mp_id"]

# About a metre on the ground. Two, as the planning layer uses for blue lines,
# closes a three-metre שביל into a line.
OFFSET = "0.00001"

PAGE = 400


# ---------------------------------------------------------------- utilities

def tls():
    """The iplan host's TLS is older than OpenSSL 3 will talk to by default.

    Same widening `build_plans.py` needs, for the same host. Certificates are
    still verified.
    """
    ctx = ssl.create_default_context()
    ctx.set_ciphers("DEFAULT@SECLEVEL=1")
    return ctx


def post(url, params, tries=3):
    body = urllib.parse.urlencode(params).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=body, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180, context=tls()) as handle:
                return json.loads(handle.read().decode("utf-8-sig"))
        except Exception as err:                       # noqa: BLE001 - retry anything
            if attempt == tries - 1:
                raise
            print(f"    ...{err}, מנסה שוב", file=sys.stderr)
            time.sleep(3)
    return None


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "derech-kitzur/1.0"})
    with urllib.request.urlopen(req, timeout=180) as handle:
        return json.loads(handle.read().decode("utf-8"))


def save_json(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=1)


def ring_area(ring):
    """Twice the signed shoelace area of a closed ring, in whatever units."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        total += x0 * y1 - x1 * y0
    return total / 2


def ring_centroid(ring):
    """Area-weighted centroid of one closed ring, and its unsigned area."""
    area = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area) < 1e-14:
        n = len(ring) or 1
        return sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n, 0.0
    return cx / (3 * area), cy / (3 * area), abs(area) / 2


def rings_centroid(rings):
    """Centroid of an esri polygon, each ring weighted by its area."""
    total = wx = wy = 0.0
    fallback = []
    for ring in rings:
        pts = [tuple(p[:2]) for p in ring]
        if len(pts) < 3:
            continue
        cx, cy, area = ring_centroid(pts)
        fallback.append((cx, cy))
        total += area
        wx += cx * area
        wy += cy * area
    if total > 0:
        return wx / total, wy / total
    if fallback:
        return (sum(p[0] for p in fallback) / len(fallback),
                sum(p[1] for p in fallback) / len(fallback))
    return None


def inside(point, ring):
    """Ray-casting point-in-polygon, for the supersession count only."""
    x, y = point
    hit = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            hit = not hit
    return hit


def when(stamp):
    """An epoch-milliseconds field as a Hebrew date, or None if it is empty."""
    if not stamp:
        return None
    try:
        return time.strftime("%d/%m/%Y", time.gmtime(int(stamp) / 1000))
    except (TypeError, ValueError, OSError):
        return None


def dunam(value):
    """A size in dunam, written the way somebody says it out loud."""
    if value is None:
        return None
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------- fetching

def boundary():
    """The municipal boundary, as an esri polygon ready to filter with.

    GeoJSON winds outer rings counter-clockwise and holes clockwise; esri wants
    the opposite of both, and a ring handed over the wrong way round is read as
    a hole - which here would have meant filtering against the doughnut instead
    of the town.
    """
    query = urllib.parse.urlencode({
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": "opendata:muni_il",
        "outputFormat": "application/json", "srsName": "EPSG:4326",
        "CQL_FILTER": f"Muni_Heb LIKE '%{LOCALITY}%'",
    })
    features = get(f"{WFS}?{query}").get("features", [])
    if not features:
        raise SystemExit(f"לא נמצא גבול מוניציפלי ל-{LOCALITY} בשכבת muni_il")

    geom = features[0]["geometry"]
    polys = (geom["coordinates"] if geom["type"] == "MultiPolygon"
             else [geom["coordinates"]])
    rings = []
    for poly in polys:
        for i, ring in enumerate(poly):
            pts = [[round(p[0], 6), round(p[1], 6)] for p in ring]
            outer = i == 0
            if (outer and ring_area(pts) > 0) or (not outer and ring_area(pts) < 0):
                pts.reverse()
            rings.append(pts)

    name = features[0]["properties"].get("Muni_Heb", LOCALITY)
    return name, {"rings": rings, "spatialReference": {"wkid": 4326}}


def landuse(area):
    """Every approved land-use cell that meets the municipal boundary, paged."""
    out = []
    offset = 0
    while True:
        doc = post(LANDUSE, {
            "geometry": json.dumps(area),
            "geometryType": "esriGeometryPolygon",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "where": f"station_desc='{APPROVED}'",
            "outFields": ",".join(FIELDS),
            "returnGeometry": "true",
            "outSR": "4326",
            "maxAllowableOffset": OFFSET,
            "orderByFields": "objectid",
            "resultOffset": str(offset),
            "resultRecordCount": str(PAGE),
            "f": "json",
        })
        if "error" in doc:
            raise SystemExit(f"שגיאה מהשירות: {doc['error']}")
        got = doc.get("features", [])
        out += got
        print(f"  התקבלו {len(out)} תאי שטח")
        if len(got) < PAGE or not doc.get("exceededTransferLimit"):
            break
        offset += PAGE
    return out


def plans(area):
    """pl_number -> {url, date}, off the blue-lines layer.

    Layer 4 names the plan that set each designation and carries nothing else
    about it. Layer 1 - which `build_plans.py` already reads - carries both the
    things this file needs, so one request with no geometry buys every polygon a
    way through to its documents *and* the date it came into force.

    `pl_date_8` is publication in רשומות, which is the day a plan becomes the
    law of that ground; `pl_date7`, the approval hearing, stands in for the
    handful that have not got one. That date is the whole basis of deciding
    which of two overlapping plans is the one in force - see `superseded`.

    Filtered by the same boundary as the land uses rather than by the town's
    name, because a designation inside the moshava can come from a plan that is
    not the moshava's: a road out of תלמי אלעזר crosses the line, and asking for
    plans whose `plan_area_name` says פרדס חנה left twenty-seven cells with no
    link and, worse, no date to test supersession against.
    """
    meta = {}
    offset = 0
    while True:
        doc = post(BLUELINES, {
            "geometry": json.dumps(area),
            "geometryType": "esriGeometryPolygon",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "where": "1=1",
            "outFields": "pl_number,pl_url,pl_date_8,pl_date7",
            "returnGeometry": "false",
            "orderByFields": "pl_number",
            "resultOffset": str(offset),
            "resultRecordCount": str(PAGE),
            "f": "json",
        })
        if "error" in doc:
            print(f"  אזהרה: אין קישורים ותאריכים ({doc['error']})", file=sys.stderr)
            return meta
        got = doc.get("features", [])
        for feat in got:
            at = feat.get("attributes", {})
            number = (at.get("pl_number") or "").strip()
            if not number:
                continue
            entry = meta.setdefault(number, {"url": None, "date": None})
            entry["url"] = entry["url"] or (at.get("pl_url") or "").strip() or None
            date = at.get("pl_date_8") or at.get("pl_date7")
            # The newest edition of a plan that has several rows here.
            if date and (entry["date"] is None or date > entry["date"]):
                entry["date"] = date
        if len(got) < PAGE or not doc.get("exceededTransferLimit"):
            break
        offset += PAGE
    return meta


# ---------------------------------------------------------------- building

def group_of():
    """designation -> (bucket name, colour)."""
    return {use: (name, colour)
            for name, colour, uses in GROUPS for use in uses}


def note_for(at, group, size, date):
    """What the detail pane says about one cell of designated land.

    Deliberately short and specific to this cell. What is true of every cell in
    the layer - that a designation is not ownership and not permission to enter,
    that blank ground is unknown rather than private - is said once in the
    layer's own note instead of five hundred and forty-three times in the file.
    """
    use = (at.get("mavat_name") or "").strip()
    bits = []

    if group == PRIVATE_GROUP:
        bits.append(f"{use}: שטח פתוח שאינו ציבורי. נראה כמו שצ\"פ ואיננו - "
                    "הקרקע פרטית, והייעוד רק קובע שלא ייבנה עליה.")
    else:
        bits.append(f"ייעוד מאושר: {use}.")

    plan = (at.get("pl_number") or "").strip()
    title = (at.get("pl_name") or "").strip()
    if plan:
        bits.append(f"תכנית {plan}" + (f", {title}." if title else "."))

    cell = (at.get("num") or "").strip()
    if cell:
        bits.append(f"תא שטח {cell}.")
    if size:
        bits.append(f"{size} דונם.")
    if date:
        bits.append(f"בתוקף מ-{date}.")
    return " ".join(bits)


def build(features, meta, area_name):
    """Sort every approved cell into a bucket, then drop what a later plan
    overrode.

    Two passes over the same features, because the second needs the first: a
    public cell is only in force where no later plan has re-designated the same
    ground for something that is not public, and "later" can only be decided
    once every cell has been paired with its plan's date.
    """
    buckets = group_of()

    def cell_of(feat):
        at = feat.get("attributes", {})
        rings = (feat.get("geometry") or {}).get("rings") or []
        centre = rings_centroid(rings) if rings else None
        plan = (at.get("pl_number") or "").strip()
        return at, rings, centre, plan, (meta.get(plan) or {}).get("date")

    # Pass one: the non-public cells, as ground somebody's plan has spoken for.
    private = []
    counts = {"defers": 0, "other": 0, "no_geometry": 0, "superseded": 0}
    unknown = {}
    for feat in features:
        at, rings, centre, _, date = cell_of(feat)
        use = (at.get("mavat_name") or "").strip()
        if not centre or use in DEFERS or use in buckets:
            continue
        unknown[use] = unknown.get(use, 0) + 1
        counts["other"] += 1
        if date is None:
            continue                    # nothing to compare against; ignore it
        ring = [(round(p[0], 6), round(p[1], 6)) for p in rings[0]]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        private.append((min(xs), min(ys), max(xs), max(ys), ring, date))

    def superseded(centre, date):
        """Whether a later plan re-designated this spot for something private.

        Conservative in both directions. With no date on either side there is
        nothing to compare and the cell stays: a שצ"פ that might have been built
        over is a better answer than a hole in the layer. The test is on the
        centre point rather than on the area, which is the same simplification
        the rest of this file makes and is right for the shape of the problem -
        the overlaps that matter are a whole cell swallowed by a later plan, not
        a corner clipped off one.
        """
        if date is None:
            return False
        x, y = centre
        return any(x0 <= x <= x1 and y0 <= y <= y1 and later > date
                   and inside((x, y), ring)
                   for x0, y0, x1, y1, ring, later in private)

    # Pass two: the public ones, minus whatever a later plan overrode.
    places = []
    seen = {}
    for feat in features:
        at, rings, centre, plan, date = cell_of(feat)
        use = (at.get("mavat_name") or "").strip()
        if not centre:
            counts["no_geometry"] += 1
            continue
        if use in DEFERS:
            counts["defers"] += 1
            continue
        if use not in buckets:
            continue                    # counted in pass one
        if superseded(centre, date):
            counts["superseded"] += 1
            continue

        group, _ = buckets[use]
        # `shape_area`, the measured area of this cell in square metres, and not
        # `legal_area` - which the register calls "שטח רשום דונם" and which
        # turns out to be the whole plan's registered area rather than the
        # cell's. Trusting the alias put 41,313 dunam of "תשתית" in a town of
        # 44,000: twenty-nine railway cells, each credited with the entire
        # length of the plan that drew it.
        size = (at.get("shape_area") or 0) / 1000 or None
        text = dunam(size)
        in_force = when(date)

        # The plan edition and the cell number inside it, which is the register's
        # own name for this piece of ground and survives a rebuild of the
        # service - unlike the row's objectid, and unlike a running index. The
        # counter is only for a plan that numbers two cells the same.
        key = f"use-{at.get('mp_id') or 0:.0f}-{(at.get('num') or '?').strip()}"
        seen[key] = seen.get(key, 0) + 1
        places.append({
            "id": key if seen[key] == 1 else f"{key}-{seen[key]}",
            "name": f"{use} · {text} דונם" if text else use,
            "group": group,
            "cats": [c for c in ["ייעוד", use, plan] if c],
            "num": (at.get("num") or "").strip() or None,
            "plan": plan or None,
            "dunam": round(size, 2) if size else None,
            "url": (meta.get(plan) or {}).get("url") or XPLAN_SITE,
            "photos": [],
            "geo": {"lat": round(centre[1], 6), "lng": round(centre[0], 6),
                    "source": "landuse"},
            "shape": [[[round(p[0], 6), round(p[1], 6)] for p in ring]
                      for ring in rings],
            "note": note_for(at, group, text, in_force),
        })

    order = {name: i for i, (name, _, _) in enumerate(GROUPS)}
    places.sort(key=lambda p: (order.get(p["group"], 9), -len(p["shape"][0]), p["name"]))

    used = {p["group"] for p in places}
    return {
        "version": 1,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": XPLAN_SITE,
        "name": "שטחים ציבוריים",
        "area": area_name,
        "groups": [{"name": n, "color": c} for n, c, _ in GROUPS if n in used],
        "places": places,
        "stats": {
            "cells": len(places),
            "vertices": sum(len(r) for p in places for r in p["shape"]),
            "dunam": {
                name: round(sum((p.get("dunam") or 0) for p in places
                                if p["group"] == name), 1)
                for name, _, _ in GROUPS if name in used
            },
            "by_group": {name: sum(1 for p in places if p["group"] == name)
                         for name, _, _ in GROUPS if name in used},
            "skipped": counts,
            "unknown": dict(sorted(unknown.items(), key=lambda kv: -kv[1])),
        },
    }


def main():
    print("מושך את הגבול המוניציפלי מ-govmap…")
    area_name, area = boundary()
    print(f"  {area_name}, {sum(len(r) for r in area['rings']):,} נקודות גבול")

    print("מושך ייעודי קרקע מאושרים ממנהל התכנון…")
    features = landuse(area)

    print("מושך תכניות, קישורים ותאריכי תוקף…")
    meta = plans(area)
    dated = sum(1 for m in meta.values() if m["date"])
    print(f"  {len(meta)} תכניות, {dated} מהן עם תאריך תוקף")

    doc = build(features, meta, area_name)
    save_json(OUT, doc)

    stats = doc["stats"]
    size = os.path.getsize(OUT) // 1024
    print(f"\nנכתבו {stats['cells']} תאי שטח ל-{os.path.relpath(OUT, HERE)}"
          f" ({size} KB, {stats['vertices']:,} נקודות)")
    for group in doc["groups"]:
        print(f"  {stats['by_group'][group['name']]:5}  {group['name']}"
              f"  ({stats['dunam'][group['name']]:,.0f} דונם)")

    skipped = stats["skipped"]
    print(f"\n  {skipped['defers']} תאים מפנים לתכנית אחרת או מחוץ לתחום התכנית, "
          "ולכן לא נצבעו")
    print(f"  {skipped['other']} ייעודים לא ציבוריים - מגורים, חקלאות, תעסוקה וכו'")
    if skipped["superseded"]:
        print(f"  {skipped['superseded']} ייעודים ציבוריים שתכנית מאוחרת יותר "
              "שינתה לייעוד לא ציבורי, ולכן ירדו")
    if skipped["no_geometry"]:
        print(f"  {skipped['no_geometry']} בלי גיאומטריה")

    if stats["unknown"]:
        print("\nייעודים שהקובץ הזה לא מכיר, ולכן נשארו בחוץ. "
              "אם אחד מהם ציבורי, מקומו ב-GROUPS:")
        for use, n in stats["unknown"].items():
            print(f"  {n:5}  {use}")


if __name__ == "__main__":
    main()
