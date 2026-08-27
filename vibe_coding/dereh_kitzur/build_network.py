#!/usr/bin/env python3
"""Turn the moshava's cycling-network shapefiles into a layer file for the app.

The source is the planning package for the Pardes Hanna-Karkur cycling network
(IOA Engineering, project 1102), as ESRI shapefiles zipped in ``network_src/``.

**The files are not organised the way the app's layers are, and their names
lie.** The 27/8/2026 pair is called "ראשית" and "משנית" - primary and secondary
- but each holds a mix of both, and 38 of their segments are identical
duplicates of one another. So everything is pooled, deduplicated by geometry,
and split by the ``סטטוס`` field into what exists and what does not. Trust the
fields, never the file names.

Output is ``web/data/layers.json``, which the app loads alongside
``data/trails.json``. The two pipelines are deliberately separate:
``build_data.py`` hits Google on every run and takes minutes, this one is
offline apart from an optional, cached street-name lookup.

    pip install pyshp pyproj
    python3 build_network.py
"""

import collections
import io
import json
import math
import os
import re
import sys
import zipfile

import shapefile          # pyshp

SRC_DIR = "network_src"
OUT = "web/data/layers.json"
NAME_CACHE = ".cache/named_roads.json"

# Which files to read. Everything in them is pooled and then split by status,
# because the planners' files are not organised the way the app's layers are:
# the 27/8/2026 pair is named "ראשית" and "משנית" but each holds a mix of both,
# and 38 of their segments are byte-identical duplicates of each other. Trust
# the סטטוס and רשת fields, not the file names.
#
# The two files from 8/2026 are left in place and unused. The newer pair covers
# every one of their segments - checked, none further than 40 m from a new
# counterpart - and adds 23 more.
SOURCES = [
    "רשת ראשית מעודכנת.zip",
    "רשת משנית מעודכנת.zip",
]

# What a rider actually wants to know is whether they can ride it today. So the
# split is by that, and the precise stage - proposed, being promoted, under
# construction - rides along as `status` and shows as a chip in the app.
#
# An unrecognised or blank status counts as not-yet: claiming a path is ridable
# when the data does not say so is the worse of the two mistakes.
EXISTING = "קיים"

# Order here is drawing order, bottom first, so the planned network sits under
# the network that already exists.
LAYERS = [
    {
        "id": "bike-proposed",
        "name": "רשת רכיבה מתוכננת",
        "short": "מתוכננת",
        "color": "#e07b00",
        "dash": True,
        "note": "מה שעוד לא בשטח: מוצע, מקודם, או בביצוע. השלב המדויק מופיע בפרטי המקטע.",
    },
    {
        "id": "bike-existing",
        "name": "רשת רכיבה קיימת",
        "short": "קיימת",
        "color": "#1565c0",
        "dash": False,
        "note": "מה שכבר סלול ורכיב היום, לפי אותה תוכנית.",
    },
]

CREDIT = "תוכנית רשת הרכיבה של פרדס חנה-כרכור, IOA Engineering"

# DBF caps field names at ten bytes, so the Hebrew headers arrive truncated and
# the tail byte is often mangled. Match on the readable prefix instead.
FIELD_ALIASES = {
    "סטטוס": "status",
    "רשת": "grade",
    "אורך ש": "src_length",
    "סוג שב": "kind",
    "הערות": "note",
    "אורך": "group_length",
}


def alias(field_name):
    """Map a possibly-truncated Hebrew DBF header to a stable ASCII key."""
    clean = field_name.replace("�", "").replace("?", "").strip()
    for prefix, key in FIELD_ALIASES.items():
        if clean and prefix.startswith(clean):
            return key
    return None


def haversine(a, b):
    """Metres between two (lat, lng) pairs."""
    radius, rad = 6371000.0, math.pi / 180
    d_lat = (b[0] - a[0]) * rad
    d_lng = (b[1] - a[1]) * rad
    h = (math.sin(d_lat / 2) ** 2 +
         math.cos(a[0] * rad) * math.cos(b[0] * rad) * math.sin(d_lng / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(h))


def path_length(path):
    return sum(haversine(a, b) for a, b in zip(path, path[1:]))


# ---------------------------------------------------------------- projection

def transformer_for(prj_text):
    """A callable (x, y) -> (lat, lng), or None when the file is already WGS84.

    The proposed layer is drawn on the Israeli TM Grid and the existing one is
    already in degrees, so the two cannot share one code path.
    """
    if "PROJCS" not in prj_text:
        return None                       # plain GEOGCS: coordinates are lng/lat

    from pyproj import CRS, Transformer
    crs = CRS.from_wkt(prj_text)
    tr = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)

    def convert(x, y):
        lng, lat = tr.transform(x, y)
        return lat, lng

    return convert


# ------------------------------------------------------------- street naming

BBOX = (32.44, 34.90, 32.50, 35.02)      # the moshava, with room to spare
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
# Overpass answers an unidentified client with 406, so the agent is required.
H = {"User-Agent": "DerechKitzur/1.0 (orimosenzon@gmail.com)"}


def named_roads():
    """Named OSM road segments as [name, lat1, lng1, lat2, lng2].

    The planners' segments carry no names at all, only a serial number, which
    makes for an unreadable list. Naming each one after the street it runs
    along is the difference between "מקטע 17" and "דרך הבנים".
    """
    if os.path.exists(NAME_CACHE):
        with open(NAME_CACHE, encoding="utf-8") as fh:
            return json.load(fh)

    try:
        import requests
    except ImportError:
        return []

    s_lat, s_lng, n_lat, n_lng = BBOX
    query = ('[out:json][timeout:120];'
             'way["highway"]["name"]'
             f'({s_lat},{s_lng},{n_lat},{n_lng});out geom;')

    elements = None
    for host in OVERPASS:
        try:
            res = requests.post(host, data={"data": query}, headers=H, timeout=180)
            res.raise_for_status()
            elements = res.json()["elements"]
            break
        except Exception as exc:                   # noqa: BLE001 - try the mirror
            print(f"  ! overpass {host} failed: {exc}", file=sys.stderr)
    if elements is None:
        print("  ! no street names available; segments keep serial names",
              file=sys.stderr)
        return []

    segs = []
    for way in elements:
        name = way.get("tags", {}).get("name")
        geom = way.get("geometry", [])
        for a, b in zip(geom, geom[1:]):
            segs.append([name, a["lat"], a["lon"], b["lat"], b["lon"]])

    os.makedirs(os.path.dirname(NAME_CACHE), exist_ok=True)
    with open(NAME_CACHE, "w", encoding="utf-8") as fh:
        json.dump(segs, fh, ensure_ascii=False)
    return segs


def point_to_segment_m(lat, lng, a_lat, a_lng, b_lat, b_lng):
    my = 111320.0
    mx = 111320.0 * math.cos(math.radians(lat))
    px, py = (lng - a_lng) * mx, (lat - a_lat) * my
    bx, by = (b_lng - a_lng) * mx, (b_lat - a_lat) * my
    span = bx * bx + by * by
    t = 0.0 if span == 0 else max(0.0, min(1.0, (px * bx + py * by) / span))
    return math.hypot(px - t * bx, py - t * by)


SNAP_M = 30      # a cycle route further than this from a street is off-road


def street_names(path, roads):
    """Street names the path runs along, most-travelled first.

    Sampling every vertex would over-weight the dense end of a polyline, so
    walk the line at a fixed spacing and score each street by how much of the
    route sits near it.
    """
    if not roads:
        return []

    votes = {}
    step = 40.0                                    # metres between probes
    for a, b in zip(path, path[1:]):
        span = haversine(a, b)
        for i in range(max(1, int(span // step)) + 1):
            t = min(1.0, i * step / span) if span else 0.0
            lat = a[0] + (b[0] - a[0]) * t
            lng = a[1] + (b[1] - a[1]) * t

            best, best_name = SNAP_M, None
            for name, a_lat, a_lng, b_lat, b_lng in roads:
                if abs(a_lat - lat) > 0.0006 and abs(b_lat - lat) > 0.0006:
                    continue                       # cheap reject before the maths
                if abs(a_lng - lng) > 0.0008 and abs(b_lng - lng) > 0.0008:
                    continue
                d = point_to_segment_m(lat, lng, a_lat, a_lng, b_lat, b_lng)
                if d < best:
                    best, best_name = d, name
            if best_name:
                votes[best_name] = votes.get(best_name, 0) + 1

    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    # Anything under a fifth of the winner is a street the route merely crosses.
    floor = ranked[0][1] / 5 if ranked else 0
    return [name for name, n in ranked if n >= floor][:3]


# ------------------------------------------------------------------ the build

def read_source(filename):
    """One shapefile zip -> [(attrs, path as [[lat, lng], ...]), ...].

    A record may hold several disjoint lines; each becomes its own entry, so a
    row in the app's list is always one continuous stretch.
    """
    path = os.path.join(SRC_DIR, filename)
    with open(path, "rb") as fh:
        archive = zipfile.ZipFile(io.BytesIO(fh.read()))

    parts = {}
    for entry in archive.namelist():
        stem, _, ext = entry.rpartition(".")
        if ext.lower() in ("shp", "dbf", "shx", "prj", "cpg"):
            parts[ext.lower()] = archive.read(entry)

    prj = parts.get("prj", b"").decode("utf-8", "replace")
    project = transformer_for(prj)

    # The .cpg holds the DBF's code page. It may be a bare number ("1255") or a
    # name ("UTF-8"); pyshp wants a codec name either way.
    cpg = parts.get("cpg", b"utf-8").decode("ascii", "replace").strip()
    encoding = f"cp{cpg}" if cpg.isdigit() else cpg

    reader = shapefile.Reader(
        shp=io.BytesIO(parts["shp"]),
        dbf=io.BytesIO(parts["dbf"]),
        shx=io.BytesIO(parts["shx"]),
        encoding=encoding,
        encodingErrors="replace",
    )
    keys = [alias(f[0]) for f in reader.fields[1:]]

    out = []
    for sr in reader.iterShapeRecords():
        attrs = {k: v for k, v in zip(keys, sr.record) if k}
        bounds = list(sr.shape.parts) + [len(sr.shape.points)]
        for start, end in zip(bounds, bounds[1:]):
            pts = sr.shape.points[start:end]
            if len(pts) < 2:
                continue
            path_ll = [(project(x, y) if project else (y, x)) for x, y in pts]
            out.append((attrs, [[round(lat, 6), round(lng, 6)] for lat, lng in path_ll]))
    return out


def build_segment(spec, attrs, path_ll, roads, index):
    streets = street_names(path_ll, roads)
    grade = (attrs.get("grade") or "").strip()
    label = f"רשת {grade}" if grade else spec["short"]
    name = f"{label} · {' / '.join(streets)}" if streets \
        else f"{label} · מקטע {index}"

    return {
        "id": f'{spec["id"]}-{index}',
        "name": name,
        "note": (attrs.get("note") or "").strip(),
        "photos": [],
        "path": path_ll,
        "length": round(path_length(path_ll)),
        "color": spec["color"],
        "layer": spec["id"],
        "grade": grade,                              # ראשית / משנית
        "kind": (attrs.get("kind") or "").strip(),   # מופרד / משולב / בדיון
        "status": (attrs.get("status") or "").strip(),
        "streets": streets,
        "entries": [
            {"lat": path_ll[0][0], "lng": path_ll[0][1]},
            {"lat": path_ll[-1][0], "lng": path_ll[-1][1]},
        ],
    }


def main():
    roads = named_roads()
    print(f"named road segments: {len(roads)}")

    # Pool every source, then drop exact geometric duplicates. The two files
    # overlap heavily by design - each is one planner's view of the same
    # network - and the shared segments carry identical attributes, so keeping
    # whichever arrives first loses nothing.
    pool = {}
    dupes = 0
    for filename in SOURCES:
        rows = read_source(filename)
        print(f"  {filename}: {len(rows)} מקטעים")
        for attrs, path_ll in rows:
            key = tuple(map(tuple, path_ll))
            if key in pool:
                dupes += 1
                continue
            pool[key] = (attrs, path_ll)
    print(f"  אחרי איחוד כפילויות: {len(pool)} מקטעים ({dupes} כפולים הוסרו)")

    by_layer = {spec["id"]: [] for spec in LAYERS}
    unknown = collections.Counter()
    for attrs, path_ll in pool.values():
        status = (attrs.get("status") or "").strip()
        target = "bike-existing" if status == EXISTING else "bike-proposed"
        if status and status != EXISTING:
            unknown[status] += 1
        elif not status:
            unknown["(ריק)"] += 1
        by_layer[target].append((attrs, path_ll))

    out_layers = []
    for spec in LAYERS:
        rows = by_layer[spec["id"]]
        segments = [build_segment(spec, attrs, path_ll, roads, i)
                    for i, (attrs, path_ll) in enumerate(rows)]
        total = sum(s["length"] for s in segments)
        named = sum(1 for s in segments if s["streets"])
        print(f'{spec["name"]}: {len(segments)} מקטעים, {total} מ׳, {named} עם שם רחוב')
        out_layers.append({
            "id": spec["id"],
            "name": spec["name"],
            "short": spec["short"],
            "color": spec["color"],
            "dash": spec["dash"],
            "note": spec["note"],
            "credit": CREDIT,
            "on": False,      # off by default: the initiative's trails come first
            "segments": segments,
        })

    if unknown:
        print("  שלבים בשכבה המתוכננת: " +
              ", ".join(f"{k}={v}" for k, v in unknown.most_common()))

    # This script owns exactly the layers in LAYERS. Anything else in the file
    # arrived from somewhere else - import_offroad.py writes recorded tracks
    # here too - so it is read back and kept, above ours, rather than dropped
    # on every rebuild.
    owned = {spec["id"] for spec in LAYERS}
    foreign = []
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            foreign = [l for l in json.load(fh).get("layers", [])
                       if l.get("id") not in owned]
    if foreign:
        print("שכבות שנשמרו כפי שהן: " +
              ", ".join(l.get("name", l.get("id", "?")) for l in foreign))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"credit": CREDIT, "layers": out_layers + foreign}, fh,
                  ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
