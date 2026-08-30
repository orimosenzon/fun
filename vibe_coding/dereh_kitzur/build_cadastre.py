#!/usr/bin/env python3
"""Build web/data/blocks.json: the cadastral blocks (גושים) of the moshava.

Yoav asked for "גושים חלקות ובעלויות". This is the first of the three, and it
is the one that pays for itself immediately: every plan in the planning layer
is named after a block and a parcel - "תוספת זכויות בניה בגוש 10105 חלקה 203" -
and until now there was no way to find out on the map which block you were
standing in. Sixty outlines with their numbers on them turn every one of those
names into a place.

Where it comes from
-------------------
The national cadastre, published by govmap as an open WFS with no key:

    open.govmap.gov.il/geoserver/opendata/ows    (opendata:SUB_GUSH_ALL)

The same service `build_shimur.py` resolves the conservation appendix against,
one layer up: SUB_GUSH_ALL is the blocks, PARCEL_ALL the parcels inside them.
Rows carry `LOCALITY_N`, so the moshava's own blocks come out in one request
without any geometry filtering.

Why the parcels are not here too
--------------------------------
They do not fit. Twenty-one of these sixty blocks - the ones the conservation
appendix happens to touch - already hold 5,041 parcels between them, so the
whole town is somewhere above ten thousand. The app loads every layer up front
and caches it in localStorage, which has about five megabytes for everything;
a parcel layer that size would not go in, and would not be readable if it did.

The blocks are 60 polygons and 11,303 points, which is a third of the
pardespedia layer. If the parcels are wanted later they want a different shape
entirely - fetched for the block you are looking at, when you zoom in past it,
rather than shipped - and that is a decision about how this app loads data, not
another builder.

    python3 build_cadastre.py
"""

import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web", "data", "blocks.json")

WFS = "https://open.govmap.gov.il/geoserver/opendata/ows"
LOCALITY = "פרדס חנה-כרכור"
GOVMAP = "https://www.govmap.gov.il/"

COLOR = "#8d6e63"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "derech-kitzur/1.0"})
    with urllib.request.urlopen(req, timeout=180) as handle:
        return json.loads(handle.read().decode("utf-8"))


def save_json(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=1)


def ring_centroid(ring):
    """Area-weighted centroid of one closed ring, and its area."""
    area = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area) < 1e-12:
        n = len(ring) or 1
        return sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n, 0.0
    return cx / (3 * area), cy / (3 * area), abs(area) / 2


def rings_of(geom):
    """Every outer ring of a (Multi)Polygon, as plain lists of [lng, lat]."""
    if not geom:
        return []
    polys = (geom["coordinates"] if geom["type"] == "MultiPolygon"
             else [geom["coordinates"]])
    return [[[round(p[0], 6), round(p[1], 6)] for p in ring]
            for poly in polys for ring in poly]


def fetch():
    query = urllib.parse.urlencode({
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": "opendata:SUB_GUSH_ALL",
        "outputFormat": "application/json", "srsName": "EPSG:4326",
        "CQL_FILTER": f"LOCALITY_N='{LOCALITY}'",
    })
    return get(f"{WFS}?{query}").get("features", [])


def build(features):
    blocks = {}
    for feat in features:
        props = feat.get("properties", {})
        rings = rings_of(feat.get("geometry"))
        if not rings:
            continue
        gush = props.get("GUSH_NUM")
        if gush is None:
            continue

        total = wx = wy = 0.0
        for ring in rings:
            cx, cy, area = ring_centroid([tuple(p) for p in ring])
            total += area
            wx += cx * area
            wy += cy * area
        if total <= 0:
            continue

        # A block split across sheets comes back as several rows. Merge them
        # rather than letting the last one win: the outline of a block is all
        # of its pieces, and its number belongs on the biggest.
        key = str(int(gush))
        entry = blocks.setdefault(key, {"rings": [], "area": 0.0, "cx": 0.0, "cy": 0.0})
        entry["rings"] += rings
        entry["area"] += total
        entry["cx"] += wx
        entry["cy"] += wy

    places = []
    for gush, e in sorted(blocks.items(), key=lambda kv: int(kv[0])):
        # Square metres, from a shoelace area in square degrees at this
        # latitude. Good to a couple of percent, which is all a "how big is
        # this block" line needs.
        dunam = e["area"] * (111320 * 93800) / 1000
        places.append({
            "id": "gush-" + gush,
            "name": "גוש " + gush,
            "cats": ["גוש"],
            "num": gush,
            "url": GOVMAP,
            "photos": [],
            "geo": {"lat": round(e["cy"] / e["area"], 6),
                    "lng": round(e["cx"] / e["area"], 6),
                    "source": "block"},
            "shape": e["rings"],
            "note": f"גוש {gush} בקדסטר הארצי, כ-{dunam:,.0f} דונם. "
                    "שמות התכניות בשכבת התכניות מפנים לגוש ולחלקה, "
                    "וזו הדרך למצוא אותם על הקרקע. "
                    "החלקות שבתוך הגוש אינן בשכבה הזאת.",
        })

    return {
        "version": 1,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": GOVMAP,
        "name": "גושים",
        # No groups: a block is a block, and every one of them is drawn in the
        # same colour. The app falls back to the layer's own colour, and the
        # legend shows the layer as one dashed line rather than a list of one.
        "groups": [],
        "places": places,
        "stats": {
            "blocks": len(places),
            "vertices": sum(len(r) for p in places for r in p["shape"]),
        },
    }


def main():
    print(f"מושך את גושי {LOCALITY} מהקדסטר הארצי…")
    features = fetch()
    print(f"  התקבלו {len(features)} שורות")
    doc = build(features)
    save_json(OUT, doc)
    size = os.path.getsize(OUT) // 1024
    print(f"\nנכתבו {doc['stats']['blocks']} גושים ל-{os.path.relpath(OUT, HERE)}"
          f" ({size} KB, {doc['stats']['vertices']:,} נקודות)")


if __name__ == "__main__":
    main()
