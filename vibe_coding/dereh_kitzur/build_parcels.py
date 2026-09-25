#!/usr/bin/env python3
"""Build web/data/parcels.json: every cadastral parcel (חלקה) in the moshava.

The blocks are in blocks.json and the parcels used to be fetched from govmap a
viewport at a time, from zoom 16 in, because "there are over ten thousand" and
a layer that size did not fit the way this app keeps layers (see the
docstring of build_cadastre.py). Counted on 25/9/2026 it is 9,126, and the
question Ori asked - which parcel is this point in, anywhere in the moshava,
at any zoom - needs all of them at once.

It fits as long as it is not treated like the other layers:

  * It is not kept in localStorage. The browser's HTTP cache holds it, and
    MapLibre fetches and parses it in its worker, off the main thread.
  * Its parcels are not list items. 9,126 rows made the list and the search
    unusable when measured; a parcel becomes an item only when it is tapped.
    See Parcels in app.js and the grid in layers.js.
  * It is fetched only when the layer is switched on.

Where it comes from
-------------------
The national cadastre as an open WFS with no key, the same service as the
blocks:

    open.govmap.gov.il/geoserver/opendata/ows    (opendata:PARCEL_ALL)

`LOCALITY_N` names the town, and filtering on it returns exactly the parcels of
the sixty blocks in blocks.json - checked: 9,126 either way, all of them
"מוסדר" (settled) but two "חדש רשום". There is no unsettled land in the
moshava, so the parcels tile it without holes.

Size
----
The raw outlines are 392,287 points, a lot of them duplicates and many more a
curve drawn in centimetre steps. Douglas-Peucker at 10cm keeps 90,808 of them:
3.0MB, 0.46MB gzipped as GitHub Pages sends it. At the closest zoom the app
allows a pixel is about 30cm, so 10cm is below anything the map can show; it
is not below what a surveyor cares about, and the detail pane says so.

    python3 build_parcels.py
"""

import json
import math
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web", "data", "parcels.json")

WFS = "https://open.govmap.gov.il/geoserver/opendata/ows"
LOCALITY = "פרדס חנה-כרכור"
PAGE = 3000
TOLERANCE_M = 0.1

# Metres per degree at the moshava's latitude, for measuring in the plane.
LAT0 = 32.47
MX = 111320 * math.cos(math.radians(LAT0))
MY = 111320


def get(params, tries=3):
    url = WFS + "?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "derech-kitzur/1.0"})
            with urllib.request.urlopen(req, timeout=300) as handle:
                return json.loads(handle.read().decode("utf-8"))
        except Exception as err:                       # noqa: BLE001 - retry anything
            if attempt == tries - 1:
                raise
            print(f"    ...{err}, מנסה שוב")
            time.sleep(5)
    return None


def fetch():
    out = []
    start = 0
    while True:
        doc = get({
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": "opendata:PARCEL_ALL",
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "propertyName": "GUSH_NUM,PARCEL,LEGAL_AREA,the_geom",
            "CQL_FILTER": f"LOCALITY_N='{LOCALITY}'",
            # A stable order, or paging can hand the same parcel out twice.
            "sortBy": "OBJECTID",
            "count": str(PAGE),
            "startIndex": str(start),
        })
        got = doc.get("features", [])
        out += got
        print(f"  התקבלו {len(out)} חלקות")
        if len(got) < PAGE:
            return out
        start += PAGE


def simplify(pts, tol):
    """Douglas-Peucker, measured in metres on the plane."""
    if len(pts) < 3:
        return pts
    ax, ay = pts[0][0] * MX, pts[0][1] * MY
    bx, by = pts[-1][0] * MX, pts[-1][1] * MY
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    worst, at = -1.0, 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i][0] * MX, pts[i][1] * MY
        if length:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length))
            d = math.hypot(ax + t * dx - px, ay + t * dy - py)
        else:                          # a closed ring: first and last coincide
            d = math.hypot(px - ax, py - ay)
        if d > worst:
            worst, at = d, i
    if worst <= tol:
        return [pts[0], pts[-1]]
    return simplify(pts[:at + 1], tol)[:-1] + simplify(pts[at:], tol)


def ring(points):
    pts = [[round(x, 6), round(y, 6)] for x, y, *_ in points]
    thin = simplify(pts, TOLERANCE_M)
    if len(thin) >= 4:                 # still a polygon: three corners and back
        pts = thin
    out = [pts[0]]
    for p in pts[1:]:
        if p != out[-1]:
            out.append(p)
    return out


def main():
    print("מושך חלקות מ-govmap…")
    raw = fetch()

    features = []
    seen = set()
    points = 0
    for f in raw:
        p = f.get("properties") or {}
        gush, parcel = p.get("GUSH_NUM"), p.get("PARCEL")
        if gush is None or parcel is None or (gush, parcel) in seen:
            continue
        seen.add((gush, parcel))
        polys = (f.get("geometry") or {}).get("coordinates") or []
        # MultiPolygon on the wire, but every one of them is a single polygon;
        # the twenty with a hole keep it.
        rings = [ring(r) for poly in polys for r in poly if len(r) >= 4]
        if not rings:
            continue
        points += sum(len(r) for r in rings)
        features.append({
            "type": "Feature",
            # The id MapLibre keys feature-state on, so the selected parcel can
            # be lit without touching the other nine thousand.
            "id": gush * 10000 + parcel,
            "properties": {"g": gush, "p": parcel, "a": p.get("LEGAL_AREA") or 0},
            "geometry": {"type": "Polygon", "coordinates": rings},
        })

    doc = {
        "type": "FeatureCollection",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "https://www.govmap.gov.il/",
        "features": features,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, separators=(",", ":"))

    blocks = len({f["properties"]["g"] for f in features})
    size = os.path.getsize(OUT) / 1e6
    print(f"\nנכתבו {len(features)} חלקות ב-{blocks} גושים, {points} נקודות, "
          f"{size:.2f}MB ל-{os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
