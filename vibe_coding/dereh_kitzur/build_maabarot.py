#!/usr/bin/env python3
"""Build the שביל המעברות layer: the heritage trail at the east edge of the
moshava, nine stations through the ground of the old immigrant camp (מחנה 87)
and the ma'abara that grew out of it.

Nobody has published the route as a track. The one source is a slide from the
presentation by בית הראשונים and עמותת המעברות, an oblique Google Earth view
with the route dashed in yellow and the nine stations numbered (it is on the
pardespedia article, קובץ:שביל המעברות - מסלול.jpg). So the route here is a
reading of that picture, matched by eye against an aerial photo and
OpenStreetMap:

* 9 stands on דרך הנדיב south of ביג; the route runs south along the road
  to the corner, then east along the street to 8, where the street east of
  the commercial block meets the dirt road.
* 7 is at the north-west corner of the orchard on that dirt road, which runs
  north-east to the corner of the industrial zone (J here), with 1 just past it.
* The loop: north-east to 3, beside the motocross track (the clearest landmark
  in the slide), north-west along the tree line past 4 to the end of the
  industrial zone's main street, south-west along it past 5 to the cross
  street at 6, and down that street and a dirt track back to J.

ANCHORS below are those readings. Between them the line follows OSM's own
ways (shortest path), so the shape is the real streets and tracks; what is a
guess is which of them the trail takes, and the stations are where the slide
puts them to within a few tens of metres. The layer says so.

    python3 build_maabarot.py            # preview: length, stations, snapping
    python3 build_maabarot.py --write    # merge into web/data/layers.json
    python3 build_maabarot.py --publish  # and put it in the data repo, which the app reads
"""

import argparse
import json
import math
import os
import urllib.parse
import urllib.request

import networkx as nx

from import_offroad import merge, path_length, simplify

LAYER_ID = "maabarot"
COLOR = "#8d6e63"
ARTICLE = "https://pardespedia.info/wiki/" + urllib.parse.quote("שביל_המעברות")
OVERPASS = "https://overpass-api.de/api/interpreter"
BBOX = (32.476, 34.962, 32.496, 34.988)
CACHE = ".cache/maabarot_osm.json"
UA = {"User-Agent": "derech-kitzur-maabarot/1.0 (https://github.com/orimosenzon/fun)"}

# Read off the slide, in walking order. A name is a station; None is a turn
# the route has to pass through so the shortest path takes the right street.
ANCHORS = [
    ("1", (32.489294, 34.977078)),
    ("2", (32.489927, 34.977722)),
    ("3", (32.490425, 34.978312)),
    ("4", (32.491239, 34.977454)),
    (None, (32.492009, 34.976542)),   # end of the industrial zone's main street
    ("5", (32.491212, 34.975684)),
    ("6", (32.490561, 34.974986)),
    (None, (32.489067, 34.976864)),   # back at the corner of the zone (J)
    ("7", (32.486760, 34.974343)),
    ("8", (32.485809, 34.972894)),
    (None, (32.484723, 34.971178)),   # the corner on דרך הנדיב
    ("9", (32.485945, 34.970534)),
]

# The station titles, from the same slide.
STATIONS = {
    "1": "סיפור המעברות",
    "2": "המעברה: מבחר סיפורים",
    "3": "המעברה: מבחר סיפורים",
    "4": "המחנה הבריטי",
    "5": "מחנה העולים",
    "6": "מהמעברה למרכז תעסוקה",
    "7": "תצפית על השטח",
    "8": "כל הסיפור בתמצית",
    "9": "הסיפור והתקומה",
}


def haversine(a, b):
    r = 6371000
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def osm_ways():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    s, w, n, e = BBOX
    q = f"[out:json][timeout:60];way({s},{w},{n},{e})[highway];out geom;"
    req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        doc = json.load(r)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    return doc


def graph(doc):
    g = nx.Graph()
    for w in doc["elements"]:
        if w["tags"].get("highway") in ("motorway", "trunk", "construction", "proposed"):
            continue
        pts = [(p["lat"], p["lon"]) for p in w["geometry"]]
        for nid, pt in zip(w["nodes"], pts):
            g.add_node(nid, pt=pt)
        for a, b, pa, pb in zip(w["nodes"], w["nodes"][1:], pts, pts[1:]):
            g.add_edge(a, b, weight=haversine(pa, pb))
    return g


def nearest(g, pt):
    nid = min(g.nodes, key=lambda n: haversine(g.nodes[n]["pt"], pt))
    return nid, haversine(g.nodes[nid]["pt"], pt)


def build():
    g = graph(osm_ways())
    snapped = []
    for name, pt in ANCHORS:
        nid, off = nearest(g, pt)
        snapped.append((name, nid, off))

    path = []
    for (_, a, _), (_, b, _) in zip(snapped, snapped[1:]):
        legs = nx.shortest_path(g, a, b, weight="weight")
        pts = [list(map(lambda v: round(v, 6), g.nodes[n]["pt"])) for n in legs]
        path += pts if not path else pts[1:]
    path = simplify(path, 2.0)
    length = round(path_length(path))

    waypoints = []
    for name, nid, off in snapped:
        if not name:
            continue
        lat, lng = (round(v, 6) for v in g.nodes[nid]["pt"])
        waypoints.append({
            "id": f"{LAYER_ID}-w{name}",
            "name": f"תחנה {name}: {STATIONS[name]}",
            "note": "המיקום משוער, לפי מפת המסלול של בית הראשונים ועמותת המעברות.",
            "photos": [],
            "lat": lat, "lng": lng,
            "entries": [{"lat": lat, "lng": lng}],
            "color": COLOR,
            "by": "בוטי",
            "links": [{"url": ARTICLE, "title": "שביל המעברות בפרדספדיה"}],
        })

    segment = {
        "id": f"{LAYER_ID}-0",
        "name": "שביל המעברות",
        "note": ("שביל מורשת שמנציח את מחנה העולים בפרדס חנה ואת המעברה שקמה בו. "
                 "תשע תחנות, מדרך הנדיב דרך השדות ועד אזור התעסוקה. נחנך ב-11 בנובמבר 2022."),
        "photos": [{"thumb": "https://pardespedia.info/images/6/6f/%D7%A9%D7%91%D7%99%D7%9C_%D7%94%D7%9E%D7%A2%D7%91%D7%A8%D7%95%D7%AA_-_%D7%9E%D7%A1%D7%9C%D7%95%D7%9C.jpg",
                    "full": "https://pardespedia.info/images/6/6f/%D7%A9%D7%91%D7%99%D7%9C_%D7%94%D7%9E%D7%A2%D7%91%D7%A8%D7%95%D7%AA_-_%D7%9E%D7%A1%D7%9C%D7%95%D7%9C.jpg"}],
        "links": [{"url": ARTICLE, "title": "שביל המעברות בפרדספדיה"}],
        "path": path,
        "length": length,
        "color": COLOR,
        "layer": LAYER_ID,
        "by": "בוטי",
        "grade": "",
        "kind": "",
        "status": "קיים",
        "streets": [],
        "entries": [{"lat": path[0][0], "lng": path[0][1]},
                    {"lat": path[-1][0], "lng": path[-1][1]}],
    }

    layer = {
        "id": LAYER_ID,
        "name": "שביל המעברות",
        "short": "מעברות",
        "color": COLOR,
        "dash": False,
        "note": ("שביל המורשת של מחנה העולים והמעברה, עם תשע התחנות שלו. "
                 "המסלול לא פורסם כהקלטה: הוא משוחזר ממפת המסלול במצגת של בית הראשונים "
                 "ועמותת המעברות, לאורך הדרכים הקיימות. מיקומי התחנות משוערים."),
        "credit": "בית הראשונים ועמותת המעברות · OpenStreetMap",
        "on": False,
        "segments": [segment],
        "waypoints": waypoints,
    }
    return layer, snapped


def main():
    ap = argparse.ArgumentParser(description="שכבת שביל המעברות")
    ap.add_argument("--write", action="store_true", help="למזג ל-web/data/layers.json")
    ap.add_argument("--publish", action="store_true",
                    help="גם לפרסם את layers.json בריפו הנתונים, שממנו האפליקציה קוראת")
    a = ap.parse_args()
    layer, snapped = build()
    seg = layer["segments"][0]
    print(f"{seg['length']} מ׳, {len(seg['path'])} נקודות")
    for name, _, off in snapped:
        print(f"  {name or '·'}: הוצמד לדרך במרחק {off:.0f} מ׳")
    if a.write or a.publish:
        print(f"layers.json: {merge(layer)} שכבות")
    if a.publish:
        # The app reads layers.json from the data repo, not from web/data, so
        # a layer that is only written locally shows up under ?local and
        # nowhere else.
        from data_repo import publish_json
        with open("web/data/layers.json", encoding="utf-8") as fh:
            doc = json.load(fh)
        print("ריפו הנתונים: " + publish_json(
            "data/layers.json", doc, "שכבת שביל המעברות (build_maabarot.py, בוטי)",
            compact=True))


if __name__ == "__main__":
    main()
