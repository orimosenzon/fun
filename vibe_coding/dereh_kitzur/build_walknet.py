#!/usr/bin/env python3
"""Build the walking network the in-app router runs on: web/data/walknet.json.

The question somebody actually asks is not "where is trail 17" but "how do I
get from here to the kindergarten". Google cannot answer it properly here,
because 83% of the shortcuts are on no map Google has seen, so it routes
*around* them. This app can: it has the shortcuts, and with the streets under
them it has the whole graph.

This script supplies the streets. Everything a person may walk on in and around
the moshava, from OpenStreetMap through Overpass, written as a compact graph:

    {
      "o":     [lat0, lng0],           origin of the integer grid
      "scale": 100000,                 1e-5 degrees, about 1.1 m
      "pts":   [dlat, dlng, ...],      every node, as integers off the origin
      "names": ["הנשיא", ...],         street names, once each
      "ways":  [[name, i, j, k], ...]  name index (-1: none), then node indices
    }

Consecutive nodes in a way are an edge. The browser joins the shortcuts on
(see web/route.js) - they are not in here, because they change every time a
resident sends one in and this file changes when OSM does.

    python3 build_walknet.py [--refresh]

The raw Overpass answer is cached in .cache/walknet/; --refresh asks again.
"""

import json
import math
import os
import sys

import build_curitiba
from build_curitiba import overpass

OUT = "web/data/walknet.json"
LANDUSE = "web/data/landuse.json"
SCALE = 100000

# Roads nobody walks along. A trunk road here is Route 65, which has no
# pavement for most of its length and is not where a shortcut ends anyway.
NO_WALK = {"motorway", "motorway_link", "trunk", "trunk_link", "construction",
           "proposed", "raceway", "bus_guideway", "abandoned", "platform",
           "corridor", "elevator", "busway"}
NO_ACCESS = {"no", "private"}

# Around the moshava rather than exactly on it: a route from the edge of town to
# the station at Binyamina still starts inside, and its first few hundred metres
# may run outside the municipal line.
MARGIN = 0.006


def moshava_bbox():
    """The extent of the land-use layer, which covers the whole municipality."""
    with open(LANDUSE, encoding="utf-8") as fh:
        doc = json.load(fh)
    s = w = math.inf
    n = e = -math.inf

    def walk(c):
        nonlocal s, w, n, e
        if isinstance(c[0], (int, float)):
            lng, lat = c[0], c[1]
            s, n, w, e = min(s, lat), max(n, lat), min(w, lng), max(e, lng)
        else:
            for x in c:
                walk(x)

    for f in doc["features"]:
        walk(f["geometry"]["coordinates"])
    return s - MARGIN, w - MARGIN, n + MARGIN, e + MARGIN


def walkable(tags):
    hw = tags.get("highway")
    if not hw or hw in NO_WALK:
        return False
    if tags.get("area") == "yes":
        return False
    foot = tags.get("foot")
    if foot in ("yes", "designated", "permissive"):
        return True
    if foot in NO_ACCESS or tags.get("access") in NO_ACCESS:
        return False
    # A driveway is somebody's yard; the service road behind the shops is not.
    if hw == "service" and tags.get("service") == "driveway":
        return False
    return True


def main():
    refresh = "--refresh" in sys.argv
    s, w, n, e = moshava_bbox()
    print(f"תחום: {s:.4f},{w:.4f} עד {n:.4f},{e:.4f}")
    query = f"""[out:json][timeout:300];
way["highway"]({s},{w},{n},{e});
out tags geom;"""
    build_curitiba.CACHE = ".cache/walknet"   # overpass() caches under its module's CACHE
    doc = overpass("ways", query, refresh)

    o_lat, o_lng = round(s, 3), round(w, 3)
    index = {}                     # (ilat, ilng) -> node index; joins ways at shared nodes
    pts, names, name_ix, ways = [], [], {}, []
    skipped = 0

    for el in doc["elements"]:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        if not walkable(tags):
            skipped += 1
            continue
        name = tags.get("name:he") or tags.get("name") or ""
        if name and name not in name_ix:
            name_ix[name] = len(names)
            names.append(name)
        row = [name_ix[name] if name else -1]
        for g in el["geometry"]:
            key = (round((g["lat"] - o_lat) * SCALE), round((g["lon"] - o_lng) * SCALE))
            i = index.get(key)
            if i is None:
                i = index[key] = len(pts) // 2
                pts.extend(key)
            if row[-1] != i or len(row) == 1:
                row.append(i)
        if len(row) > 2:
            ways.append(row)

    out = {"version": 1, "source": "OpenStreetMap contributors, ODbL",
           "o": [o_lat, o_lng], "scale": SCALE, "pts": pts,
           "names": names, "ways": ways}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(OUT) / 1024
    edges = sum(len(r) - 2 for r in ways)
    print(f"{len(ways)} דרכים, {len(pts) // 2} צמתים, {edges} קשתות, "
          f"{len(names)} שמות, {skipped} דרכים שלא הולכים בהן. {size:.0f} KB -> {OUT}")


if __name__ == "__main__":
    main()
