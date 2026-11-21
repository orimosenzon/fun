"""Build web/data/hoods.json: the moshava's neighbourhoods, as the app draws them.

The division is the one Irit Rotem drew in November 2026 and pardespedia
adopted on 21/11/2026: sixteen neighbourhoods, nine in Pardes Hanna and seven
in Karkur. Pardespedia keeps it as its single source in
pardespedia/hood_maps/hoods.json (hood_extract.py writes it from her map), and
its colours in PALETTE in pardespedia/hood_maps.py, the colours of the wiki's
clickable map. Both are read from there rather than copied, so a correction
made for the wiki reaches this layer on the next build.

What is written is a places document (the shape layers.js already knows from
the blocks and the plans): one place per neighbourhood, with its outline, the
point its name is written at, its colour, its district as the group, its area,
its neighbours and its pardespedia article.

    python3 build_hoods.py

No network. The file is served next to the app (Store.load reads it the way
it reads the canopy and the tree survey), not from the data repo: it is a
build product, and nothing in the app edits it.
"""
import ast
import json
import os
import re
import urllib.parse

from shapely.geometry import Polygon
from shapely.ops import polylabel

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.join(HERE, "..", "pardespedia")
SRC = os.path.join(PP, "hood_maps", "hoods.json")
PALETTE_SRC = os.path.join(PP, "hood_maps.py")
OUT = os.path.join(HERE, "web", "data", "hoods.json")
WIKI = "https://pardespedia.info/wiki/"


def palette():
    """PALETTE from hood_maps.py, read as a literal: importing the module would
    pull in its rendering dependencies for the sake of sixteen colours."""
    text = open(PALETTE_SRC, encoding="utf-8").read()
    m = re.search(r"^PALETTE\s*=\s*(\{.*?\})", text, re.S | re.M)
    if not m:
        raise SystemExit("PALETTE not found in hood_maps.py")
    return ast.literal_eval(m.group(1))


def article_url(title):
    return WIKI + urllib.parse.quote(title.replace(" ", "_"))


def main():
    src = json.load(open(SRC, encoding="utf-8"))
    colours = palette()
    places = []
    for h in src["hoods"]:
        name = h["name"]
        if name not in colours:
            raise SystemExit(f"no colour for {name} in hood_maps.PALETTE")
        ring = h["rings"][0]                       # one ring each; checked below
        if len(h["rings"]) != 1:
            raise SystemExit(f"{name}: {len(h['rings'])} rings, expected one")
        poly = Polygon([(lng, lat) for lat, lng in ring])
        if not poly.is_valid:
            poly = poly.buffer(0)
        # The name goes where the shape is widest, not at its centroid, which
        # for a crescent like מזרח כרכור falls outside it.
        at = polylabel(poly, tolerance=0.00002)
        shape = [[[round(x, 6), round(y, 6)] for x, y in poly.exterior.coords]]
        # A merged neighbourhood (יובלים וקורן) has no article of its own but
        # one for each part; the first is the link, all of them are listed.
        parts = [p["name"] for p in h.get("parts", [])]
        article = h.get("article") or (h["parts"][0]["article"] if h.get("parts") else None)
        places.append({
            "id": "hood-" + re.sub(r"\W+", "-", name).strip("-"),
            "name": name,
            "group": h["district"],
            "color": colours[name],
            "cats": ["שכונה", h["district"]],
            "hood": True,
            "area_km2": h["area_km2"],
            "neighbours": h.get("neighbours", []),
            "parts": parts,
            "article": article,
            "url": article_url(article) if article else WIKI + urllib.parse.quote("רשימת_שכונות"),
            "photos": [],
            "geo": {"lat": round(at.y, 6), "lng": round(at.x, 6), "source": "hood"},
            "shape": shape,
        })
    doc = {
        "version": 1,
        "source": "https://pardespedia.info/wiki/" + urllib.parse.quote("רשימת_שכונות"),
        "name": "שכונות",
        "about": src.get("source", ""),
        "groups": [],
        "places": places,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{len(places)} neighbourhoods -> {os.path.relpath(OUT, HERE)} "
          f"({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    main()
