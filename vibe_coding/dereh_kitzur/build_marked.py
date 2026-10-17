#!/usr/bin/env python3
"""Build data/marked.json: every marked hiking trail in Israel, from OpenStreetMap.

The trails are marked on the ground by the Israel Trails Committee (הוועדה
לשבילי ישראל, ITC) in four colours - red, blue, green, black - each a colour
stripe between two white ones, plus the long trails with marks of their own:
שביל ישראל, שביל ישו, שביל הגולן, שביל הים, שביל הסנהדרין and some forty
others. The committee does not publish them as open data; its maps are its
own. OpenStreetMap carries them anyway, mapped by the Israel Hiking Map
community, which keeps them in step with the committee's markings: of the
1,920 hiking routes in the Israel extract (16/10/2026), 1,785 carry
operator=itc and almost all an `osmc:symbol` such as `red:white:red_stripe`,
which is the mark itself written down. That is the colour drawn here.

"As up to date as possible" (Ori, 17/10/2026) is the reason for the source.
Geofabrik rebuilds the Israel extract every night from the live OSM database,
so a trail the committee marks or reroutes reaches this layer the night after
somebody maps it. Overpass would do the same for one area, but the whole
country is a heavy query and the public server turns it away when busy; the
extract is 120 MB, a plain download that does not fail that way.

One feature per route relation: its ways merged into as few lines as they
make, simplified to about 3 m and rounded to 1 m, so the country fits in a
few megabytes. The app fetches the file only when the layer is switched on.
A route that is part of another one (שביל ישו is two relations, שביל חיפה a
superroute of three) is drawn as itself; one with no ways of its own (the
superroute) only lends its name to an unnamed child.

Most committee trails have no name in OSM, only their colour, so each is
named after its colour and the nearest settlement: "שביל אדום ליד עין הוד".

    python3 build_marked.py            # download if stale, write web/data/marked.json
    python3 build_marked.py --publish  # also put it in the data repo (nightly cron)
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from datetime import date

import osmium
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

HERE = os.path.dirname(os.path.abspath(__file__))
PBF_URL = "https://download.geofabrik.de/asia/israel-and-palestine-latest.osm.pbf"
PBF = os.path.join(HERE, ".cache", "il.osm.pbf")
OUT = os.path.join(HERE, "web", "data", "marked.json")
STALE_H = 20                 # Geofabrik publishes once a night, around 23:20 UTC

# The mark's colour as the trail sign has it. A little deeper than the pure
# hues, so that the white casing around them reads as the white stripes.
COLOURS = {
    "red": "#e53935", "blue": "#1e88e5", "green": "#2e9e44", "black": "#222222",
    "orange": "#fb8c00", "purple": "#8e24aa", "yellow": "#fdd835",
    "brown": "#8d6e63", "white": "#ffffff", "gray": "#9e9e9e", "grey": "#9e9e9e",
    "maroon": "#8e2430", "olive": "#808000",
}
COLOUR_HE = {
    "red": "אדום", "blue": "כחול", "green": "ירוק", "black": "שחור",
    "orange": "כתום", "purple": "סגול", "yellow": "צהוב", "brown": "חום",
    "maroon": "בורדו", "olive": "זית",
}
NO_COLOUR = "#6d4c41"


def download():
    """Fetch the extract unless the copy here is from the last night's build."""
    if os.path.exists(PBF) and time.time() - os.path.getmtime(PBF) < STALE_H * 3600:
        return
    os.makedirs(os.path.dirname(PBF), exist_ok=True)
    tmp = PBF + ".part"
    req = urllib.request.Request(PBF_URL, headers={"User-Agent": "derech-kitzur/1.0"})
    with urllib.request.urlopen(req, timeout=600) as res, open(tmp, "wb") as f:
        while True:
            chunk = res.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, PBF)


def mark_of(tags):
    """(main colour name, stripe colour name or None) of a route's mark.

    `osmc:symbol` is waycolour:background[:foreground...]. The committee's
    own trails are `red:white:red_stripe`; a long trail has a mark of its own,
    שביל ישראל `orange:orange:white_right:blue_stripe` (orange, white and
    blue), שביל הגולן `green:green:white_right:blue_stripe`."""
    sym = (tags.get("osmc:symbol") or "").split(":")
    main = sym[0] if sym and sym[0] in COLOURS else None
    if not main:
        c = (tags.get("colour") or "").lower()
        main = c if c in COLOURS else None
    stripe = None
    for part in sym[2:]:
        if part.endswith("_stripe"):
            s = part[:-len("_stripe")]
            if s in COLOURS and s != main:
                stripe = s
    return main, stripe


class Relations(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.rels = {}

    def relation(self, r):
        t = {k: v for k, v in r.tags}
        if t.get("route") not in ("hiking", "foot"):
            return
        self.rels[r.id] = {
            "tags": t,
            "ways": [m.ref for m in r.members if m.type == "w"],
            "kids": [m.ref for m in r.members if m.type == "r"],
        }


class Ways(osmium.SimpleHandler):
    def __init__(self, wanted):
        super().__init__()
        self.wanted = wanted
        self.geom = {}
        self.places = []

    def node(self, n):
        place = n.tags.get("place")
        if place in ("city", "town", "village", "hamlet", "suburb", "neighbourhood"):
            name = n.tags.get("name:he") or n.tags.get("name")
            if name:
                weight = {"city": 0, "town": 0, "village": 1, "hamlet": 1}.get(place, 2)
                self.places.append((n.location.lon, n.location.lat, name, weight))

    def way(self, w):
        if w.id not in self.wanted:
            return
        try:
            self.geom[w.id] = [(n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            pass


def simplify(geom):
    """About 3 m, in degrees at Israel's latitude, then rounded to about 1 m."""
    g = geom.simplify(0.00003, preserve_topology=False)
    lines = list(g.geoms) if isinstance(g, MultiLineString) else [g]
    out = []
    for ln in lines:
        pts = []
        for x, y in ln.coords:
            p = [round(x, 5), round(y, 5)]
            if not pts or pts[-1] != p:
                pts.append(p)
        if len(pts) > 1:
            out.append(pts)
    return out


def length_m(parts):
    total = 0.0
    for pts in parts:
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            k = math.cos(math.radians((y1 + y2) / 2))
            total += math.hypot((x2 - x1) * k, y2 - y1) * 111320
    return total


def nearest_place(places, lon, lat):
    best, bd = None, 1e9
    k = math.cos(math.radians(lat))
    for x, y, name, weight in places:
        # A town counts a little nearer than a neighbourhood the same distance
        # off: "ליד זכרון יעקב" says more than "ליד שכונת הנדיב".
        d = math.hypot((x - lon) * k, y - lat) * 111.32 + weight * 0.6
        if d < bd:
            bd, best = d, name
    return best


def build():
    download()
    rh = Relations()
    rh.apply_file(PBF)
    rels = rh.rels
    wanted = {w for r in rels.values() for w in r["ways"]}
    wh = Ways(wanted)
    wh.apply_file(PBF, locations=True, idx="flex_mem")

    parent = {}
    for rid, r in rels.items():
        for kid in r["kids"]:
            parent.setdefault(kid, rid)

    features = []
    counts = {}
    for rid, r in sorted(rels.items()):
        t = r["tags"]
        lines = [LineString(wh.geom[w]) for w in r["ways"] if len(wh.geom.get(w, [])) > 1]
        if not lines:
            continue
        merged = linemerge(MultiLineString(lines))
        parts = simplify(merged)
        if not parts:
            continue
        main, stripe = mark_of(t)
        up = rels.get(parent.get(rid), {}).get("tags", {})
        name = (t.get("name:he") or t.get("name") or up.get("name:he") or up.get("name") or "")
        net = t.get("network") or ""
        long_trail = net in ("nwn", "iwn", "rwn")
        # The middle of the longest part, for the name of the nearest town and
        # for the detail pane's pin.
        longest = max(parts, key=len)
        mid = longest[len(longest) // 2]
        near = nearest_place(wh.places, mid[0], mid[1])
        props = {
            "id": rid,
            "n": name,
            "c": COLOURS.get(main, NO_COLOUR),
            "k": "l" if long_trail else "t",      # long trail / committee-style trail
            "m": main or "",
            "km": round(length_m(parts) / 1000, 1),
            "near": near or "",
        }
        if stripe:
            props["s"] = COLOURS[stripe]
        if t.get("name:en") or up.get("name:en"):
            props["en"] = t.get("name:en") or up.get("name:en")
        # The mark as OSM writes it, which the app turns into a sentence
        # about what to look for on the rocks.
        if t.get("osmc:symbol"):
            props["sym"] = t["osmc:symbol"]
        for key, short in (("ref", "ref"), ("operator", "op"), ("website", "web"),
                           ("wikipedia:he", "wiki"), ("description:he", "d"),
                           ("description", "d"), ("roundtrip", "round")):
            if t.get(key) and short not in props:
                props[short] = t[key]
        if props.get("wiki", "").startswith("he:"):
            props["wiki"] = props["wiki"][3:]
        geometry = ({"type": "LineString", "coordinates": parts[0]} if len(parts) == 1
                    else {"type": "MultiLineString", "coordinates": parts})
        features.append({"type": "Feature", "id": rid, "properties": props,
                         "geometry": geometry})
        counts[props["m"] or "?"] = counts.get(props["m"] or "?", 0) + 1

    # Long trails last, so that they are drawn over the local ones they share
    # a path with - the order of a GeoJSON source is its drawing order.
    features.sort(key=lambda f: f["properties"]["k"] == "l")
    # The moment of the OSM database the extract was cut from, which is what
    # "up to date" means here - not when this machine downloaded it.
    header = osmium.io.Reader(PBF, osmium.osm.osm_entity_bits.NOTHING).header()
    stamp = (header.get("osmosis_replication_timestamp") or "")[:10] or \
        time.strftime("%Y-%m-%d", time.gmtime(os.path.getmtime(PBF)))
    doc = {
        "type": "FeatureCollection",
        "name": "שבילים מסומנים",
        "source": "OpenStreetMap, דרך Geofabrik",
        "osm": stamp,
        "updated": date.today().isoformat(),
        "features": features,
    }
    return doc, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--publish", action="store_true",
                        help="also write data/marked.json to the data repo")
    args = parser.parse_args()
    doc, counts = build()
    body = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(body)
    km = sum(f["properties"]["km"] for f in doc["features"])
    longs = sum(1 for f in doc["features"] if f["properties"]["k"] == "l")
    print(f"{len(doc['features'])} trails ({longs} long), {km:,.0f} km, "
          f"{len(body) / 1e6:.1f} MB, OSM of {doc['osm']}")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))
    if args.publish:
        sys.path.insert(0, HERE)
        from data_repo import publish_json
        print(publish_json("data/marked.json", doc,
                           f"שבילים מסומנים: עדכון מ-OSM של {doc['osm']}",
                           ignore=("updated", "osm"), compact=True))


if __name__ == "__main__":
    main()
