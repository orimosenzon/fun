#!/usr/bin/env python3
"""Build web/data/gaps.json: public ground that exists on paper and not on the ground.

Ori, 13/11/2026: "אנחנו צריכים לדעת על הפקעות ישנות עבור דרכים וגם איפה אפשר
לדרוש מהמועצה לפתוח שביל חדש". Three layers this app already has answer it
when laid over one another:

    landuse.json     what the approved plans designate each lot for
    ownership.json   who the land registry says owns each parcel
    walknet.json     what anyone can walk on today (OSM), plus the
    trails.json      shortcuts the initiative has mapped

and a fourth, read here from the Israel OSM extract build_marked.py keeps in
.cache/: the footprint of every building.

For every lot designated as a path, a road or a public open space (שצ"פ) it
asks how much of it is owned by the public (the council, or the state through
רמ"י), how much of its length has anything walkable along it, and how much of
it is under a building. The gaps between the answers are the findings:

    closed      a path the public already owns, with nothing to walk on. The
                council can open it by a decision: no expropriation, no
                compensation. The strongest thing דרך קיצור can ask for.
    unexp_path  a path still in private hands and not walkable: the plan
                designated it, nobody ever expropriated it.
    unexp_road  the same for a road, if it is wider than a widening strip
                along a street that is already there (SLIVER). Roads that
                are in private hands but already paved are counted, not
                drawn: they are a registration backlog, not a missing road.
    open_priv   a שצ"פ still in private hands.
    open_built  a שצ"פ with buildings on a tenth of it or more.

Neighbouring path lots of the same finding are joined into one corridor, and
each corridor is given the walk it would save: the walking distance between its
two ends today, on the network, less its own length. That is the order the list
is in.

What it cannot see
------------------
* A path that is walked but not in OSM looks closed. The first sample against
  the aerial photo (13/11/2026): of the twelve corridors that save the most,
  nine were really closed or overgrown, two were private strips between
  houses, one was a strip along the railway with no registered owner. Lots whose
  ground is mostly unregistered are left out for that reason.
* A fence. OSM has six in the whole moshava.
* Whether a "closed" path was closed by a neighbour or was never opened. The
  ground answers that; the list is where to go and look.
* The date and purpose of the expropriation, for the ones that were done.
  That is in ילקוט הפרסומים, not here.

    python3 build_gaps.py              # the buildings from .cache/gaps/osm.json
    python3 build_gaps.py --refresh    # read them off .cache/il.osm.pbf again
"""

import collections
import heapq
import json
import math
import os
import sys
import time
from datetime import date

from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "web", "data")
OUT = os.path.join(D, "gaps.json")
CACHE = os.path.join(HERE, ".cache", "gaps")
PBF = os.path.join(HERE, ".cache", "il.osm.pbf")

TO_ITM = Transformer.from_crs(4326, 2039, always_xy=True).transform
TO_WGS = Transformer.from_crs(2039, 4326, always_xy=True).transform

PUBLIC = ("m", "i", "l")          # council, Israel lands, Israel lands leased (an estimate)
WALK_NEAR = 3                     # m: a walkable line this close to a lot runs along it
NOT_WALKED = 0.3                  # under this share of its length walkable: closed
WALKED = 0.6
SLIVER = 3                        # m: a road lot narrower than this is a widening strip
BUILT = 0.1                       # share of a שצ"פ under buildings
REGISTERED = 0.5                  # below this share of a lot in the registry: unknown, left out
SNAP = 35                         # m from a corridor's end to the network
SEARCH = 3000                     # m: farther than that on the network, give up

CLASSES = {
    "closed": "שביל ציבורי בלי מעבר",
    "unexp_path": "שביל שלא הופקע",
    "unexp_road": "דרך שלא הופקעה ולא נסללה",
    "open_priv": 'שצ"פ בבעלות פרטית',
    "open_built": 'שצ"פ שבנוי בחלקו',
}


def itm(geom):
    return transform(TO_ITM, geom)


def load(name):
    with open(os.path.join(D, name), encoding="utf-8") as fh:
        return json.load(fh)


def buildings(refresh):
    path = os.path.join(CACHE, "osm.json")
    if os.path.exists(path) and not refresh:
        with open(path) as fh:
            return json.load(fh)["buildings"]
    import osmium
    from osmium.filter import KeyFilter
    s, w, n, e = 32.43, 34.92, 32.51, 35.02
    out = []
    t = time.time()
    fp = osmium.FileProcessor(PBF).with_locations().with_areas().with_filter(KeyFilter("building"))
    for o in fp:
        if not (o.is_area() and "building" in o.tags):
            continue
        try:
            for ring in o.outer_rings():
                c = [(nd.lon, nd.lat) for nd in ring]
                if c and s <= c[0][1] <= n and w <= c[0][0] <= e:
                    out.append({"t": o.tags["building"], "c": c})
        except Exception:      # a ring with a node outside the extract
            pass
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"buildings": out}, fh)
    print(f"{len(out)} בניינים מ-OSM ({time.time() - t:.0f} שניות)")
    return out


def network():
    """The walk graph in ITM, its lines, and the street names along it."""
    w = load("walknet.json")
    o, sc, p = w["o"], w["scale"], w["pts"]
    nodes = {}

    def node(i):
        if i not in nodes:
            nodes[i] = TO_ITM(o[1] + p[2 * i + 1] / sc, o[0] + p[2 * i] / sc)
        return nodes[i]

    edges = collections.defaultdict(list)
    lines, named = [], []
    for row in w["ways"]:
        ids = row[1:]
        for a, b in zip(ids, ids[1:]):
            d = math.dist(node(a), node(b))
            edges[a].append((b, d))
            edges[b].append((a, d))
        line = LineString([node(i) for i in ids])
        lines.append(line)
        if row[0] >= 0:
            named.append((line, w["names"][row[0]]))
    for seg in load("trails.json")["segments"]:
        lines.append(itm(LineString([(q[1], q[0]) for q in seg["path"]])))
    return nodes, edges, lines, named


def shortest(edges, a, b):
    dist, heap = {a: 0}, [(0, a)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == b:
            return d
        if d > SEARCH:
            return None
        if d > dist[u]:
            continue
        for v, w in edges[u]:
            if d + w < dist.get(v, math.inf):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return None


def ends(geom):
    """The midpoints of the short sides of the smallest rectangle around it."""
    c = list(geom.minimum_rotated_rectangle.exterior.coords)
    sides = sorted(range(4), key=lambda i: math.dist(c[i], c[i + 1]))
    mid = lambda i: ((c[i][0] + c[i + 1][0]) / 2, (c[i][1] + c[i + 1][1]) / 2)
    return mid(sides[0]), mid(sides[1])


def main():
    refresh = "--refresh" in sys.argv
    landuse = load("landuse.json")["features"]
    owners = [(itm(shape(f["geometry"])).buffer(0), f["properties"]["o"])
              for f in load("ownership.json")["features"] if "o" in f["properties"]]  # not the KKL forests
    owner_tree = STRtree([g for g, _ in owners])
    blds = [itm(Polygon(b["c"])).buffer(0) for b in buildings(refresh) if len(b["c"]) >= 4]
    bld_tree = STRtree(blds)
    nodes, edges, lines, named = network()
    line_tree = STRtree(lines)
    name_tree = STRtree([l for l, _ in named])
    node_ids = list(nodes)
    node_tree = STRtree([Point(nodes[i]) for i in node_ids])
    print(f"{len(landuse)} מגרשים, {len(owners)} חלקות, {len(blds)} בניינים, {len(lines)} קווי הליכה")

    def street(g):
        i = name_tree.nearest(g)
        return named[i][1] if named[i][0].distance(g) < 120 else ""

    def snap(pt):
        near = [(math.dist(nodes[node_ids[j]], pt), node_ids[j])
                for j in node_tree.query(Point(pt).buffer(SNAP))]
        near = [x for x in near if x[0] <= SNAP]
        return min(near) if near else None

    # ---- every public lot, measured
    lots = []
    for f in landuse:
        p = f["properties"]
        if p["k"] not in ("path", "newroad", "road", "open"):
            continue
        g = itm(shape(f["geometry"])).buffer(0)
        if g.area < 20:
            continue
        by = collections.Counter()
        for i in owner_tree.query(g):
            a = g.intersection(owners[i][0]).area
            if a > 0.5:
                by[owners[i][1]] += a
        area = g.area
        a, b = ends(g)
        length = math.dist(a, b) or 1
        near = g.buffer(WALK_NEAR)
        walked = sum(near.intersection(lines[i]).length for i in line_tree.query(near))
        built = sum(g.intersection(blds[i]).area for i in bld_tree.query(g))
        lots.append({
            "k": p["k"], "lot": p["l"], "plan": p["t"], "use": p["u"], "g": g, "area": area,
            "reg": sum(by.values()) / area,
            "pub": sum(by[k] for k in PUBLIC) / area, "mun": by["m"] / area,
            "priv": by["p"] / area, "walk": min(1, walked / length), "built": built / area,
            "width": area / length,
        })

    def finding(x):
        if x["reg"] < REGISTERED:
            return None
        if x["k"] == "path" and x["walk"] < NOT_WALKED:
            if x["pub"] >= 0.5:
                return "closed"
            if x["priv"] >= 0.5:
                return "unexp_path"
        if (x["k"] in ("newroad", "road") and x["priv"] >= 0.5 and x["walk"] < NOT_WALKED
                and x["width"] >= SLIVER):
            return "unexp_road"
        if x["k"] == "open":
            if x["priv"] >= 0.5:
                return "open_priv"
            if x["built"] >= BUILT:
                return "open_built"
        return None

    for x in lots:
        x["c"] = finding(x)

    # ---- join neighbouring lots of the same finding; measure each corridor
    feats = []
    for cls in CLASSES:
        mine = [x for x in lots if x["c"] == cls]
        if not mine:
            continue
        tree = STRtree([x["g"] for x in mine])
        merged = unary_union([x["g"].buffer(1) for x in mine])
        for part in getattr(merged, "geoms", [merged]):
            members = [mine[i] for i in tree.query(part)
                       if part.contains(mine[i]["g"].representative_point())]
            if not members:
                continue
            geom = unary_union([m["g"] for m in members])
            area = sum(m["area"] for m in members)
            share = lambda k: round(sum(m[k] * m["area"] for m in members) / area, 2)
            props = {
                "c": cls, "a": round(area), "pub": share("pub"), "mun": share("mun"),
                "priv": share("priv"), "walk": share("walk"), "built": share("built"),
                "plans": sorted({m["plan"] for m in members}),
                "lots": [m["lot"] for m in members], "st": street(geom.centroid),
            }
            if cls in ("closed", "unexp_path"):
                a, b = ends(geom)
                props["len"] = round(math.dist(a, b))
                na, nb = snap(a), snap(b)
                props["ends"] = (na is not None) + (nb is not None)
                if na and nb and na[1] != nb[1]:
                    net = shortest(edges, na[1], nb[1])
                    if net is not None:
                        props["net"] = round(net)
                        props["save"] = max(0, round(net - (na[0] + props["len"] + nb[0])))
            c = geom.centroid
            props["at"] = [round(v, 6) for v in TO_WGS(c.x, c.y)[::-1]]
            out = transform(TO_WGS, geom.simplify(0.2))
            feats.append({"type": "Feature", "properties": props,
                          "geometry": json.loads(json.dumps(mapping(out)), parse_float=lambda s: round(float(s), 6))})

    feats.sort(key=lambda f: (list(CLASSES).index(f["properties"]["c"]),
                              -f["properties"].get("save", -1), -f["properties"]["a"]))

    # ---- the summary the page opens with: every public designation by owner and by state
    summary = {}
    for k in ("path", "newroad", "road", "open"):
        mine = [x for x in lots if x["k"] == k and x["reg"] >= REGISTERED]
        dunam = lambda xs: round(sum(x["area"] for x in xs) / 1000, 1)
        summary[k] = {
            "lots": len(mine), "dunam": dunam(mine),
            "priv": dunam([x for x in mine if x["priv"] >= 0.5]),
            "priv_walked": dunam([x for x in mine if x["priv"] >= 0.5 and x["walk"] >= WALKED]),
            "pub_closed": dunam([x for x in mine if x["pub"] >= 0.5 and x["walk"] < NOT_WALKED]),
            "built": dunam([x for x in mine if x["built"] >= BUILT]),
        }
    counts = collections.Counter(f["properties"]["c"] for f in feats)
    doc = {
        "type": "FeatureCollection", "version": 1, "updated": date.today().isoformat(),
        "classes": {k: {"name": v, "n": counts[k]} for k, v in CLASSES.items()},
        "summary": summary,
        "source": "ייעודי קרקע: הקומפילציה של הוועדה המקומית ו-Xplan · בעלות: פנקסי המקרקעין דרך govmap · "
                  "מעבר ובניינים: OpenStreetMap ומיפוי דרך קיצור",
        "features": feats,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, separators=(",", ":"))
    print({k: counts[k] for k in CLASSES})
    for k, v in summary.items():
        print(k, v)
    print(f"{os.path.getsize(OUT) / 1024:.0f} KB -> {OUT}")


if __name__ == "__main__":
    main()
