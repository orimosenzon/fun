#!/usr/bin/env python3
"""Build web/data/ownership.json: who owns every registered parcel in the moshava.

Ori, 20/10/2026: a layer "בעלות קרקע" - which ground is the council's, which
is קק"ל's, which is the state's (רמ"י) - and anything else about ownership that
can be had.

build_public.py said in September that ownership "is not available". It is,
one level down from where we looked: govmap's viewer has a layer "סוג בעלות
בחלקות רשומות" (layer 6, `parcel_ownership_new`), which the open WFS does not
carry. Its rows are the parcels of the national cadastre with the land
registry's summary of who holds them:

    פרטית            private owners
    רשות מקומית       the local authority - here, the council
    מקרקעי ישראל      "Israel lands" in the sense of חוק יסוד: מקרקעי ישראל: the
                     state, קק"ל and רשות הפיתוח together, all three managed by
                     רמ"י
    מעורבת           more than one kind of owner in the same parcel

So the state and קק"ל are one colour here, and not by choice. The registry
knows which of the three it is - the national extract has a code per owner,
seen in קק"ל's own copies of it for a few Galilee industrial zones - but it is
not published for anywhere else, and a נסח טאבו per parcel is paid. The layer
says so rather than guess.

Two more kinds turned up in the moshava: "אחרת" (other) and "לא ידוע"
(unknown), a few dozen parcels between them.

Where it comes from
-------------------
The layer's WFS and GetFeatureInfo are closed (403). What is open is what the
viewer itself uses:

1. **WMS GetMap**, with a style of our own. govmap sends the WMS an SLD_BODY to
   highlight a parcel, so the server takes one; this one paints each kind of
   owner in a pure colour of its own (PAINT) with antialiasing off. The
   moshava is 36 tiles of 2048 pixels, under a metre a pixel, and every parcel
   in parcels.json (ids `block*10000+parcel`) is read off them at a few points
   inside it, well clear of its edges. A parcel whose points disagree is
   printed, and goes by the majority.

2. **entitiesByPoint**, the answer to a click:

       POST www.govmap.gov.il/api/layers-catalog/entitiesByPoint
            {"point": [x, y] (EPSG:3857), "layers": [{"layerId": "6"}],
             "tolerance": t}

   It names block, parcel and kind outright, but at most 100 parcels an answer
   and it starts answering 429 after about fifty requests; covering 9,126
   parcels that way is well over a thousand requests. It wants an
   `x-fingerprint-id`/`x-user-id` header, which the site makes up per
   browser; any hex string will do. Here it is the check: `--check N` asks it
   about N parcels picked at random and compares with what the images said.

Everything fetched is cached in .cache/ownership/, so a rebuild without
--refresh asks nothing.

קק"ל's forests
--------------
The other thing about ownership that is published openly: the forests קק"ל
manages, off its own ArcGIS service (`yaar_VIEW_OD`), cut to the municipal
boundary. Managed, which is not owned - a קק"ל forest can stand on state land
- and the layer's note says that too.

    python3 build_ownership.py             # from the cache where there is one
    python3 build_ownership.py --refresh   # ask govmap again
    python3 build_ownership.py --check 40  # and compare 40 parcels with the click answer
"""

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import urllib.parse
import urllib.request
import uuid
from collections import Counter

from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point, mapping, shape
from shapely.ops import transform

import build_public as bp

HERE = os.path.dirname(os.path.abspath(__file__))
PARCELS = os.path.join(HERE, "web", "data", "parcels.json")
OUT = os.path.join(HERE, "web", "data", "ownership.json")
CACHE = os.path.join(HERE, ".cache", "ownership")

API = "https://www.govmap.gov.il/api/layers-catalog/entitiesByPoint"
WMS = "https://www.govmap.gov.il/api/geoserver/ows/public/"
LAYER = "6"
LAYER_NAME = "govmap:layer_parcel_ownership_new"
TOLERANCE = 2           # EPSG:3857 units: a click on one parcel
TILE = 2000             # EPSG:3857 units a tile, ~1.7km on the ground
TILE_PX = 2048
PAUSE = 2

KKL = ("https://services2.arcgis.com/utNNrmXb4IZOLXXs/arcgis/rest/services/"
       "yaar_VIEW_OD/FeatureServer/0/query")

TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True).transform

# The registry's types -> the short code in the file, its colour, its name in
# the app. A type not in here fails the build: a new kind of owner is a
# decision, not something to paint grey.
KINDS = {
    "רשות מקומית": ("m", "#1565c0", "המועצה המקומית"),
    "מקרקעי ישראל": ("i", "#e65100", 'מקרקעי ישראל (מדינה, קק"ל, רשות הפיתוח)'),
    "פרטית": ("p", "#f2d16b", "פרטית"),
    "מעורבת": ("x", "#8e24aa", "מעורבת"),
    "אחרת": ("o", "#00897b", "אחרת"),
    "לא ידוע": ("u", "#9e9e9e", "לא ידוע"),
}

# The colour each kind is painted in on the images the WMS renders for us:
# pure and far apart, with antialiasing off, so a pixel is one kind or none.
PAINT = {
    "רשות מקומית": (0, 0, 255), "מקרקעי ישראל": (255, 0, 0), "פרטית": (0, 255, 0),
    "מעורבת": (255, 255, 0), "אחרת": (0, 255, 255), "לא ידוע": (128, 128, 128),
}
ELSE = (255, 0, 255)            # a kind nobody has told this file about

FINGERPRINT = uuid.uuid4().hex


def ask(x, y):
    body = json.dumps({"point": [x, y], "layers": [{"layerId": LAYER}],
                       "tolerance": TOLERANCE}).encode()
    headers = {
        "User-Agent": bp.UA, "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.govmap.gov.il", "Referer": "https://www.govmap.gov.il/",
        "x-fingerprint-id": FINGERPRINT, "x-user-id": FINGERPRINT,
        "x-trace-id": uuid.uuid4().hex,
    }
    for attempt in range(3):
        try:
            req = urllib.request.Request(API, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as handle:
                return json.loads(handle.read().decode("utf-8"))
        except Exception as err:                       # noqa: BLE001 - retry anything
            if attempt == 2:
                raise
            print(f"    ...{err}, מחכה חצי דקה", file=sys.stderr)
            time.sleep(30)
    return None


def rows(doc):
    """The parcels of one answer, as (id, fields by name)."""
    for layer in doc.get("data") or []:
        for ent in layer.get("entities") or []:
            at = {f["fieldName"]: f["fieldValue"] for f in ent["fields"]}
            if at.get("תת גוש"):
                continue                # no sub-blocks in the moshava's parcels
            yield int(at["גוש"]) * 10000 + int(at["חלקה"]), at


def sld():
    """A style that paints each kind of owner in its PAINT colour, flat."""
    def fill(colour):
        hexc = "#%02x%02x%02x" % colour
        return (f'<PolygonSymbolizer><Fill><CssParameter name="fill">{hexc}</CssParameter>'
                f'</Fill></PolygonSymbolizer>')
    rules = "".join(
        f"<Rule><ogc:Filter><ogc:PropertyIsEqualTo><ogc:PropertyName>ownership"
        f"</ogc:PropertyName><ogc:Literal>{name}</ogc:Literal></ogc:PropertyIsEqualTo>"
        f"</ogc:Filter>{fill(colour)}</Rule>" for name, colour in PAINT.items())
    rules += f"<Rule><ElseFilter/>{fill(ELSE)}</Rule>"
    return ('<?xml version="1.0" encoding="UTF-8"?><StyledLayerDescriptor version="1.0.0" '
            'xmlns="http://www.opengis.net/sld" xmlns:ogc="http://www.opengis.net/ogc">'
            f"<NamedLayer><Name>{LAYER_NAME}</Name><UserStyle><FeatureTypeStyle>{rules}"
            "</FeatureTypeStyle></UserStyle></NamedLayer></StyledLayerDescriptor>")


def tile(tx, ty, refresh):
    """The tile whose south-west corner is (tx, ty), from the cache or the WMS."""
    path = os.path.join(CACHE, f"tile_{tx}_{ty}.png")
    if refresh or not os.path.exists(path):
        body = urllib.parse.urlencode({
            "REQUEST": "GetMap", "SERVICE": "WMS", "VERSION": "1.1.1",
            "FORMAT": "image/png", "TRANSPARENT": "true", "LAYERS": LAYER_NAME,
            "SLD_BODY": sld(), "SRS": "EPSG:3857",
            "BBOX": f"{tx},{ty},{tx + TILE},{ty + TILE}",
            "WIDTH": TILE_PX, "HEIGHT": TILE_PX, "FORMAT_OPTIONS": "antialias:none",
        }).encode()
        headers = {"User-Agent": bp.UA, "Referer": "https://www.govmap.gov.il/",
                   "Content-Type": "application/x-www-form-urlencoded"}
        for attempt in range(3):
            try:
                req = urllib.request.Request(WMS, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as handle:
                    data = handle.read()
                Image.open(io.BytesIO(data)).verify()      # an error page is not a PNG
                break
            except Exception as err:                       # noqa: BLE001 - retry anything
                if attempt == 2:
                    raise
                print(f"    ...{err}, מחכה חצי דקה", file=sys.stderr)
                time.sleep(30)
        with open(path, "wb") as handle:
            handle.write(data)
        print(f"  אריח {tx},{ty}")
        time.sleep(PAUSE)
    return Image.open(path).convert("RGB")


def pull(features, refresh):
    """Every parcel's kind of owner, read off the WMS images.

    -> {parcel id: kind, or None where no point inside it was painted}."""
    os.makedirs(CACHE, exist_ok=True)
    by_colour = {colour: name for name, colour in PAINT.items()}
    shapes = {f["id"]: transform(TO_3857, shape(f["geometry"])) for f in features}
    x0 = min(s.bounds[0] for s in shapes.values()) // TILE * TILE
    y0 = min(s.bounds[1] for s in shapes.values()) // TILE * TILE
    tiles = {}
    known, split, strange = {}, [], Counter()
    rng = random.Random(5)
    for fid, poly in shapes.items():
        # A metre and a half clear of the edges where the parcel is wide enough
        # for it, less where it is a narrow strip of road.
        inner = poly
        for clear in (1.8, 0.8, 0.3):
            shrunk = poly.buffer(-clear)
            if not shrunk.is_empty:
                inner = shrunk
                break
        pts = [inner.representative_point()]
        minx, miny, maxx, maxy = inner.bounds
        for _ in range(60):
            if len(pts) >= 5:
                break
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if inner.contains(Point(x, y)):
                pts.append(Point(x, y))
        votes = Counter()
        for pt in pts:
            tx = int((pt.x - x0) // TILE * TILE + x0)
            ty = int((pt.y - y0) // TILE * TILE + y0)
            if (tx, ty) not in tiles:
                tiles[(tx, ty)] = tile(tx, ty, refresh)
            im = tiles[(tx, ty)]
            px = min(TILE_PX - 1, int((pt.x - tx) / TILE * TILE_PX))
            py = min(TILE_PX - 1, int((ty + TILE - pt.y) / TILE * TILE_PX))
            colour = im.getpixel((px, py))
            if colour in by_colour:
                votes[by_colour[colour]] += 1
            elif colour != (0, 0, 0) and colour != (255, 255, 255):
                strange[colour] += 1
        if len(votes) > 1:
            split.append((fid, dict(votes)))
        known[fid] = votes.most_common(1)[0][0] if votes else None
    print(f"  {len(tiles)} אריחים; {len(split)} חלקות שהנקודות בהן חלוקות")
    if strange:
        raise SystemExit(f"צבעים לא מוכרים על התמונות (סוג בעלות חדש?): {strange.most_common(5)}")
    # Where the points disagree the image cannot settle it - the registry's
    # outline of the parcel has moved since parcels.json, or a neighbour's
    # edge runs through it - so the click API is asked, a handful of requests.
    split += [(fid, {}) for fid, kind in known.items() if kind is None]
    for fid, votes in split:
        answer = clicked(shapes[fid].representative_point()).get(fid)
        print(f"    גוש {fid // 10000} חלקה {fid % 10000}: {votes} -> {answer or 'אין תשובה'}")
        if answer:
            known[fid] = answer
    return known, shapes


def clicked(pt):
    """The click API's answer at a point, through the cache: {parcel id: kind}."""
    key = hashlib.sha1(f"{pt.x:.1f},{pt.y:.1f},{TOLERANCE}".encode()).hexdigest()[:16]
    path = os.path.join(CACHE, f"{key}.json")
    if not os.path.exists(path):
        bp.save_json(path, ask(pt.x, pt.y))
        time.sleep(PAUSE)
    with open(path, encoding="utf-8") as handle:
        return {gid: at["סוג בעלות"] for gid, at in rows(json.load(handle))}


def check(known, shapes, n):
    """Ask the click API about n random parcels, and about any it answered for
    before, and compare. Exits on a disagreement."""
    truth = {}
    for name in os.listdir(CACHE):
        if name.endswith(".json"):
            with open(os.path.join(CACHE, name), encoding="utf-8") as handle:
                truth.update({gid: at["סוג בעלות"] for gid, at in rows(json.load(handle))})
    for fid in random.Random(11).sample(sorted(shapes), n):
        truth.update(clicked(shapes[fid].representative_point()))
    both = [fid for fid in truth if fid in known]
    wrong = [fid for fid in both if known[fid] != truth[fid]]
    print(f"  בדיקה מול תשובת הלחיצה: {len(both) - len(wrong)} מתוך {len(both)} חלקות תואמות")
    for fid in wrong[:15]:
        print(f"    גוש {fid // 10000} חלקה {fid % 10000}: בתמונה {known[fid]}, בלחיצה {truth[fid]}")
    if len(wrong) > len(both) * 0.01:
        raise SystemExit("יותר מאחוז אי-התאמות: לא כותב את הקובץ")


def forests(area):
    """קק"ל's forests inside the municipal boundary, cut to it."""
    query = urllib.parse.urlencode({
        "where": "1=1", "geometry": json.dumps(area), "geometryType": "esriGeometryPolygon",
        "inSR": 4326, "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FOR_Num,FOR_Name", "returnGeometry": "true", "outSR": 4326,
        "f": "geojson",
    })
    req = urllib.request.Request(KKL, data=query.encode(), headers={"User-Agent": bp.UA})
    with urllib.request.urlopen(req, timeout=120) as handle:
        doc = json.loads(handle.read().decode("utf-8"))
    town = shape({"type": "Polygon", "coordinates": area["rings"]}).buffer(0)
    out = []
    for f in doc.get("features", []):
        cut = shape(f["geometry"]).buffer(0).intersection(town)
        if cut.is_empty:
            continue
        # Web Mercator stretches by 1/cos(latitude) each way.
        m2 = transform(TO_3857, cut).area * math.cos(math.radians(32.47)) ** 2
        if m2 < 2000:                    # a sliver where a forest edge meets the line
            continue
        cut = cut.simplify(0.000003)
        out.append({
            "type": "Feature", "id": -int(f["properties"]["FOR_Num"]),
            "properties": {"k": "f", "n": f["properties"]["FOR_Name"], "a": round(m2)},
            "geometry": mapping(cut),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true", help="ask govmap again")
    parser.add_argument("--check", type=int, default=0, metavar="N",
                        help="compare N random parcels with the click API")
    args = parser.parse_args()

    with open(PARCELS, encoding="utf-8") as handle:
        parcels = json.load(handle)
    features = parcels["features"]
    print(f"{len(features)} חלקות ב-parcels.json")
    known, shapes = pull(features, args.refresh)
    check(known, shapes, args.check)

    out, tally, missing = [], {}, []
    for f in features:
        kind = known.get(f["id"])
        if not kind:
            missing.append(f["id"])
            continue
        code = KINDS[kind][0]
        props = dict(f["properties"], o=code)
        out.append({"type": "Feature", "id": f["id"], "properties": props,
                    "geometry": f["geometry"]})
        tally[code] = tally.get(code, [0, 0])
        tally[code][0] += 1
        tally[code][1] += props.get("a", 0)

    _, area = bp.boundary()
    woods = forests(area)

    print("\nלפי סוג בעלות:")
    for name, (code, _, label) in KINDS.items():
        n, m2 = tally.get(code, [0, 0])
        print(f"  {label}: {n} חלקות, {m2 / 1000:,.0f} דונם")
    print(f"  בלי נתון: {len(missing)} חלקות")
    print(f'יערות קק"ל בתחום: {", ".join(w["properties"]["n"] for w in woods) or "אין"}')

    doc = {
        "type": "FeatureCollection",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "https://www.govmap.gov.il/?lay=6",
        "kinds": {code: {"name": label, "color": colour,
                         "parcels": tally.get(code, [0, 0])[0],
                         "dunam": round(tally.get(code, [0, 0])[1] / 1000)}
                  for code, colour, label in KINDS.values()},
        "missing": len(missing),
        "features": out + woods,
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, separators=(",", ":"))
    print(f"\nנכתב {OUT}: {len(out)} חלקות ו-{len(woods)} יערות, "
          f"{os.path.getsize(OUT) / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
