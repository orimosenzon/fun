#!/usr/bin/env python3
"""Build web/data/canopy.json and web/overlays/canopy/: where the trees are.

Every other layer on this map is about the ground. This one is about what is
over your head, and for somebody choosing a way to walk in August it is the
question that decides the route: which of two shortcuts is under trees. The
map draws it as a picture draped over the moshava - every tree canopy as a
green patch at about four metres a pixel - rather than as thousands of
polygons, because a picture is what it is: a machine-learning model run over
aerial photographs, and the shapes it drew are pixels before they are anything.

Where it comes from
-------------------
The Survey of Israel (מפ"י) flew the country in 2021 and ran a model over the
photographs to outline every tree canopy in every built-up area, and published
the result as an open national dataset:

    https://data.gov.il/dataset/nationalcanopytrees      (358 MB, "Other (Open)")

GoInfo (goinfo.co.il), an independent civic data site, rendered that dataset
into one PNG per city for its own map of trees and heat, and publishes the
figures it computed from it under CC BY 4.0. This script takes the moshava's
PNG and GoInfo's numbers rather than the 358 MB file, which is the
convenient path and the one chosen on 13/9/2026: an hour's work for a layer
somebody can look at, and if the layer earns its keep the next step is to
build from מפ"י's file directly - our own resolution, and a shade index per
shortcut - which is a different script and a different afternoon.

What it takes from GoInfo, and why each is safe to take:

  src/canopy-blobs.js       the PNG's own frame, in ITM metres. Without it the
                            picture is a picture; with it, it is a map.
  src/tree-canopy-cities.js the moshava's canopy cover, tree count and area,
                            as GoInfo computed them off the survey. These are
                            the numbers the layer sheet shows, and they carry
                            the CC BY 4.0 credit the licence asks for.
  assets/blobs/canopy/<city>.png   the picture itself.

None of it can be read from the app at run time: goinfo.co.il sends no CORS
header, so a browser on another origin is refused. Hence a build step, the
same as every other layer here, and the tiles ship with the app rather than
being fetched from GoInfo.

What is done to the picture
---------------------------
Cropped to the municipal boundary plus a margin. GoInfo's blob is a ten
kilometre square with the moshava in the middle of it and slivers of Binyamina,
Kfar Pines and Ein Iron cut off at its edges - each clipped to its own city
line and then to the square. The crop keeps what is ours and a little around
it, and the frame is recomputed from the crop's pixel edges so nothing moves.

Recoloured to the layer's own green, keeping the alpha - the source encodes
the canopy in the alpha channel and the colour is a single flat value, so the
swap costs nothing and means the swatch in the layer sheet is the colour on
the map.

Why tiles and not one picture
-----------------------------
The first version was one PNG with four corners, MapLibre's `image` source,
and it drew beautifully - until the map was tilted. With terrain on, MapLibre
4.7 paints every layer into the terrain's own tiles, and past a certain zoom
some of those tiles simply never find the one image: at zoom 16.5 the canopy
stopped dead along a horizontal line through the middle of town, and with the
terrain switched off it came back. The satellite basemap does not suffer from
this, because it is tiles, so the canopy is tiles too: a small pyramid, cut
here, served from next to the app like the CSS.

Cutting them means putting the picture into Web Mercator, which is what a tile
is. ITM is a transverse Mercator on a different meridian, so each tile pixel
is walked back through both projections to the source pixel it lands on -
nearest neighbour, because the finest tiles are at the picture's own
resolution and there is nothing to interpolate between. Coarser zooms are the
finest one halved, so a thin line of trees fades rather than flickers out.

The map overzooms the finest tile past its zoom, the same as it does with the
satellite imagery. What that costs is softness: at zoom 17 a four-metre pixel
is eight screen pixels wide, and MapLibre blends across it. For a canopy that
is the honest look - the model drew blobs, not leaves.

    python3 build_canopy.py
"""

import json
import math
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "web", "data", "canopy.json")
OUT_DIR = os.path.join(HERE, "web", "overlays", "canopy")
# What the app fetches, relative to web/index.html.
TILES_PATH = "overlays/canopy/{z}/{x}/{y}.png"

GOINFO = "https://goinfo.co.il/"
BLOBS_JS = GOINFO + "src/canopy-blobs.js"
CITIES_JS = GOINFO + "src/tree-canopy-cities.js"
CANOPY_PAGE = GOINFO + "canopy-map.html"
SURVEY = "https://data.gov.il/dataset/nationalcanopytrees"

# GoInfo's spelling of the city, with spaces around the hyphen: this is the key
# in both of its tables and the file name of the PNG.
CITY = "פרדס חנה - כרכור"

WFS = "https://open.govmap.gov.il/geoserver/opendata/ows"
# govmap spells the council "פרדס חנה - כרכור", with the spaces; the cadastre
# spells its locality without them. The prefix matches both.
LOCALITY = "פרדס חנה"
MARGIN_M = 300

# The layer's colour, and now the picture's. A shade of green darker than the
# public-land layer's, so the two do not read as one thing when both are on.
COLOR = "#1b5e20"

# 512-pixel tiles, finest at zoom 14: four metres a pixel at this latitude,
# which is what the source has. Five zooms is thirty-odd files; 256-pixel
# tiles to zoom 15 would say the same thing in four times as many.
TILE = 512
MINZ = 10
MAXZ = 14

UA = "derech-kitzur/1.0 (build_canopy.py)"

ITM = Transformer.from_crs("EPSG:4326", "EPSG:2039", always_xy=True)
WGS = Transformer.from_crs("EPSG:2039", "EPSG:4326", always_xy=True)


# ---------------------------------------------------------------- fetching

def get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as handle:
        body = handle.read()
    return body if binary else body.decode("utf-8")


def js_export(source, name):
    """The object literal a `export const NAME = {...}` line assigns.

    GoInfo's data modules are one JSON object each behind an ES export, with a
    header comment above. Brace-matching rather than a regex, because the
    values hold Hebrew strings and nothing else - no braces inside strings.
    """
    start = source.find(f"export const {name} = ")
    if start < 0:
        raise SystemExit(f"לא נמצא {name} במודול של GoInfo")
    body = source[source.index("{", start):]
    depth = 0
    for i, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(body[:i + 1])
    raise SystemExit(f"{name}: הסוגריים לא נסגרים")


def computed_at(source):
    """The date in the module's header, "computed 2026-08-05T14:40+0300"."""
    match = re.search(r"computed (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}[+-]\d{4})", source)
    return match.group(1) if match else None


def boundary_bbox():
    """The municipal boundary's bounding box, in ITM metres, plus the margin."""
    query = urllib.parse.urlencode({
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": "opendata:muni_il",
        "outputFormat": "application/json", "srsName": "EPSG:4326",
        "CQL_FILTER": f"Muni_Heb LIKE '%{LOCALITY}%'",
    })
    features = json.loads(get(f"{WFS}?{query}")).get("features", [])
    if not features:
        raise SystemExit(f"לא נמצא גבול מוניציפלי ל-{LOCALITY} בשכבת muni_il")
    geom = features[0]["geometry"]
    polys = (geom["coordinates"] if geom["type"] == "MultiPolygon"
             else [geom["coordinates"]])
    pts = [p for poly in polys for ring in poly for p in ring]
    xs, ys = zip(*(ITM.transform(p[0], p[1]) for p in pts))
    return (min(xs) - MARGIN_M, min(ys) - MARGIN_M,
            max(xs) + MARGIN_M, max(ys) + MARGIN_M)


# ---------------------------------------------------------------- the crop

def crop_to(image, frame, box):
    """The part of the picture inside `box`, and the exact frame of that part.

    `frame` is GoInfo's: x, y of the top-left corner in ITM metres - y stored
    negated, the way an SVG counts downwards - and w, h in metres. `box` is
    (east0, north0, east1, north1). Pixel edges are rounded outwards so the crop
    contains the box, and the returned frame is the crop's own, read back off
    those integer edges - which is what keeps the picture where it was.
    """
    e0, n_top = frame["x"], -frame["y"]
    ppm_x = image.width / frame["w"]
    ppm_y = image.height / frame["h"]

    left = max(0, int((box[0] - e0) * ppm_x))
    right = min(image.width, -int(-(box[2] - e0) * ppm_x))
    top = max(0, int((n_top - box[3]) * ppm_y))
    bottom = min(image.height, -int(-(n_top - box[1]) * ppm_y))
    if right <= left or bottom <= top:
        raise SystemExit("המושבה מחוץ לתמונה של GoInfo")

    cropped = image.crop((left, top, right, bottom))
    exact = {
        "e0": e0 + left / ppm_x, "e1": e0 + right / ppm_x,
        "n0": n_top - bottom / ppm_y, "n1": n_top - top / ppm_y,
    }
    return cropped, exact


def corners(exact):
    """Top-left, top-right, bottom-right, bottom-left, as [lng, lat]."""
    order = [(exact["e0"], exact["n1"]), (exact["e1"], exact["n1"]),
             (exact["e1"], exact["n0"]), (exact["e0"], exact["n0"])]
    return [[round(v, 6) for v in WGS.transform(e, n)] for e, n in order]


# ---------------------------------------------------------------- the tiles

def merc_px(lng, lat, z):
    """Global pixel coordinates at zoom z, TILE pixels to a tile."""
    n = TILE * (1 << z)
    x = (lng + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def px_merc(xs, ys, z):
    """The inverse of merc_px, over numpy arrays."""
    n = TILE * (1 << z)
    lng = xs / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * ys / n))))
    return lng, lat


def tile_range(cs, z):
    """The tiles at zoom z that the four corners fall into, inclusive."""
    px = [merc_px(lng, lat, z) for lng, lat in cs]
    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    return (int(min(xs) // TILE), int(min(ys) // TILE),
            int(max(xs) // TILE), int(max(ys) // TILE))


def reproject(alpha, exact, cs, z):
    """The crop's alpha laid out in Web Mercator at zoom z, over whole tiles.

    Returns the image and the tile it starts at. Every pixel of the output is
    walked back to the source: pixel centre -> lng/lat -> ITM -> source pixel,
    nearest neighbour. About six million points at the finest zoom, which
    pyproj takes in a few seconds when handed all of them at once.
    """
    tx0, ty0, tx1, ty1 = tile_range(cs, z)
    width = (tx1 - tx0 + 1) * TILE
    height = (ty1 - ty0 + 1) * TILE

    col = (np.arange(width) + 0.5) + tx0 * TILE
    row = (np.arange(height) + 0.5) + ty0 * TILE
    xs, ys = np.meshgrid(col, row)
    lng, lat = px_merc(xs.ravel(), ys.ravel(), z)
    east, north = ITM.transform(lng, lat)

    src_h, src_w = alpha.shape
    ppm_x = src_w / (exact["e1"] - exact["e0"])
    ppm_y = src_h / (exact["n1"] - exact["n0"])
    sx = np.floor((east - exact["e0"]) * ppm_x).astype(np.int64)
    sy = np.floor((exact["n1"] - north) * ppm_y).astype(np.int64)
    inside = (sx >= 0) & (sx < src_w) & (sy >= 0) & (sy < src_h)

    out = np.zeros(width * height, dtype=np.uint8)
    out[inside] = alpha[sy[inside], sx[inside]]
    return out.reshape(height, width), (tx0, ty0)


def coloured(alpha_img):
    r, g, b = (int(COLOR[i:i + 2], 16) for i in (1, 3, 5))
    out = Image.new("RGBA", alpha_img.size, (r, g, b, 0))
    out.putalpha(alpha_img)
    return out


def write_tiles(alpha, exact, cs):
    """The pyramid, finest zoom first and each coarser one halved from it.

    Halving with a box filter rather than resampling the source again at each
    zoom, so that a single line of trees becomes a fainter line at the next
    zoom out instead of being present or absent by where the sample fell.
    """
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    grid, (tx0, ty0) = reproject(alpha, exact, cs, MAXZ)
    layer = Image.fromarray(grid, "L")
    # The grid's top-left corner in global pixels at MAXZ; halved with it.
    ox, oy = tx0 * TILE, ty0 * TILE

    count = 0
    for z in range(MAXZ, MINZ - 1, -1):
        if z < MAXZ:
            layer = layer.resize((layer.width // 2, layer.height // 2), Image.BOX)
            ox, oy = ox / 2, oy / 2
        picture = coloured(layer)
        r, g, b = (int(COLOR[i:i + 2], 16) for i in (1, 3, 5))

        for ty in range(int(oy // TILE), int(math.ceil((oy + layer.height) / TILE))):
            for tx in range(int(ox // TILE), int(math.ceil((ox + layer.width) / TILE))):
                left = int(round(tx * TILE - ox))
                top = int(round(ty * TILE - oy))
                tile = Image.new("RGBA", (TILE, TILE), (r, g, b, 0))
                part = picture.crop((max(0, left), max(0, top),
                                     min(picture.width, left + TILE),
                                     min(picture.height, top + TILE)))
                tile.paste(part, (max(0, -left), max(0, -top)))
                path = os.path.join(OUT_DIR, str(z), str(tx), f"{ty}.png")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tile.save(path, optimize=True)
                count += 1
        print(f"  זום {z}: {layer.width}x{layer.height}, "
              f"{count} אריחים עד כה")
    return count


# ---------------------------------------------------------------- main

def main():
    print("קורא את הטבלאות של GoInfo…")
    blobs_src = get(BLOBS_JS)
    blobs = js_export(blobs_src, "CANOPY_BLOBS")
    cities_src = get(CITIES_JS)
    cities = js_export(cities_src, "CITY_CANOPY")
    if CITY not in blobs or CITY not in cities:
        raise SystemExit(f"'{CITY}' לא נמצאת בטבלאות של GoInfo: "
                         f"{[k for k in blobs if 'פרדס' in k]}")
    frame = blobs[CITY]
    city = cities[CITY]
    rendered = computed_at(blobs_src)
    computed = computed_at(cities_src)

    png_url = GOINFO + urllib.parse.quote(frame["src"].lstrip("./"))
    print(f"מוריד {png_url}")
    image = Image.open(BytesIO(get(png_url, binary=True))).convert("RGBA")
    print(f"  {image.width}x{image.height}, "
          f"{frame['w'] / image.width:.2f} מ׳ לפיקסל")

    print("מושך את הגבול המוניציפלי…")
    box = boundary_bbox()
    cropped, exact = crop_to(image, frame, box)
    cs = corners(exact)
    print(f"  נחתך ל-{cropped.width}x{cropped.height}")

    print("חותך אריחים…")
    alpha = np.asarray(cropped.getchannel("A"))
    count = write_tiles(alpha, exact, cs)

    lngs = [c[0] for c in cs]
    lats = [c[1] for c in cs]
    doc = {
        "version": 2,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "name": "צל עצים",
        "color": COLOR,
        "tiles": TILES_PATH,
        "tileSize": TILE,
        "minzoom": MINZ,
        "maxzoom": MAXZ,
        # West, south, east, north: the box the map asks for tiles inside.
        "bounds": [round(min(lngs), 6), round(min(lats), 6),
                   round(max(lngs), 6), round(max(lats), 6)],
        "tileCount": count,
        "metres_per_pixel": round(frame["w"] / image.width, 2),
        "survey": {"year": 2021, "by": "המרכז למיפוי ישראל", "url": SURVEY},
        "via": {"name": "GoInfo", "url": CANOPY_PAGE, "license": "CC BY 4.0",
                "rendered": rendered, "computed": computed},
        "stats": {
            "pct": city.get("pct"),
            "trees": city.get("treeCount"),
            "canopy_m2": city.get("canopyAreaM2"),
            "city_m2": city.get("cityAreaM2"),
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=1)

    size = sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(OUT_DIR) for f in fs) / 1024
    print(f"\nנכתבו {count} אריחים ({size:.0f}KB) ל-{os.path.relpath(OUT_DIR, HERE)}/ "
          f"ו-{os.path.relpath(OUT_JSON, HERE)}")
    print(f"  כיסוי חופות: {city.get('pct')}% · {city.get('treeCount'):,} עצים"
          f" · סקר {doc['survey']['year']}, עיבוד GoInfo {computed}")
    print(f"  תחום: {doc['bounds']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
