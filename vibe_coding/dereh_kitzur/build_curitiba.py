#!/usr/bin/env python3
"""Build the Curitiba layer: a Brazilian city's transit spine, bike network and
pedestrian core, as the second worked example next to Houten.

Houten answers "what if a town were laid out for the bicycle" by building a town
from scratch, in a rich country, for fifty thousand people. That is a fair
answer and an easy one to wave away. Curitiba is the same argument made inside a
city that was already there: two million people in the south of Brazil, no money
to dig a metro, and a mayor who was an architect. It is the city planners point
at when they want to say that the shape of a place is a decision rather than a
budget.

Three ideas of theirs matter to this project, and each is a group in the layer:

  the canaletas    From 1974 the city gave the bus its own road down the middle
                   of five structural axes, with the car traffic pushed to
                   one-way streets either side and the tall buildings zoned
                   along the axis. Buses arrive every ninety seconds, board at
                   platform level through five doors, and the fare is paid
                   before boarding - which is a subway, at a twentieth of the
                   price. The rest of the world calls it Bus Rapid Transit and
                   copied it from here.

  the ciclovias    Around 250 km of bike structures, threaded through the parks
                   the city built in the flood plains it was not going to be
                   able to stop flooding. A shortcut across a park is the same
                   idea as a shortcut between two streets in the moshava, at a
                   different scale.

  the calçadões    In 1972 Rua XV de Novembro was closed to cars over a weekend,
                   before the shopkeepers could get an injunction, and became
                   the first pedestrian street in Brazil. It is still the
                   busiest street in the city.

Everything geometric comes from OpenStreetMap through Overpass. The pictures
come from Wikimedia Commons, resolved at build time so that the credit and the
licence are whatever Commons currently says rather than whatever was true when
this file was written. The videos are YouTube ids, checked against the oEmbed
endpoint on every build so that a clip taken down does not leave a dead tile.

    python3 build_curitiba.py [--refresh] [--skip-media]

Raw answers are cached under ``.cache/curitiba/``; ``--refresh`` throws them
away and asks again. ``--skip-media`` builds the geometry alone, which is what
to use when there is no network beyond Overpass.
"""

import collections
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request

OUT = "web/data/curitiba.json"
CACHE = ".cache/curitiba"

API = "https://overpass-api.de/api/interpreter"
STATUS = "https://overpass-api.de/api/status"
COMMONS = "https://commons.wikimedia.org/w/api.php"
OEMBED = "https://www.youtube.com/oembed"
UA = "derech-kitzur/build_curitiba.py (github.com/orimosenzon/fun)"

# The municipality of Curitiba, OSM relation 297514. Overpass area ids are the
# relation id plus 3600000000.
AREA = 3600297514

CREDIT = "OpenStreetMap contributors, ODbL · תמונות מוויקישיתוף"

CURITIBA_ID = "curitiba"

# ------------------------------------------------------------------ the groups
#
# Order is drawing order, bottom first, and it is the reverse of the argument:
# the canaletas go down first because they are the thing everything else was
# arranged around, and the pedestrian core goes on top because it is fifteen
# streets in a city of two million and would vanish under anything.

CANAL = "הקנאלטות של האוטובוס"
PATHS = "שבילי אופניים"
WALK = "מדרחובות"
ROUTES = "מסלולי האופניים המסומנים"

GROUPS = [
    {"name": CANAL, "color": "#b0392f"},
    {"name": PATHS, "color": "#1565c0"},
    {"name": WALK, "color": "#e08a00"},
    {"name": ROUTES, "color": "#2e7d32"},
]

COLOR = {g["name"]: g["color"] for g in GROUPS}

ABOUT = {
    CANAL: "נתיב שמור לאוטובוס בלבד, במרכז אחד מצירי המבנה של העיר. התנועה "
           "הפרטית נדחקת לרחובות חד-סטריים משני הצדדים, והבנייה הגבוהה מותרת "
           "לאורך הציר. זה מה שהעולם מכיר בשם BRT, והוא הומצא כאן ב-1974.",
    PATHS: "שביל אופניים מהרשת העירונית, שאורכה כ-250 ק״מ. חלק גדול ממנה עובר "
           "בתוך הפארקים שהעירייה הקימה בגדות הנחלים במקום לנסות לכלוא אותם.",
    WALK: "מדרחוב. הראשון שבהם, רחוב XV de Novembro, נסגר לתנועה בסוף שבוע אחד "
          "ב-1972, לפני שבעלי החנויות הספיקו להוציא צו מניעה.",
    ROUTES: "מסלול רכיבה מסומן, מתוך רשת המסלולים שמחברת את הפארקים זה לזה.",
}


# ------------------------------------------------------------------- overpass

def wait_for_slot(limit=900):
    """Overpass allows two queries at a time per address. Ask, do not hammer."""
    deadline = time.time() + limit
    while time.time() < deadline:
        req = urllib.request.Request(STATUS, headers={"User-Agent": UA})
        txt = urllib.request.urlopen(req, timeout=30).read().decode()
        free = re.search(r"(\d+) slots available now", txt)
        if free and int(free.group(1)) > 0:
            return
        waits = [int(m) for m in re.findall(r"in (\d+) seconds", txt)]
        nap = min(waits) + 2 if waits else 15
        print(f"  ממתין {nap} שניות לתור ב-Overpass", flush=True)
        time.sleep(nap)


def overpass(name, query, refresh=False):
    """One query, cached on disk under its own name."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.json")
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    body = b""
    for attempt in range(6):
        wait_for_slot()
        req = urllib.request.Request(
            API, data=urllib.parse.urlencode({"data": query}).encode(),
            headers={"User-Agent": UA})
        # `wait_for_slot` asks and is still sometimes wrong: the status endpoint
        # is a moment behind, and a slot that was free when it answered can be
        # taken by the time the query goes out, which comes back as a bare 429.
        # Retrying is the whole handling - the request is idempotent and the
        # alternative is losing four cached answers to the fifth.
        try:
            body = urllib.request.urlopen(req, timeout=900).read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as err:
            print(f"  {name}: {err}, מנסה שוב", flush=True)
            time.sleep(30 * (attempt + 1))
            continue
        # A refused query answers with an HTML error page and status 200, so the
        # only reliable check is what the first byte looks like.
        if body.lstrip()[:1] in (b"{", b"["):
            with open(path, "wb") as fh:
                fh.write(body)
            doc = json.loads(body)
            print(f"  {name}: {len(doc['elements'])} רכיבים "
                  f"({len(body) / 1024:.0f} KB)", flush=True)
            return doc
        print(f"  {name}: נדחה, מנסה שוב", flush=True)
        time.sleep(20 * (attempt + 1))
    sys.exit(f"Overpass kept refusing {name}: {body[:300]!r}")


# `highway=busway` is the canaleta itself. Curitiba's are mapped as their own
# ways rather than as a lane tag on the road, which is what they are on the
# ground: a separate carriageway with its own bridges and its own stations.
Q_CANAL = f"""[out:json][timeout:600];
area({AREA})->.a;
(
  way(area.a)["highway"="busway"];
);
out geom;"""

Q_BIKE = f"""[out:json][timeout:900];
area({AREA})->.a;
(
  way(area.a)["highway"="cycleway"];
  way(area.a)["highway"~"^(path|footway|track|service|residential|unclassified|living_street|pedestrian)$"]["bicycle"="designated"];
);
out geom;"""

Q_WALK = f"""[out:json][timeout:600];
area({AREA})->.a;
(
  way(area.a)["highway"="pedestrian"];
);
out geom;"""

Q_ROUTES = f"""[out:json][timeout:600];
area({AREA})->.a;
(
  rel(area.a)["route"="bicycle"];
);
out geom;"""

# Curitiba's seventy-five bairros, as administrative boundaries.
#
# Most of the bike network carries no name in OSM - a cycleway alongside an
# avenue is usually just a cycleway - and without this the app's list showed a
# hundred and forty-eight rows all called "שביל", numbered. That is the same
# problem `build_network.py` solved for the moshava by naming its segments after
# the streets they run along, and the same answer: a segment has to say where it
# is, or the list is unusable.
Q_BAIRROS = f"""[out:json][timeout:600];
area({AREA})->.a;
(
  rel(area.a)["boundary"="administrative"]["admin_level"="10"];
);
out geom;"""

# The named things worth a pin of their own. Asked for by name rather than by
# id, so that a retagging in OSM is a name that stops matching - which the build
# reports - rather than a pin that quietly moves to whatever took the id over.
LANDMARK_NAMES = [
    "Jardim Botânico de Curitiba",
    "Ópera de Arame",
    "Parque Barigui",
    "Parque Tanguá",
    "Parque Tingui",
    "Museu Oscar Niemeyer",
]

Q_LANDMARKS = f"""[out:json][timeout:300];
area({AREA})->.a;
(
""" + "\n".join(
    f'  nwr(area.a)["name"="{n}"];' for n in LANDMARK_NAMES
) + """
);
out center tags;"""


# ------------------------------------------------------------------- geometry

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


def geom(way):
    """A way's geometry as [[lat, lng], ...], rounded the way the app stores it."""
    return [[round(p["lat"], 6), round(p["lon"], 6)] for p in way.get("geometry", [])]


def key(point):
    """Two fragments meet when their endpoints round to the same spot. Six
    decimals is about 10 cm, finer than two ways that share a node differ by."""
    return (round(point[0], 6), round(point[1], 6))


# How far a point may be from the line drawn without it before it has to stay.
#
# This layer is the only one here that needs thinning, and the reason is that
# `Store.load()` fetches every document at boot whether its layer is switched on
# or not - so a visitor in the moshava looking for a shortcut downloads Curitiba
# too, on whatever signal they have. Unsimplified the file was 1.3 MB, five
# times the whole cycling plan of the moshava.
#
# Four metres, because this is a reference mesh looked at from city scale and
# nobody navigates by it: the app's own trails, which people do walk, are not
# touched by any of this. At four metres the drawn line is within half a lane of
# the surveyed one everywhere, and the file drops by more than half.
SIMPLIFY_M = 4.0


def simplify(path, tolerance=SIMPLIFY_M):
    """Ramer-Douglas-Peucker, iterative so a long avenue cannot blow the stack.

    Endpoints are never moved, which is what lets this run *after* `chain`:
    fragments are joined on endpoints that round to the same spot, and thinning
    a joined line leaves both of its ends exactly where they were.

    It runs *before* `line_id`, so an item's id is the hash of the line the app
    actually draws. That is the more stable of the two choices: an OSM edit that
    nudges a node by a metre disappears here and the id, and the photo hanging
    off it, survive.
    """
    if len(path) < 3:
        return path

    def far(a, b, p):
        """Perpendicular distance from p to the segment a-b, in metres."""
        # Flat earth over a few hundred metres, with longitude squeezed by the
        # latitude so that a degree east and a degree north are the same length.
        scale = math.cos(math.radians(a[0]))
        ax, ay = a[1] * scale, a[0]
        bx, by = b[1] * scale, b[0]
        px, py = p[1] * scale, p[0]
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return haversine(a, p)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        return haversine(p, [ay + t * dy, (ax + t * dx) / scale])

    keep = [False] * len(path)
    keep[0] = keep[-1] = True
    stack = [(0, len(path) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        worst, at = 0.0, lo
        for i in range(lo + 1, hi):
            d = far(path[lo], path[hi], path[i])
            if d > worst:
                worst, at = d, i
        if worst > tolerance:
            keep[at] = True
            stack.append((lo, at))
            stack.append((at, hi))
    return [p for p, k in zip(path, keep) if k]


def chain(paths, labels=None):
    """Join fragments that meet end to end into as few polylines as possible.

    The same greedy walk as `build_houten.py`, and for the same reason: Overpass
    cuts a way at every junction, so one avenue arrives as forty pieces and each
    would otherwise be its own row in the app's list. At a junction where
    several fragments meet, the chain takes the one carrying the same name and
    otherwise stops rather than guessing.

    Returns (line, members) so the caller can name a chain after whatever its
    pieces were called.
    """
    ends = collections.defaultdict(list)
    for i, path in enumerate(paths):
        if len(path) < 2:
            continue
        ends[key(path[0])].append(i)
        ends[key(path[-1])].append(i)

    name_of = (lambda i: labels[i] if labels else None)

    used = set()
    out = []
    for start in range(len(paths)):
        if start in used or len(paths[start]) < 2:
            continue
        used.add(start)
        line = list(paths[start])
        members = [start]
        label = name_of(start)

        for _ in range(2):                       # forwards, then backwards
            while True:
                here = key(line[-1])
                nxt = [i for i in ends[here] if i not in used]
                if len(nxt) > 1 and label:
                    named = [i for i in nxt if name_of(i) == label]
                    nxt = named if len(named) == 1 else nxt
                if len(nxt) != 1:
                    break
                i = nxt[0]
                piece = paths[i]
                used.add(i)
                members.append(i)
                label = label or name_of(i)
                line += (piece[1:] if key(piece[0]) == here
                         else list(reversed(piece))[1:])
            line.reverse()

        out.append((line, members))
    return out


# ------------------------------------------------------------------ where am I

def bairro_rings(doc):
    """Each bairro as (name, outer ring).

    A boundary relation arrives as a heap of member ways in no particular order,
    so the rings are assembled with the same `chain` that joins the bike paths -
    a closed ring is simply a chain whose two ends meet. The largest ring wins
    where a bairro has several, which happens when a boundary follows a river
    and picks up an island.
    """
    rings = []
    for rel in doc["elements"]:
        name = rel.get("tags", {}).get("name")
        if not name:
            continue
        members = [m for m in rel.get("members", [])
                   if m["type"] == "way" and m.get("geometry")
                   and m.get("role") in ("outer", "", None)]
        closed = [line for line, _ in chain([geom(m) for m in members])
                  if len(line) > 3 and key(line[0]) == key(line[-1])]
        if closed:
            rings.append((name, max(closed, key=len)))
    return rings


def in_ring(point, ring):
    """Ray casting. The ring is closed, so the wrap-around pair is the last
    vertex back to the first and the loop below covers it."""
    lat, lng = point
    inside_it = False
    for (y1, x1), (y2, x2) in zip(ring, ring[1:]):
        if (y1 > lat) != (y2 > lat):
            cut = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lng < cut:
                inside_it = not inside_it
    return inside_it


def bairro_of(point, rings):
    """Which bairro a point is in, or ''.

    A containment test and not the nearest centre: several of these bairros are
    long and thin along a river, and the nearest centre to a path inside one of
    them is regularly the centre of the next one over. A label that is quietly
    wrong is worse than no label - the same finding as the addresses in
    `build_places.py`, which is why nothing here says "approximately".
    """
    for name, ring in rings:
        if in_ring(point, ring):
            return name
    return ""


def midpoint(path):
    """The point half way along the line, by distance rather than by index: a
    line whose first half is finely mapped and second half is two long straights
    has its middle index nowhere near its middle."""
    total = path_length(path)
    if total <= 0:
        return path[0]
    walked = 0.0
    for a, b in zip(path, path[1:]):
        step = haversine(a, b)
        if walked + step >= total / 2:
            return a
        walked += step
    return path[-1]


# --------------------------------------------------------------- the segments

def note_for(tags, group):
    """The one line under a segment's name: what the group is, plus whatever
    this particular stretch says about itself that somebody would care about."""
    bits = []
    if tags.get("width"):
        bits.append(f'רוחב {tags["width"]} מ׳')
    if tags.get("lit") == "yes":
        bits.append("מואר")
    elif tags.get("lit") == "no":
        bits.append("לא מואר")
    if tags.get("surface") and tags["surface"] not in ("asphalt", "paved"):
        bits.append(f'מצע {tags["surface"]}')
    if tags.get("maxspeed"):
        bits.append(f'{tags["maxspeed"]} קמ״ש')
    if tags.get("tunnel"):
        bits.append("מנהרה")
    elif tags.get("bridge"):
        bits.append("גשר")
    if tags.get("segregated") == "yes":
        bits.append("מופרד מהולכי הרגל")
    return ABOUT[group] + (" · " + " · ".join(bits) if bits else "")


def line_id(prefix, path):
    """A segment's id, derived from the line itself rather than from its
    position in the answer Overpass happened to give.

    The app attaches photos and videos to items by id, in `data/media.json`, and
    this file is rebuilt from OpenStreetMap. Numbering by position would mean
    that one new cycleway mapped anywhere in Curitiba renumbers everything after
    it and moves somebody's photo onto a different road, silently. Five decimals
    is about a metre: finer than any edit that leaves the line meaning the same
    thing, coarser than the noise.
    """
    digest = hashlib.sha1(
        ";".join(f"{lat:.5f},{lng:.5f}" for lat, lng in path).encode()
    ).hexdigest()[:8]
    return f"{prefix}-{digest}"


def assign_ids(items, prefix):
    """Name every item after its geometry, in one pass so that two identical
    lines get told apart rather than overwriting each other."""
    taken = set()
    for it in items:
        path = it.get("path") or [[it["lat"], it["lng"]]]
        ident = base = line_id(prefix, path)
        n = 2
        while ident in taken:
            ident = f"{base}-{n}"
            n += 1
        taken.add(ident)
        it["id"] = ident
    return items


def segment(group, name, path, tags, streets):
    return {
        "id": "",                                # filled in by assign_ids
        "name": name,
        "note": note_for(tags, group),
        "photos": [],
        "path": path,
        "length": round(path_length(path)),
        "color": COLOR[group],
        "layer": CURITIBA_ID,
        "group": group,
        "kind": "",
        "streets": streets,
        "entries": [
            {"lat": path[0][0], "lng": path[0][1]},
            {"lat": path[-1][0], "lng": path[-1][1]},
        ],
    }


def chained_group(ways, group, label, rings, floor=15):
    """Chain a pile of Overpass ways into readable segments.

    Chaining runs over the whole pile at once rather than per name, so that an
    unnamed link between two named stretches does not come out as its own stub;
    street names then come off whatever the chain actually ran along, at most
    three of them.
    """
    paths = [geom(w) for w in ways]
    labels = [w["tags"].get("name") for w in ways]
    out = []
    for line, members in chain(paths, labels):
        if len(line) < 2 or path_length(line) < floor:
            continue

        pieces = [ways[i] for i in members]
        # Longest piece speaks for the chain: on a stretch that is asphalt for
        # 400 m and paving stones for 20, asphalt is the answer.
        tags = max(pieces, key=lambda w: len(w.get("geometry", [])))["tags"]

        streets = []
        for way in pieces:
            name = way["tags"].get("name")
            if name and name not in streets:
                streets.append(name)

        # The streets it runs along, else the bairro it is in, else nothing and
        # the numbering in `number_duplicates` is all there is to go on.
        if streets:
            title = f"{label} · {' / '.join(streets[:3])}"
        else:
            where = bairro_of(midpoint(line), rings)
            title = f"{label} · {where}" if where else label
        out.append(segment(group, title, simplify(line), tags, streets))
    return number_duplicates(out)


# The bounding box of the municipality, from Nominatim on relation 297514.
#
# Overpass answers `rel(area.a)` with every relation that so much as touches
# Curitiba, and `out geom` then hands back all of its geometry - so a regional
# route that passes through town arrives in full, hundreds of kilometres of it.
# The first build drew 1,572 km of "signed routes" across the state of Paraná.
SOUTH, WEST, NORTH, EAST = -25.6435, -49.3891, -25.3451, -49.1843

# Route relations that are not routes, or not Curitiba's.
#
# `Rede Ciclovias de Curitiba` is the whole city network gathered into one
# relation - 1,926 ways, which is the mesh itself rather than a route along it.
# Taking it as a route both drew the entire network in the routes colour and,
# because a way drawn as part of a route is dropped from the mesh, emptied the
# bike layer down to 44 km. `São José dos Pinhais` is the next town along.
NOT_A_ROUTE = ("Rede Ciclovias de Curitiba",
               "Rede de Ciclovias de São José dos Pinhais")


def inside(point):
    return SOUTH <= point[0] <= NORTH and WEST <= point[1] <= EAST


def local_routes(doc):
    """The relations worth drawing as named routes: the city's own signed ones.

    `network=lcn` is a local cycle network and `rcn` a regional one, and that
    tag is exactly the distinction wanted here - the regional ones are the
    long-distance rides out of town (Caminhos da Graciosa, Anel do Pinhão) which
    belong to Paraná rather than to Curitiba, and which are the whole reason the
    first build came out at fifteen hundred kilometres.
    """
    return [r for r in doc["elements"]
            if r.get("tags", {}).get("network") == "lcn"
            and r.get("tags", {}).get("name") not in NOT_A_ROUTE]


def route_way_ids(doc):
    """The ways drawn as part of a named route, which the mesh then leaves
    alone so that nothing is drawn twice."""
    return {m["ref"] for rel in local_routes(doc)
            for m in rel.get("members", []) if m["type"] == "way"}


def route_segments(doc):
    """The signed cycle routes, one entry per continuous piece."""
    out = []
    for rel in sorted(local_routes(doc),
                      key=lambda r: r.get("tags", {}).get("name", "")):
        tags = rel.get("tags", {})
        title = tags.get("name") or tags.get("ref") or "מסלול רכיבה"
        members = [m for m in rel.get("members", [])
                   if m["type"] == "way" and m.get("geometry")]
        # Clipped by dropping whole member ways that start outside the city,
        # rather than by cutting a line at the boundary: a cut leaves a stub
        # ending in the middle of nowhere, and a way is short enough that
        # keeping or dropping it whole is accurate to a block.
        members = [m for m in members if inside(geom(m)[0])]
        pieces = [line for line, _ in chain([geom(m) for m in members])]
        # A route arrives as one line when its ways are in order and unbroken,
        # and as a handful when the relation skips a junction. Longest first, so
        # the piece that carries the route's name is the route itself.
        pieces.sort(key=path_length, reverse=True)
        for i, path in enumerate(pieces):
            if len(path) < 2 or path_length(path) < 60:
                continue
            name = title if not i else f"{title} · קטע {i + 1}"
            out.append(segment(ROUTES, name, simplify(path), tags, []))
    return out


def number_duplicates(segments):
    """A name that lands on more than one segment gets a running number.

    The canaletas are the reason: an axis arrives as a dozen pieces all called
    the same thing, which in a list is a dozen rows that look like one row. The
    numbers run longest first, so `· 1` is the main run.
    """
    counts = collections.Counter(s["name"] for s in segments)
    order = collections.defaultdict(int)
    for seg in sorted(segments, key=lambda s: -s["length"]):
        total = counts[seg["name"]]
        if total > 1:
            order[seg["name"]] += 1
            seg["name"] = f'{seg["name"]} · {order[seg["name"]]} מתוך {total}'
    return segments


# ------------------------------------------------------------------- the pins

def landmark(name, lat, lng, note):
    return {
        "id": "",
        "name": name,
        "note": note,
        "photos": [],
        "lat": round(lat, 6),
        "lng": round(lng, 6),
        "color": COLOR[ROUTES],
        "layer": CURITIBA_ID,
        "links": [],
    }


# What each pin is, in Hebrew, and which picture and clip belong to it. The
# Commons titles and the YouTube ids are written out rather than searched for at
# build time on purpose: a search that returns something different next month
# would silently put a different picture on the map, and the point of a curated
# handful is that somebody looked at each one.
LANDMARKS = {
    "Jardim Botânico de Curitiba": {
        "he": "הגן הבוטני",
        "note": "הגן הבוטני של קוריטיבה, שנפתח ב-1991 על שטח שהיה מזבלה. "
                "החממה מזכוכית ומתכת היא העתק מוקטן של קריסטל פאלאס הלונדוני, "
                "והיא הדימוי שהעיר משתמשת בו כדי לספר על עצמה. שבילי אופניים "
                "מגיעים אליו משתי כניסות.",
        "file": "File:Estufa principal do Jardim Botânico de Curitiba 02.jpg",
        "yt": "N2LodwS_mU0",
    },
    "Ópera de Arame": {
        "he": "אופרת התיל",
        "note": "אולם הופעות מצינורות פלדה וזכוכית שנבנה ב-1992 בתוך מחצבה "
                "נטושה, בשישים ימים. הגג שקוף, מסביבו אגם, ומגיעים אליו דרך "
                "גשר. הקו שחוזר בכל הפרויקטים של העיר: לא להרוס מה שנשאר "
                "מאחור אלא למצוא לו שימוש.",
        "file": "File:Ópera De Arame - Curitiba - panoramio.jpg",
        "yt": "dsRU0acVCVY",
    },
    "Parque Barigui": {
        "he": "פארק בריגווי",
        "note": "הגדול שבפארקי העיר, 1.4 קמ״ר סביב אגם על נחל בריגווי. הוא לא "
                "נבנה בשביל הנוף אלא בשביל השיטפונות: במקום לתעל את הנחל, "
                "העירייה קנתה את גדותיו והפכה אותן לפארק שמותר לו להציף. "
                "קפיברות מסתובבות בו חופשי, ומקיף אותו מסלול רכיבה.",
        "file": "File:Parque Barigui, Curitiba, Paraná.jpg",
        "yt": "9gD02ZsGwcw",
    },
    "Parque Tanguá": {
        "he": "פארק טנגואה",
        "note": "עוד מחצבה שהפכה לפארק, ב-1996. מפל יורד מקיר הסלע אל אגם, "
                "ומנהרה חצובה מחברת בין שני חלקי המחצבה. יש בו טיילת רכיבה "
                "לאורך הקצה העליון, ומשם רואים את כל צפון העיר.",
        "file": "File:Tanguá Curitiba.jpg",
        "yt": "uoSFCQPto5U",
    },
    "Parque Tingui": {
        "he": "פארק טינגווי",
        "note": "פארק לאורך נחל בריגווי, קרוי על שם השבט שישב כאן לפני "
                "המתיישבים. בתוכו האנדרטה לזכר ההגירה האוקראינית, שהיא כנסייה "
                "מעץ בסגנון גליציאני. מסלול הרכיבה שלו מחובר לזה של בריגווי.",
        "file": "File:Parque Tingui Curitiba DSC03970.JPG",
        "yt": "495FbjQpvQY",
    },
    "Museu Oscar Niemeyer": {
        "he": "מוזיאון אוסקר נימאייר",
        "note": "מוזיאון האמנות של המדינה, שהעיר קוראת לו פשוט \"מוזיאון "
                "העין\" בגלל האגף שנימאייר הוסיף ב-2002: גוף בצורת עין על "
                "עמוד אחד, מעל בריכה. הבניין המקורי, משנת 1967, הוא גם שלו.",
        "file": "File:Museu Oscar Niemeyer 2 Curitiba Brasil.jpg",
        "yt": "GIfMZK-p8jU",
    },
}


# And the media that belongs to a line rather than to a point. Each rule names
# a group and a pattern; the picture and the clip go on the longest segment in
# that group whose name matches, or on the longest in the group when the pattern
# is empty. Longest and not first, so the choice does not depend on the order
# Overpass answered in.
LINE_MEDIA = [
    {
        "group": WALK,
        "match": r"XV de Novembro",
        "file": "File:Rua-xv-bondinho.jpg",
        "yt": "hVOTNS91N2M",
        "note": "המדרחוב הראשון בברזיל. ב-19 במאי 1972 סגרה העירייה את הרחוב "
                "לתנועה ביום שישי אחר הצהריים והספיקה לרצף אותו עד יום שני "
                "בבוקר, כדי שבעלי החנויות שהתנגדו לא יספיקו להוציא צו מניעה. "
                "בסוף השבוע הראשון הם ביקשו להאריך אותו. היום הוא הרחוב "
                "ההומה בעיר.",
    },
    {
        "group": CANAL,
        "match": r"",
        "file": "File:Ônibus biarticulado Curitiba 20250603.jpg",
        "yt": "VQY1VTQJWRo",
        "note": "הקטע הרצוף הארוך ביותר של נתיב האוטובוס הבלעדי. אוטובוס "
                "דו-מפרקי באורך 28 מטר נוסע כאן בתדירות של פעם בדקה וחצי, "
                "הנוסעים משלמים לפני העלייה ועולים מרציף בגובה הרצפה דרך חמש "
                "דלתות. זו רכבת תחתית שנוסעת על גלגלים, בעלות של אחד חלקי "
                "עשרים.",
    },
    {
        "group": PATHS,
        "match": r"",
        "file": "File:Ciclo via na marechal floriano peixoto - curitiba.JPG",
        "yt": "9GEomqYMwdc",
        "note": "הקטע הרצוף הארוך ביותר ברשת האופניים של העיר. הרשת כולה "
                "כ-250 ק״מ, וחלק גדול ממנה נסלל לצד הקנאלטות ובתוך הפארקים, "
                "כלומר בדיוק במקומות שבהם ממילא לא עוברות מכוניות.",
    },
]


# ------------------------------------------------------------------ the media

def http_json(url, what):
    """One GET that answers JSON, or None. Never fatal: a build with no
    pictures is a worse layer, and a build that died is no layer."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        return json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as err:                     # network, 404, rate limit
        print(f"  {what}: {err}", flush=True)
        return None


def commons_photos(titles):
    """Resolve Commons file titles to a thumbnail, a display copy and a credit.

    Two renditions and never the original: several of these are 5000 px wide and
    weigh megabytes, and the app would fetch that to fill a phone screen. The
    thumbnailer takes any width, so 500 for the strip and 1600 for the lightbox
    matches exactly what `build_data.py` writes for the initiative's own photos.

    Credit is read off Commons rather than written here, because that is the
    condition the pictures are offered under and a copy of it would go stale.
    """
    if not titles:
        return {}

    def ask(width):
        url = COMMONS + "?" + urllib.parse.urlencode({
            "action": "query", "format": "json", "titles": "|".join(titles),
            "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": width,
        })
        doc = http_json(url, "ויקישיתוף")
        pages = (doc or {}).get("query", {}).get("pages", {}) or {}
        return {p["title"]: (p.get("imageinfo") or [{}])[0]
                for p in pages.values() if p.get("title")}

    small, large = ask(500), ask(1600)
    strip = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()

    # Commons hangs `?utm_source=…&utm_campaign=imageinfo` off every url the API
    # hands out. Dropping it is not only tidiness: it is a campaign tag that
    # would be sent by every visitor's browser on every load, and it says
    # nothing this app needs the image server to know.
    plain = lambda u: u.split("?", 1)[0] if u else u

    out = {}
    for title in titles:
        thumb, full = small.get(title, {}), large.get(title, {})
        if not thumb.get("thumburl"):
            print(f"  חסרה תמונה בוויקישיתוף: {title}", flush=True)
            continue
        meta = full.get("extmetadata", thumb.get("extmetadata", {})) or {}
        artist = strip(meta.get("Artist", {}).get("value"))
        lic = strip(meta.get("LicenseShortName", {}).get("value"))
        credit = " · ".join(x for x in (artist, lic) if x)
        out[title] = {
            "thumb": plain(thumb["thumburl"]),
            "full": plain(full.get("thumburl") or thumb["thumburl"]),
            "cap": credit or "ויקישיתוף",
        }
    return out


def youtube_ok(vid):
    """Does this video still exist and still allow embedding.

    oEmbed is the free way to ask: it needs no key, and it answers 404 both for
    a video that was taken down and for one whose owner has switched embedding
    off - which are different things to a person and the same thing to this map,
    because both come out as a tile that plays nothing.
    """
    url = OEMBED + "?" + urllib.parse.urlencode({
        "format": "json", "url": f"https://www.youtube.com/watch?v={vid}"})
    doc = http_json(url, f"יוטיוב {vid}")
    return bool(doc and doc.get("title"))


def video_entry(vid):
    return {"yt": vid, "thumb": f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"}


def attach_media(wanted, skip):
    """Hang the curated picture and clip on the items that asked for one.

    `wanted` is a list of (item, spec) pairs - a list and not a dictionary
    keyed by item, because the items are plain dicts and a dict cannot be a key.
    Both halves of a spec are optional in the result: a picture that has gone
    from Commons and a video that has been taken down each drop out on their
    own, and the item keeps whatever survived. The picture goes first, because a
    strip that opens with a YouTube thumbnail reads as a video gallery.
    """
    if skip or not wanted:
        return
    titles = sorted({w["file"] for _, w in wanted if w.get("file")})
    photos = commons_photos(titles)
    checked = {}
    for it, want in wanted:
        shot = photos.get(want.get("file"))
        if shot:
            it["photos"].append(dict(shot))
        vid = want.get("yt")
        if vid:
            if vid not in checked:
                checked[vid] = youtube_ok(vid)
                time.sleep(0.4)              # one clip at a time, politely
            if checked[vid]:
                it["photos"].append(video_entry(vid))


# ------------------------------------------------------------------- assembly

def build_landmarks(doc):
    """The pins, from whatever Overpass matched by name."""
    found = {}
    for el in doc["elements"]:
        name = el.get("tags", {}).get("name")
        if name not in LANDMARKS or name in found:
            continue
        centre = el.get("center") or el
        if centre.get("lat") is None:
            continue
        found[name] = (centre["lat"], centre["lon"])

    out = []
    for name, spec in LANDMARKS.items():
        if name not in found:
            print(f"  לא נמצא ב-OSM: {name}", flush=True)
            continue
        lat, lng = found[name]
        out.append(landmark(spec["he"], lat, lng, spec["note"]))
    return out


def pick_line_media(segments):
    """Which segment each line rule lands on: the longest in its group whose
    name matches, or the longest in the group when there is no pattern.

    Returns a list of (segment, rule). Longest and not first, so that the choice
    does not depend on the order Overpass answered in - which is the same
    reasoning as the ids, one level up.
    """
    picked = []
    for rule in LINE_MEDIA:
        pool = [s for s in segments if s["group"] == rule["group"]]
        if rule["match"]:
            pool = [s for s in pool
                    if re.search(rule["match"], s["name"], re.I)
                    or any(re.search(rule["match"], t, re.I) for t in s["streets"])]
        if not pool:
            print(f'  אין מקטע מתאים לכלל: {rule["group"]} / '
                  f'{rule["match"] or "הארוך ביותר"}', flush=True)
            continue
        picked.append((max(pool, key=lambda s: s["length"]), rule))
    return picked


def main():
    refresh = "--refresh" in sys.argv
    skip_media = "--skip-media" in sys.argv

    canal = overpass("canaletas", Q_CANAL, refresh)
    bike = overpass("bike", Q_BIKE, refresh)
    walk = overpass("pedestrian", Q_WALK, refresh)
    routes = overpass("routes", Q_ROUTES, refresh)
    marks = overpass("landmarks", Q_LANDMARKS, refresh)
    rings = bairro_rings(overpass("bairros", Q_BAIRROS, refresh))
    print(f"  שכונות: {len(rings)}", flush=True)

    # A signed route runs along the very cycleways in `bike`, so the ways it is
    # made of are dropped from the mesh: otherwise every marked route is drawn
    # twice, once as itself and once as the paths underneath it, and clicking
    # the line picks whichever happened to end up on top.
    on_route = route_way_ids(routes)
    paths = [w for w in bike["elements"] if w["id"] not in on_route]

    segments = []
    segments += chained_group(canal["elements"], CANAL, "קנאלטה", rings, floor=40)
    segments += chained_group(paths, PATHS, "שביל", rings)
    segments += chained_group(walk["elements"], WALK, "מדרחוב", rings, floor=25)
    segments += route_segments(routes)
    assign_ids(segments, CURITIBA_ID)

    waypoints = assign_ids(build_landmarks(marks), CURITIBA_ID + "-p")

    print()
    for group in (g["name"] for g in GROUPS):
        rows = [s for s in segments if s["group"] == group]
        km = sum(s["length"] for s in rows) / 1000
        print(f"  {group}: {len(rows)} מקטעים, {km:.1f} ק״מ")
    print(f"  נקודות ציון: {len(waypoints)}")

    bike_km = sum(s["length"] for s in segments if s["group"] == PATHS) / 1000
    canal_km = sum(s["length"] for s in segments if s["group"] == CANAL) / 1000
    print(f"\n  רשת האופניים: {bike_km:.1f} ק״מ · הקנאלטות: {canal_km:.1f} ק״מ")

    # The pictures and the clips, last, so that a build with no network beyond
    # Overpass still produces the whole layer.
    wanted = []
    for seg, rule in pick_line_media(segments):
        seg["note"] = rule["note"]
        wanted.append((seg, rule))
    by_he = {spec["he"]: spec for spec in LANDMARKS.values()}
    for wp in waypoints:
        spec = by_he.get(wp["name"])
        if spec:
            wanted.append((wp, spec))
    attach_media(wanted, skip_media)
    shots = sum(len(x["photos"]) for x in segments + waypoints)
    print(f"  תמונות וסרטונים משובצים: {shots}")

    # What the app flies to when the layer is switched on. The built-up city and
    # not the municipality: Curitiba's boundary reaches well past anything this
    # layer draws, and framing all of it puts the network in the middle of empty
    # ground. The canaletas are exactly the spine of the built city.
    pts = [p for s in segments if s["group"] == CANAL for p in s["path"]]
    lats = [p[0] for p in pts]
    lngs = [p[1] for p in pts]
    bounds = [[round(min(lats), 5), round(min(lngs), 5)],
              [round(max(lats), 5), round(max(lngs), 5)]]

    layer = {
        "id": CURITIBA_ID,
        "name": "קוריטיבה",
        "short": "קוריטיבה",
        "color": COLOR[ROUTES],
        "dash": False,
        "note": "העיר הברזילאית שנתנה לאוטובוס כביש משלו במרכז, דחקה את "
                "המכוניות לצדדים, סגרה את רחוב המסחר שלה למכוניות ב-1972 "
                "והפכה את גדות הנחלים לפארקים במקום לתעל אותם. שכבת הדגמה, "
                "לא נתונים על המושבה: הדלקה שלה מטיסה את המפה לברזיל.",
        "credit": CREDIT,
        "groups": GROUPS,
        "bounds": bounds,
        "on": False,
        "segments": segments,
        "waypoints": waypoints,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"credit": CREDIT, "layers": [layer]}, fh,
                  ensure_ascii=False, separators=(",", ":"))
    print(f"\nwrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB, "
          f"{len(segments)} מקטעים, {len(waypoints)} נקודות)")


if __name__ == "__main__":
    main()
