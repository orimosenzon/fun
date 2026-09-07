#!/usr/bin/env python3
"""Build the Houten layer: a Dutch town's bike network, as a worked example.

Houten (Utrecht province, ~50,000 people) is the town this project keeps being
pointed at. It was laid out from 1979 around one rule: **the bicycle goes
through, the car goes around.** Every neighbourhood hangs off a ring road that
cars must return to in order to reach the next one, while the bike paths run
straight across the middle to the railway station. The result is that for most
trips inside the town the bike is not the virtuous choice, it is the *short*
one - which is this project's own name, built at the scale of a whole town.

So the layer is not only bike paths. It carries the car ring road as well, in
its own colour, because the bike network on its own is just a dense mesh: the
lesson is only visible in the contrast between the two.

Everything comes from OpenStreetMap through Overpass, in four groups:

  the eight signed routes   relations tagged ``cycle_network=Fietsnet Houten``,
                            the town's own numbered bike network
  bike streets              ``cyclestreet=yes`` - a street where cars are
                            guests, 30 km/h, and may not overtake a bicycle
  bike paths                everything else tagged ``highway=cycleway`` or
                            designated for bicycles
  the car ring road         ``name=Rondweg`` - the detour the cars take

Overpass hands out ways chopped at every junction, so 969 fragments become a
few hundred readable segments by chaining whatever meets end to end.

    python3 build_houten.py [--refresh]

Raw Overpass answers are cached under ``.cache/houten/``; ``--refresh`` throws
them away and asks again.
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

OUT = "web/data/houten.json"
CACHE = ".cache/houten"

API = "https://overpass-api.de/api/interpreter"
STATUS = "https://overpass-api.de/api/status"
UA = "derech-kitzur/build_houten.py (github.com/orimosenzon/fun)"

# The municipality of Houten, OSM relation 419211. Overpass area ids are the
# relation id plus 3600000000.
AREA = 3600419211

CREDIT = "OpenStreetMap contributors, ODbL"

# ------------------------------------------------------------------ the groups
#
# Order is drawing order, bottom first: the ring road goes down first so the
# bike network is never hidden under it, and the eight signed routes go last so
# they read on top of the mesh they run along.

RING = "כביש הטבעת של המכוניות"
PATHS = "שבילי אופניים"
STREETS = "רחובות אופניים"
ROUTES = "שמונת המסלולים המסומנים"

GROUPS = [
    {"name": RING, "color": "#b0392f"},
    {"name": PATHS, "color": "#1565c0"},
    {"name": STREETS, "color": "#e08a00"},
    {"name": ROUTES, "color": "#2e7d32"},
]

COLOR = {g["name"]: g["color"] for g in GROUPS}

# What each group is, in one line, shown on the segment itself.
ABOUT = {
    RING: "הכביש שהמכוניות נוסעות בו כדי לעבור משכונה לשכונה. אין דרך לחצות את "
          "העיר במכונית בלעדיו, וזה מה שהופך את האופניים לקצרים יותר.",
    PATHS: "שביל אופניים מופרד. רוב הרשת של האוטן היא כזאת: תוואי משלה, לא שוליים "
           "של כביש.",
    STREETS: "רחוב אופניים (fietsstraat): המכוניות אורחות בו, מוגבלות ל-30 קמ״ש "
             "ואסור להן לעקוף רוכב.",
    ROUTES: "אחד המסלולים הממוספרים של רשת האופניים העירונית, מסומן בשילוט בשטח.",
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

    for attempt in range(5):
        wait_for_slot()
        req = urllib.request.Request(
            API, data=urllib.parse.urlencode({"data": query}).encode(),
            headers={"User-Agent": UA})
        body = urllib.request.urlopen(req, timeout=600).read()
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


Q_BIKE = f"""[out:json][timeout:300];
area({AREA})->.a;
(
  way(area.a)["highway"="cycleway"];
  way(area.a)["highway"~"^(path|footway|track|service|residential|unclassified|living_street|pedestrian)$"]["bicycle"="designated"];
);
out geom;"""

Q_EXTRA = f"""[out:json][timeout:300];
area({AREA})->.a;
(
  way(area.a)["name"="Rondweg"]["highway"];
  way(area.a)["cyclestreet"="yes"];
);
out geom;"""

Q_ROUTES = """[out:json][timeout:300];
rel["cycle_network"="Fietsnet Houten"];
out geom;"""


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


# Two fragments count as meeting when their endpoints round to the same spot.
# Six decimals is about 10 cm, which is finer than any two ways that genuinely
# share a node will ever differ by.
def key(point):
    return (round(point[0], 6), round(point[1], 6))


def chain(paths, labels=None):
    """Join fragments that meet end to end into as few polylines as possible.

    Overpass hands out a way per junction, so a single path across town arrives
    as twenty pieces. Left as they are, every one of them is a separate row in
    the app's list and a separate thing to tap - the town's main north-south
    bike route would read as twenty nameless stubs.

    Greedy and not optimal: start from a fragment, extend at both ends for as
    long as one unused fragment continues it, then start again. At a junction
    where several fragments meet, the chain takes the one carrying the same
    street name and otherwise stops rather than guessing - so a path that runs
    dead straight through four side turnings stays one line, while a genuine
    fork stays two. Without the name rule the mesh came out as a hundred junction
    stubs under fifty metres, one row each in the app's list.

    `labels` is the name to match on, one per path; pass none to chain on
    geometry alone. Returns a list of (line, members) - the joined polyline and
    the indices that went into it, so the caller can name the result after
    whatever the pieces were called.
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

        # Extend forwards, then backwards off the same growing line.
        for _ in range(2):
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
                # An unnamed link picks up the name of what it joins, so a chain
                # does not lose its thread crossing one.
                label = label or name_of(i)
                line += (piece[1:] if key(piece[0]) == here
                         else list(reversed(piece))[1:])
            line.reverse()

        out.append((line, members))
    return out


# --------------------------------------------------------------- the segments

def note_for(tags, group):
    """The one line under a segment's name: what the group is, plus whatever
    this particular stretch says about itself that a rider would care about."""
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
    return ABOUT[group] + (" · " + " · ".join(bits) if bits else "")


HOUTEN_ID = "houten"


def line_id(prefix, path):
    """A segment's id, derived from the line itself.

    This used to be the segment's position in the answer Overpass happened to
    give - ``houten-0``, ``houten-1`` - which is fine for as long as nothing
    else refers to it, and stopped being fine on 7/9/2026, when the app grew a
    way to attach a photo or a video to any item on the map. Those attachments
    live in ``data/media.json`` keyed by id, and this file is rebuilt from
    OpenStreetMap: one new cycleway mapped anywhere in Houten would have
    renumbered everything after it and moved somebody's photo onto a different
    road, silently and with no way to notice.

    Five decimals is about a metre, which is finer than any edit that leaves
    the line meaning the same thing and coarser than the noise. So the id
    survives a rebuild and changes when, and only when, the line does.
    """
    digest = hashlib.sha1(
        ";".join(f"{lat:.5f},{lng:.5f}" for lat, lng in path).encode()
    ).hexdigest()[:8]
    return f"{prefix}-{digest}"


def assign_ids(segments, prefix):
    """Name every segment after its geometry, in one pass so that two identical
    lines - which the ring road and a cycleway alongside it can genuinely be -
    get told apart rather than overwriting each other."""
    taken = set()
    for seg in segments:
        ident = line_id(prefix, seg["path"])
        base, n = ident, 2
        while ident in taken:
            ident = f"{base}-{n}"
            n += 1
        taken.add(ident)
        seg["id"] = ident
    return segments


def segment(index, group, name, path, tags, streets):
    return {
        # Filled in by `assign_ids` once every segment exists.
        "id": "",
        "name": name,
        "note": note_for(tags, group),
        "photos": [],
        "path": path,
        "length": round(path_length(path)),
        "color": COLOR[group],
        "layer": HOUTEN_ID,
        "group": group,
        "kind": "",
        "streets": streets,
        "entries": [
            {"lat": path[0][0], "lng": path[0][1]},
            {"lat": path[-1][0], "lng": path[-1][1]},
        ],
    }


def chained_group(ways, group, label, index):
    """Chain a pool of ways into segments and name each after where it runs.

    Chaining is by connectivity alone and ignores the names, which matters: an
    earlier version pooled by name first, and every short unnamed link between
    two named paths - there are over a hundred of them - came out as its own
    stub, because its neighbours had been chained away into other pools. The
    mesh reads far better as "Kooikerspad / Imkerspad" than as three rows.

    Street names then come off whatever the chain actually ran along, at most
    three of them, the same way `build_network.py` names the moshava's own
    segments.
    """
    paths = [geom(w) for w in ways]
    labels = [w["tags"].get("name") for w in ways]
    out = []
    for line, members in chain(paths, labels):
        if len(line) < 2 or path_length(line) < 15:
            continue

        pieces = [ways[i] for i in members]
        # Longest piece speaks for the chain: on a stretch that is asphalt for
        # 400 m and paving stones for 20, asphalt is the answer.
        tags = max(pieces, key=lambda w: len(w.get("geometry", [])))["tags"]

        # In the order they are ridden, not alphabetically, and without the
        # repeat when a street is left and rejoined.
        streets = []
        for way in pieces:
            name = way["tags"].get("name")
            if name and name not in streets:
                streets.append(name)

        title = f"{label} · {' / '.join(streets[:3])}" if streets \
            else f"{label} · מקטע {index}"
        out.append(segment(index, group, title, line, tags, streets))
        index += 1
    return number_duplicates(out), index


def number_duplicates(segments):
    """A name that lands on more than one segment gets a running number.

    The ring road is the reason: it arrives as fourteen pieces all called
    Rondweg, which in a list is fourteen rows that look like the same row. The
    numbers run longest first, so `· 1` is the main run rather than whichever
    stub Overpass happened to return first.
    """
    seen = collections.Counter(s["name"] for s in segments)
    order = collections.Counter()
    for s in sorted(segments, key=lambda s: -s["length"]):
        if seen[s["name"]] > 1:
            order[s["name"]] += 1
            s["name"] = f'{s["name"]} · {order[s["name"]]} מתוך {seen[s["name"]]}'
    return segments


def route_segments(doc, index):
    """One signed route per relation, chained out of its member ways."""
    out = []
    for rel in sorted(doc["elements"], key=lambda r: r["tags"].get("ref", "")):
        tags = rel["tags"]
        ref = tags.get("ref", "?")
        members = [m for m in rel.get("members", [])
                   if m["type"] == "way" and m.get("geometry")]
        pieces = [line for line, _ in chain([geom(m) for m in members])]
        # A route arrives as one line when its ways are in order and unbroken,
        # and as a handful when the relation skips a junction. Longest first, so
        # the piece that carries the route's name is the route itself.
        pieces.sort(key=path_length, reverse=True)
        for i, path in enumerate(pieces):
            if len(path) < 2 or path_length(path) < 40:
                continue
            name = f"מסלול {ref} של רשת האוטן"
            if i:
                name += f" · קטע {i + 1}"
            out.append(segment(index, ROUTES, name, path, tags, []))
            index += 1
    return out, index


def main():
    refresh = "--refresh" in sys.argv

    bike = overpass("bike", Q_BIKE, refresh)
    extra = overpass("extra", Q_EXTRA, refresh)
    routes = overpass("routes", Q_ROUTES, refresh)

    # A way can be all three things at once - a cycleway, a bike street, and a
    # leg of a signed route - and drawing it three times would put three lines
    # on top of each other in three colours. Each way lands in exactly one
    # group, the most specific one it qualifies for.
    on_route = {m["ref"] for rel in routes["elements"]
                for m in rel.get("members", []) if m["type"] == "way"}

    ring, streets = [], []
    for way in extra["elements"]:
        tags = way.get("tags", {})
        if tags.get("name") == "Rondweg":
            ring.append(way)
        elif way["id"] not in on_route:
            streets.append(way)

    bike_street_ids = {w["id"] for w in streets}
    paths = [w for w in bike["elements"]
             if w["id"] not in on_route and w["id"] not in bike_street_ids]

    index = 0
    segments = []
    for ways, group, label in ((ring, RING, "כביש הטבעת"),
                               (paths, PATHS, "שביל"),
                               (streets, STREETS, "רחוב אופניים")):
        made, index = chained_group(ways, group, label, index)
        segments += made
    made, index = route_segments(routes, index)
    segments += made
    assign_ids(segments, HOUTEN_ID)

    print()
    for group in (g["name"] for g in GROUPS):
        rows = [s for s in segments if s["group"] == group]
        km = sum(s["length"] for s in rows) / 1000
        print(f"  {group}: {len(rows)} מקטעים, {km:.1f} ק״מ")
    bike_km = sum(s["length"] for s in segments if s["group"] != RING) / 1000
    ring_km = sum(s["length"] for s in segments if s["group"] == RING) / 1000
    print(f"\n  רשת האופניים: {bike_km:.1f} ק״מ מול {ring_km:.1f} ק״מ של כביש הטבעת")

    # What the app flies to. Bounds and not a centre-and-zoom, because a guessed
    # zoom is wrong on every screen shape and this is right on all of them.
    #
    # The ring road's bounds and not the whole layer's. The municipality reaches
    # ten kilometres south into the polder, and framing all of it puts the town -
    # the entire reason to look at this - as a small knot in the middle of empty
    # fields. The ring is exactly the town's own edge, which is the point being
    # made; the paths running out into the countryside stay a zoom-out away.
    ring_pts = [p for s in segments if s["group"] == RING for p in s["path"]]
    lats = [p[0] for p in ring_pts]
    lngs = [p[1] for p in ring_pts]
    bounds = [[round(min(lats), 5), round(min(lngs), 5)],
              [round(max(lats), 5), round(max(lngs), 5)]]

    layer = {
        "id": HOUTEN_ID,
        "name": "האוטן",
        "short": "האוטן",
        "color": COLOR[ROUTES],
        "dash": False,
        "note": "העיר ההולנדית שבנתה את עצמה סביב הכלל שהאופניים עוברות בתוך "
                "העיר והמכונית מקיפה אותה מבחוץ. שכבת הדגמה, לא נתונים על "
                "המושבה: הדלקה שלה מטיסה את המפה להולנד.",
        "credit": CREDIT,
        "groups": GROUPS,
        "bounds": bounds,
        "on": False,
        "segments": segments,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"credit": CREDIT, "layers": [layer]}, fh,
                  ensure_ascii=False, separators=(",", ":"))
    print(f"\nwrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB, "
          f"{len(segments)} מקטעים)")


if __name__ == "__main__":
    main()
