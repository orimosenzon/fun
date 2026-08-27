#!/usr/bin/env python3
"""Import a recorded track from off-road.io as a layer in the app.

off-road.io is an Israeli off-road community site. Its tracks are GPS
recordings uploaded by users, which makes them a different kind of source from
everything else the app carries: the initiative's own trails are surveyed, and
the cycling network comes from a planner's shapefiles. A recording is one
person's actual walk on one day, so it wanders where they wandered.

The site is a React app with no download button, but its backend is a plain
Google Cloud Endpoints API that answers without a key or a login:

    /_ah/api/offroadApi/v2/tracks/{id}              metadata, no geometry
    /_ah/api/offroadApi/v2/trackLayers/{layerKey}   the actual points

The metadata carries ``trackLayerKey``, which is what the second call wants -
the track id itself returns 404 there. Points arrive as full GPS fixes with
altitude, speed and timestamps; only lat/lng survive into the app.

**Attribution is not optional here.** Every track has a named owner, and the
layer records them in ``credit`` so the app shows it under the layer list.
Check that the owner is happy before publishing someone else's recording.

    python3 import_offroad.py https://off-road.io/track/6177368023236608
    python3 import_offroad.py 6177368023236608 --write

Written layers are merged into ``web/data/layers.json`` by id, leaving the
cycling-network layers alone. ``build_network.py`` returns the favour.
"""

import argparse
import datetime
import json
import math
import os
import re
import sys
import urllib.request

API = "https://api.off-road.io/_ah/api/offroadApi/v2"
OUT = "web/data/layers.json"

# Points of interest a track's owner pinned along it. The endpoint comes from
# the API's own discovery document, which lists all 175 methods:
#   /_ah/api/discovery/v1/apis/offroadApi/v2/rest
# Most tracks have none - it returns an empty collection rather than a 404 -
# so an import with no waypoints is the ordinary case, not a failure.
MAP_ITEMS = "mapItems/track"

# Raw GPS altitude wanders by a metre or two per fix, and summing every rise
# turns that noise into hundreds of metres of imaginary climbing. Averaging
# over this many fixes first costs real detail only on climbs shorter than
# about fifty metres, which do not belong in a summary line anyway.
ALT_WINDOW = 5

# Metres. GPS fixes on foot jitter by a few metres even standing still, and at
# the zoom the moshava is viewed at, anything under this is invisible. Douglas-
# Peucker at 4 m drops roughly two thirds of the points and moves the line by
# less than the width it is drawn with.
SIMPLIFY_M = 4.0

# off-road.io grades 1-5. The app has no difficulty field, so it goes into the
# note where a walker will actually read it.
DIFFICULTY = {1: "קלה מאוד", 2: "קלה", 3: "בינונית", 4: "קשה", 5: "קשה מאוד"}


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


def fetch(url):
    # The API answers curl but returns 403 to urllib's default agent string,
    # so it is filtering on the header rather than on anything about the caller.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


# ---------------------------------------------------------------- simplify

def point_to_segment_m(p, a, b):
    """Metres from p to the segment ab, flat-earth over a few hundred metres."""
    rad = math.pi / 180
    scale = math.cos(a[0] * rad)
    ax, ay = a[1] * scale, a[0]
    bx, by = b[1] * scale, b[0]
    px, py = p[1] * scale, p[0]
    dx, dy = bx - ax, by - ay
    span = dx * dx + dy * dy
    t = 0.0 if span == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / span))
    near = (ay + t * dy, (ax + t * dx) / scale)
    return haversine(p, near)


def simplify(path, epsilon):
    """Douglas-Peucker, iterative so a 900-point recording cannot blow the stack."""
    if len(path) < 3:
        return list(path)

    keep = [False] * len(path)
    keep[0] = keep[-1] = True
    stack = [(0, len(path) - 1)]
    while stack:
        first, last = stack.pop()
        worst, worst_at = 0.0, None
        for i in range(first + 1, last):
            d = point_to_segment_m(path[i], path[first], path[last])
            if d > worst:
                worst, worst_at = d, i
        if worst_at is not None and worst > epsilon:
            keep[worst_at] = True
            stack.append((first, worst_at))
            stack.append((worst_at, last))
    return [p for p, k in zip(path, keep) if k]


# ---------------------------------------------------------------- build

def track_id(argument):
    """Accept a full track URL or a bare id."""
    match = re.search(r"(\d{6,})", argument)
    if not match:
        sys.exit(f"לא זיהיתי מזהה מסלול ב-{argument!r}")
    return match.group(1)


def elevation_gain(points):
    """Metres climbed, after smoothing the altitude noise out of the fixes."""
    alts = [p["altitude"] for p in points if p.get("altitude") is not None]
    if len(alts) < ALT_WINDOW:
        return None
    half = ALT_WINDOW // 2
    smooth = [sum(alts[max(0, i - half):i + half + 1]) /
              len(alts[max(0, i - half):i + half + 1]) for i in range(len(alts))]
    return round(sum(max(0.0, b - a) for a, b in zip(smooth, smooth[1:])))


def recorded_span(points):
    """Wall-clock hours the recording covers, or None if it carries no clocks."""
    stamps = [p["timestamp"] for p in points if p.get("timestamp")]
    if len(stamps) < 2:
        return None
    first, last = (datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
                   for s in (stamps[0], stamps[-1]))
    return (last - first).total_seconds() / 3600


def median_speed_kmh(points):
    """Typical speed over the whole recording. Median, not mean, because a
    handful of GPS jumps read as 45 km/h and would drag an average with them."""
    speeds = sorted(p["speed"] * 3.6 for p in points if p.get("speed") is not None)
    return speeds[len(speeds) // 2] if speeds else None


def photos_of(url):
    """off-road.io serves one image per track, at full size and nothing else.
    Both slots point at it: the app wants a thumb, and a wrong-sized thumb
    beats the grey placeholder it shows when the list has none."""
    return [{"thumb": url, "full": url}] if url else []


def to_waypoint(item, index, layer_id, color):
    """One off-road.io map item as an app waypoint."""
    point = item.get("point") or {}
    lat, lng = point.get("latitude"), point.get("longitude")
    if lat is None or lng is None:
        return None
    photo = item.get("serveUrl") or item.get("resourceUrl")
    owner = (item.get("ownerDisplayName") or "").strip()
    note = (item.get("description") or "").strip()
    if owner:
        note = f"{note}\nנוסף בידי {owner}.".strip()
    return {
        "id": f"{layer_id}-w{index}",
        "name": (item.get("title") or "נקודת עניין").strip(),
        "note": note,
        "photos": photos_of(photo),
        "lat": round(lat, 6),
        "lng": round(lng, 6),
        "entries": [{"lat": round(lat, 6), "lng": round(lng, 6)}],
        "color": color,
        "links": ([{"url": item["externalUrl"], "title": "off-road.io"}]
                  if item.get("externalUrl") else []),
    }


def build_layer(track, points, items, layer_id, color, on):
    path = [[round(p["latitude"], 6), round(p["longitude"], 6)] for p in points]
    raw_km = path_length(path) / 1000
    path = simplify(path, SIMPLIFY_M)

    title = (track.get("title") or "מסלול").strip()
    owner = (track.get("ownerDisplayName") or "").strip()
    credit = f"{title}, הוקלט בידי {owner}, off-road.io" if owner \
        else f"{title}, off-road.io"

    length = round(path_length(path))
    loop = bool(track.get("activities", {}).get("roundTrip"))
    difficulty = DIFFICULTY.get(track.get("difficultyLevel"))
    climb = elevation_gain(points)
    hours = recorded_span(points)
    speed = median_speed_kmh(points)
    when = (track.get("created") or "")[:10]

    # Everything the recording actually proves, in the order a walker asks it.
    # The site's own `duration` field is ignored: it reads 8 hours here against
    # a recording that took 2:16, so it is a guess the uploader typed, while
    # the timestamps on the fixes are measurements.
    #
    # This goes on the segment rather than the layer, because the segment is
    # what opens when the line is tapped, and these are facts about the walk.
    facts = []
    if loop:
        facts.append("מסלול מעגלי")
    if climb:
        facts.append(f"טיפוס מצטבר {climb} מ׳")
    if hours:
        facts.append(f"ההקלטה ארכה {int(hours)}:{int(hours % 1 * 60):02d} שעות")
    if difficulty:
        facts.append(f"דרגת קושי {difficulty} לפי המקליט")

    recorded = f"הוקלט בידי {owner}" if owner else "הוקלט"
    if when:
        recorded += f" ב-{when[8:10]}/{when[5:7]}/{when[:4]}"
    detail = f"{recorded}. " + (", ".join(facts) + "." if facts else "")

    # The recording moves far too fast for the "Walking" label the site files it
    # under, and this app is about walking. Saying so is the honest thing: the
    # line is a real route, but nobody walked it at this pace.
    if speed and speed > 8:
        detail += (f" מסומן באתר כמסלול הליכה, אך ההקלטה מתקדמת בקצב חציוני של "
                   f"{speed:.0f} קמ״ש, כלומר נרשמה ברכיבה ולא בהליכה.")

    note = "הקלטת GPS של מסלול שלם מ-off-road.io, לא רשת מקטעים מתוכננת."

    segment = {
        "id": f"{layer_id}-0",
        "name": title,
        "note": detail.strip(),
        "photos": photos_of(track.get("backgroundServeUrl")),
        "path": path,
        "length": length,
        "color": color,
        "layer": layer_id,
        # Deliberately empty. `grade` is the cycling network's tier, and the app
        # renders it as "רשת <value>" (app.js), so a difficulty here would read
        # as "רשת בינונית". The difficulty is in the note instead.
        "grade": "",
        "kind": "",
        "status": "קיים",
        "streets": [],
        # A loop has one useful way in, where the recording began. Adding the
        # end point too would drop two pins 46 m apart on the same spot.
        "entries": ([{"lat": path[0][0], "lng": path[0][1]}] if loop else
                    [{"lat": path[0][0], "lng": path[0][1]},
                     {"lat": path[-1][0], "lng": path[-1][1]}]),
    }

    waypoints = [w for w in (to_waypoint(it, i, layer_id, color)
                             for i, it in enumerate(items)) if w]

    layer = {
        "id": layer_id,
        "name": title,
        "short": title.split()[0] if title else "מסלול",
        "color": color,
        "dash": False,
        "note": note,
        "credit": credit,
        "on": on,
        "segments": [segment],
        # Same layer as the line, so switching the track on brings its points
        # with it. layers.js gives every layer a `waypoints` array anyway.
        "waypoints": waypoints,
    }
    return layer, raw_km, len(points)


def merge(layer):
    """Replace this layer by id in layers.json, leaving every other layer alone."""
    try:
        with open(OUT, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        doc = {"credit": "", "layers": []}

    layers = [l for l in doc.get("layers", []) if l.get("id") != layer["id"]]
    layers.append(layer)          # on top: a single walked route reads best above the network
    doc["layers"] = layers

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, separators=(",", ":"))
    return len(layers)


def main():
    parser = argparse.ArgumentParser(description="ייבוא מסלול מ-off-road.io")
    parser.add_argument("track", help="כתובת המסלול או המזהה שלו")
    parser.add_argument("--id", dest="layer_id", default=None,
                        help="מזהה השכבה באפליקציה (ברירת מחדל: offroad-<track id>)")
    parser.add_argument("--color", default="#6a1b9a")
    parser.add_argument("--on", action="store_true",
                        help="להדליק את השכבה כברירת מחדל")
    parser.add_argument("--write", action="store_true",
                        help="לכתוב ל-web/data/layers.json (אחרת רק תצוגה מקדימה)")
    args = parser.parse_args()

    tid = track_id(args.track)
    track = fetch(f"{API}/tracks/{tid}")

    key = track.get("trackLayerKey")
    if not key:
        sys.exit(f"למסלול {tid} אין trackLayerKey, אין ממה לבנות גיאומטריה")
    bundle = fetch(f"{API}/trackLayers/{key}")

    points = [p for sub in bundle.get("layers", []) for p in sub.get("path", [])]
    if not points:
        sys.exit(f"שכבת המסלול {key} חזרה בלי נקודות")

    items = fetch(f"{API}/{MAP_ITEMS}/{tid}").get("items", [])

    layer_id = args.layer_id or f"offroad-{tid}"
    layer, raw_km, raw_points = build_layer(track, points, items, layer_id,
                                            args.color, args.on)
    path = layer["segments"][0]["path"]

    print(f'{layer["name"]}  ({track.get("ownerDisplayName", "?")}, {track.get("created", "")[:10]})')
    print(f"  נקודות מסלול: {raw_points} → {len(path)} אחרי פישוט ב-{SIMPLIFY_M:.0f} מ׳")
    print(f'  אורך: {raw_km:.2f} ק״מ → {layer["segments"][0]["length"] / 1000:.2f} ק״מ')
    print(f'  תמונות: {len(layer["segments"][0]["photos"])}')
    print(f'  נקודות עניין: {len(layer["waypoints"])}'
          + ('  (למסלול הזה לא הוצמדו נקודות באתר)' if not items else ''))
    print(f'  מקור: {layer["credit"]}')
    print(f'  הערה: {layer["segments"][0]["note"]}')

    if not args.write:
        print("\nתצוגה מקדימה בלבד. הוסף --write כדי לכתוב ל-" + OUT)
        return

    total = merge(layer)
    print(f"\nנכתב {OUT}: {total} שכבות, {os.path.getsize(OUT) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
