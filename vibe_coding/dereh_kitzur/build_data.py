#!/usr/bin/env python3
"""Turn the live "דרך קיצור" My Maps layer into the app's data file.

The map is the source of truth and changes often, so this is a regenerator:
re-run it and web/data/trails.json is current. Nothing here is hand-edited.

Photos are mirrored into web/img/ rather than hotlinked. That is not a
preference: Google serves the My Maps photos with
`cross-origin-resource-policy: same-site`, so a browser on any other origin
refuses to render them (curl does not enforce it, which makes the problem easy
to miss). Mirroring also lets us re-encode 6 MB PNGs into webp a phone can
actually load. Downloads are cached by URL, so re-runs only fetch new photos.

Usage:
    python3 build_data.py [output.json]
    python3 build_data.py --no-photos      # data only, skip image mirroring
"""
import hashlib
import html
import io
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import requests
from PIL import Image

MID = "19q4SgSScTh3pW3lRqQO9dQ8kw8z2wTg"
KML_URL = f"https://www.google.com/maps/d/kml?mid={MID}&forcekml=1"
MAP_URL = f"https://www.google.com/maps/d/viewer?hl=iw&mid={MID}"
NS = {"k": "http://www.opengis.net/kml/2.2"}
H = {"User-Agent": "DerechKitzur/1.0 (orimosenzon@gmail.com)"}
DEFAULT_OUT = "web/data/trails.json"
IMG_DIR = "web/img"

# My Maps serves its hosted photos through a resizing proxy: the trailing
# "fife" parameter picks the rendition. Ask it for a sane size up front so we
# are not pulling 6 MB PNGs across the wire just to shrink them here.
FETCH_AS = "s2000-rw"

# What we store. The full rendition is what the lightbox opens; 1600px is the
# point where more pixels stop being visible on a phone but keep costing data.
THUMB_PX, THUMB_Q = 500, 78
FULL_PX, FULL_Q = 1600, 82


def haversine(a, b):
    """Metres between two (lon, lat) pairs."""
    r = 6371000.0
    la1, la2 = math.radians(a[1]), math.radians(b[1])
    dla = la2 - la1
    dlo = math.radians(b[0] - a[0])
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def bearing(a, b):
    """Compass degrees from (lon, lat) a towards b."""
    la1, la2 = math.radians(a[1]), math.radians(b[1])
    dlo = math.radians(b[0] - a[0])
    y = math.sin(dlo) * math.cos(la2)
    x = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlo)
    return round((math.degrees(math.atan2(y, x)) + 360) % 360)


def coords(el):
    txt = el.findtext("k:coordinates", default="", namespaces=NS).strip()
    out = []
    for tok in txt.split():
        parts = tok.split(",")
        if len(parts) >= 2:
            out.append((float(parts[0]), float(parts[1])))
    return out


def resize(url, rendition):
    """Swap whatever fife rendition the KML carries for one we chose."""
    if "fife=" in url:
        return re.sub(r"fife=[^&]*", f"fife={rendition}", url)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}fife={rendition}"


def encode(img, longest, quality, path):
    out = img.copy()
    out.thumbnail((longest, longest), Image.LANCZOS)
    if out.mode not in ("RGB", "L"):
        out = out.convert("RGB")
    out.save(path, "WEBP", quality=quality, method=5)


def mirror(url, stats):
    """Download a photo and write the two renditions the app serves.

    Files are named after a hash of the image bytes, not of the URL: Google
    re-signs every photo URL on every KML fetch, so no two runs ever see the
    same URL for the same picture. Content addressing keeps rebuilds idempotent
    - same photos in, same filenames out, no duplicates and no git churn.
    That does mean every run re-downloads; the encode step is what we skip.
    """
    try:
        res = requests.get(resize(url, FETCH_AS), headers=H, timeout=90)
        res.raise_for_status()
        raw = res.content
    except Exception as exc:                       # noqa: BLE001 - report and skip
        stats["failed"] += 1
        print(f"  ! photo download failed: {exc}", file=sys.stderr)
        return None

    key = hashlib.sha1(raw).hexdigest()[:14]
    rel = {"thumb": f"img/{key}_t.webp", "full": f"img/{key}.webp"}
    abs_thumb = os.path.join(IMG_DIR, f"{key}_t.webp")
    abs_full = os.path.join(IMG_DIR, f"{key}.webp")
    stats["seen"].update([f"{key}_t.webp", f"{key}.webp"])

    if os.path.exists(abs_thumb) and os.path.exists(abs_full):
        stats["cached"] += 1
        return rel

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:                       # noqa: BLE001 - report and skip
        stats["failed"] += 1
        print(f"  ! photo {key} unreadable: {exc}", file=sys.stderr)
        return None

    encode(img, FULL_PX, FULL_Q, abs_full)
    encode(img, THUMB_PX, THUMB_Q, abs_thumb)
    stats["fetched"] += 1
    stats["bytes"] += os.path.getsize(abs_full) + os.path.getsize(abs_thumb)
    return rel


def prune(stats):
    """Delete mirrored photos the current map no longer references."""
    removed = 0
    for name in os.listdir(IMG_DIR):
        if name.endswith(".webp") and name not in stats["seen"]:
            os.remove(os.path.join(IMG_DIR, name))
            removed += 1
    return removed


def photos(desc_raw, stats, mirror_images=True):
    seen, out = set(), []
    for src in re.findall(r'<img[^>]+src="([^"]+)"', desc_raw):
        src = html.unescape(src)
        if src in seen:
            continue
        seen.add(src)
        if not mirror_images:
            continue
        entry = mirror(src, stats)
        if entry:
            out.append(entry)
    return out


def note(desc_raw):
    """The human sentence, with the photo markup taken out."""
    text = re.sub(r"<img[^>]*>", " ", desc_raw)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def kml_color(value):
    """KML stores aabbggrr; CSS wants #rrggbb."""
    v = (value or "").strip()
    if len(v) != 8:
        return None
    return "#" + v[6:8] + v[4:6] + v[2:4]


def line_colors(root):
    """Map every style id to the line colour it paints, following StyleMaps."""
    direct = {}
    for style in root.findall(".//k:Style", NS):
        sid = style.get("id")
        line = style.find("k:LineStyle", NS)
        if sid and line is not None:
            colour = kml_color(line.findtext("k:color", namespaces=NS))
            if colour:
                direct[sid] = colour
    for smap in root.findall(".//k:StyleMap", NS):
        sid = smap.get("id")
        for pair in smap.findall("k:Pair", NS):
            if pair.findtext("k:key", namespaces=NS) == "normal":
                target = (pair.findtext("k:styleUrl", namespaces=NS) or "").lstrip("#")
                if sid and target in direct:
                    direct[sid] = direct[target]
    return direct


# --------------------------------------------------------------------------
# Street View aiming
#
# Street View only exists where the camera car drove, so a viewpoint sitting in
# the middle of a footpath usually resolves to nothing. For every access point
# we find the nearest drivable road in OpenStreetMap and, when that road is far
# enough away to matter, move the viewpoint onto it and aim the camera back at
# the trail mouth. That is also the more useful picture: it shows the entrance
# the way you meet it when walking up the street.
# --------------------------------------------------------------------------

OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
ROAD_CACHE = ".cache/roads.json"
DRIVABLE = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service"
SNAP_MIN_M = 12       # below this the entrance is already street-side; leave it
COVER_WARN_M = 35     # beyond this, warn that there may be no panorama


def fetch_roads(bounds):
    if os.path.exists(ROAD_CACHE):
        with open(ROAD_CACHE, encoding="utf-8") as fh:
            return json.load(fh)

    (s_lat, s_lng), (n_lat, n_lng) = bounds
    pad = 0.004
    query = (f'[out:json][timeout:120];way["highway"~"^({DRIVABLE})$"]'
             f'({s_lat - pad},{s_lng - pad},{n_lat + pad},{n_lng + pad});out geom;')

    for host in OVERPASS:
        try:
            res = requests.post(host, data={"data": query}, headers=H, timeout=180)
            res.raise_for_status()
            elements = res.json()["elements"]
            break
        except Exception as exc:                    # noqa: BLE001 - try the mirror
            print(f"  ! overpass {host} failed: {exc}", file=sys.stderr)
    else:
        return []

    segs = []
    for way in elements:
        geom = way.get("geometry", [])
        for a, b in zip(geom, geom[1:]):
            segs.append([a["lat"], a["lon"], b["lat"], b["lon"]])

    os.makedirs(os.path.dirname(ROAD_CACHE), exist_ok=True)
    with open(ROAD_CACHE, "w", encoding="utf-8") as fh:
        json.dump(segs, fh)
    return segs


def closest_on_segment(lat, lon, a_lat, a_lon, b_lat, b_lon):
    """Nearest point on a road segment, in metres and as a coordinate."""
    my = 111320.0
    mx = 111320.0 * math.cos(math.radians(lat))
    px, py = (lon - a_lon) * mx, (lat - a_lat) * my
    bx, by = (b_lon - a_lon) * mx, (b_lat - a_lat) * my
    span = bx * bx + by * by
    t = 0.0 if span == 0 else max(0.0, min(1.0, (px * bx + py * by) / span))
    near_lat = a_lat + (b_lat - a_lat) * t
    near_lon = a_lon + (b_lon - a_lon) * t
    return math.hypot(px - t * bx, py - t * by), near_lat, near_lon


def nearest_road(lat, lon, segs):
    best = (1e9, None, None)
    for a_lat, a_lon, b_lat, b_lon in segs:
        if abs(a_lat - lat) > 0.004 and abs(b_lat - lat) > 0.004:
            continue
        cand = closest_on_segment(lat, lon, a_lat, a_lon, b_lat, b_lon)
        if cand[0] < best[0]:
            best = cand
    return best


def aim(point, inward, segs):
    """Decide where the Street View camera should stand and where it should look.

    `point` is the access point (lon, lat); `inward` a neighbouring point on the
    trail, so that a street-side entrance still looks along the path.
    """
    lat, lon = point[1], point[0]
    view_lat, view_lon = lat, lon
    heading = bearing(point, inward)
    dist, r_lat, r_lon = nearest_road(lat, lon, segs) if segs else (None, None, None)

    if dist is not None and SNAP_MIN_M <= dist < 1e9:
        view_lat, view_lon = r_lat, r_lon
        heading = bearing((r_lon, r_lat), (lon, lat))

    return {
        "lat": round(lat, 6),
        "lng": round(lon, 6),
        "view": [round(view_lat, 6), round(view_lon, 6)],
        "heading": heading,
        "road": None if dist is None else round(dist),
        "likely": dist is not None and dist <= COVER_WARN_M,
    }


SPLIT = re.compile(r"\s*[-–—]\s*|\s+ו\s+")
NOISE = ("קיצור דרך:", "קיצור דרך", "מסלול", "דרך")


def endpoints(name):
    """Best-effort split of a segment name into the two places it connects."""
    cleaned = name
    for token in NOISE:
        if cleaned.startswith(token):
            cleaned = cleaned[len(token):]
    parts = [p.strip(" ()") for p in SPLIT.split(cleaned.strip(" :")) if p.strip(" ()")]
    return [p for p in parts if len(p) > 1]


def build(kml_bytes, mirror_images=True, use_roads=True):
    root = ET.fromstring(kml_bytes)
    colours = line_colors(root)
    segments, waypoints = [], []
    stats = {"fetched": 0, "cached": 0, "failed": 0, "bytes": 0, "seen": set()}

    if mirror_images:
        os.makedirs(IMG_DIR, exist_ok=True)

    for idx, pm in enumerate(root.findall(".//k:Placemark", NS)):
        name = (pm.findtext("k:name", default="", namespaces=NS) or "").strip()
        desc_raw = pm.findtext("k:description", default="", namespaces=NS) or ""
        style = (pm.findtext("k:styleUrl", default="", namespaces=NS) or "").lstrip("#")
        line = pm.find(".//k:LineString", NS)
        point = pm.find(".//k:Point", NS)

        common = {
            "id": f"p{idx}",
            "name": name or "ללא שם",
            "note": note(desc_raw),
            "photos": photos(desc_raw, stats, mirror_images),
        }

        if line is not None:
            pts = coords(line)
            if len(pts) < 2:
                continue
            length = sum(haversine(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
            beeline = haversine(pts[0], pts[-1])
            segments.append({
                **common,
                "path": [[round(lat, 6), round(lon, 6)] for lon, lat in pts],
                "length": round(length),
                "detour": round(length / beeline, 2) if beeline > 1 else None,
                "connects": endpoints(name),
                "color": colours.get(style, "#2e7d32"),
                # Filled in below, once the road network is available.
                "_ends": [(pts[0], pts[1]), (pts[-1], pts[-2])],
            })
        elif point is not None:
            pts = coords(point)
            if not pts:
                continue
            lon, lat = pts[0]
            waypoints.append({
                **common,
                "lat": round(lat, 6),
                "lng": round(lon, 6),
            })

    segments.sort(key=lambda s: -s["length"])

    lats = [p[0] for s in segments for p in s["path"]] + [w["lat"] for w in waypoints]
    lngs = [p[1] for s in segments for p in s["path"]] + [w["lng"] for w in waypoints]
    bounds = [[min(lats), min(lngs)], [max(lats), max(lngs)]]

    roads = fetch_roads(bounds) if use_roads else []
    if use_roads:
        print(f"roads: {len(roads)} drivable segments"
              + ("" if roads else " (unavailable, falling back to raw endpoints)"))

    for seg in segments:
        seg["entries"] = [aim(end, inward, roads) for end, inward in seg.pop("_ends")]
    for wp in waypoints:
        here = (wp["lng"], wp["lat"])
        # A waypoint has no direction of its own, so look at it from the road.
        wp["entries"] = [aim(here, here, roads)]

    return segments, waypoints, bounds, stats


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mirror_images = "--no-photos" not in sys.argv
    use_roads = "--no-roads" not in sys.argv
    out = args[0] if args else DEFAULT_OUT

    res = requests.get(KML_URL, headers=H, timeout=60)
    res.raise_for_status()
    segments, waypoints, bounds, pstats = build(res.content, mirror_images, use_roads)

    data = {
        "source": MAP_URL,
        "segments": segments,
        "waypoints": waypoints,
        "stats": {
            "segments": len(segments),
            "waypoints": len(waypoints),
            "total_length": sum(s["length"] for s in segments),
            "photos": sum(len(s["photos"]) for s in segments)
                      + sum(len(w["photos"]) for w in waypoints),
        },
        "bounds": bounds,
    }

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)

    s = data["stats"]
    print(f"wrote {out}: {s['segments']} segments, {s['waypoints']} waypoints, "
          f"{s['total_length']} m, {s['photos']} photos")
    if mirror_images:
        removed = prune(pstats)
        print(f"photos: {pstats['fetched']} new, {pstats['cached']} unchanged, "
              f"{pstats['failed']} failed, {removed} stale removed, "
              f"{pstats['bytes'] / 1e6:.1f} MB written to {IMG_DIR}/")

    entries = [e for s in segments for e in s["entries"]] + \
              [e for w in waypoints for e in w["entries"]]
    if use_roads and entries:
        likely = sum(1 for e in entries if e["likely"])
        snapped = sum(1 for e in entries if [e["lat"], e["lng"]] != e["view"])
        print(f"street view: {likely}/{len(entries)} access points within "
              f"{COVER_WARN_M} m of a road, {snapped} aimed from the road")


if __name__ == "__main__":
    main()
