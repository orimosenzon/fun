#!/usr/bin/env python3
"""Draw the אמנות במושבה hub map: the moshava's street grid with 11 numbered pins.

Positioning, because a map must not claim more than it knows. No free geocoder
resolves house numbers in פרדס חנה-כרכור: Nominatim returns the middle of the
street (האורנים 12 and האורנים 40 come back as the same point, 900 m apart in
reality), OSM holds no addr:housenumber object in the town, and govmap's search
endpoint is closed. What does exist is `data/events_geo.json`, the geocoded
programme of קהילילה לבן 2026, with exact coordinates for 67 local venues. Every
pin here is placed from the best evidence available for it, and `place()`
reports which:

  exact      the venue itself appears in the קהילילה לבן data
  fit        two or more known house numbers on the same street, fitted linearly
  anchor     one known house number on the street, offset along its geometry
  street     nothing better than the centre of the street

Numbers only, no Hebrew in the image: the legend lives in the wikitext table.
Output: hub_map.png, hub_placement.json
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

S = os.path.dirname(os.path.abspath(__file__))
ways = json.load(open(f"{S}/data/streets_geom.json", encoding="utf-8"))["elements"]
cent = json.load(open(f"{S}/data/street_centroids.json", encoding="utf-8"))
kehilila = json.load(open(f"{S}/data/events_geo.json", encoding="utf-8"))

# (pin, hub title, festival address, OSM street name, house number)
HUBS = [
    (1, "מתחם הידית", "חרושת 1", "החרושת", 1),
    (2, "מתחם האורנים", "האורנים 40", "האורנים", 40),
    (3, "מתחם הקשת", "קדמה 8ב", "קדמה", 8),
    (4, "מתחם התורמוסים", "הבוטנים 25", "הבוטנים", 25),
    (5, "מתחם הדרים Dream", "הדרים 77א", "הדרים", 77),
    (6, "השוק הישן", "האורנים 12", "האורנים", 12),
    (7, "בית תרבות נרבתא", "ביל״ו 10", 'ביל"ו', 10),
    (8, "הבוטנים 7", "הבוטנים 7", "הבוטנים", 7),
    (9, "מתחם שאננים", "שאננים 9", "שאננים", 9),
    (10, "הדקלים 131", "הדקלים 131", "הדקלים", 131),
    (11, "המזמור 11", "המזמור 11", "המזמור", 11),
]

# Venues the two festivals share, so a קהילילה לבן coordinate pins the hub
# outright. קבב ואחיו is the useful one: אמנות במושבה lists it at שאננים 9 and
# קהילילה לבן at המייסדים 48, i.e. the same corner plot seen from two streets.
EXACT_VIA = {
    "חרושת 1": "מתחם הידית",
    "האורנים 12": "השוק הישן",
    "ביל״ו 10": "נרבתא - בית תרבות",
    "שאננים 9": "קבב ואחיו",
}
LAT0 = 32.475
KX = math.cos(math.radians(LAT0))


def kehilila_pt(business):
    for e in kehilila:
        if e.get("business") == business and e.get("lat"):
            return e["lat"], e["lon"]
    return None


def anchors_on(street):
    """Known (house number -> point) pairs on a street, from קהילילה לבן."""
    out = {}
    bare = street.lstrip("ה")
    for e in kehilila:
        if not e.get("lat"):
            continue
        addr = e["address"]
        for form in (street, bare, "ה" + bare):
            if addr.startswith(form + " "):
                tail = addr[len(form) + 1:]
                num = ""
                for ch in tail:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                if num:
                    out.setdefault(int(num), (e["lat"], e["lon"]))
                break
    return out


def street_points(street):
    pts = []
    for form in (street, street.lstrip("ה"), "ה" + street.lstrip("ה")):
        for w in ways:
            if w.get("tags", {}).get("name") == form:
                pts.extend((p["lat"], p["lon"]) for p in w["geometry"])
        if pts:
            break
    return pts


def place(street, house, addr):
    if addr in EXACT_VIA:
        p = kehilila_pt(EXACT_VIA[addr])
        if p:
            return p[0], p[1], "exact"

    known = anchors_on(street)
    if len(known) >= 2:
        # least squares of lat and lon against house number
        ns = sorted(known)
        n_mean = sum(ns) / len(ns)
        den = sum((n - n_mean) ** 2 for n in ns)
        out = []
        for i in (0, 1):
            v_mean = sum(known[n][i] for n in ns) / len(ns)
            num = sum((n - n_mean) * (known[n][i] - v_mean) for n in ns)
            slope = num / den if den else 0.0
            out.append(v_mean + slope * (house - n_mean))
        return out[0], out[1], "fit"

    pts = street_points(street)
    if len(known) == 1 and pts:
        # one anchor: keep its position but slide along the street towards the
        # end the numbering runs to, proportionally to the number difference
        (kn, kp), = known.items()
        lat_span = max(p[0] for p in pts) - min(p[0] for p in pts)
        lon_span = (max(p[1] for p in pts) - min(p[1] for p in pts)) * KX
        axis = 1 if lon_span > lat_span else 0
        pts.sort(key=lambda p: p[axis])
        i_near = min(range(len(pts)),
                     key=lambda i: (pts[i][0] - kp[0]) ** 2 + ((pts[i][1] - kp[1]) * KX) ** 2)
        step = (house - kn) / max(kn, 1)
        j = max(0, min(len(pts) - 1, int(i_near + step * len(pts) * 0.5)))
        return pts[j][0], pts[j][1], "anchor"

    if pts:
        p = pts[len(pts) // 2]
        return p[0], p[1], "street"
    c = cent.get(street) or cent.get(street.lstrip("ה"))
    return (c[0], c[1], "street") if c else (None, None, "unknown")


placed = []
for num, title, addr, street, house in HUBS:
    lat, lon, how = place(street, house, addr)
    placed.append({"pin": num, "title": title, "address": addr,
                   "lat": lat, "lon": lon, "how": how})
    print(f"{num:2d} {title:22s} {addr:14s} {lat:.5f},{lon:.5f}  {how}")
json.dump(placed, open(f"{S}/hub_placement.json", "w"), ensure_ascii=False, indent=1)

lats = [p["lat"] for p in placed]
lons = [p["lon"] for p in placed]
bounds = (min(lats) - 0.010, max(lats) + 0.010, min(lons) - 0.012, max(lons) + 0.012)

fig, ax = plt.subplots(figsize=(11, 8.2), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("#fbfaf7")

MAJOR = {"motorway", "trunk", "primary", "secondary"}
for w in ways:
    g = w.get("geometry")
    if not g:
        continue
    hw = w.get("tags", {}).get("highway")
    xs = [p["lon"] for p in g]
    ys = [p["lat"] for p in g]
    if hw in MAJOR:
        ax.plot(xs, ys, color="#c3b8a4", lw=2.0, solid_capstyle="round", zorder=2)
    elif hw in ("tertiary", "unclassified"):
        ax.plot(xs, ys, color="#d6cec0", lw=1.1, solid_capstyle="round", zorder=1)
    else:
        ax.plot(xs, ys, color="#e6e1d7", lw=0.6, solid_capstyle="round", zorder=1)

# Solid pin = the venue's own coordinate. Hollow pin = right street, position
# along it inferred. Saying that visually beats a footnote nobody reads.
for p in placed:
    solid = p["how"] == "exact"
    ax.scatter([p["lon"]], [p["lat"]], s=470,
               c="#21659c" if solid else "#ffffff",
               edgecolors="#ffffff" if solid else "#21659c",
               linewidths=2.2, zorder=5)
    ax.text(p["lon"], p["lat"], str(p["pin"]), ha="center", va="center",
            fontsize=13, fontweight="bold",
            color="white" if solid else "#21659c", zorder=6)

ax.set_xlim(bounds[2], bounds[3])
ax.set_ylim(bounds[0], bounds[1])
ax.set_aspect(1 / KX)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_edgecolor("#ccc5b8")

ax.text(0.5, 0.982, "P A R D E S   H A N N A  -  K A R K U R", transform=ax.transAxes,
        ha="center", va="top", fontsize=11.5, color="#8a8172", fontweight="bold")
ax.text(0.994, 0.012, "map data © OpenStreetMap contributors (ODbL)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
        color="#9a9385", path_effects=[withStroke(linewidth=2.5, foreground="white")])

fig.tight_layout(pad=0.4)
fig.savefig(f"{S}/hub_map.png", facecolor="white")
print("wrote hub_map.png")
