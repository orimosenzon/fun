#!/usr/bin/env python3
"""Resolve exact coordinates for the אמנות במושבה hubs.

Neither Nominatim nor OSM knows house numbers in פרדס חנה-כרכור, so the usual
geocoders return the middle of the street: האורנים 12 and האורנים 40 come back
as the same point, 900 m apart in reality. The festival's own pages, though,
link each participant to Waze, and Waze's short links redirect to a URL that
carries the coordinates the organisers themselves pinned:

    https://waze.com/ul/hsvbbspe4t
      -> https://www.waze.com/live-map/directions?to=ll.32.47329,34.984825

So: for every participant, pull the Waze href off their page, resolve the short
ones, and take the median per hub address. Output: data/hub_coords.json
"""
import concurrent.futures as cf
import json
import os
import re
import statistics
import sys
import urllib.parse
import urllib.request

from scrape_pardesart import get

S = os.path.dirname(os.path.abspath(__file__))
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
LL = re.compile(r"ll\.(-?\d+\.\d+)%?2?C?,?(-?\d+\.\d+)")


def waze_href(url):
    t = get(url)
    m = re.search(r'href="(https://waze\.com/ul[^"]*)"', t)
    return m.group(1) if m else ""


def resolve(short):
    """Follow the redirect chain; the final URL carries ll.<lat>,<lon>."""
    req = urllib.request.Request(short, headers={"User-Agent": UA})
    try:
        final = urllib.request.urlopen(req, timeout=45).geturl()
    except Exception:
        return None
    m = LL.search(urllib.parse.unquote(final))
    return (float(m.group(1)), float(m.group(2))) if m else None


def main():
    recs = (json.load(open(f"{S}/data/artists.json", encoding="utf-8"))
            + json.load(open(f"{S}/data/foods.json", encoding="utf-8")))
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        hrefs = list(ex.map(lambda r: waze_href(r["url"]), recs))
    shorts = [(r["address_label"].strip(), h) for r, h in zip(recs, hrefs)
              if "/ul/" in h and "?q=" not in h]
    print(f"{len(shorts)} short Waze links out of {len(recs)} participants", file=sys.stderr)

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        coords = list(ex.map(lambda x: resolve(x[1]), shorts))

    by_addr = {}
    for (addr, _), c in zip(shorts, coords):
        if c:
            by_addr.setdefault(addr, []).append(c)

    out = {}
    for addr, pts in sorted(by_addr.items()):
        lat = statistics.median(p[0] for p in pts)
        lon = statistics.median(p[1] for p in pts)
        spread = max((abs(p[0] - lat) + abs(p[1] - lon)) for p in pts) if len(pts) > 1 else 0
        out[addr] = {"lat": round(lat, 6), "lon": round(lon, 6),
                     "n": len(pts), "spread_deg": round(spread, 6)}
        print(f"  {addr:34s} {lat:.5f},{lon:.5f}  n={len(pts)} spread={spread:.5f}",
              file=sys.stderr)

    json.dump(out, open(f"{S}/data/hub_coords.json", "w"), ensure_ascii=False, indent=1)
    print(f"wrote data/hub_coords.json ({len(out)} addresses)", file=sys.stderr)


if __name__ == "__main__":
    main()
