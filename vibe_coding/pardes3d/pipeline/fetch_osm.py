"""מושך מ-OpenStreetMap את כל מה שצריך לבניית העולם.

גבול המועצה, בניינים, כבישים, מסילה, שימושי קרקע, עצים ומקומות עם שם.
שומר ל-data/raw/osm.json. הנתונים ברישיון ODbL ומחייבים ייחוס.
"""
import json
import pathlib
import sys
import time

import requests

from geo import BOUNDARY_REL

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "osm.json"
MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
UA = {"User-Agent": "pardes3d/0.1 (github.com/orimosenzon/fun)"}

# גבול המועצה עם שוליים של כ-300 מטר, כדי שהבניינים בקצה לא ייחתכו
BBOX = (32.4419, 34.9386, 32.5078, 35.0181)  # s, w, n, e


def query(q):
    last = None
    for url in MIRRORS:
        for attempt in range(3):
            try:
                r = requests.post(url, data={"data": q}, headers=UA, timeout=300)
                if r.status_code == 200 and r.text.lstrip().startswith("{"):
                    return r.json()
                last = f"{url}: HTTP {r.status_code} {r.text[:200]}"
            except requests.RequestException as e:
                last = f"{url}: {e}"
            time.sleep(5 * (attempt + 1))
    sys.exit(f"Overpass נכשל: {last}")


def main():
    s, w, n, e = BBOX
    q = f"""
[out:json][timeout:240][bbox:{s},{w},{n},{e}];
(
  way[building]; rel[building];
  way[highway]; way[railway];
  way[landuse]; rel[landuse];
  way[natural]; rel[natural];
  way[leisure]; rel[leisure];
  way[amenity];
  way[waterway];
  way[barrier~"^(wall|fence|retaining_wall)$"];
  node[natural=tree];
  node[name];
  node[highway~"^(bus_stop|street_lamp|traffic_signals|stop|give_way|crossing)$"];
  node[amenity~"^(bench|waste_basket|drinking_water|post_box|bicycle_parking)$"];
  node[man_made~"^(water_tower|mast|tower|flagpole)$"];
  node[power~"^(pole|tower)$"];
  way[power=line];
  rel({BOUNDARY_REL});
);
out geom;
"""
    data = query(q)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False))
    kinds = {}
    for el in data["elements"]:
        t = el.get("tags", {})
        k = next((k for k in ("building", "highway", "railway", "landuse", "natural", "leisure", "amenity")
                  if k in t), "other")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"נשמרו {len(data['elements'])} אלמנטים ל-{OUT.relative_to(ROOT)}: {kinds}")


if __name__ == "__main__":
    main()
