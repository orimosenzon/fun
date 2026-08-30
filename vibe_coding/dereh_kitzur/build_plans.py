#!/usr/bin/env python3
"""Build web/data/plans.json: the planning schemes in process over the moshava.

Yoav asked for "תכניות שהופקדו והסטטוס שלהן" and called it the point, or one of
them. This is that: what somebody is trying to build here, where, and how far
along it has got - including the ones still inside their objection window, which
is the only stretch of a plan's life when a resident can do anything about it.

Where it comes from
-------------------
The planning administration publishes every online scheme in the country as a
public ArcGIS service, no key and no login:

    ags.iplan.gov.il/arcgisiplan/rest/services/PlanningPublic/Xplan/MapServer/1

Layer 1 is "קוים כחולים-תכניות מקוונות" - the blue line, the outer boundary of
each scheme, one polygon per plan. Note the path: `/arcgisiplan/`, not
`/arcgis/`. The second is a different host on the same domain and answers every
request with the government WAF's "the operation is not supported" page, which
is what sent an earlier attempt at this looking for a service that was there all
along.

What is left out, and why
-------------------------
413 plans name פרדס חנה כרכור. Only the 95 still in process are here:

    בהפקדה           פרסום הפקדה, בתהליך הפקדה         the objection window
    בדיון והחלטות    החלטה בדיון, בתהליך אישור/פרסום   decided, not yet published
    בתנאי סף         קיום תנאי סף                       filed, at the threshold

The 224 that are approved and the 94 marked "סיום טיפול" are not. An approved
plan is the legal reality of the ground rather than news about it, and 224 more
polygons over a small town is a wall you cannot see the moshava through; a
closed file is not information about anything. Both are one `STAGES` entry away
if that turns out to be wrong.

    python3 build_plans.py
    python3 build_plans.py --all      # every status, including approved and closed
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web", "data", "plans.json")

XPLAN = ("https://ags.iplan.gov.il/arcgisiplan/rest/services"
         "/PlanningPublic/Xplan/MapServer/1/query")
XPLAN_SITE = "https://ags.iplan.gov.il/xplan/"

# The service is behind the government WAF, which answers a plain urllib with an
# HTML error page. A browser user-agent is enough.
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128.0 Safari/537.36")

# Three fields name the town, and they disagree slightly: plan_area_name has
# 413, jurstiction_area_name 398, plan_county_name 383. The widest is the right
# one here - a plan that straddles the boundary is still a plan about the
# moshava - and the blue lines are drawn anyway, so anything that wandered in
# from a neighbour is visible for what it is.
AREA = "plan_area_name LIKE '%פרדס חנה%'"

# The buckets, and the order they are drawn and listed in. The colours are a
# blue ramp because these are the blue lines and because it keeps them off the
# conservation layer's oranges; within the layer the deposited ones are the
# darkest, which is the point of the layer.
STAGES = [
    ("בהפקדה", "#01579b", ["פרסום הפקדה", "בתהליך הפקדה"]),
    ("בדיון והחלטות", "#00838f", ["החלטה בדיון", "בתהליך אישור", "בתהליך פרסום"]),
    ("בתנאי סף", "#8d6e63", ["קיום תנאי סף"]),
]

APPROVED = ("אושרו", "#558b2f", ["פרסום אישור", "התכנית אושרה"])
CLOSED = ("סיום טיפול", "#90a4ae", ["סיום טיפול"])

FIELDS = [
    "pl_number", "pl_name", "pl_objectives", "internet_short_status",
    "station_desc", "entity_subtype_desc", "plan_charactor_name", "pl_url",
    "pl_area_dunam", "pl_landuse_string", "pl_by_auth_of",
    "depositing_date", "pl_last_deposit_date", "receiving_date",
    "last_update_date", "pl_id", "mp_id",
]

PAGE = 200


# ---------------------------------------------------------------- utilities

def tls():
    """The server's TLS is older than what OpenSSL 3 will talk to by default.

    A plain urllib gets SSLV3_ALERT_HANDSHAKE_FAILURE off this host while curl
    on the same machine connects; the difference is Ubuntu's security level, and
    dropping it to 1 for this one connection is enough. Certificates are still
    verified - this widens the cipher list, it does not turn checking off.
    """
    ctx = ssl.create_default_context()
    ctx.set_ciphers("DEFAULT@SECLEVEL=1")
    return ctx


def get(url, params, tries=3):
    body = urllib.parse.urlencode(params).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=body, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120, context=tls()) as handle:
                return json.loads(handle.read().decode("utf-8-sig"))
        except Exception as err:                       # noqa: BLE001 - retry anything
            if attempt == tries - 1:
                raise
            print(f"    ...{err}, מנסה שוב", file=sys.stderr)
            time.sleep(3)
    return None


def save_json(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=1)


def ring_centroid(ring):
    """Area-weighted centroid of one closed ring, and its area.

    The shoelace centroid rather than the mean of the corners, so that a plan
    shaped like a road - long, thin, bent - is represented by a point on it
    instead of one dragged off to the side of the bend.
    """
    area = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area) < 1e-12:
        n = len(ring) or 1
        return sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n, 0.0
    return cx / (3 * area), cy / (3 * area), abs(area) / 2


def rings_centroid(rings):
    """Centroid of an esri polygon, weighting each ring by its area.

    Esri gives outer rings clockwise and holes counter-clockwise, and both come
    through here with a positive area. A hole big enough to move the centroid
    noticeably would have to be most of the plan, which does not happen on a
    blue line, so they are not told apart.
    """
    total = wx = wy = 0.0
    fallback = []
    for ring in rings:
        pts = [tuple(p[:2]) for p in ring]
        if len(pts) < 3:
            continue
        cx, cy, area = ring_centroid(pts)
        fallback.append((cx, cy))
        total += area
        wx += cx * area
        wy += cy * area
    if total > 0:
        return wx / total, wy / total
    if fallback:
        return (sum(p[0] for p in fallback) / len(fallback),
                sum(p[1] for p in fallback) / len(fallback))
    return None


def when(stamp):
    """An epoch-milliseconds field as a Hebrew date, or None if it is empty."""
    if not stamp:
        return None
    try:
        return time.strftime("%d/%m/%Y", time.gmtime(int(stamp) / 1000))
    except (TypeError, ValueError, OSError):
        return None


# ---------------------------------------------------------------- fetching

def statuses(want_all):
    buckets = list(STAGES) + ([APPROVED, CLOSED] if want_all else [])
    return buckets, {s: name for name, _, group in buckets for s in group}


def fetch(want_all):
    """Every plan in the chosen statuses, with its blue line, paged."""
    buckets, _ = statuses(want_all)
    wanted = [s for _, _, group in buckets for s in group]
    quoted = ",".join("'" + s + "'" for s in wanted)
    where = f"{AREA} AND internet_short_status IN ({quoted})"

    out = []
    offset = 0
    while True:
        doc = get(XPLAN, {
            "where": where,
            "outFields": ",".join(FIELDS),
            "returnGeometry": "true",
            "outSR": "4326",
            # About two metres on the ground. The blue lines are simple - five
            # or six corners each - so this changes almost nothing, and stops a
            # plan traced along a wadi from carrying a thousand points.
            "maxAllowableOffset": "0.00002",
            "orderByFields": "pl_number",
            "resultOffset": str(offset),
            "resultRecordCount": str(PAGE),
            "f": "json",
        })
        if "error" in doc:
            raise SystemExit(f"שגיאה מהשירות: {doc['error']}")
        got = doc.get("features", [])
        out += got
        print(f"  התקבלו {len(out)} תכניות")
        if len(got) < PAGE or not doc.get("exceededTransferLimit"):
            break
        offset += PAGE
    return out


# ---------------------------------------------------------------- building

def note_for(at):
    """What the detail pane says about a plan, in the order somebody reads it.

    The objectives field is the planner's own sentence about what the plan is
    for, and it is written with `^` where the form had a line break; it is worth
    far more than anything this script could assemble from the other columns.
    """
    bits = []

    objectives = (at.get("pl_objectives") or "").replace("^", " ").strip()
    name = (at.get("pl_name") or "").strip()
    # The objectives of a small plan are often literally its name again.
    if objectives and objectives.rstrip(" .") != name.rstrip(" ."):
        bits.append(objectives)

    stage = (at.get("station_desc") or "").strip()
    if stage:
        bits.append(f"השלב הנוכחי: {stage}.")

    deposit = when(at.get("pl_last_deposit_date")) or when(at.get("depositing_date"))
    if deposit:
        bits.append(f"הופקדה ב-{deposit}.")
    received = when(at.get("receiving_date"))
    if received and not deposit:
        bits.append(f"נקלטה ב-{received}.")

    dunam = at.get("pl_area_dunam")
    if dunam:
        size = f"{dunam:.0f}" if dunam >= 10 else f"{dunam:.1f}".rstrip("0").rstrip(".")
        bits.append(f"שטח התכנית: {size} דונם.")

    uses = (at.get("pl_landuse_string") or "").strip()
    if uses:
        bits.append(f"ייעודי קרקע: {uses}.")

    bits.append("הגבול המסומן הוא הקו הכחול של התכנית, כפי שהוא במאגר מנהל התכנון. "
                "המסמכים המלאים נמצאים בקישור למבא\"ת.")
    return " ".join(bits)


def build(features, want_all):
    buckets, bucket_of = statuses(want_all)

    plans = []
    skipped = 0
    for feat in features:
        at = feat.get("attributes", {})
        rings = (feat.get("geometry") or {}).get("rings") or []
        centre = rings_centroid(rings)
        if not centre:
            skipped += 1
            continue

        number = (at.get("pl_number") or "").strip()
        name = (at.get("pl_name") or "").strip() or number
        status = (at.get("internet_short_status") or "").strip()

        plans.append({
            "id": "plan-" + (number or str(at.get("pl_id"))),
            "name": name,
            "group": bucket_of.get(status, "בתהליך"),
            "cats": [c for c in ["תכנית", status] if c],
            "num": number,
            # Both render as chips in the detail pane.
            "status": status,
            "kind": (at.get("entity_subtype_desc") or "").strip() or None,
            "url": at.get("pl_url") or XPLAN_SITE,
            "photos": [],
            "geo": {"lat": round(centre[1], 6), "lng": round(centre[0], 6),
                    "source": "plan"},
            # The blue line itself, drawn as a filled outline under the dots.
            "shape": [[[round(p[0], 6), round(p[1], 6)] for p in ring]
                      for ring in rings],
            "note": note_for(at),
        })

    order = {name: i for i, (name, _, _) in enumerate(buckets)}
    plans.sort(key=lambda p: (order.get(p["group"], 9), p["name"]))

    used = {p["group"] for p in plans}
    doc = {
        "version": 1,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": XPLAN_SITE,
        "name": "תכניות בתהליך" if not want_all else "תכניות",
        "groups": [{"name": n, "color": c} for n, c, _ in buckets if n in used],
        "places": plans,
        "stats": {
            "plans": len(plans),
            "no_geometry": skipped,
            "by_status": {s: sum(1 for p in plans if p["status"] == s)
                          for s in sorted({p["status"] for p in plans})},
        },
    }
    return doc


def main():
    ap = argparse.ArgumentParser(description="בניית שכבת התכניות מ-Xplan")
    ap.add_argument("--all", action="store_true",
                    help="גם תכניות מאושרות וגם סיום טיפול, לא רק מה שבתהליך")
    args = ap.parse_args()

    print("מושך תכניות ממנהל התכנון…")
    features = fetch(args.all)
    doc = build(features, args.all)
    save_json(OUT, doc)

    print(f"\nנכתבו {doc['stats']['plans']} תכניות ל-{os.path.relpath(OUT, HERE)}")
    if doc["stats"]["no_geometry"]:
        print(f"  {doc['stats']['no_geometry']} בלי גבול מסומן, לא נכללו")
    for group in doc["groups"]:
        n = sum(1 for p in doc["places"] if p["group"] == group["name"])
        print(f"  {group['name']}: {n}")
    print()
    for status, n in sorted(doc["stats"]["by_status"].items(), key=lambda kv: -kv[1]):
        print(f"    {n:4}  {status}")


if __name__ == "__main__":
    main()
