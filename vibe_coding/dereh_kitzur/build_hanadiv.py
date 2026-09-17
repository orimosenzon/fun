#!/usr/bin/env python3
"""Build data/hanadiv.json: the events of פסטיבל דרך הנדיב, one pin per event.

The festival site (2026.hanadiv.org) is an Angular app over a plain JSON API
at api.hanadiv.org, open, no key: /festival, /event and /place, each taking
`festival=<subdomain>`. An event carries its street and house number; a place
carries the same address plus the coordinates the festival platform geocoded
for it. So the raw material is good - better than the wiki, which has no
coordinates at all - and the work here is checking it.

What the check found (17/9/2026), which is why FIXES exists:

    ברקאי 7        0,0 in the API. OSM has the building (way 684389749) and
                   B144 agrees to ten metres.
    שלדג 33        the API put it in the middle of the moshava, 2.4 km from
                   the only שלדג in OpenStreetMap. B144 puts it on that street.
    הצפירה 10      the API point is 580 m up the street from where B144 puts
                   number 10. B144's numbering along הצפירה is monotonic and
                   number 40 is still short of the API point, so the API had
                   the wrong end of a long street.
    המייסדים 90    API and B144 disagreed by 80 m. gushim.co.il gives the
                   parcel (גוש 10074 חלקה 45) and govmap's open cadastre puts
                   its centre 20 m from B144's point, 75 m from the API's.
    ארבל 14        API and B144 40 m apart; OSM has the building, and it is
                   where B144 says.
    המעלה 4        the amphitheatre. The API is 40 m off the marker Amudanan
                   keeps for בית העם, which is the building itself.
    הדקלים 90      "רחבת יד לבנים"; OSM has the memorial 65 m from the API's
                   point, and the plaza is the memorial's.

Everything else agreed with a second source to within a few tens of metres,
or sat on its street with no second source to be had. Two cases where the
second source was *wrong* and the API right, recorded so nobody "fixes" them:
B144 puts הבוטנים 54 (the community centre) 220 m south of where OSM has the
Moshe Meir arts centre that is in it, and המעלה 4 (the amphitheatre) 170 m
west of Amudanan's marker. A geocoder's house number is an interpolation; a
landmark's own marker beats it.

    python3 build_hanadiv.py            # fetch, check, write web/data/hanadiv.json
    python3 build_hanadiv.py --report   # also print every venue against B144

The B144 check is a report for whoever runs this, not an automatic decision -
see the two cases above. Its results are cached in .cache/ so a rerun is free.
"""

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request

from build_places import load_json, save_json

API = "https://api.hanadiv.org"
SITE = "https://2026.hanadiv.org/"
FESTIVAL = "2026"                       # the subdomain is the festival key
OUT = "web/data/hanadiv.json"
B144_CACHE = ".cache/hanadiv_b144.json"
UA = {"User-Agent": "derech-kitzur-hanadiv/1.0 (https://github.com/orimosenzon/fun; orimosenzon@gmail.com)",
      "Origin": "https://2026.hanadiv.org"}

# The moshava plus a margin; a coordinate outside it is a geocoder's guess at
# a street of the same name in another town (the API had a place in Jerusalem).
BOX = (32.435, 34.925, 32.515, 35.005)          # south, west, north, east

# Verified positions, by normalised "street number". See the module docstring
# for how each was decided; `source` is what the app shows under the pin.
FIXES = {
    "ברקאי 7": dict(lat=32.475970, lng=35.000690, source="osm",
                    why="OSM building way 684389749; B144 within 10 m"),
    "שלדג 33": dict(lat=32.461936, lng=34.952337, source="b144",
                    why="API had the town centre; B144 on the OSM שלדג way"),
    "הצפירה 10": dict(lat=32.466534, lng=34.974731, source="b144",
                      why="API 580 m up the street; B144 numbering monotonic"),
    "המייסדים 90": dict(lat=32.478745, lng=34.991746, source="parcel",
                        why="גוש 10074 חלקה 45 via gushim.co.il, centre off govmap WFS"),
    "ארבל 14": dict(lat=32.475380, lng=35.001300, source="osm",
                    why="OSM building; B144 agrees"),
    "המעלה 4": dict(lat=32.474976, lng=34.969912, source="landmark",
                    why="Amudanan marker for בית העם - אמפיתיאטרון"),
    "הדקלים 90": dict(lat=32.475430, lng=34.973730, source="osm",
                      why="OSM node יד לבנים; the plaza is the memorial's"),
}

# The festival's fourteen genres, folded into groups a legend can hold. The
# genre itself stays on the pin as the second chip.
GROUPS = [
    {"name": "מוסיקה", "color": "#f6a11b", "genres": ["מוסיקה"]},
    {"name": "סדנאות ויצירה", "color": "#00acac",
     "genres": ["סדנא חווייתית", "אמנות פלסטית", "ציור", "אוכל"]},
    {"name": "גוף ותנועה", "color": "#3ec28f", "genres": ["ספורט", "תנועה ומחול"]},
    {"name": "הרצאות ומילים", "color": "#7b5ea7", "genres": ["הרצאה", "הגות ושירה"]},
    {"name": "ילדים ומשפחה", "color": "#e45462", "genres": ["שעת סיפור", "ילדים והורים"]},
    {"name": "במה ותערוכות", "color": "#c2185b", "genres": ["תיאטרון", "תערוכה"]},
    {"name": "אירוח", "color": "#9b8f80", "genres": ["אירוח"]},
]
OTHER = {"name": "אחר", "color": "#607d8b"}

DAYS = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]   # datetime.weekday()


def fetch(path, **params):
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{API}/{path}?{query}", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_street(value):
    """'מתנס הבוטנים 54 פרדס חנה ' -> 'הבוטנים'. The hosts typed these."""
    s = text(value)
    s = re.sub(r"^מתנ.?ס\s+", "", s)
    s = re.sub(r"\s*פרדס חנה.*$", "", s)
    s = s.replace("״", '"').replace("”", '"')
    return s


def address_key(street, number):
    """'הבוטנים 54 פרדס חנה', 0  ->  'הבוטנים 54': a host who typed the
    number into the street field and left the number field empty."""
    street = norm_street(street)
    number = int(number or 0)
    tail = re.search(r"^(.*?)\s+(\d+)$", street)
    if tail and not number:
        street, number = tail.group(1), int(tail.group(2))
    return f"{street} {number}"


def in_box(lat, lng):
    return BOX[0] <= lat <= BOX[2] and BOX[1] <= lng <= BOX[3]


def metres(a, b):
    k = 111320
    return math.hypot((a[0] - b[0]) * k, (a[1] - b[1]) * k * math.cos(math.radians(32.47)))


def duration_text(minutes):
    if not minutes:
        return ""
    if minutes < 60:
        return f"{minutes} דק׳"
    hours, rest = divmod(minutes, 60)
    words = {1: "שעה", 2: "שעתיים"}
    base = words.get(hours, f"{hours} שעות")
    if rest == 30:
        return base + " וחצי"
    if rest:
        return f"{base} ו-{rest} דק׳"
    return base


def when_text(date, hour, minutes):
    d = dt.date.fromisoformat(date)
    bits = [f"יום {DAYS[d.weekday()]} {d.day}.{d.month}", hour]
    dur = duration_text(minutes)
    if dur:
        bits.append(dur)
    return " · ".join(bits)


def group_of(genre):
    for g in GROUPS:
        if genre in g["genres"]:
            return g["name"]
    return OTHER["name"]


def youtube_id(url):
    m = re.search(r"(?:youtu\.be/|[?&]v=|/shorts/|/embed/|/live/)([\w-]{11})", url or "")
    return m.group(1) if m else ""


def link_title(url):
    host = urllib.parse.urlparse(url).hostname or ""
    host = host.replace("www.", "").replace("open.", "")
    return {"youtube.com": "יוטיוב", "youtu.be": "יוטיוב", "spotify.com": "ספוטיפיי",
            "instagram.com": "אינסטגרם", "facebook.com": "פייסבוק",
            "drive.google.com": "גוגל דרייב"}.get(host, host or "קישור")


def clean_links(*raw):
    """The hosts pasted these into a form, sometimes with the form's own
    placeholder text around them. Keep the URL, drop the rest."""
    out = []
    for value in raw:
        for m in re.finditer(r"https?://\S+", str(value or "")):
            url = m.group(0).rstrip(".,)")
            if url not in out:
                out.append(url)
    return out


def b144(street, number, cache):
    """Bezeq's map site prints one coordinate pair per address page. A second
    opinion for the report, nothing more: its house numbers are interpolated
    along the street and it puts two landmarks in the wrong place."""
    key = f"{street} {number}"
    if key in cache:
        return cache[key]
    url = "https://www.b144.co.il/maps/" + urllib.parse.quote(f"פרדס-חנה-כרכור/{street}/{number}/")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120"})
    try:
        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
        lat = re.findall(r"3[12]\.\d{5,}", page)
        lng = re.findall(r"3[45]\.\d{5,}", page)
        cache[key] = [float(lat[0]), float(lng[0])] if lat and lng else None
    except Exception as err:                       # noqa: BLE001 - a report line, not a failure
        print(f"  ! b144 {key}: {err}", file=sys.stderr)
        cache[key] = None
    time.sleep(1.1)
    return cache[key]


def spread(places):
    """Every pile onto a ring, whatever its source. build_places.spread leaves
    exact positions alone, which is right for a street of businesses; here a
    pile is nine concerts on one stage, and the ring is the only way to reach
    the second one. The venue's own position is the ring's centre."""
    piles = {}
    for place in places:
        geo = place.get("geo")
        if geo:
            piles.setdefault((geo["lat"], geo["lng"]), []).append(place)
    for (lat, lng), pile in piles.items():
        if len(pile) < 2:
            continue
        radius = 10 + 1.2 * len(pile)                  # metres
        for i, place in enumerate(pile):               # already in date order
            angle = 2 * math.pi * i / len(pile)
            place["geo"]["lat"] = round(lat + radius * math.cos(angle) / 111320, 6)
            place["geo"]["lng"] = round(lng + radius * math.sin(angle) / (111320 * math.cos(math.radians(lat))), 6)
            place["geo"]["spread"] = True


def locate(place_rec, key):
    """The position for a venue and where it came from.

    The API's own coordinate is the default, because the platform geocoded a
    full address and most of them check out. FIXES overrides it where a
    better source was found; a coordinate that is missing or outside the
    moshava and has no fix is reported and the pin goes out unplaced."""
    fix = FIXES.get(key)
    if fix:
        return {"lat": fix["lat"], "lng": fix["lng"], "source": fix["source"]}
    if place_rec:
        lat, lng = float(place_rec.get("lat") or 0), float(place_rec.get("lng") or 0)
        if in_box(lat, lng):
            return {"lat": round(lat, 6), "lng": round(lng, 6), "source": "festival"}
    print(f"  ! no position for {key}", file=sys.stderr)
    return None


def build(report):
    festival = fetch("festival", festival=FESTIVAL)
    events = fetch("event", festival=FESTIVAL)
    places = fetch("place", festival=FESTIVAL)
    print(f"{festival['name']}: {festival['fromDate']} עד {festival['untilDate']}, "
          f"{len(events)} אירועים, {len(places)} מקומות")

    # Venues by address. Several place records share an address (the community
    # centre has five); any of them gives the coordinate, and the one whose
    # description matches the event's gives the photo.
    by_address = {}
    for p in places:
        by_address.setdefault(address_key(p["street"], p["houseNumber"]), []).append(p)

    cache = load_json(B144_CACHE, {})
    seen_titles = {}
    for e in events:
        seen_titles.setdefault(text(e["title"]), set()).add(e["date"])

    out = []
    venues = {}
    for e in sorted(events, key=lambda e: (e["date"], e["hour"])):
        if int(e.get("status", 1)) != 1 or e.get("virtual"):
            continue
        key = address_key(e["street"], e["houseNumber"])
        recs = by_address.get(key, [])
        rec = next((r for r in recs if text(r["description"]) == text(e["placeDescription"])), recs[0] if recs else None)
        geo = locate(rec, key)
        venues[key] = (rec, geo)

        title = text(e["title"])
        # A title is a label on the map. One host wrote a paragraph into the
        # field; the label takes its first clause and the note keeps the rest.
        short = title
        if len(short) > 48:
            cut = re.split(r"\s[-–·!|]\s|!|\.\s|,\s", short, maxsplit=1)[0].strip()
            short = cut if 8 <= len(cut) <= 48 else short[:45].rsplit(" ", 1)[0] + "…"
        name = short
        if len(seen_titles[title]) > 1:          # the same thing on two days
            d = dt.date.fromisoformat(e["date"])
            name = f"{short} · יום {DAYS[d.weekday()]}"

        leader = text(f"{e.get('initiativeOwnerFirstName', '')} {e.get('initiativeOwnerLastName', '')}")
        host = text(rec.get("host")) if rec else ""
        note_bits = ([title] if short != title else []) \
            + [text(e.get("details")), text(e.get("initiativeDescription"))]
        # Half the festival is people hosting their own thing; say it once.
        if leader and host and leader != host:
            note_bits.append(f"בהנחיית {leader}, באירוח {host}")
        elif leader or host:
            note_bits.append(f"בהנחיה ובאירוח של {leader or host}" if leader == host
                             else (f"בהנחיית {leader}" if leader else f"באירוח {host}"))
        if text(e.get("placeNotes")):
            note_bits.append(text(e["placeNotes"]))
        note = " · ".join(dict.fromkeys(b for b in note_bits if b))

        address = f"{norm_street(e['street'])} {int(e['houseNumber'] or 0) or ''}".strip()
        if text(e.get("entrance")) and len(text(e["entrance"])) < 12:
            address += f", כניסה {text(e['entrance'])}"
        if text(e.get("placeDescription")):
            address += f" · {text(e['placeDescription'])}"

        access = []
        if e.get("isAccessible"):
            access.append("נגיש")
        if e.get("hasParking"):
            access.append("יש חניה")
        if e.get("petFriendly"):
            access.append("ידידותי לחיות")
        if e.get("areaSetting") in ("בפנים", "בחוץ"):
            access.append(e["areaSetting"])

        photos = []
        if e.get("path"):
            photos.append({"thumb": e["path"], "full": e["path"]})
        links = []
        for url in clean_links(e.get("externalLink1"), e.get("externalLink2")):
            yt = youtube_id(url)
            if yt:
                photos.append({"yt": yt, "thumb": f"https://i.ytimg.com/vi/{yt}/hqdefault.jpg"})
            else:
                links.append({"url": url, "title": link_title(url)})

        place = {
            "id": f"hanadiv-{e['eventID']}",
            "name": name,
            "group": group_of(e.get("genre")),
            "craft": text(e.get("genre")),
            "when": when_text(e["date"], e["hour"], e.get("duration")),
            "date": e["date"],
            "hour": e["hour"],
            "audience": text(e.get("public")),
            "note": note,
            "url": f"{SITE}event-page/{e['eventID']}",
            "links": links,
            "photos": photos,
            "geo": dict(geo) if geo else None,
            "address": address,
        }
        if access:
            place["access"] = " · ".join(access)
        if e.get("maxNumberOfPeople"):
            place["seats"] = int(e["maxNumberOfPeople"])
        if not geo:
            del place["geo"]
        out.append(place)

    if report:
        print("\nvenue                     api                   b144                  apart  used")
        for key, (rec, geo) in sorted(venues.items()):
            street, number = key.rsplit(" ", 1)
            second = b144(street, number, cache)
            api = (float(rec["lat"]), float(rec["lng"])) if rec and in_box(float(rec.get("lat") or 0), float(rec.get("lng") or 0)) else None
            apart = f"{metres(api, second):5.0f} m" if api and second else "      "
            used = f"{geo['source']}" if geo else "UNPLACED"
            fmt = lambda p: f"{p[0]:.5f},{p[1]:.5f}" if p else "-"
            print(f"{key:24} {fmt(api):21} {fmt(second):21} {apart}  {used}")
        save_json(B144_CACHE, cache)

    # One event at a venue is one pin; nine at the community centre are a
    # pile, and a pile is one pin you can click. Same ring as everywhere else.
    spread(out)

    doc = {
        "version": 1,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": SITE,
        "name": f"אירועי דרך הנדיב {FESTIVAL}",
        "festival": {"name": festival["name"], "from": festival["fromDate"], "until": festival["untilDate"]},
        "groups": [{"name": g["name"], "color": g["color"]} for g in GROUPS] + [OTHER],
        "places": out,
        "stats": {
            "events": len(out),
            "venues": len(venues),
            "placed": sum(1 for p in out if p.get("geo")),
            "photos": sum(len(p["photos"]) for p in out),
        },
    }
    save_json(OUT, doc)
    print(f"\nנכתב {OUT}: {doc['stats']}")
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report", action="store_true",
                        help="print every venue against B144 (slow, cached)")
    args = parser.parse_args()
    build(args.report)


if __name__ == "__main__":
    main()
