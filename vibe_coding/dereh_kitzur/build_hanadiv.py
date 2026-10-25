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

The 6/10/2026 rerun (114 events, 41 venues) added five FIXES - four venues
with no API position, placed by B144, and האולם on ההגנה, which has no house
number and is a named building in OSM. Left as the API has them, on purpose:
המושב 1 and 5, where B144 puts 1, 5 and 44 within 250 m of each other while the
API runs monotonically down the OSM street (1 at its south end, 44 midway);
and אתרוג 8, the Neve Pardesim community centre, 340 m from B144 but on its
street, with no OSM or Amudanan marker to settle it. One event stays
unplaced: חוות החזון הירוק, "a dirt road past Zen Garden near BIG", which no
open map has.

The 13/10/2026 rerun (152 events, 58 venues) found /place sending null for
lat, lng and street on every record. The positions it gave on 6/10 are kept
in API_POSITIONS; a new venue with none falls back to B144 automatically
(source b144), and four that B144 lacked got FIXES. Unplaced: חוות החזון
הירוק again, and בית המחול מנשה in קריית החינוך מנשה, found on no open map.

    python3 build_hanadiv.py            # fetch, check, write web/data/hanadiv.json
    python3 build_hanadiv.py --report   # also print every venue against B144
    python3 build_hanadiv.py --publish  # also put it in the data repo (nightly cron)

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
from data_repo import publish_json

API = "https://api.hanadiv.org"
SITE = "https://2026.hanadiv.org/"
FESTIVAL = "2026"                       # the subdomain is the festival key
OUT = "web/data/hanadiv.json"
B144_CACHE = ".cache/hanadiv_b144.json"
# What /place said each address was, by "street number". Between 6/10 and
# 13/10/2026 the API stopped sending lat, lng and street for places (all 80
# null), so the positions it gave earlier live here, recovered from the 6/10
# output (a pile's ring is centred on the venue, so the mean of its pins is
# the API point exactly). Every run adds whatever the API still sends.
API_POSITIONS = "hanadiv_src/api_positions.json"
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
    # Added 6/10/2026, when the programme grew from 46 events to 114.
    "ההגנה 0": dict(lat=32.474830, lng=34.981190, source="osm",
                    why="no house number; OSM way 1525472043 is האולם itself, 180 m from the API point"),
    "האורנים 1": dict(lat=32.478080, lng=34.968570, source="b144",
                      why="no API position; B144, monotonic with האורנים 66 and 84"),
    "הדקלים 100": dict(lat=32.476070, lng=34.972280, source="b144",
                       why="no API position; B144, beside הדקלים 90"),
    "הנשיא 14": dict(lat=32.473960, lng=34.987950, source="b144",
                     why="no API position; B144"),
    "נעורים 23": dict(lat=32.470110, lng=34.993560, source="b144",
                      why="no API position; B144, 50 m from נעורים 27 where the API agrees"),
    # Added 13/10/2026 (152 events), when the API had stopped sending any
    # positions and B144 had nothing for these. Street-level, not house-level.
    "הבנים רחבת המועצה 20": dict(lat=32.475455, lng=34.974988, source="b144",
                                 why="the council plaza; B144 for הבנים 20, the council's address"),
    "לאה גולדברג 12": dict(lat=32.466657, lng=34.983502, source="osm",
                           why="centre of OSM way 489280873, a 250 m street; B144 has no page"),
    "ברקת 27": dict(lat=32.466120, lng=34.980500, source="osm",
                    why="'a grove between ברקת and יהלום': midway between OSM ways 477494365 and 477494607"),
    "סמטת צאלון 4": dict(lat=32.473589, lng=34.963170, source="osm",
                         why="centre of OSM way 183324944 (צאלון); B144 has no page"),
    # Added 16/10/2026 (169 events). The first two are one meeting point, "הזולה
    # בכרכור", in the fields past the end of ברקאי; numbering starts at the
    # west end (ברקאי 7 is there), so the end is the east one.
    "סוף רחוב ברקאי בשדות 1": dict(lat=32.475820, lng=35.004000, source="osm",
                                   why="just past the east end of OSM way 112976696 (ברקאי), into the fields"),
    "ברקאי 100": dict(lat=32.475820, lng=35.004000, source="osm",
                      why="ברקאי has no 100; same venue as the line above, by its description"),
    "המעלה פינת רבי עקיבא 4": dict(lat=32.474700, lng=34.970180, source="osm",
                                   why="'the plaza between the theatre and רבי עקיבא': midway from the amphitheatre "
                                       "marker to where OSM way 68706427 meets המעלה"),
    'עמ"ל 45': dict(lat=32.463774, lng=34.976019, source="osm",
                    why="centre of OSM way 68726800 (עמל), 630 m; B144 has no page"),
    # Added 22/10/2026 (228 events). האולם came back with Waze directions in
    # the street field, so "ההגנה 0" above no longer catches it. Left unplaced:
    # ירושלים 52 (הבוסתן הקהילתי, no B144 page, nothing in OSM) and שפת הים 13
    # in Caesarea (a beach, not an address).
    'בווייז - האולם בית תרבות מקומי פרכו"ר. להגיע לרחבת האוטובוסים ברח׳ ההגנה פינת צנחנים, ולהמשיך לפי השילוט במקום. 0':
        dict(lat=32.474830, lng=34.981190, source="osm", why="האולם, the same building as ההגנה 0"),
    "השופטים פינת נילי 34": dict(lat=32.474662, lng=34.987145, source="osm",
                                 why="OSM way 1366208564, גינה קהילתית עמק החובזות itself"),
    "גן עוז 12": dict(lat=32.462600, lng=34.965841, source="osm",
                      why="'the garden at the end of גן עוז': the dead end of OSM way 236101086"),
    "הצפירה פינת הבוטנים 15": dict(lat=32.467031, lng=34.975362, source="osm",
                                   why="the node הצפירה and הבוטנים share in OSM"),
    "גינת אגוז - דירה 35 קומה 9": dict(lat=32.467710, lng=34.955967, source="osm",
                                       why="a flat on גינת אגוז; centre of OSM way 69024212"),
    # Added 25/10/2026. הידית האחורית turned into a procession: it gathers at
    # 17:30 at רחבת יד לבנים (the venue it now names) and walks to the theatre,
    # with a sentence in the street field. The pin goes where people arrive.
    "לאחר התהלוכה - האירוע הראשי בתיאטרון הידית - רחוב מעלה פינת רבי עקיבא 4":
        dict(lat=32.475430, lng=34.973730, source="osm",
             why="the procession's start, רחבת יד לבנים, as הדקלים 90; the theatre is in the description"),
    "מעלה פינת רבי עקיבא 4": dict(lat=32.475430, lng=34.973730, source="osm",
                                  why="the same procession, rewritten the same day: 'מפגש ברחבת בית יד לבנים'"),
    "רחוב המעלה 4": dict(lat=32.474976, lng=34.969912, source="landmark",
                         why="the theatre, as המעלה 4 with 'רחוב' in front"),
}

# The festival's fourteen genres, folded into groups a legend can hold. The
# genre itself stays on the pin as the second chip.
GROUPS = [
    {"name": "מוסיקה", "color": "#f6a11b", "genres": ["מוסיקה"]},
    {"name": "סדנאות ויצירה", "color": "#00acac",
     "genres": ["סדנא חווייתית", "אמנות פלסטית", "ציור", "אוכל", "יצירה"]},
    {"name": "גוף ותנועה", "color": "#3ec28f", "genres": ["ספורט", "תנועה ומחול"]},
    {"name": "הרצאות ומילים", "color": "#7b5ea7", "genres": ["הרצאה", "הגות ושירה", "כתיבה"]},
    {"name": "ילדים ומשפחה", "color": "#e45462", "genres": ["שעת סיפור", "ילדים והורים", "משחק"]},
    {"name": "במה ותערוכות", "color": "#c2185b", "genres": ["תיאטרון", "תערוכה"]},
    # Ten screenings in 2026, the festival's own film strand: a group of its own.
    {"name": "קולנוע", "color": "#3949ab", "genres": ["קולנוע"]},
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
    tail = re.search(r"^(.*?),?\s+(\d+)$", street)
    # 'לאה גולדברג, 12' with 12 in the number field too: one number, not two.
    if tail and (not number or int(tail.group(2)) == number):
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
            "drive.google.com": "גוגל דרייב", "chat.whatsapp.com": "קבוצת ווטסאפ",
            "api.whatsapp.com": "ווטסאפ", "wa.me": "ווטסאפ"}.get(host, host or "קישור")


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


def locate(place_rec, key, known, cache):
    """The position for a venue and where it came from.

    The API's own coordinate is the default, because the platform geocoded a
    full address and most of them check out; `known` keeps the ones it gave
    before it stopped giving them. FIXES overrides both where a better source
    was found. A venue with neither falls back to B144, which is right for a
    house on a street and wrong for a landmark (see the docstring) - those
    show as source b144 in the report and get a FIXES line if they need one.
    Nothing at all, and the pin goes out unplaced."""
    fix = FIXES.get(key)
    if fix:
        return {"lat": fix["lat"], "lng": fix["lng"], "source": fix["source"]}
    if place_rec:
        lat, lng = float(place_rec.get("lat") or 0), float(place_rec.get("lng") or 0)
        if in_box(lat, lng):
            known[key] = [round(lat, 6), round(lng, 6)]
    if key in known:
        lat, lng = known[key]
        return {"lat": lat, "lng": lng, "source": "festival"}
    street, number = key.rsplit(" ", 1)
    second = b144(street, number, cache) if int(number) else None
    if second and in_box(*second):
        return {"lat": round(second[0], 6), "lng": round(second[1], 6), "source": "b144"}
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
    known = load_json(API_POSITIONS, {})
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
        geo = locate(rec, key, known, cache)
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

    save_json(API_POSITIONS, dict(sorted(known.items())))
    save_json(B144_CACHE, cache)

    if report:
        print("\nvenue                     api                  b144                  apart  used")
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
    parser.add_argument("--publish", action="store_true",
                        help="put the file in the data repo, where the app reads it")
    args = parser.parse_args()
    doc = build(args.report)
    if args.publish:
        s = doc["stats"]
        msg = (f"דרך הנדיב: {s['events']} אירועים ב-{s['venues']} כתובות, "
               f"{s['placed']} ממוקמים (build_hanadiv.py, בוטי)")
        print(f"ריפו הנתונים: {publish_json('data/hanadiv.json', doc, msg)}")


if __name__ == "__main__":
    main()
