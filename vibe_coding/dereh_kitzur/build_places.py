#!/usr/bin/env python3
"""Build the pardespedia places layer: web/data/places.json.

Pardespedia is the community wiki of Pardes Hanna-Karkur, and it already holds
several hundred articles about physical places in the moshava - cafes, parks,
playgrounds, schools, memorials, neighbourhoods. Each one has a summary, most
have a photo, and every one has a permanent URL. That is most of a map layer.

The one thing it does not have is coordinates. The wiki runs no GeoData
extension and no location template, so nothing in it knows where anything is.
This script works the position out from what the articles do contain, in
descending order of trust:

    manual    a pin an editor dropped in the app. Never overwritten.
    google    Google Places, behind --google and a key you supply. The only
              source that actually knows where a small business is. Free in
              practice: see GooglePlaces for the arithmetic and the guard rails.
    osm       the article title matches a named feature in OpenStreetMap
              exactly. Precise when it hits, which is mostly for the places
              somebody bothered to map.
    address   a house number resolved through Nominatim. Kept for the day this
              becomes possible - see below - but it currently never fires.
    street    the street is known and the house number is not, so the position
              is the middle of the road. Marked approximate in the app.
    nearby    no address of its own, but the article names another place we
              have already located ("in the Hayadit compound"). Inherits its
              position, also approximate.

**No house number in this town resolves.** Every one of the ninety in-area
results Nominatim returned was of class `highway` - the road itself, handed
back because the number it was asked for does not exist in OpenStreetMap. So
street level is the ceiling for geocoding here, not a fallback from something
better, and a business "at Hameyasdim 16" is placed in the middle of
Hameyasdim. Do not read `street` as a near miss; read it as the best any free
geocoder can do for Pardes Hanna-Karkur today.

Free-text geocoding is worse still, and was tried first: two hits in thirty,
against fourteen for the same addresses split into street and number.

Whatever is left over is written out with no position at all. Those articles
still appear in the app, in a list of places waiting for someone to place them,
and an editor pins them from the map in a few seconds each. That is the honest
division of labour here: the script does what can be automated and hands the
rest over as a small, finite job rather than guessing.

Coordinates already in places.json are kept. Re-running refreshes the text and
the photos and leaves every position alone unless --regeocode is passed.

    pip install requests
    python3 build_places.py              # refresh text, geocode what is new
    python3 build_places.py --regeocode  # throw away derived positions, redo
    python3 build_places.py --no-net     # wiki only, no Overpass, no Nominatim
    GOOGLE_MAPS_API_KEY=... python3 build_places.py --google   # the accurate pass
"""

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.parse

import requests

WIKI = "https://pardespedia.info/api.php"
WIKI_PAGE = "https://pardespedia.info/wiki/"
OVERPASS = "https://overpass-api.de/api/interpreter"
NOMINATIM = "https://nominatim.openstreetmap.org/search"

# Nominatim's usage policy asks for a real contact address and at most one
# request a second. Both are cheap to honour and the results are cached, so a
# rerun costs nothing.
UA = {"User-Agent": "derech-kitzur-places/1.0 (https://github.com/orimosenzon/fun; orimosenzon@gmail.com)"}
NOMINATIM_DELAY = 1.1

OUT = "web/data/places.json"
OSM_CACHE = ".cache/osm_named.json"
GEO_CACHE = ".cache/geocode.json"

# The moshava plus a margin. Anything a geocoder returns outside this is a
# different town with a street of the same name, which happens constantly:
# every third settlement in Israel has a המייסדים.
BOX = (32.435, 34.925, 32.515, 35.005)          # south, west, north, east

HE = r"֐-׿"

# Which categories describe something you can stand in front of, and what to
# call the resulting group. A place in several of these takes the first match,
# so the order is the priority: what the place *is* beats what it also is.
GROUPS = [
    ("אוכל ושתייה", "#d84315", [
        "בתי קפה", "מסעדות", "ברים ופאבים", "פלאפליות וסביח", "פודטראק",
        "חנויות לממכר מזון", "מאפיות",
    ]),
    ("טבע וגנים", "#2e7d32", [
        "גני משחקים", "טבע עירוני", "כיכרות", "הגנת הסביבה", "קיימות",
    ]),
    ("חינוך", "#1565c0", [
        "מוסדות חינוך", "בית הספר החקלאי פרדס חנה",
    ]),
    ("תרבות וקהילה", "#6a1b9a", [
        "מרחבים קהילתיים", "אמנויות הבמה", "אמנות", "תרבות מקומית",
        "מרחבי פנאי וזיכרון בקהילה", "קהילות", "פעילות קהילתית",
        "יוזמה קהילתית", "אקטיביזם במושבה", "מלאכת יד",
    ]),
    ("היסטוריה והנצחה", "#5d4037", [
        "הנצחה", "מבנים לשימור", "היסטוריה", "סמלים היסטוריים",
    ]),
    ("שירותים ובריאות", "#00838f", [
        "בריאות", "קופות חולים", "מתקנים ציבוריים", "שרותי דת",
        "מסגרות לגיל השלישי", "ספורט", "אמנויות לחימה", "תחבורה",
        "מועצה מקומית פרדס חנה כרכור",
    ]),
    ("שכונות", "#455a64", [
        "שכונות", "יובלים (שכונה)",
    ]),
    ("עסקים", "#ef6c00", [
        "עסקים ותעשייה בפרדס חנה-כרכור", "חנויות ובוטיקים",
    ]),
]

# Streets are deliberately left out. They are lines rather than points, the
# basemap already labels them, and 64 pins sitting in the middle of 64 roads
# would bury everything else on the layer.
SKIP_CATEGORIES = {
    "אישים בפרדס חנה-כרכור", "רחובות", "תבניות", "דף מיזם", "מושגים",
    "תמונות", "תמונות אירועים", "תמונות מוויקישיתוף", "תמונות בשימוש הוגן",
    "תמונות של אישים", "שימוש הוגן", "תחזוקה", "ערכים להרחבה",
}

CATEGORIES = [c for _, _, cats in GROUPS for c in cats]

# Titles that are a survey of a topic rather than a place you can visit. They
# carry the same categories as the places they describe, so nothing but a list
# separates them.
NOT_A_PLACE = re.compile(
    r"^(רשימת|בתי הקפה של|מסעדות של|אולמות ומרחבי|גני המשחקים|"
    r"תוכנית המתאר|תכנית המתאר|היסטוריה של)|"
    r"(בפרדס חנה-כרכור|במושבה)$")


# ---------------------------------------------------------------- wiki

def wiki(**params):
    params.setdefault("format", "json")
    params.setdefault("action", "query")
    res = requests.post(WIKI, data=params, headers=UA, timeout=60)
    res.raise_for_status()
    return res.json()


def category_members(name):
    out, cont = [], {}
    while True:
        got = wiki(list="categorymembers", cmtitle="קטגוריה:" + name,
                   cmlimit=500, cmnamespace=0, **cont)
        out += [m["title"] for m in got["query"]["categorymembers"]]
        if "continue" not in got:
            return out
        cont = got["continue"]


def collect_titles():
    """Every article in a place category, with the categories it came from."""
    found = {}
    for group, _, cats in GROUPS:
        for cat in cats:
            try:
                members = category_members(cat)
            except Exception as err:                       # a category that no longer exists
                print(f"  ! קטגוריה {cat}: {err}", file=sys.stderr)
                continue
            for title in members:
                found.setdefault(title, {"group": group, "cats": []})["cats"].append(cat)
    for title in category_members("אישים בפרדס חנה-כרכור"):
        found.pop(title, None)
    return {t: v for t, v in found.items() if not NOT_A_PLACE.search(t)}


def fetch_pages(titles):
    """Wikitext for every article, in batches of forty."""
    pages = {}
    for i in range(0, len(titles), 40):
        batch = titles[i:i + 40]
        got = wiki(prop="revisions", rvprop="content", rvslots="main",
                   titles="|".join(batch))
        for page in got["query"]["pages"].values():
            if "revisions" in page:
                pages[page["title"]] = page["revisions"][0]["slots"]["main"]["*"]
        print(f"\r  ערכים: {len(pages)}/{len(titles)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return pages


def image_urls(filenames):
    """Resolve file names to a thumbnail and a full-size URL.

    These are hotlinked rather than mirrored. The My Maps photos had to be
    copied because Google serves them with cross-origin-resource-policy:
    same-site, which blocks them in any other page; pardespedia sets no such
    header, so its own images load straight from it and cost this repo nothing.
    """
    out = {}
    names = list(filenames)
    for i in range(0, len(names), 40):
        batch = names[i:i + 40]
        got = wiki(prop="imageinfo", iiprop="url|size", iiurlwidth=520,
                   titles="|".join("קובץ:" + n for n in batch))
        for page in got["query"]["pages"].values():
            info = (page.get("imageinfo") or [None])[0]
            if not info:
                continue
            name = page["title"].split(":", 1)[1]
            out[name] = {
                "thumb": info.get("thumburl") or info["url"],
                "full": info["url"],
                "page": WIKI_PAGE + urllib.parse.quote(page["title"].replace(" ", "_")),
            }
    return out


# ------------------------------------------------------- wikitext parsing

FILE_RE = re.compile(r"\[\[\s*(?:קובץ|File|Image|תמונה)\s*:\s*([^|\]]+?)\s*[|\]]")
GMAPS_RE = re.compile(r"https?://(?:www\.)?google\.com/maps/[^\s\]]*?[?&]query=([^&\s\]]+)")
BOLD_ADDR_RE = re.compile(r"'''\s*כתובת\s*:?\s*'''\s*:?\s*(.+)")
SECT_ADDR_RE = re.compile(r"==+\s*כתובת[^=\n]*==+\s*\n+\s*(.+)")


def strip_markup(text):
    """Wikitext down to something a person can read in a map popup."""
    text = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\[\[\s*(?:קובץ|File|Image|תמונה)\s*:[^\]]*\]\]", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("'''", "").replace("''", "")
    return re.sub(r"[ \t]+", " ", text)


def summary(wikitext, limit=320):
    """The lead, which on this wiki is the sentence that says what the place is."""
    body = strip_markup(wikitext)
    for para in body.split("\n"):
        para = para.strip()
        if len(para) < 25 or para.startswith(("=", "*", "|", "{", "#", "!")):
            continue
        if len(para) <= limit:
            return para
        cut = para[:limit]
        stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        return (cut[:stop + 1] if stop > limit * 0.5 else cut.rsplit(" ", 1)[0]) + "…"
    return ""


def lead_image(wikitext):
    match = FILE_RE.search(wikitext)
    return match.group(1).strip().replace("_", " ") if match else None


def raw_address(wikitext):
    """The address as the article states it, from wherever it states it."""
    match = GMAPS_RE.search(wikitext)
    if match:
        return urllib.parse.unquote_plus(match.group(1))
    for pattern in (BOLD_ADDR_RE, SECT_ADDR_RE):
        match = pattern.search(wikitext)
        if match:
            return strip_markup(match.group(1))
    return None


def street_in_text(wikitext, roads):
    """An address buried in the prose, rather than in the article's own fields.

    Plenty of articles never fill in a כתובת line but say "on Hadkalim 131" in
    passing. Any number in Hebrew text could be a year, a house number or a
    road number, so the words in front of it are checked against the streets
    that actually exist in this town - which is what makes בשנת 1948 and
    כביש 65 fall out on their own.
    """
    words = strip_markup(wikitext).split()
    for i, word in enumerate(words):
        number = re.fullmatch(r"(\d{1,3})[,.]?", word)
        if not number or i == 0:
            continue
        for back in (2, 1):
            if i - back < 0:
                continue
            name = " ".join(words[i - back:i]).strip(" ,.-'\"()")
            name = re.sub(r"^(רחוב|רח'|שדרות|שד')\s+", "", name)
            if len(name) < 3:
                continue
            for candidate in (name, "ה" + name):
                if normalise(candidate) in roads:
                    return f"{number.group(1)} {name}", normalise(candidate)
    return None


def external_links(wikitext):
    """Official site, Instagram, Facebook - whatever the article already links."""
    links = []
    seen = set()
    for url, label in re.findall(r"\[(https?://[^\s\]]+)\s*([^\]]*)\]", wikitext):
        if "google.com/maps" in url or "pardespedia" in url:
            continue
        host = urllib.parse.urlparse(url).netloc.replace("www.", "")
        if host in seen:
            continue
        seen.add(host)
        links.append({"url": url, "title": (label.strip() or host)[:60]})
    return links[:4]


# ------------------------------------------------------------ coordinates

def normalise(name):
    """Fold the spelling differences between a wiki title and an OSM name."""
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"\(.*?\)", " ", name)
    name = (name.replace("־", "-").replace("–", "-")
                .replace("“", '"').replace("”", '"').replace("״", '"')
                .replace("’", "'").replace("׳", "'"))
    name = re.sub(rf"[^\w{HE}]+", " ", name)
    return " ".join(name.split()).lower()


def inside(lat, lng):
    return BOX[0] < lat < BOX[2] and BOX[1] < lng < BOX[3]


# The town and its halves. A business "in Karkur" has told you nothing about
# where it is, and left in the anchor list these three swallow everything.
SETTLEMENTS = {normalise(n) for n in
               ("פרדס חנה-כרכור", "פרדס חנה", "כרכור", "פרדס חנה כרכור")}

# How many places may borrow one anchor's position. A courtyard really can hold
# two dozen businesses; a name that a hundred articles mention is a figure of
# speech rather than a location.
CROWD = 40


def spread(places):
    """Pull apart every pile of pins that landed on one identical point.

    Both approximations produce piles. Two dozen businesses in the Orvot
    HaOmanim yard inherit the yard's coordinates; every shop we could place
    only to the street gets that street's midpoint. A stack of pins on one
    point is one pin - you click the top of the pile and never learn the rest
    are under it.

    So the members of a pile go onto a ring around it. This does not make any
    of them more accurate, and it is not meant to: they are all flagged as
    approximate either way. It makes each one reachable.
    """
    piles = {}
    for place in places:
        geo = place.get("geo")
        if geo and geo["source"] not in ("manual", "google", "osm"):
            piles.setdefault((geo["lat"], geo["lng"]), []).append(place)

    for (lat, lng), pile in piles.items():
        if len(pile) < 2:
            continue
        radius = 10 + 1.2 * len(pile)                  # metres
        for i, place in enumerate(sorted(pile, key=lambda p: p["name"])):
            angle = 2 * math.pi * i / len(pile)
            d_lat = radius * math.cos(angle) / 111320
            d_lng = radius * math.sin(angle) / (111320 * math.cos(math.radians(lat)))
            place["geo"]["lat"] = round(lat + d_lat, 6)
            place["geo"]["lng"] = round(lng + d_lng, 6)
            place["geo"]["spread"] = True


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return default


def save_json(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=1)


# The app writes to places.json as well as reading it: dragging a pin in the
# arrange tool stores a `manual` position straight into the published copy.
# Those pins survive a rebuild only because `previous` carries them over - so
# `previous` has to be read from the published document and not from
# web/data/, which is an offline mirror and goes stale the moment somebody
# pins a place from their phone.
#
# Measured 2/9/2026: the mirror was six days behind the published copy and one
# place further short. A rebuild from it would have dropped that place with no
# error and no way to notice.
PUBLISHED_PLACES = ("https://raw.githubusercontent.com/orimosenzon/"
                    "derech-kitzur-data/main/data/places.json")


def published_places(offline=False):
    """The places document the app is actually serving, for `previous`.

    Falls back to the local mirror when the network is not there, and says so
    out loud: building offline is fine, building offline without knowing it is
    how a pin gets lost.
    """
    local = load_json(OUT, {})
    if offline:
        print("  --no-net: הבסיס נלקח מהעותק המקומי", file=sys.stderr)
        return local

    try:
        # raw sits behind a five minute CDN cache, which is exactly long enough
        # to serve a copy from before the pin that prompted this rebuild.
        res = requests.get(PUBLISHED_PLACES, params={"t": int(time.time())},
                           timeout=20)
        res.raise_for_status()
        remote = res.json()
    except Exception as err:                       # noqa: BLE001 - any failure
        print(f"  אזהרה: לא הצלחתי לקרוא את places.json המפורסם ({err})",
              file=sys.stderr)
        print("  ממשיך מהעותק המקומי. מיקומים שנעצו מהאפליקציה עלולים ללכת לאיבוד.",
              file=sys.stderr)
        return local

    pinned = lambda doc: {p["id"] for p in doc.get("places", [])          # noqa: E731
                          if (p.get("geo") or {}).get("source") == "manual"}
    gained = pinned(remote) - pinned(local)
    if gained:
        print(f"  {len(gained)} מיקומים נעצו מהאפליקציה מאז העותק המקומי, "
              "והם נשמרים", file=sys.stderr)
    return remote


def osm_index(offline=False):
    """Every named node and way in the moshava, keyed by normalised name.

    Highways are kept apart from everything else. A street called אחוזה and a
    synagogue called אחוזה are both in here, and matching an article about the
    synagogue to the street would put the pin a kilometre away with no sign
    that anything went wrong.
    """
    elements = load_json(OSM_CACHE, None)
    if elements is None:
        if offline:
            return {}, {}
        south, west, north, east = BOX
        query = f"""[out:json][timeout:180];
(
  node["name"]({south},{west},{north},{east});
  way["name"]({south},{west},{north},{east});
);
out center tags;"""
        print("  מושך שמות מ-OpenStreetMap…", file=sys.stderr)
        # Overpass answers 406 to a request with no User-Agent.
        res = requests.post(OVERPASS, data={"data": query}, headers=UA, timeout=200)
        res.raise_for_status()
        elements = res.json()["elements"]
        save_json(OSM_CACHE, elements)

    pois, roads = {}, {}
    for el in elements:
        point = el.get("center") or el
        if "lat" not in point:
            continue
        target = roads if "highway" in el.get("tags", {}) else pois
        for key in ("name", "name:he", "alt_name", "official_name"):
            value = el["tags"].get(key)
            if value:
                target.setdefault(normalise(value), (point["lat"], point["lon"]))
    return pois, roads


class GooglePlaces:
    """Positions from Google, which is the only source that actually has them.

    OpenStreetMap knows 442 named points in this whole town and no house
    numbers at all, so for a cafe on a side street there is no free answer.
    Google's own position for the same cafe is usually right.

    Cost, because this is the one part of the project that can charge money:
    a Text Search asking for `places.location` bills against the Places API
    Pro tier, whose free allowance is 5,000 calls a month. This project has
    about 450 places to ask about, once. Every answer is cached, so a rerun
    costs nothing. The realistic bill is zero, and it stays zero as long as
    nothing turns this into a loop - which is what MAX_CALLS is for.

    Even so, before enabling billing on a Google Cloud project, set a hard
    daily quota on the Places API in the console (APIs & Services > Quotas).
    A budget alert only tells you afterwards; a quota is what actually stops.

    The key is read from the environment and never written anywhere.
    """

    ENDPOINT = "https://places.googleapis.com/v1/places:searchText"
    FIELDS = "places.location,places.displayName,places.formattedAddress"
    MAX_CALLS = 600                # a ceiling a bug cannot spend past

    def __init__(self, key, cache_path=".cache/google_places.json"):
        self.key = key
        self.path = cache_path
        self.cache = load_json(cache_path, {})
        self.calls = 0
        self.stopped = False

    def save(self):
        save_json(self.path, self.cache)

    def find(self, name):
        """{'lat','lng','name','address'} or None."""
        if name in self.cache:
            hit = self.cache[name]
            return hit or None
        if self.stopped:
            return None
        if self.calls >= self.MAX_CALLS:
            print(f"  ! עצירה: {self.MAX_CALLS} קריאות לגוגל בריצה אחת", file=sys.stderr)
            self.stopped = True
            return None

        body = {
            "textQuery": f"{name} פרדס חנה כרכור",
            "languageCode": "he",
            "regionCode": "IL",
            "maxResultCount": 3,
            # A bias, not a filter: a wrong-town result is still thrown out
            # below, but biasing means the right one usually comes first.
            "locationBias": {"circle": {
                "center": {"latitude": 32.4755, "longitude": 34.966},
                "radius": 6000.0
            }},
        }
        headers = {**UA, "Content-Type": "application/json",
                   "X-Goog-Api-Key": self.key, "X-Goog-FieldMask": self.FIELDS}
        self.calls += 1
        try:
            res = requests.post(self.ENDPOINT, json=body, headers=headers, timeout=40)
            if res.status_code in (401, 403):
                print(f"  ! גוגל דחתה את המפתח ({res.status_code}). עוצר.", file=sys.stderr)
                self.stopped = True
                return None
            res.raise_for_status()
            results = res.json().get("places", [])
        except Exception as err:
            print(f"  ! google {name}: {err}", file=sys.stderr)
            return None

        found = None
        for place in results:
            loc = place.get("location") or {}
            lat, lng = loc.get("latitude"), loc.get("longitude")
            if lat is None or not inside(lat, lng):
                continue
            title = (place.get("displayName") or {}).get("text", "")
            # Text search answers something for almost any string. Requiring a
            # word in common with what we asked for is what stops a bakery in
            # Hadera from becoming the position of a bakery here.
            if not (toks(title) & toks(name)):
                continue
            found = {"lat": lat, "lng": lng, "name": title,
                     "address": place.get("formattedAddress", "")}
            break

        self.cache[name] = found
        if self.calls % 20 == 0:
            self.save()
        return found


def toks(text):
    """Words worth comparing: the generic half of a Hebrew business name is
    shared by half the businesses in town."""
    drop = {"בית", "בתי", "גן", "מרכז", "חנות", "סטודיו", "קפה", "בר", "של",
            "מסעדה", "מסעדת", "פרדס", "חנה", "כרכור", "the", "and", "of"}
    return {w for w in normalise(text).split() if len(w) > 2 and w not in drop}


class Geocoder:
    """Structured Nominatim lookups, cached on disk.

    Structured is the whole point. Handing Nominatim the address as free text -
    which is how it comes out of the article, business name and all - resolved
    two addresses out of thirty. Splitting out the street and the house number
    and naming the town resolved fourteen.
    """

    STREET = re.compile(
        rf"(?:רחוב|רח'|דרך|שדרות|שד'|סמטת)?\s*([{HE}][{HE}\s'\"-]{{2,25}}?)\s+(\d{{1,3}})\b")
    STREET_ONLY = re.compile(rf"(?:רחוב|רח'|דרך|שדרות|סמטת)\s+([{HE}][{HE}\s'\"-]{{2,20}})")
    NOISE = re.compile(r"\b(פרדס חנה[- ]כרכור|פרדס חנה|כרכור|ישראל|קומה|דירה|ת\.ד)\b")

    def parse(self, address, title):
        """An address string down to (street+number, street)."""
        text = strip_markup(address)
        text = re.sub(r"\(.*?\)", " ", text)
        text = text.replace(strip_markup(title), " ")
        text = self.NOISE.sub(" ", text)
        text = re.sub(r"\d{5,}", " ", text)                 # postcode, not a house
        text = " ".join(text.split())

        match = self.STREET.search(text)
        if match:
            street = re.sub(r"^(רחוב|רח'|דרך|שדרות|סמטת)\s+", "", match.group(1).strip(" ,.-"))
            if len(street) >= 3:
                return f"{match.group(2)} {street}", street
        match = self.STREET_ONLY.search(text)
        if match:
            return None, match.group(1).strip(" ,.")
        return None, None

    # Nominatim answers a house-number query it cannot satisfy with the road
    # itself rather than with nothing, and says so only in the result's class.
    # Israeli house numbers are patchy in OSM, so this is the common case: nine
    # cafes on Derech HaBanim all came back with one set of coordinates while
    # claiming house-number precision. Trust the class, not the question.
    VERSION = 2

    def __init__(self, offline=False):
        cached = load_json(GEO_CACHE, {})
        self.cache = cached.get("hits", {}) if cached.get("v") == self.VERSION else {}
        self.offline = offline
        self.calls = 0

    def save(self):
        save_json(GEO_CACHE, {"v": self.VERSION, "hits": self.cache})

    def query(self, street, city):
        key = f"{street}|{city}"
        if key not in self.cache:
            if self.offline:
                return None
            params = {"street": street, "city": city, "country": "Israel",
                      "format": "json", "limit": 2, "countrycodes": "il",
                      "accept-language": "he"}
            time.sleep(NOMINATIM_DELAY)
            self.calls += 1
            try:
                res = requests.get(NOMINATIM, params=params, headers=UA, timeout=40)
                res.raise_for_status()
                hits = res.json()
            except Exception as err:
                print(f"  ! nominatim {street}: {err}", file=sys.stderr)
                return None
            self.cache[key] = [{"lat": float(h["lat"]), "lng": float(h["lon"]),
                                "cls": h.get("class", "")} for h in hits]
            if self.calls % 20 == 0:
                self.save()
        for hit in self.cache[key]:
            if inside(hit["lat"], hit["lng"]):
                return hit
        return None

    def locate(self, address, title):
        """(lat, lng, precision) or None.

        Precision separates a real house number from the middle of a road, and
        the app shows the reader which it got rather than hiding it."""
        full, street = self.parse(address, title)
        for candidate, exact in ((full, True), (street, False)):
            if not candidate:
                continue
            for city in ("פרדס חנה-כרכור", "כרכור", "פרדס חנה"):
                hit = self.query(candidate, city)
                if hit:
                    precise = exact and hit["cls"] != "highway"
                    return hit["lat"], hit["lng"], ("address" if precise else "street")
        return None


# ------------------------------------------------------------------ build

def build(args):
    print("קטגוריות…", file=sys.stderr)
    found = collect_titles()
    titles = sorted(found)
    print(f"  {len(titles)} ערכי מקומות", file=sys.stderr)

    pages = fetch_pages(titles)

    previous = {p["id"]: p for p in published_places(args.no_net).get("places", [])}
    pois, roads = osm_index(args.no_net)
    geo = Geocoder(args.no_net)

    places = []
    for title in titles:
        wikitext = pages.get(title)
        if not wikitext:
            continue
        meta = found[title]
        places.append({
            "id": "pp-" + re.sub(r"\s+", "_", title),
            "name": title,
            "group": meta["group"],
            "cats": meta["cats"],
            "note": summary(wikitext),
            "url": WIKI_PAGE + urllib.parse.quote(title.replace(" ", "_")),
            "links": external_links(wikitext),
            "_file": lead_image(wikitext),
            "_address": raw_address(wikitext),
        })

    # Photos, in one batch rather than one request per article.
    files = {p["_file"] for p in places if p["_file"]}
    print(f"תמונות: {len(files)} קבצים…", file=sys.stderr)
    images = image_urls(files) if files else {}
    for place in places:
        image = images.get(place.pop("_file") or "")
        place["photos"] = [image] if image else []

    # -- positions
    stats = {"manual": 0, "google": 0, "osm": 0, "address": 0, "street": 0,
             "nearby": 0, "none": 0}

    google = None
    if args.google:
        key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
        if not key:
            print("  ! --google בלי GOOGLE_MAPS_API_KEY בסביבה. מדלג.", file=sys.stderr)
        else:
            google = GooglePlaces(key)
            print("  שואל את גוגל על מקומות שאין להם מיקום מדויק…", file=sys.stderr)

    for place in places:
        old = previous.get(place["id"], {})
        keep = old.get("geo") and (
            old["geo"]["source"] == "manual"          # a person put it there
            or (not args.regeocode
                # Google beats anything this script can derive, so with --google
                # a derived guess is worth replacing - but a Google answer we
                # already have is not worth paying for twice.
                and not (google and old["geo"]["source"] not in ("google",))))
        if keep:
            place["geo"] = old["geo"]
            stats[old["geo"]["source"]] = stats.get(old["geo"]["source"], 0) + 1
            place.pop("_address", None)
            continue

        if google:
            hit = google.find(place["name"])
            if hit:
                place["geo"] = {"lat": round(hit["lat"], 6), "lng": round(hit["lng"], 6),
                                "source": "google"}
                if hit["address"]:
                    place["address"] = hit["address"][:90]
                stats["google"] += 1
                place.pop("_address", None)
                continue
            # No answer from Google either. Fall through to the free sources
            # rather than leaving it unplaced.
            if old.get("geo") and not args.regeocode:
                place["geo"] = old["geo"]
                stats[old["geo"]["source"]] = stats.get(old["geo"]["source"], 0) + 1
                place.pop("_address", None)
                continue

        key = normalise(place["name"])
        hit = pois.get(key) or (roads.get(key) if place["group"] == "שכונות" else None)
        if hit:
            place["geo"] = {"lat": round(hit[0], 6), "lng": round(hit[1], 6), "source": "osm"}
            stats["osm"] += 1
            place.pop("_address", None)
            continue

        address = place.pop("_address", None)
        if address:
            place["address"] = " ".join(strip_markup(address).split())[:90]
        located = geo.locate(address, place["name"]) if address else None

        if not located:
            # No address field, or one the geocoder could not resolve. The
            # street may still be sitting in the prose.
            prose = street_in_text(pages[place["name"]], roads)
            if prose:
                query, road_key = prose
                located = geo.locate(query, place["name"])
                if not located:
                    # The street exists but not that house number. The middle of
                    # the road is a couple of hundred metres out, which is worth
                    # saying as "roughly here" and not worth pretending is exact.
                    lat, lng = roads[road_key]
                    located = (lat, lng, "street")

        if located:
            lat, lng, how = located
            place["geo"] = {"lat": round(lat, 6), "lng": round(lng, 6), "source": how}
            stats[how] += 1
        else:
            place["_text"] = strip_markup(pages[place["name"]])[:1200]
    geo.save()
    if google:
        google.save()
        print(f"  גוגל: {google.calls} קריאות בפועל (השאר מהמטמון)", file=sys.stderr)

    # -- a place that names a compound we already found sits roughly where it does
    #
    # "Olika is a roastery in the Orvot HaOmanim compound" really does say where
    # Olika is. "X is a bar in Karkur" says nothing, and a first attempt at this
    # pinned 191 businesses onto the Karkur article and 52 more onto a pub
    # called המקומית, matched inside the words המועצה המקומית. So an anchor has
    # to be a multi-word name, which is what compounds are called and what
    # districts are not, and no anchor may claim a crowd.
    anchors = {normalise(p["name"]): p["geo"] for p in places
               if p.get("geo") and p["group"] != "שכונות"
               and normalise(p["name"]) not in SETTLEMENTS
               and len(normalise(p["name"]).split()) >= 2}

    claimed = {}
    for place in places:
        text = place.pop("_text", None)
        if place.get("geo") or not text:
            continue
        haystack = normalise(text)
        for name, geo_ref in anchors.items():
            if re.search(rf"(?<![^\s]){re.escape(name)}(?![^\s])", haystack):
                claimed.setdefault(name, []).append((place, geo_ref))
                break

    for name, group in claimed.items():
        if len(group) > CROWD:
            print(f"  ~ מדלג על {name}: {len(group)} מקומות טוענים שהם שם",
                  file=sys.stderr)
            continue
        for place, geo_ref in group:
            place["geo"] = {"lat": geo_ref["lat"], "lng": geo_ref["lng"],
                            "source": "nearby", "near": name}
            stats["nearby"] += 1
    for place in places:
        place.pop("_text", None)
        place.pop("_address", None)
        if not place.get("geo"):
            stats["none"] += 1
    spread(places)

    places.sort(key=lambda p: (p["group"], p["name"]))
    doc = {
        "version": 1,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "https://pardespedia.info",
        "groups": [{"name": name, "color": colour} for name, colour, _ in GROUPS],
        "places": places,
        "stats": {
            "places": len(places),
            "located": sum(1 for p in places if p.get("geo")),
            "photos": sum(len(p["photos"]) for p in places),
        },
    }
    save_json(OUT, doc)

    print(f"\nנכתב {OUT}", file=sys.stderr)
    print(f"  {len(places)} מקומות, {doc['stats']['located']} עם מיקום, "
          f"{doc['stats']['photos']} תמונות", file=sys.stderr)
    for source, count in stats.items():
        if count:
            print(f"    {source}: {count}", file=sys.stderr)
    if stats["none"]:
        print(f"  {stats['none']} ממתינים לנעיצה ידנית באפליקציה.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regeocode", action="store_true",
                        help="derive every position afresh; hand-placed pins still survive")
    parser.add_argument("--no-net", action="store_true",
                        help="wiki only: no Overpass, no Nominatim, cached lookups only")
    parser.add_argument("--google", action="store_true",
                        help="ask Google Places for the positions the free sources cannot "
                             "give. Needs GOOGLE_MAPS_API_KEY in the environment. "
                             "~450 calls against a 5,000/month free allowance, cached, "
                             "capped at 600 per run. Never overwrites a hand-placed pin.")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
