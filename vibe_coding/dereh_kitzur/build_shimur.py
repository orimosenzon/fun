#!/usr/bin/env python3
"""Build the two conservation layers: web/data/shimur.json and web/data/makom_shamur.json.

The moshava has two lists of what is worth keeping, and they are different in
kind, which is why they are two layers and not one:

    shimur.json       the conservation appendix of the comprehensive master
                      plan (תכנית 353-0138586). Statutory: a site on this list
                      carries a legal conservation grade, and the plan says what
                      may and may not be done to it.

    makom_shamur.json מקום שמור, Ilana Pelada's documentation project. Not
                      statutory at all - an invitation. Its own list page says
                      "כל האתרים הם בגדר הצעה", and it includes buildings
                      nobody has protected and some that are already gone.

Where the data comes from
-------------------------
The appendix is a 287-page scan with no text layer, published by the planning
administration at

    https://apps.land.gov.il/IturTabotData/nispachim/haifa/3008729/12.pdf

Its last twenty pages are "נספח 2: כרטסת השימור לפי מפתח כתובת" - one row per
site with name, address, block and parcel, conservation grade and the site's
number on the conservation map. That table was transcribed from the page images
into `shimur_src/nispach_shimur.tsv`, which is the input here. Re-run this
script after editing that file; do not edit the JSON.

מקום שמור is a WordPress.com site, so its two list pages and its sixty-odd
documented buildings come off the public WordPress API rather than the HTML.

The thing neither source has is coordinates
-------------------------------------------
The appendix gives a block and parcel (גוש וחלקה) for every site, which is
better than an address: it resolves against the national parcel layer to an
actual polygon. So a site's position here is the area-weighted centroid of its
parcels, taken from

    https://open.govmap.gov.il/geoserver/opendata/ows   (opendata:PARCEL_ALL)

That is the centre of the land, not the centre of the building. On a small
town plot the two are metres apart; on the former agricultural school, whose
compound is sixteen parcels, the pin lands in the middle of the campus. Sites
whose row spans several parcels - the avenues, the compounds - are flagged
`approx` in the app for exactly this reason, and their pin should be read as
"this site is around here", not "the door is here".

מקום שמור has neither coordinates nor parcels, only names. Those are matched
against the appendix, then against the pardespedia places layer, then against
street names in OpenStreetMap for the entries that are a street. That places
eighteen of its eighty-two. The rest are written out with no position at all
and show up in the app's list of places waiting to be pinned - which is not a
failure of the matching but the shape of the list: most of what is left is
private houses known only by a family name, and no index anywhere holds them.

    python3 build_shimur.py                 # both layers
    python3 build_shimur.py --no-net        # cached parcel lookups only
    python3 build_shimur.py --refresh-cache # re-fetch the parcel geometry
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
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "shimur_src", "nispach_shimur.tsv")
OUT_SHIMUR = os.path.join(HERE, "web", "data", "shimur.json")
OUT_MAKOM = os.path.join(HERE, "web", "data", "makom_shamur.json")
CACHE = os.path.join(HERE, ".cache", "parcels.json")

WFS = "https://open.govmap.gov.il/geoserver/opendata/ows"
WP = "https://public-api.wordpress.com/rest/v1.1/sites/makomshamur.com"

PLAN = "353-0138586"
APPENDIX_URL = "https://apps.land.gov.il/IturTabotData/nispachim/haifa/3008729/12.pdf"
MAKOM_URL = "https://makomshamur.com/"

# The two list pages on makomshamur.com. Both are ordinary published posts that
# the site links to with a stale `?preview=true` query, which is why they are
# addressed here by id rather than by the link in the page.
MAKOM_LISTS = {652: "פרדס חנה", 650: "כרכור"}

# The narratives the appendix sorts its sites into. The first number of a site's
# code is its narrative - 1.3 is a water site, 4.1 a British army one - and the
# appendix's own chapter headings (5.7 to 5.15) are where these names come from.
# The colours are this app's, chosen to stay apart from the trail greens and
# from the pardespedia purple. The appendix's own conservation-map colours are
# a different scheme entirely and are deliberately not reproduced: they mean
# something on that map's legend and nothing on this one.
NARRATIVES = {
    "1": ("מים", "#0277bd"),
    "2": ("חקלאות ופרדסנות", "#558b2f"),
    "3": ("המושבה", "#ef6c00"),
    "4": ("הצבא הבריטי", "#546e7a"),
    "5": ("עמדות שמירה", "#37474f"),
    "6": ("קליטת עלייה", "#c62828"),
    "7": ("מוסדות ייחודיים", "#6d4c41"),
}

HE = "֐-׿"

# What each borrowed position gets told about itself in the detail pane. A
# visitor standing in front of the wrong house deserves to know which of these
# put them there.
WHENCE = {
    "shimur": 'המיקום לפי אתר {num} בנספח השימור של תוכנית המתאר ("{name}").',
    "pardespedia": 'המיקום לפי הערך "{name}" בשכבת המקומות מפרדספדיה.',
    "osm": 'המיקום הוא מרכז "{name}" לפי OpenStreetMap, ולכן מקורב.',
    "street": 'המיקום הוא אמצע רחוב {name}, ולכן מקורב.',
}


# ---------------------------------------------------------------- utilities

def get(url, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "derech-kitzur/1.0"})
            with urllib.request.urlopen(req, timeout=120) as handle:
                return json.loads(handle.read().decode("utf-8"))
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


def normalise(name):
    """Fold the spelling differences between the appendix and a blog title.

    The appendix writes ביה"כ where the blog writes בית הכנסת, and both are
    inconsistent about geresh and gershayim. Abbreviations are expanded before
    the punctuation is stripped, because stripping first turns ביה"ס into
    ביהס and loses the join.
    """
    name = unicodedata.normalize("NFKC", name)
    name = (name.replace("־", "-").replace("–", "-")
                .replace("“", '"').replace("”", '"').replace("״", '"')
                .replace("’", "'").replace("׳", "'"))
    for long, short in (("בית הספר", 'ביה"ס'), ("בית הכנסת", 'ביה"כ'),
                        ("בית כנסת", 'ביה"כ')):
        name = name.replace(long, short)
    name = re.sub(r"\(.*?\)", " ", name)
    name = re.sub(rf"[^\w{HE}]+", " ", name)
    return " ".join(name.split()).lower()


def strip_html(text):
    text = re.sub(r"<(script|style).*?</\1>", " ", text or "", flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#8211;", "–")
                .replace("&#8217;", "'").replace("&quot;", '"').replace("&#8220;", '"')
                .replace("&#8221;", '"').replace("&lt;", "<").replace("&gt;", ">"))
    return " ".join(text.split())


# ------------------------------------------------------------ parcel lookup

def ring_centroid(ring):
    """Area-weighted centroid of one closed ring, and its area.

    The shoelace centroid, not the average of the corners: a parcel with a long
    thin tail would otherwise be represented by a point dragged out along the
    tail. Degenerate rings (a sliver, or a duplicate-point ring) have zero area
    and fall back to the mean of their corners so they still produce something.
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


def geometry_centroid(geom):
    """Centroid of a (Multi)Polygon, weighting each part by its area."""
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    total = wx = wy = 0.0
    fallback = []
    for poly in polys:
        if not poly:
            continue
        cx, cy, area = ring_centroid([tuple(p[:2]) for p in poly[0]])
        fallback.append((cx, cy))
        total += area
        wx += cx * area
        wy += cy * area
    if total > 0:
        return wx / total, wy / total, total
    if fallback:
        return (sum(p[0] for p in fallback) / len(fallback),
                sum(p[1] for p in fallback) / len(fallback), 0.0)
    return None


class Parcels:
    """The national parcel layer, one request per block and then cached.

    Asking for a whole block at a time rather than for each parcel keeps this
    to about twenty requests for the whole appendix, and means a site that adds
    a parcel later costs nothing new. The cache holds centroids and areas only,
    never the polygons - it is a lookup table, not a copy of the cadastre.
    """

    def __init__(self, offline=False):
        self.offline = offline
        self.blocks = {}
        if os.path.exists(CACHE):
            with open(CACHE, encoding="utf-8") as handle:
                self.blocks = json.load(handle)
        self.fetched = 0

    def save(self):
        save_json(CACHE, self.blocks)

    def block(self, gush):
        key = str(gush)
        if key in self.blocks:
            return self.blocks[key]
        if self.offline:
            return {}
        query = urllib.parse.urlencode({
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeName": "opendata:PARCEL_ALL", "outputFormat": "application/json",
            "srsName": "EPSG:4326", "CQL_FILTER": f"GUSH_NUM={gush}",
        })
        doc = get(f"{WFS}?{query}")
        self.fetched += 1
        out = {}
        for feature in doc.get("features", []):
            props, geom = feature["properties"], feature.get("geometry")
            if not geom:
                continue
            found = geometry_centroid(geom)
            if not found:
                continue
            lng, lat, area = found
            # Two rows for one parcel happen where a parcel is split across
            # sheets; keep the bigger piece rather than whichever came last.
            parcel = str(props["PARCEL"])
            if parcel not in out or area > out[parcel][2]:
                out[parcel] = [round(lat, 6), round(lng, 6), area]
        self.blocks[key] = out
        return out

    # How far along the numbering a vanished parcel may borrow from. Adjacent
    # numbers in a block of this town are a median 30-60 m apart, so a window
    # of four is tens of metres of error - the size of a back garden, and far
    # tighter than the only alternative, which is the middle of the block.
    NEAR = 4

    def near(self, table, parcel):
        """The nearest surviving parcel numbers to one that no longer exists.

        The appendix was written against the 2017 cadastre and a dozen of its
        parcels have since been merged or renumbered away. Parcel numbers
        within a block run in spatial order closely enough that the neighbours
        of a vanished number are the best free answer to where it was.
        """
        try:
            target = int(parcel)
        except ValueError:
            return []
        out = []
        for offset in range(1, self.NEAR + 1):
            for candidate in (target - offset, target + offset):
                hit = table.get(str(candidate))
                if hit:
                    out.append(hit)
            if out:                                    # closest ring wins
                return out
        return []

    def locate(self, spec):
        """Position for one `גוש:חלקה,חלקה;גוש:חלקה` cell.

        The answer is the area-weighted centroid of every parcel that resolved,
        together with how many were asked for, how many were found on the nose
        and how many had to be borrowed from a neighbouring number. The caller
        needs those counts to decide how honest the pin is allowed to look.
        """
        want = exact = borrowed = 0
        total = wx = wy = 0.0

        def add(hit, weight_scale=1.0):
            nonlocal total, wx, wy
            lat, lng, area = hit
            weight = (area or 1e-9) * weight_scale
            total += weight
            wx += lng * weight
            wy += lat * weight

        for part in spec.split(";"):
            part = part.strip()
            if not part or ":" not in part:
                continue
            gush, parcels = part.split(":", 1)
            table = self.block(gush.strip())
            for parcel in parcels.split(","):
                parcel = parcel.strip()
                if not parcel:
                    continue
                want += 1
                hit = table.get(parcel)
                if hit:
                    exact += 1
                    add(hit)
                    continue
                neighbours = self.near(table, parcel)
                if neighbours:
                    borrowed += 1
                    for one in neighbours:
                        add(one, 1.0 / len(neighbours))
        if not (exact or borrowed) or not total:
            return None
        return round(wy / total, 6), round(wx / total, 6), want, exact, borrowed


# ------------------------------------------------------------- the appendix

def read_appendix():
    """The transcribed table, as a list of dicts, in file order."""
    rows = []
    with open(SRC, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cells = line.split("\t")
            if cells[0] == "num":                        # the header row
                continue
            cells += [""] * (5 - len(cells))
            num, name, grade, address, parcels = (c.strip() for c in cells[:5])
            if not num or not name:
                continue
            rows.append({"num": num, "name": name, "grade": grade,
                         "address": address, "parcels": parcels})
    return rows


def build_shimur(parcels):
    sites = []
    stats = {"located": 0, "borrowed": 0, "unplaced": 0}

    for row in read_appendix():
        narrative, colour = NARRATIVES.get(row["num"].split(".")[0],
                                           ("אחר", "#6a1b9a"))
        site = {
            "id": "shimur-" + row["num"],
            "name": row["name"],
            "group": narrative,
            "cats": [row["grade"]] if row["grade"] else [],
            "num": row["num"],
            "grade": row["grade"],
            "plan": PLAN,
            "url": APPENDIX_URL,
            "photos": [],
        }
        if row["address"]:
            site["address"] = row["address"]

        note = [f'אתר {row["num"]} בנספח השימור של תוכנית המתאר הכוללנית.']
        if row["grade"]:
            note.append(f'דרגת השימור שנקבעה: {row["grade"]}.')
        located = parcels.locate(row["parcels"]) if row["parcels"] else None

        if located:
            lat, lng, want, exact, borrowed = located
            site["geo"] = {"lat": lat, "lng": lng,
                           "source": "parcel" if not borrowed else "neighbour"}
            if want > 1:
                # An avenue or a compound: the row is a run of parcels and the
                # centroid is the middle of the run, which is nowhere in
                # particular. Say so rather than let the dot claim precision.
                site["geo"]["spread"] = True
                note.append("המיקום הוא מרכז השטח, שמשתרע על כמה חלקות.")
            else:
                note.append("המיקום נגזר מגבולות החלקה, ולכן הוא מרכז המגרש ולא בהכרח המבנה.")
            if borrowed:
                site["geo"]["spread"] = True
                note.append(f"{borrowed} מתוך {want} החלקות שבנספח כבר אינן קיימות "
                            "בקדסטר הנוכחי, והמיקום שלהן משוער לפי החלקות הסמוכות במספור.")
                stats["borrowed"] += 1
            stats["located"] += 1
        else:
            stats["unplaced"] += 1

        if row["parcels"]:
            blocks = [p.split(":")[0] for p in row["parcels"].split(";") if ":" in p]
            note.append("גוש " + ", ".join(blocks) + ".")
        site["note"] = " ".join(note)
        sites.append(site)

    sites.sort(key=lambda s: (s["group"], s["name"]))
    doc = {
        "version": 1,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": APPENDIX_URL,
        "name": "אתרים לשימור: תוכנית המתאר",
        "groups": [{"name": n, "color": c} for n, c in NARRATIVES.values()],
        "places": sites,
        "stats": {"places": len(sites), "located": stats["located"]},
    }
    save_json(OUT_SHIMUR, doc)
    return doc, stats


# ------------------------------------------------------------- מקום שמור

def makom_names():
    """The two list posts, as (name, half-of-the-town) pairs.

    The lists are prose with one site per line, and the first line of each is
    the sentence that introduces it rather than a site. A line is taken as a
    site if it reads like a name: short, and not ending in a full stop.
    """
    out = []
    for post_id, half in MAKOM_LISTS.items():
        doc = get(f"{WP}/posts/{post_id}")
        body = re.sub(r"<br\s*/?>", "\n", doc.get("content", ""))
        body = re.sub(r"</(p|li|div)>", "\n", body)
        for line in strip_html_lines(body):
            if line.endswith(":") or line.endswith(".") or len(line) > 45:
                continue
            out.append((line, half))
    return out


def strip_html_lines(body):
    lines = []
    for chunk in body.split("\n"):
        text = strip_html(chunk)
        if text:
            lines.append(text)
    return lines


def makom_posts():
    """Every article on makomshamur.com, newest first."""
    posts, page = [], 0
    while True:
        doc = get(f"{WP}/posts/?number=100&offset={page * 100}"
                  "&fields=ID,title,URL,date,categories,excerpt,featured_image")
        batch = doc.get("posts", [])
        posts.extend(batch)
        page += 1
        if len(batch) < 100 or page > 5:
            break
    return posts


def anchor_sources(shimur_doc, offline=False):
    """Everything in this project that already knows where something is.

    Tried in this order, because that is the order of how much each one knows
    about the specific building:

        shimur      the same site in the statutory appendix, resolved to its
                    parcel. Half a dozen of these are exact.
        pardespedia a place the wiki layer already carries, including any pin
                    an editor dropped by hand in the app.
        osm         a street or a neighbourhood of that name. Only ever used
                    for the entries that *are* a street - מקום שמור documents
                    whole roads alongside single houses - and it puts the pin
                    in the middle of the road, which is what it is.

    OpenStreetMap comes through build_places.py rather than a second copy of
    the Overpass query, and reads that script's cache; the town's named
    features have not changed since it was written.
    """
    sources = []

    shimur = {}
    for site in shimur_doc["places"]:
        if site.get("geo"):
            shimur.setdefault(normalise(site["name"]), site)
    sources.append(("shimur", shimur))

    places_path = os.path.join(HERE, "web", "data", "places.json")
    if os.path.exists(places_path):
        with open(places_path, encoding="utf-8") as handle:
            wiki = {}
            for place in json.load(handle)["places"]:
                if place.get("geo"):
                    wiki.setdefault(normalise(place["name"]), place)
        sources.append(("pardespedia", wiki))

    try:
        sys.path.insert(0, HERE)
        import build_places                              # noqa: PLC0415 - optional
        pois, roads = build_places.osm_index(offline=offline)
        # build_places folds names its own way; re-key so one normalise rules.
        entry = lambda k, v: {"geo": {"lat": v[0], "lng": v[1]}, "name": k}
        sources.append(("osm", {normalise(k): entry(k, v) for k, v in pois.items()}))
        sources.append(("street", {normalise(k): entry(k, v) for k, v in roads.items()}))
    except Exception as err:                             # noqa: BLE001
        print(f"  ~ בלי OpenStreetMap: {err}", file=sys.stderr)

    return sources


def build_makom(shimur_doc, offline=False):
    """The מקום שמור layer, positioned off whatever already knows the place."""
    sources = anchor_sources(shimur_doc, offline)

    posts = makom_posts()
    by_name = {}
    for post in posts:
        by_name.setdefault(normalise(strip_html(post["title"])), post)

    places, seen = [], set()
    stats = {"located": 0, "unplaced": 0, "documented": 0}

    for name, half in makom_names():
        key = normalise(name)
        if not key or key in seen:
            continue
        seen.add(key)

        place = {
            "id": "makom-" + re.sub(r"\s+", "-", key)[:60],
            "name": name,
            "group": half,
            "cats": [],
            "url": MAKOM_URL,
            "photos": [],
        }
        note = ["אתר מרשימת פרויקט מקום שמור. הרשימה היא הצעה, ואינה מעמד סטטוטורי."]

        # The project wrote a full article about some of its sites; those carry
        # the article's own words, its photo and a link straight to it.
        post = by_name.get(key) or partial_post(key, by_name)
        if post:
            place["url"] = post["URL"]
            place["cats"] = sorted(post.get("categories", {}).keys())
            excerpt = strip_html(post.get("excerpt", ""))
            if excerpt:
                note.append(excerpt[:400])
            if post.get("featured_image"):
                place["photos"] = [{"thumb": post["featured_image"],
                                    "full": post["featured_image"]}]
            stats["documented"] += 1

        for kind, index in sources:
            if kind == "street":
                anchor = street_anchor(key, index)
            else:
                anchor = index.get(key) or partial_anchor(key, index)
            if not anchor:
                continue
            geo = dict(anchor["geo"])
            geo["source"] = kind
            # Nothing here is the project's own survey: every position is
            # borrowed from something that happens to share the name, so all of
            # it reads as approximate in the app.
            geo["spread"] = True
            place["geo"] = geo
            note.append(WHENCE[kind].format(name=anchor["name"],
                                            num=anchor.get("num", "")))
            stats[kind] = stats.get(kind, 0) + 1
            stats["located"] += 1
            break
        else:
            stats["unplaced"] += 1

        place["note"] = " ".join(note)
        places.append(place)

    places.sort(key=lambda p: (p["group"], p["name"]))
    doc = {
        "version": 1,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": MAKOM_URL,
        "name": "מקום שמור",
        "groups": [{"name": "פרדס חנה", "color": "#ad1457"},
                   {"name": "כרכור", "color": "#00838f"}],
        "places": places,
        "stats": {"places": len(places), "located": stats["located"],
                  "documented": stats["documented"]},
    }
    save_json(OUT_MAKOM, doc)
    return doc, stats


# The words a name can open with that say what kind of thing it is rather than
# which one. מקום שמור writes "רחוב הדקלים חלקו הצפוני" where OpenStreetMap
# writes "הדקלים", and only a name that announces itself as a street this way
# is allowed to match a street (see street_anchor).
CLASSIFIERS = ("רחוב", "שדרות", "שדרת", "סמטת", "סמטה", "שכונת", "שכונה")


def tokens(key):
    """A name as a set of words, with the one-letter prefixes folded away.

    The appendix says "מגדל המים בתל שלום" and מקום שמור says "מגדל מים תל
    שלום": same building, four words each, not one of them spelled the same
    way twice. Dropping a leading ה, ב, ל, מ or ו from each word makes those
    two identical without making anything else collide - a prefix is one
    letter, so this cannot merge two names that differ in a real word.
    """
    out = set()
    for word in key.split():
        if len(word) > 2 and word[0] in "הבלמו":
            word = word[1:]
        out.add(word)
    return out


def matches(key, name):
    """Whether two names denote the same place, generously but not loosely.

    Every word of the shorter name has to appear in the longer one. That
    catches "בית אחוזה" against the appendix's "בית אחוזה בכרכור", and it
    refuses "בית פרח" against "בית פרס", which differ in a word rather than in
    a qualifier. Two-word minimum, because בית or גן on its own is half the
    town.
    """
    a, b = tokens(key), tokens(name)
    if len(a) < 2 or len(b) < 2:
        return False
    return a <= b or b <= a


def partial_post(key, by_name):
    """The project's own article about this site, where it wrote one.

    "מדרשית נעם" on the list is "מדרשית נעם – פרדס חנה" as an article.
    """
    for title, post in by_name.items():
        if matches(key, title):
            return post
    return None


def partial_anchor(key, anchors):
    """The same, against something that already knows where the place is."""
    for name, anchor in anchors.items():
        if matches(key, name):
            return anchor
    return None


def street_anchor(key, roads):
    """A road of this name, for the entries that are a road.

    מקום שמור documents whole streets next to single houses, and a street is
    the one thing a single-word match is safe for - but only when the name
    says it is a street. "רחוב המגדל" may take the road called המגדל; the
    house called "בית המגדל" may not.
    """
    words = key.split()
    if not words or words[0] not in CLASSIFIERS:
        return None
    rest = " ".join(words[1:])
    if not rest:
        return None
    if rest in roads:
        return roads[rest]
    for name, road in roads.items():
        if name and (rest.startswith(name + " ") or matches(rest, name)):
            return road
    return None


# ---------------------------------------------------------------------- cli

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-net", action="store_true",
                        help="use the cached parcel geometry and skip מקום שמור")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="throw away the cached parcel centroids and re-fetch")
    args = parser.parse_args()

    if args.refresh_cache and os.path.exists(CACHE):
        os.remove(CACHE)

    parcels = Parcels(offline=args.no_net)
    shimur, stats = build_shimur(parcels)
    parcels.save()

    print(f"\nנכתב {OUT_SHIMUR}", file=sys.stderr)
    print(f"  {shimur['stats']['places']} אתרים, {stats['located']} עם מיקום, "
          f"{stats['unplaced']} ללא", file=sys.stderr)
    if stats["borrowed"]:
        print(f"  {stats['borrowed']} אתרים שחלקה שלהם כבר לא בקדסטר, "
              "וממוקמים לפי החלקות השכנות", file=sys.stderr)
    if parcels.fetched:
        print(f"  {parcels.fetched} גושים נשלפו משכבת החלקות הארצית", file=sys.stderr)

    if args.no_net:
        print("\n--no-net: מקום שמור לא נבנה (דורש רשת).", file=sys.stderr)
        return

    makom, mstats = build_makom(shimur)
    print(f"\nנכתב {OUT_MAKOM}", file=sys.stderr)
    print(f"  {makom['stats']['places']} אתרים, {mstats['located']} עם מיקום, "
          f"{mstats['documented']} עם ערך באתר הפרויקט", file=sys.stderr)
    for kind in ("shimur", "pardespedia", "osm", "street"):
        if mstats.get(kind):
            print(f"    {kind}: {mstats[kind]}", file=sys.stderr)
    print(f"  {mstats['unplaced']} ממתינים לנעיצה ידנית.", file=sys.stderr)


if __name__ == "__main__":
    main()
