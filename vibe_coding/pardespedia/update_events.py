#!/usr/bin/env python3
"""Update the events board on the pardespedia main page ("עמוד ראשי").

Pulls events from several structured sources in the moshava, keeps the ones in
the next N days, merges + de-duplicates them, and rebuilds the auto-managed
events board inside the "אירועי תרבות ובילוי קרובים" section of the main page
(between the AUTO markers). Content outside the AUTO markers is preserved.

The board is the full table — image and video columns included — and is
rendered collapsible-but-*open* (``mw-collapsible`` without ``mw-collapsed``),
so a visitor sees the whole board on arrival and can fold it with "הסתר".
(The old standalone page "אירועי תרבות ובילוי בפרדס חנה-כרכור" was retired in
favour of showing the board directly on the main page.)

Each row gets, where possible:
  * an image (first column) — the event's flyer, downloaded + uploaded to the
    wiki at low resolution under a fair-use rationale (external hotlinking is
    disabled on pardespedia). Filenames are stable per event, so re-runs skip
    images that already exist.
  * a video (last column) — looked up in event_videos.json, a hand-maintained
    map of event-name → YouTube id (the feeds carry no video data). The script
    reads it every run, so curated videos persist across automatic updates.

Sources (each is best-effort — a failing source is logged and skipped):
  * eventschedule.com hub  — clean JSON feed, aggregates many venues.
  * מרכז אמנויות הבמה (מתנ"ס) — schema.org JSON-LD embedded in its Tickchak page.

Usage:
    python3 update_events.py [--days 14] [--dry-run] [--no-images]

Deterministic (no LLM) — safe to run unattended from cron.
"""
import argparse
import calendar
import datetime as dt
import hashlib
import io
import json
import os
import re
import sys
import urllib.parse

import requests

from wiki_client import WikiClient, API_URL

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_MAP_FILE = os.path.join(HERE, "event_videos.json")
MANUAL_FILE = os.path.join(HERE, "manual_events.json")
MANUAL_HORIZON_DAYS = 70  # how far ahead to expand recurring manual events

MAIN_TITLE = "עמוד ראשי"
MAIN_HEADING = "אירועי תרבות ובילוי קרובים"
TODAY_HEADING = "היום במושבה"
MAIN_ANCHOR = "== מידע עירוני =="  # fallback: insert the section just before this

ES_FEED_URL = "https://pardeshanna.eventschedule.com/api/calendar-events"
ES_HUB_URL = "https://pardeshanna.eventschedule.com"
ES_LABEL = f"[{ES_HUB_URL} לוח האירועים של פרדס חנה-כרכור (eventschedule)]"
MATNAS_URL = "https://live.tickchak.co.il/matans-pardes-hana"
MATNAS_LABEL = "[https://live.tickchak.co.il/matans-pardes-hana מרכז אמנויות הבמה — מתנ\"ס פרדס חנה]"
MATNAS_VENUE = "מרכז אמנויות הבמה (מתנ\"ס)"
HAULAM_URL = "https://haulam-phk.smarticket.co.il/"
HAULAM_LABEL = "[https://haulam-phk.smarticket.co.il האולם — בית תרבות מקומי פרכו\"ר]"
HAULAM_VENUE = "האולם — בית תרבות מקומי פרכו\"ר"

AUTO_START = "<!-- AUTO:START — אזור מתעדכן אוטומטית, אל תערוך ידנית -->"
AUTO_END = "<!-- AUTO:END -->"
TODAY_AUTO_START = "<!-- AUTO:TODAY:START — אזור מתעדכן אוטומטית, אל תערוך ידנית -->"
TODAY_AUTO_END = "<!-- AUTO:TODAY:END -->"
TODAY_MAX = 3  # at most this many events surfaced in the "today" highlight box

HE_WEEKDAYS = {0: "שני", 1: "שלישי", 2: "רביעי", 3: "חמישי", 4: "שישי", 5: "שבת", 6: "ראשון"}
HE_MONTHS = {
    1: "ינואר", 2: "פברואר", 3: "מרץ", 4: "אפריל", 5: "מאי", 6: "יוני",
    7: "יולי", 8: "אוגוסט", 9: "ספטמבר", 10: "אוקטובר", 11: "נובמבר", 12: "דצמבר",
}
# The closed vocabulary of the board's "סוג" column. That column is sortable,
# which is the whole reason it must be closed: a reader sorting it to find the
# community events should land on one block, not on four — "קהילה",
# "פעילות קהילתית", "שוק קהילתי" and "ספורט וקהילה" all sorted apart while
# meaning nearly the same thing, because the feeds and manual_events.json each
# invented their own wording. Everything now passes through canonical_category()
# in _row(), so there is exactly one place where a label can enter the board.
#
# Keep this list short. A type with one member is not a type — the event's own
# name already says "שוק קהילתי"; the column only has to answer "what kind".
CATEGORIES = [
    "מופעים והופעות", "תיאטרון", "סטנדאפ וקומדיה", "מסיבות ופסטיבלים",
    "אמנות ותרבות", "הרצאות וסדנאות", "ספורט", "קהילה", "משפחה וילדים",
    "רוחניות והתפתחות אישית",
]

# alias -> canonical. English keys are matched case-insensitively.
CATEGORY_HE = {
    "Concerts": "מופעים והופעות", "MusicEvent": "מופעים והופעות",
    "Event": "מופעים והופעות", "מופע": "מופעים והופעות",
    "מוזיקה": "מופעים והופעות",
    "TheaterEvent": "תיאטרון",
    "ComedyEvent": "סטנדאפ וקומדיה", "סטנד-אפ": "סטנדאפ וקומדיה",
    "סטנדאפ": "סטנדאפ וקומדיה",
    "Parties & Festivals": "מסיבות ופסטיבלים", "מסיבה": "מסיבות ופסטיבלים",
    "פסטיבל": "מסיבות ופסטיבלים", "מסיבות ומועדונים": "מסיבות ופסטיבלים",
    "Art & Culture": "אמנות ותרבות", "תערוכה": "אמנות ותרבות",
    "אמנות": "אמנות ותרבות", "קולנוע": "אמנות ותרבות",
    "Workshops": "הרצאות וסדנאות", "סדנאות": "הרצאות וסדנאות",
    "סדנה": "הרצאות וסדנאות", "הרצאה": "הרצאות וסדנאות",
    "Sports": "ספורט", "ספורט וקהילה": "ספורט",
    "Community": "קהילה", "פעילות קהילתית": "קהילה",
    "שוק קהילתי": "קהילה", "שוק": "קהילה",
    "מפגש קהילתי": "קהילה",
    "מופע ילדים": "משפחה וילדים", "פעילות משפחתית": "משפחה וילדים",
    "ילדים": "משפחה וילדים", "משפחה": "משפחה וילדים",
    "Personal Growth": "רוחניות והתפתחות אישית",
    "התפתחות אישית": "רוחניות והתפתחות אישית",
    "צמיחה אישית": "רוחניות והתפתחות אישית",
    "Spirituality": "רוחניות והתפתחות אישית",
    "רוחניות": "רוחניות והתפתחות אישית",
}
_CATEGORY_LOOKUP = {k.casefold(): v for k, v in CATEGORY_HE.items()}
_CATEGORY_LOOKUP.update({c.casefold(): c for c in CATEGORIES})  # canonical is its own alias


def canonical_category(name) -> str:
    """Fold a source's category wording into the board's vocabulary.

    Warns once per unknown label. The venues register themselves on the
    eventschedule hub and pick their own categories, so new wordings keep
    arriving — this log line is what makes the next one visible instead of
    letting it quietly open an eleventh column value.
    """
    name = (name or "").strip()
    if not name:
        return ""
    canon = _CATEGORY_LOOKUP.get(name.casefold())
    if canon:
        return canon
    if name not in _SEEN_UNKNOWN_CATEGORIES:
        _SEEN_UNKNOWN_CATEGORIES.add(name)
        print(f"  category outside the vocabulary, passed through as-is: "
              f"{name!r}", file=sys.stderr)
    return name


category_he = canonical_category  # older call sites

_SEEN_UNKNOWN_CATEGORIES = set()


def he_date(d: dt.date) -> str:
    return f"{d.day} ב{HE_MONTHS[d.month]} {d.year}"


def wiki_escape(text: str) -> str:
    return (text or "").replace("|", "‖").strip()


# eventschedule venue strings are pipe-separated and end with the locality:
#   "מרכז ללימוד תורה | פרדס חנה-כרכור"
#   "סוזן דלל | אולם אריסון | תל אביב-יפו"
# The board is a moshava board, but the hub is open to self-registration and
# performers list out-of-town dates on it too — a Suzanne Dellal show in Tel
# Aviv reached the main page this way. So the locality decides inclusion.
# HOME localities render as before (bare venue name, maps link into the
# moshava); NEARBY ones are kept but must say where they are, or a reader
# assumes the moshava. Anything else is dropped and logged, never silently.
HOME_LOCALITIES = {
    "פרדס חנה-כרכור", "פרדס חנה כרכור", "פרדס חנה", "כרכור",
    "pardes hanna-karkur", "pardes hanna karkur", "pardes hanna", "karkur",
}
NEARBY_LOCALITIES = {
    "חדרה": "חדרה", "מעיין צבי": "מעיין צבי", "maayan tzvi": "מעיין צבי",
    "בנימינה": "בנימינה", "בנימינה-גבעת עדה": "בנימינה", "גבעת עדה": "גבעת עדה",
    "כפר פינס": "כפר פינס", "עין שמר": "עין שמר", "גן שמואל": "גן שמואל",
    "משמרות": "משמרות", "תלמי אלעזר": "תלמי אלעזר", "מנשה": "מנשה",
    "אור עקיבא": "אור עקיבא", "זכרון יעקב": "זכרון יעקב", "קיסריה": "קיסריה",
}
_SEEN_UNKNOWN_LOCALITIES = set()


def split_venue(name: str):
    """Split an eventschedule venue string into (display name, locality).

    Locality is "" when the string carries no pipe and is not itself a bare
    locality — the caller then has nothing to judge by.
    """
    parts = [p.strip() for p in (name or "").split("|") if p.strip()]
    if not parts:
        return "", ""
    if len(parts) == 1:
        only = parts[0]
        return ("", only) if only.casefold() in HOME_LOCALITIES else (only, "")
    return " ".join(parts[:-1]).strip(), parts[-1]


def locality_kind(locality: str) -> str:
    """"home" / "nearby" / "away" / "unknown" for a locality string."""
    key = (locality or "").strip().casefold()
    if not key:
        return "unknown"
    if key in HOME_LOCALITIES:
        return "home"
    if key in NEARBY_LOCALITIES:
        return "nearby"
    return "away"


def clean_venue(name: str) -> str:
    """Display name for a venue: bare in the moshava, "(town)" outside it."""
    venue, locality = split_venue(name)
    kind = locality_kind(locality)
    if kind == "nearby":
        town = NEARBY_LOCALITIES[locality.strip().casefold()]
        return f"{venue} ({town})" if venue else town
    return venue


def in_area(name: str) -> bool:
    """Whether an event at this venue belongs on a Pardes Hanna-Karkur board.

    An unparseable or missing venue is kept: the hub is the moshava's own, so
    a nameless venue is far more likely to be local than not — but it is
    logged, so a wrong guess is visible in the cron log rather than invisible.
    """
    _, locality = split_venue(name)
    kind = locality_kind(locality)
    if kind in ("home", "nearby"):
        return True
    if kind == "away":
        print(f"  מחוץ לאזור, האירוע לא נכלל: {name!r}", file=sys.stderr)
        return False
    if name and name not in _SEEN_UNKNOWN_LOCALITIES:
        _SEEN_UNKNOWN_LOCALITIES.add(name)
        print(f"  יישוב לא מזוהה בשם המקום, האירוע נכלל בכל זאת: {name!r}",
              file=sys.stderr)
    return True


def maps_link(venue: str) -> str:
    """Google Maps search link for a venue, forced to Hebrew UI (hl=iw).

    `venue` is a display name from clean_venue(), so an out-of-town one already
    carries its town in parentheses — searching that verbatim beats appending
    "פרדס חנה-כרכור" to a venue that is not there. Only a known town counts:
    plenty of venue names end in an ordinary parenthetical (מרכז אמנויות הבמה
    (מתנ"ס)) that must not be mistaken for a locality.
    """
    venue = (venue or "").strip()
    towns = set(NEARBY_LOCALITIES.values())
    m = re.search(r"^(.*?)\s*\(([^()]+)\)$", venue)
    if m and m.group(2).strip() in towns:
        query = f"{m.group(1).strip()}, {m.group(2).strip()}"
    elif any(venue.casefold().endswith(loc) for loc in HOME_LOCALITIES):
        query = venue  # the venue string already names the moshava
    else:
        query = f"{venue}, פרדס חנה-כרכור"
    return f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(query)}&hl=iw"


def price_label(price, is_free: bool = False) -> str:
    """Entry-fee label — "" when the source simply did not say.

    A feed that sends is_free=false with ticket_price=null is withholding the
    price, not declaring one: eventschedule sends exactly that for every Torah
    Learning Center class. Returning "בתשלום" there made the board assert that
    free community classes cost money, so silence now stays silence and
    build_table() renders it as "—".
    """
    if is_free:
        return "חינם"
    if price in (None, ""):
        return ""
    if price not in ("0.00", "0", 0):
        try:
            val = float(price)
            if val > 0:
                return f"₪{val:.0f}" if val == int(val) else f"₪{val:.2f}"
        except (TypeError, ValueError):
            pass
    return "בתשלום"


def norm(s: str) -> str:
    """Normalize a name for fuzzy matching / de-dup (drop spaces & punctuation)."""
    return re.sub(r"[^0-9a-zא-ת]", "", (s or "").lower())


def _sort_key(d, time) -> str:
    """Canonical "<ISO date>T<HH:MM>" ordering key.

    Sources hand us different shapes for the same instant — eventschedule emits
    "2026-07-28 18:00:00", the others "2026-07-28T17:30" — and since " " < "T",
    string-sorting the raw values put every space-form event ahead of every
    T-form one on the same day, whatever the hour. Rebuild the key from the
    date and the leading HH:MM so ordering is time-based. Events with no time
    ("all day") sort to the end of their day.
    """
    m = re.match(r"\s*(\d{1,2}):(\d{2})", time or "")
    hhmm = f"{int(m.group(1)):02d}:{m.group(2)}" if m else "99:99"
    return f"{d.isoformat()}T{hhmm}"


def _row(d, time, name, url, category, venue, entry, key, image_url, date_label=None) -> dict:
    return {"date": d, "time": time, "name": wiki_escape(name), "url": url or "",
            # every source funnels its category wording through one gate here,
            # so the sortable "סוג" column keeps a single vocabulary
            "category": canonical_category(category),
            "venue": wiki_escape(venue), "entry": entry,
            "sort": _sort_key(d, time), "key": key, "image_url": image_url or "",
            "image_file": None, "video_id": None, "date_label": date_label,
            # set by merge_venue_day() when several events at one venue on one
            # day are folded into this row: "parts" keeps the originals (the
            # "today" cards still show one card per event), "name_markup" is the
            # pre-rendered multi-line cell for the board.
            # set by cap_per_venue() on the last surviving row of an over-full
            # venue: how many of that venue's later events the cap held back.
            "parts": None, "name_markup": None, "more_count": 0}


# --- sources ---------------------------------------------------------------

def fetch_eventschedule() -> list:
    r = requests.get(ES_FEED_URL, headers={"Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    rows = []
    for e in r.json().get("events", []):
        ds = e.get("local_date")
        if not ds:
            continue
        if not in_area(e.get("venue_name")):
            continue
        try:
            d = dt.date.fromisoformat(ds)
        except ValueError:
            continue
        lsa = e.get("local_starts_at") or ""
        t = ""
        if " " in lsa:
            hm = lsa.split(" ")[1][:5]
            if hm and hm != "00:00":
                t = hm
        rows.append(_row(
            d, t, e.get("name"), e.get("guest_url"),
            category_he(e.get("category_name")),
            clean_venue(e.get("venue_name")),
            price_label(e.get("ticket_price"), e.get("is_free")),
            f"es-{e.get('id')}",
            e.get("image_url") or e.get("flyer_url"),
        ))
    return rows


def _ldjson_nodes(html: str) -> list:
    nodes = []
    for block in re.findall(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for it in list(items):
            if isinstance(it, dict) and "@graph" in it:
                items += it["@graph"]
        nodes += [it for it in items if isinstance(it, dict)]
    return nodes


def fetch_matnas() -> list:
    r = requests.get(MATNAS_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0 Pardespedia-Bot"})
    r.raise_for_status()
    rows = []
    for it in _ldjson_nodes(r.text):
        t = it.get("@type")
        is_event = t in ("Event", "MusicEvent", "ComedyEvent", "TheaterEvent") or \
            (isinstance(t, list) and any(x.endswith("Event") for x in t))
        if not is_event or not it.get("startDate"):
            continue
        try:
            sd = dt.datetime.fromisoformat(it["startDate"])
        except ValueError:
            continue
        offers = it.get("offers") or {}
        if isinstance(offers, list):
            offers = offers[0] if offers else {}
        cat_key = t if isinstance(t, str) else next((x for x in t if x.endswith("Event")), "Event")
        img = it.get("image")
        if isinstance(img, list):
            img = img[0] if img else ""
        url = offers.get("url") or it.get("@id", "")
        rows.append(_row(
            sd.date(), sd.strftime("%H:%M") if (sd.hour or sd.minute) else "",
            it.get("name"), url, CATEGORY_HE.get(cat_key, "מופעים והופעות"),
            MATNAS_VENUE, price_label(offers.get("price")),
            f"mt-{hashlib.md5((url or it.get('name','')).encode()).hexdigest()[:8]}", img,
        ))
    return rows


def fetch_haulam() -> list:
    """האולם sells tickets directly (not through the מתנ"ס/tickchak feed) — its
    own site lists upcoming shows on the homepage, each with schema.org
    Event JSON-LD on its own /event/<id> page."""
    headers = {"User-Agent": "Mozilla/5.0 Pardespedia-Bot"}
    r = requests.get(HAULAM_URL, timeout=30, headers=headers)
    r.raise_for_status()
    ids = sorted(set(re.findall(r"\?id=(\d+)", r.text)), key=int)
    rows = []
    for eid in ids:
        try:
            er = requests.get(f"{HAULAM_URL}event/{eid}", timeout=30, headers=headers)
            er.raise_for_status()
        except requests.RequestException:
            continue
        for it in _ldjson_nodes(er.text):
            if it.get("@type") != "Event" or not it.get("startDate"):
                continue
            try:
                sd = dt.datetime.fromisoformat(it["startDate"])
            except ValueError:
                continue
            offers = it.get("offers") or {}
            if isinstance(offers, list):
                offers = offers[0] if offers else {}
            img = it.get("image")
            if isinstance(img, list):
                img = img[0] if img else ""
            url = offers.get("url") or f"{HAULAM_URL}event/{eid}"
            # האולם opens a separate event page per showtime and bakes the hour
            # into the title ("אסתר 09:30"). Drop it — the שעה column carries it,
            # and the bare name lets merge_showtimes() fold the showings together.
            name = re.sub(r"[\s,-]*\d{1,2}:\d{2}\s*$", "", it.get("name") or "")
            rows.append(_row(
                sd.date(), sd.strftime("%H:%M") if (sd.hour or sd.minute) else "",
                name, url, "מופעים והופעות",
                HAULAM_VENUE, price_label(offers.get("price")),
                f"hl-{eid}", img,
            ))
    return rows


# --- manual events (community/Facebook events the feeds miss) ---------------

_WD = {"sunday": 6, "monday": 0, "tuesday": 1, "wednesday": 2,
       "thursday": 3, "friday": 4, "saturday": 5}
_ORD = {"first": 0, "second": 1, "third": 2, "fourth": 3}


def _nth_weekday(year, month, weekday, ordinal):
    days = [d for d in range(1, calendar.monthrange(year, month)[1] + 1)
            if dt.date(year, month, d).weekday() == weekday]
    if ordinal == "last":
        return days[-1] if days else None
    idx = _ORD.get(ordinal)
    return days[idx] if (idx is not None and idx < len(days)) else None


def _expand_recurrence(rec: str, start: dt.date, end: dt.date) -> list:
    """rec like 'first_friday' / 'last_friday' / 'every_wednesday' -> matching dates within window."""
    try:
        ordinal, wd_name = rec.split("_", 1)
        weekday = _WD[wd_name]
    except (ValueError, KeyError):
        return []
    if ordinal == "every":
        out, d = [], start
        while d <= end:
            if d.weekday() == weekday:
                out.append(d)
            d += dt.timedelta(days=1)
        return out
    out, y, m = [], start.year, start.month
    for _ in range(4):  # cover up to ~4 months ahead
        day = _nth_weekday(y, m, weekday, ordinal)
        if day:
            d = dt.date(y, m, day)
            if start <= d <= end:
                out.append(d)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _format_date_range(dates: list) -> str:
    """['2026-07-13','2026-07-14'] -> '13-14.7' (or '13.7, 15.7' if not consecutive)."""
    consecutive = all((dates[i + 1] - dates[i]).days == 1 for i in range(len(dates) - 1))
    if consecutive and dates[0].month == dates[-1].month:
        return f"{dates[0].day}-{dates[-1].day}.{dates[0].month}"
    return ", ".join(f"{d.day}.{d.month}" for d in dates)


def fetch_manual() -> list:
    if not os.path.exists(MANUAL_FILE):
        return []
    with open(MANUAL_FILE, encoding="utf-8") as f:
        data = json.load(f)
    today = dt.date.today()
    horizon = today + dt.timedelta(days=MANUAL_HORIZON_DAYS)
    rows = []
    for ev in data.get("events", []):
        name = ev.get("name")
        if not name:
            continue
        t = ev.get("time", "")
        # "dates": [...] -> multiple nights of the same event, merged into one row
        # (e.g. a play running two consecutive evenings) rather than one row per date.
        if ev.get("dates"):
            try:
                dl = sorted(dt.date.fromisoformat(x) for x in ev["dates"])
            except ValueError:
                dl = []
            if dl:
                d0 = dl[0]
                key = "man-" + hashlib.md5(f"{name}{d0}".encode()).hexdigest()[:8]
                r = _row(d0, t, name, ev.get("url"), ev.get("category", ""),
                         ev.get("venue", ""), ev.get("entry", "חינם"),
                         key, ev.get("image_url"),
                         date_label=_format_date_range(dl))
                r["image_file"] = ev.get("image_file")
                r["video_id"] = ev.get("video")
                r["pin"] = bool(ev.get("pin"))
                rows.append(r)
            continue
        # resolve dates: explicit one-off or a monthly/weekly recurrence
        dates = []
        if ev.get("date"):
            try:
                dates = [dt.date.fromisoformat(ev["date"])]
            except ValueError:
                dates = []
        elif ev.get("recurrence"):
            dates = _expand_recurrence(ev["recurrence"], today, horizon)
        for d in dates:
            key = "man-" + hashlib.md5(f"{name}{d}".encode()).hexdigest()[:8]
            r = _row(d, t, name, ev.get("url"), ev.get("category", ""),
                     ev.get("venue", ""), ev.get("entry", "חינם"),
                     key, ev.get("image_url"))
            r["image_file"] = ev.get("image_file")  # reference an existing wiki file
            r["video_id"] = ev.get("video")          # inline curated video
            r["pin"] = bool(ev.get("pin"))           # show even if beyond the window
            rows.append(r)
    return rows

SOURCES = [("eventschedule", fetch_eventschedule), ("מתנ\"ס", fetch_matnas),
           ("האולם", fetch_haulam), ("ידני", fetch_manual)]


def _clock_times(s: str) -> set:
    """The HH:MM instants in a time string, however it is punctuated."""
    return set(re.findall(r"\d{1,2}:\d{2}", s or ""))


def _same_event(a, b) -> bool:
    """Same date, no conflicting times, and one name contains the other
    (e.g. feed's "מעגל מתנות קהילתי" vs. the manual "מעגל מתנות").

    Times are compared as the SET of clock readings, not as raw strings: the
    feed writes two showings as "09:30, 14:00" while a hand-written entry may
    say "09:30 ו-14:00". Comparing the strings made those look like different
    events, and the pre-premiere screening of "עצמאות" was listed twice."""
    if a["date"] != b["date"]:
        return False
    ta, tb = _clock_times(a["time"]), _clock_times(b["time"])
    if ta and tb and not (ta & tb):        # no shared showing -> different events
        return False
    na, nb = norm(a["name"]), norm(b["name"])
    return bool(na and nb) and (na in nb or nb in na)


def merge_showtimes(rows: list) -> list:
    """Fold several showings of the same show, same day, same venue into one
    row ("09:30, 14:00"). _same_event() deliberately keeps events with
    conflicting times apart — two different shows at one venue are two rows —
    but an identical name at the same venue is one show playing twice, and two
    near-identical rows with the same poster read as a bug to a visitor."""
    merged = []
    index = {}
    for r in rows:
        gid = (r["date"], norm(r["name"]), norm(r["venue"]))
        if not r["time"] or gid not in index:
            index.setdefault(gid, len(merged))
            merged.append(r)
            continue
        kept = merged[index[gid]]
        times = [t for t in kept["time"].split(", ") if t]
        if r["time"] not in times:
            kept["time"] = ", ".join(sorted(times + [r["time"]]))
    return merged


VENUE_MAX_EVENTS = 3


def cap_per_venue(rows: list, limit: int = VENUE_MAX_EVENTS) -> list:
    """Keep at most `limit` events per venue, dropping the LATEST ones.

    A venue can publish a standing weekly programme with no end date — the
    Torah Learning Center registered on the eventschedule hub on 2026-07-31
    with twelve entries, seven of them weekly-forever, and instantly owned a
    third of a board meant to show the whole moshava.

    Cutting from the far end rather than the near one is what keeps this fair
    over time: the events nearest today always survive, and as each one passes
    out of the window the next in line takes its place. Nothing is suppressed
    permanently — it simply waits its turn.

    Pinned manual events are exempt; they are curated by hand and are pinned
    precisely because they must show. Rows with no venue are never pooled
    together, since "unknown" is not a place.

    A cut leaves a trace: the last surviving row of an over-full venue carries
    "more_count", which build_table() renders as "ועוד N אירועים במקום זה".
    A reader who wants that venue's full programme should be told it exists
    rather than shown a board that quietly pretends it doesn't.
    """
    seen, out, last = {}, [], {}
    for r in sorted(rows, key=lambda r: r["sort"]):
        venue = norm(r["venue"])
        if not venue or r.get("pin"):
            out.append(r)
            continue
        seen[venue] = seen.get(venue, 0) + 1
        if seen[venue] <= limit:
            out.append(r)
            last[venue] = r
    over = {v: n - limit for v, n in seen.items() if n > limit}
    for venue, extra in over.items():
        if venue in last:          # always true: a capped venue kept `limit` rows
            last[venue]["more_count"] = extra
    dropped = len(rows) - len(out)
    if dropped:
        print(f"  venue cap: dropped {dropped} later event(s) from {len(over)} "
              f"over-full venue(s)", file=sys.stderr)
    return sorted(out, key=lambda r: r["sort"])


def _covered_keys(r) -> set:
    """Every source event a (possibly merged) row stands for."""
    return {p["key"] for p in (r.get("parts") or [r])}


def merge_venue_day(rows: list) -> list:
    """Fold everything one venue holds on one day into a single row.

    merge_showtimes() only folds *the same* event playing twice. That leaves a
    venue running a daily programme with one near-identical row per session —
    the Torah Learning Center posts nine classes a fortnight under one logo, so
    the board showed three consecutive rows carrying the same blue TLC image
    and reading, to a visitor, like a rendering bug.

    These really are different events, so nothing is dropped: the row lists
    every session behind its own hour ("'''10:00''' שם האירוע"), and the hour
    column carries the full list. The originals stay on "parts" so the
    "קורה היום" box can still render one card per session.
    """
    groups, order = {}, []
    for i, r in enumerate(rows):
        # an unknown venue is not a venue — never group those together
        gid = (r["date"], norm(r["venue"])) if norm(r["venue"]) else ("solo", i)
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append(r)

    out = []
    for gid in order:
        g = sorted(groups[gid], key=lambda r: r["sort"])
        if len(g) == 1:
            out.append(g[0])
            continue
        head = dict(g[0])
        head["parts"] = g
        labels = []
        for r in g:
            name = f"[{r['url']} {r['name']}]" if r["url"] else r["name"]
            labels.append(f"'''{r['time']}''' {name}" if r["time"] else name)
        head["name_markup"] = "<br />".join(labels)
        head["time"] = ", ".join(dict.fromkeys(r["time"] for r in g if r["time"]))
        for field, sep in (("category", " · "), ("entry", " · ")):
            vals = [v for v in dict.fromkeys(r[field] for r in g) if v]
            head[field] = sep.join(vals)
        # one poster for the row: the first session that has one
        for field in ("image_url", "image_file", "video_id"):
            head[field] = next((r[field] for r in g if r[field]), head[field])
        # the venue-cap note rides on the group's last session, not its first
        head["more_count"] = max(r.get("more_count", 0) for r in g)
        out.append(head)
    return out


def collect(start: dt.date, end: dt.date) -> list:
    all_rows = []
    for name, fn in SOURCES:
        try:
            rows = fn()
            all_rows += rows
            print(f"  source {name}: {len(rows)} events", file=sys.stderr)
        except Exception as exc:
            print(f"  source {name}: FAILED ({exc})", file=sys.stderr)
    # events inside the window, plus any pinned manual event still in the future
    # (pinned events show regardless of the window, up to MANUAL_HORIZON_DAYS).
    windowed = [r for r in all_rows
                if (start <= r["date"] <= end) or (r.get("pin") and r["date"] >= start)]
    deduped = []
    for r in sorted(windowed, key=lambda r: r["sort"]):
        dup = next((i for i, kept in enumerate(deduped) if _same_event(r, kept)), None)
        if dup is None:
            deduped.append(r)
        elif r["key"].startswith("man-") and not deduped[dup]["key"].startswith("man-"):
            deduped[dup] = r  # the manual entry carries curated details + wiki link
    # cap before merge_venue_day so the limit counts events, not table rows
    return merge_venue_day(cap_per_venue(merge_showtimes(deduped)))


# --- images (download + upload, fair use) ----------------------------------

def _file_exists(client, filename: str) -> bool:
    r = client.session.get(API_URL, params={
        "action": "query", "titles": f"קובץ:{filename}",
        "prop": "imageinfo", "format": "json"})
    page = next(iter(r.json()["query"]["pages"].values()))
    return "missing" not in page


def ensure_images(client, rows: list) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("  PIL not available — skipping images", file=sys.stderr)
        return
    for r in rows:
        if not r["image_url"]:
            continue
        fn = f"אירוע-{r['key']}.jpg"
        try:
            if _file_exists(client, fn):
                r["image_file"] = fn
                continue
            resp = requests.get(r["image_url"], timeout=30, headers={"User-Agent": "Mozilla/5.0 Pardespedia-Bot"})
            resp.raise_for_status()
            im = Image.open(io.BytesIO(resp.content)).convert("RGB")
            w, h = im.size
            nw = min(360, w)
            im = im.resize((nw, int(h * nw / w)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=82)
            desc = (
                f"תמונת קידום של האירוע \"{r['name']}\" ({r['venue']}).\n\n"
                f"מקור: [{r['url']} עמוד האירוע].\n\n"
                "התמונה מובאת בשימוש הוגן לצורך המחשה בלוח אירועים אנציקלופדי, ברזולוציה נמוכה.\n\n"
                "[[קטגוריה: תמונות אירועים]]\n"
            )
            token = client._csrf_token()
            up = client.session.post(API_URL, data={
                "action": "upload", "filename": fn,
                "comment": "תמונת אירוע (שימוש הוגן) ללוח אירועי התרבות",
                "text": desc, "token": token, "ignorewarnings": "1", "format": "json",
            }, files={"file": (fn, buf.getvalue(), "image/jpeg")})
            if up.json().get("upload", {}).get("result") == "Success":
                r["image_file"] = fn
                print(f"  image uploaded: {fn}", file=sys.stderr)
            else:
                print(f"  image FAILED: {fn} -> {up.json().get('error')}", file=sys.stderr)
        except Exception as exc:
            print(f"  image error for {r['key']}: {exc}", file=sys.stderr)


# --- videos (curated map) --------------------------------------------------

def load_video_map() -> dict:
    if not os.path.exists(VIDEO_MAP_FILE):
        return {}
    try:
        with open(VIDEO_MAP_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return {norm(k): v for k, v in (data.get("videos") or {}).items()}
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  video map unreadable: {exc}", file=sys.stderr)
        return {}


def attach_videos(rows: list) -> None:
    vmap = load_video_map()
    if not vmap:
        return
    for r in rows:
        if r["video_id"]:  # keep an inline (manual) video
            continue
        n = norm(r["name"])
        for key, vid in vmap.items():
            if key and (key in n or n in key):
                r["video_id"] = vid
                break


# --- rendering -------------------------------------------------------------

def build_table(rows: list, collapsible: bool = False) -> str:
    if not rows:
        return "''אין כרגע אירועים מתוזמנים בטווח הקרוב. בדקו שוב בקרוב.''"
    # mw-collapsible (without mw-collapsed) → the table starts *open*; the
    # "הסתר" toggle in the caption lets a reader fold it away.
    cls = "wikitable sortable mw-collapsible" if collapsible else "wikitable sortable"
    lines = [f'{{| class="{cls}" style="width:100%"']
    if collapsible:
        lines.append('|+ אירועי השבועיים הקרובים — לחצו על "הסתר" כדי לקפל את הלוח (ומיינו לפי "סוג" כדי למצוא סוג אירוע)')
    # עמודת התמונה אינה ניתנת למיון (חסר-משמעות). התאריך ממוין כרונולוגית דרך
    # data-sort-value בפורמט ISO בכל תא (אחרת המחרוזת העברית תמוין אלפביתית).
    lines.append('! class="unsortable" | תמונה !! תאריך !! class="unsortable" | שעה !! אירוע !! סוג !! מקום !! כניסה')
    for r in rows:
        when = r.get("date_label") or f"יום {HE_WEEKDAYS[r['date'].weekday()]}, {r['date'].day}.{r['date'].month}"
        iso = r["date"].isoformat()
        name = r.get("name_markup") or (f"[{r['url']} {r['name']}]" if r["url"] else r["name"])
        # בלי |link=: לחיצה על הכרזה פותחת את דף הקובץ, שם היא מוצגת בגדול
        # ולצידה המקור והרישיון. קודם הלחיצה קפצה לאתר האירוע, וזה הפריע —
        # מי שלוחץ על תמונה מצפה לראות אותה, לא לעזוב את הדף. הקישור לאירוע
        # עצמו לא אבד: הוא נשאר על שם האירוע בעמודת "אירוע".
        img = f"[[קובץ:{r['image_file']}|90px]]" if r["image_file"] else "—"
        venue = f'[{maps_link(r["venue"])} {r["venue"]}]' if r["venue"] else "—"
        if r.get("more_count"):
            n = r["more_count"]
            what = "אירוע אחד" if n == 1 else f"{n} אירועים"
            venue += f'<br /><small>[{ES_HUB_URL} ועוד {what} במקום זה]</small>'
        lines += ["|-", f'| {img} || data-sort-value="{iso}" | {when} || {r["time"] or "—"} || {name} || '
                  f'{r["category"]} || {venue} || {r["entry"] or "—"}']
    lines.append("|}")
    # מעטפת גלילה: הלוח רחב מדי למסך טלפון. בלי המעטפת הטבלה דוחפת את כל הדף
    # הצידה, ובדף RTL זה מסתיר את כל הטקסט מחוץ למסך (המבקר חושב שהאתר ריק).
    # עם המעטפת הטבלה גוללת בתוך עצמה והדף נשאר במקומו.
    return '<div style="overflow-x:auto;">\n' + "\n".join(lines) + "\n</div>"


def build_today_block(today_rows: list, today: dt.date) -> str:
    """Compact highlight box for events happening *today* (at most TODAY_MAX),
    meant to sit above the fold so a visitor immediately sees what's on today.
    Lives between the TODAY_AUTO markers, near the top of the main page.
    """
    heading = f"📅 קורה היום, {he_date(today)}"
    if not today_rows:
        body = (
            f'<div style="background:#fdf3ef; border:1px solid #e3bcac; border-radius:10px; '
            f'padding:10px 16px; margin:0 0 16px;">\n'
            f'<div style="font-size:118%; font-weight:bold; color:#a8674e;">{heading}</div>\n'
            f'<div style="font-size:94%; margin-top:4px;">אין כרגע אירועים מתוזמנים להיום. '
            f'למה שקורה בשבועיים הקרובים ראו את [[#אירועי תרבות ובילוי קרובים|לוח האירועים המלא]] בהמשך הדף.</div>\n'
            f'</div>'
        )
        return f"{TODAY_AUTO_START}\n{body}\n{TODAY_AUTO_END}"

    cards = []
    for r in today_rows:
        name = f"[{r['url']} {r['name']}]" if r["url"] else r["name"]
        venue = f'[{maps_link(r["venue"])} {r["venue"]}]' if r["venue"] else "—"
        # כמו בטבלה: התמונה מובילה לדף הקובץ ולא לאתר האירוע.
        img = f"[[קובץ:{r['image_file']}|46px]]" if r["image_file"] else ""
        time_part = f"'''{r['time']}''' — " if r["time"] else ""
        cards.append(
            f'<div style="display:flex; align-items:center; gap:8px; background:#fff; '
            f'border-radius:6px; padding:5px 8px; margin-top:6px; font-size:92%;">'
            f'{img}<div>{time_part}{name} <span style="color:#888;">· {venue}</span>'
            f'{" · " + r["entry"] if r["entry"] else ""}</div></div>'
        )

    body = (
        f'<div style="background:#fdf3ef; border:1px solid #e3bcac; border-radius:10px; '
        f'padding:10px 16px; margin:0 0 16px;">\n'
        f'<div style="font-size:118%; font-weight:bold; color:#a8674e;">{heading}</div>\n'
        + "\n".join(cards) +
        f'\n</div>'
    )
    return f"{TODAY_AUTO_START}\n{body}\n{TODAY_AUTO_END}"


def build_main_block(rows: list, today: dt.date, days: int) -> str:
    """The full events board (image + video columns) for the main page.

    Lives between the AUTO markers inside the "אירועי תרבות ובילוי קרובים"
    section. The table is collapsible but starts *open*, per the user's wish:
    a reader sees the whole board on arrival and can fold it with "הסתר".
    """
    end = today + dt.timedelta(days=days)
    # say the cap out loud, but only when it actually bit — a rule stated on a
    # board it isn't affecting just reads as noise.
    capped = sum(1 for r in rows if r.get("more_count"))
    cap_note = (
        f"כדי שהלוח ייצג את כל המושבה ולא מקום אחד, הוא מציג עד {VENUE_MAX_EVENTS} אירועים "
        f"לכל מקום; מה שמעבר לכך ממתין לתורו ועולה ללוח ככל שהתאריכים מתקרבים. "
        if capped else ""
    )
    body = (
        f"מה קורה במושבה — '''אירועי תרבות ובילוי''' בשבועיים הקרובים, עם תמונות וסרטונים. "
        f"'''עודכן לאחרונה:''' {he_date(today)} · מציג אירועים עד {he_date(end)}.\n\n"
        f"{build_table(rows, collapsible=True)}\n\n"
        f"{cap_note}"
        f"מקורות הנתונים: {ES_LABEL}; {MATNAS_LABEL}; {HAULAM_LABEL}. ייתכנו אירועים נוספים שאינם מופיעים בלוחות אלה."
    )
    return f"{AUTO_START}\n{body}\n{AUTO_END}"


# --- statistics (manual numbers in the main-page stats table) ---------------

STATS_NOTE = ("''כל הנתונים מתעדכנים אוטומטית (יחד עם לוח האירועים). "
              "לסטטיסטיקה מלאה: [[מיוחד:סטטיסטיקות]].''")


def compute_stats(client) -> tuple:
    """Return (talk_pages, articles_with_table).

    Both are figures the wiki magic words can't express, so the stats table on
    the main page used to carry them as hand-updated numbers. Here we recompute
    them from the live wiki on every run:
      * talk_pages — pages in the שיחה namespace (1).
      * articles_with_table — content pages (namespace 0) whose wikitext
        contains a wiki-table (``{|``).
    """
    talk = len(client.list_pages(namespace=1))
    titles = [p["title"] for p in client.list_pages(namespace=0,
                                                    redirects="nonredirects")]
    tables = 0
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        r = client.session.get(API_URL, params={
            "action": "query", "titles": "|".join(chunk),
            "prop": "revisions", "rvprop": "content", "rvslots": "main",
            "format": "json"})
        for pg in r.json()["query"]["pages"].values():
            revs = pg.get("revisions")
            if revs and "{|" in revs[0]["slots"]["main"]["*"]:
                tables += 1
    return talk, tables


def update_stats_block(text: str, talk: int, tables: int, today: dt.date) -> str:
    """Refresh the manual numbers + 'as of' date in the main-page stats table."""
    text = re.sub(r"(נכון ל־)[^)]*(\))",
                  lambda m: m.group(1) + he_date(today) + m.group(2), text, count=1)
    text = re.sub(r"(\|\s*דפי שיחה\s*\|\|\s*)\d+", r"\g<1>" + str(talk), text, count=1)
    text = re.sub(r"(\|\s*ערכים שמכילים טבלה\s*\|\|\s*)\d+",
                  r"\g<1>" + str(tables), text, count=1)
    # the old footnote called these numbers manual — they are automatic now
    text = re.sub(r"''דפי תוכן, סך כל הדפים.*?\[\[מיוחד:סטטיסטיקות\]\]\.''",
                  STATS_NOTE, text, count=1, flags=re.S)
    return text


def update_main_page(client, rows: list, days: int, dry_run: bool) -> None:
    page = client.get_page(MAIN_TITLE)
    if not page["exists"]:
        print("  main page missing — skipping events board", file=sys.stderr)
        return
    existing = page["wikitext"]
    today = dt.date.today()

    # rows are already chronologically sorted (collect() sorts by "sort").
    # Pull out at most TODAY_MAX of today's events for the top highlight box;
    # the full board below shows everything else (no duplication).
    # The box shows one card per event, so a venue-day row is expanded back into
    # its sessions here (merging is a board-layout concern, not a today concern).
    today_events = [p for r in rows if r["date"] == today
                    for p in (r["parts"] or [r])]
    today_shown = today_events[:TODAY_MAX]
    shown_keys = {r["key"] for r in today_shown}
    # drop a row from the board only once *all* of its events made the box
    rest_rows = [r for r in rows if not _covered_keys(r) <= shown_keys]

    today_block = build_today_block(today_shown, today)
    main_block = build_main_block(rest_rows, today, days)

    if TODAY_AUTO_START in existing and TODAY_AUTO_END in existing:
        new = splice_marked(existing, TODAY_AUTO_START, TODAY_AUTO_END, today_block)
    else:
        # first-ever run: no dedicated slot yet — insert right before the main
        # events heading (or at the end, as a last resort).
        section = f"== {TODAY_HEADING} ==\n{today_block}\n\n"
        marker = f"== {MAIN_HEADING} =="
        new = existing.replace(marker, section + marker, 1) if marker in existing \
            else existing.rstrip() + "\n\n" + section

    if AUTO_START in new and AUTO_END in new:
        new = splice_marked(new, AUTO_START, AUTO_END, main_block)
    else:
        section = f"== {MAIN_HEADING} ==\n{main_block}\n\n"
        if MAIN_ANCHOR in new:
            new = new.replace(MAIN_ANCHOR, section + MAIN_ANCHOR, 1)
        else:
            new = new.rstrip() + "\n\n" + section

    # refresh the statistics table in the same edit
    try:
        talk, tables = compute_stats(client)
        new = update_stats_block(new, talk, tables, today)
        print(f"  stats: {talk} talk pages, {tables} articles with a table", file=sys.stderr)
    except Exception as exc:
        print(f"  stats update skipped: {exc}", file=sys.stderr)
    if dry_run:
        # A 1500-char preview stops short of the events table itself, which is
        # the part a dry run exists to check. Write the whole page next to the
        # script so it can be diffed, and keep the short preview on stdout.
        out = os.path.join(HERE, "dry_run_main_page.txt")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(new)
        print("--- DRY RUN (עמוד ראשי) ---\n" + new[:1500])
        print(f"\n... [הדף המלא נכתב אל {out}]")
        return
    if new == existing:
        print("Main page summary — no change.")
        return
    client.edit_page(MAIN_TITLE, new,
                     summary="עדכון אוטומטי: לוח אירועי תרבות ובילוי + סטטיסטיקה בעמוד הראשי")


def splice_marked(existing: str, start_marker: str, end_marker: str, block: str) -> str:
    s = existing.find(start_marker)
    e = existing.find(end_marker)
    if s != -1 and e != -1 and e > s:
        return existing[:s] + block + existing[e + len(end_marker):]
    return existing.rstrip() + f"\n\n{block}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-images", action="store_true", help="skip image upload (faster dry runs)")
    args = ap.parse_args()

    today = dt.date.today()
    end = today + dt.timedelta(days=args.days)
    print(f"Collecting events for window {today}..{end}", file=sys.stderr)
    rows = collect(today, end)
    attach_videos(rows)
    print(f"{len(rows)} events after merge+dedup", file=sys.stderr)

    client = WikiClient()
    client.login()
    if not args.no_images:
        ensure_images(client, rows)

    # The events board lives directly on the main page now (the old standalone
    # page was retired). All publishing happens through update_main_page.
    update_main_page(client, rows, args.days, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
