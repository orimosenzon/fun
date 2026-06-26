#!/usr/bin/env python3
"""Update the "אירועי תרבות ובילוי בפרדס חנה-כרכור" guide on pardespedia.

Pulls events from several structured sources in the moshava, keeps the ones in
the next N days, merges + de-duplicates them, and rebuilds an auto-managed
section of the wiki page. Content outside the AUTO markers is preserved.

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

import requests

from wiki_client import WikiClient, API_URL

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_MAP_FILE = os.path.join(HERE, "event_videos.json")
MANUAL_FILE = os.path.join(HERE, "manual_events.json")
MANUAL_HORIZON_DAYS = 70  # how far ahead to expand recurring manual events

PAGE_TITLE = "אירועי תרבות ובילוי בפרדס חנה-כרכור"
MAIN_TITLE = "עמוד ראשי"
MAIN_HEADING = "אירועי תרבות ובילוי קרובים"
MAIN_ANCHOR = "== מידע עירוני =="  # the summary section is inserted just before this

ES_FEED_URL = "https://pardeshanna.eventschedule.com/api/calendar-events"
ES_LABEL = "[https://pardeshanna.eventschedule.com לוח האירועים של פרדס חנה-כרכור (eventschedule)]"
MATNAS_URL = "https://live.tickchak.co.il/matans-pardes-hana"
MATNAS_LABEL = "[https://live.tickchak.co.il/matans-pardes-hana מרכז אמנויות הבמה — מתנ\"ס פרדס חנה]"
MATNAS_VENUE = "מרכז אמנויות הבמה (מתנ\"ס)"

AUTO_START = "<!-- AUTO:START — אזור מתעדכן אוטומטית, אל תערוך ידנית -->"
AUTO_END = "<!-- AUTO:END -->"

HE_WEEKDAYS = {0: "שני", 1: "שלישי", 2: "רביעי", 3: "חמישי", 4: "שישי", 5: "שבת", 6: "ראשון"}
HE_MONTHS = {
    1: "ינואר", 2: "פברואר", 3: "מרץ", 4: "אפריל", 5: "מאי", 6: "יוני",
    7: "יולי", 8: "אוגוסט", 9: "ספטמבר", 10: "אוקטובר", 11: "נובמבר", 12: "דצמבר",
}
CATEGORY_HE = {
    "Art & Culture": "אמנות ותרבות", "אמנות ותרבות": "אמנות ותרבות",
    "Concerts": "מופעים והופעות", "Parties & Festivals": "מסיבות ופסטיבלים",
    "Personal Growth": "התפתחות אישית", "Sports": "ספורט", "Workshops": "סדנאות",
    "MusicEvent": "מופעים והופעות", "ComedyEvent": "סטנדאפ וקומדיה",
    "TheaterEvent": "תיאטרון", "Event": "מופעים והופעות",
}


def he_date(d: dt.date) -> str:
    return f"{d.day} ב{HE_MONTHS[d.month]} {d.year}"


def wiki_escape(text: str) -> str:
    return (text or "").replace("|", "‖").strip()


def clean_venue(name: str) -> str:
    return name.split("|")[0].strip() if name else ""


def price_label(price, is_free: bool = False) -> str:
    if is_free:
        return "חינם"
    if price not in (None, "", "0.00", "0", 0):
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


def _row(d, time, name, url, category, venue, entry, sort, key, image_url) -> dict:
    return {"date": d, "time": time, "name": wiki_escape(name), "url": url or "",
            "category": category or "", "venue": wiki_escape(venue), "entry": entry,
            "sort": sort, "key": key, "image_url": image_url or "",
            "image_file": None, "video_id": None}


# --- sources ---------------------------------------------------------------

def fetch_eventschedule() -> list:
    r = requests.get(ES_FEED_URL, headers={"Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    rows = []
    for e in r.json().get("events", []):
        ds = e.get("local_date")
        if not ds:
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
            CATEGORY_HE.get(e.get("category_name"), e.get("category_name") or ""),
            clean_venue(e.get("venue_name")),
            price_label(e.get("ticket_price"), e.get("is_free")),
            lsa or ds, f"es-{e.get('id')}",
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
            MATNAS_VENUE, price_label(offers.get("price")), sd.isoformat(),
            f"mt-{hashlib.md5((url or it.get('name','')).encode()).hexdigest()[:8]}", img,
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
    """rec like 'first_friday' / 'last_friday' -> matching dates within window."""
    try:
        ordinal, wd_name = rec.split("_", 1)
        weekday = _WD[wd_name]
    except (ValueError, KeyError):
        return []
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
        # resolve dates: explicit one-off or a monthly recurrence
        dates = []
        if ev.get("date"):
            try:
                dates = [dt.date.fromisoformat(ev["date"])]
            except ValueError:
                dates = []
        elif ev.get("recurrence"):
            dates = _expand_recurrence(ev["recurrence"], today, horizon)
        t = ev.get("time", "")
        for d in dates:
            key = "man-" + hashlib.md5(f"{name}{d}".encode()).hexdigest()[:8]
            r = _row(d, t, name, ev.get("url"), ev.get("category", ""),
                     ev.get("venue", ""), ev.get("entry", "חינם"),
                     f"{d}T{t or '00:00'}", key, ev.get("image_url"))
            r["image_file"] = ev.get("image_file")  # reference an existing wiki file
            r["video_id"] = ev.get("video")          # inline curated video
            rows.append(r)
    return rows


SOURCES = [("eventschedule", fetch_eventschedule), ("מתנ\"ס", fetch_matnas),
           ("ידני", fetch_manual)]


def collect(start: dt.date, end: dt.date) -> list:
    all_rows = []
    for name, fn in SOURCES:
        try:
            rows = fn()
            all_rows += rows
            print(f"  source {name}: {len(rows)} events", file=sys.stderr)
        except Exception as exc:
            print(f"  source {name}: FAILED ({exc})", file=sys.stderr)
    windowed = [r for r in all_rows if start <= r["date"] <= end]
    seen, deduped = set(), []
    for r in sorted(windowed, key=lambda r: r["sort"]):
        k = (norm(r["name"]), r["date"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    return deduped


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

def build_table(rows: list) -> str:
    if not rows:
        return "''אין כרגע אירועים מתוזמנים בטווח הקרוב. בדקו שוב בקרוב.''"
    lines = ['{| class="wikitable" style="width:100%"',
             "! תמונה !! תאריך !! שעה !! אירוע !! קטגוריה !! מקום !! כניסה !! סרטון"]
    for r in rows:
        when = f"יום {HE_WEEKDAYS[r['date'].weekday()]}, {r['date'].day}.{r['date'].month}"
        name = f"[{r['url']} {r['name']}]" if r["url"] else r["name"]
        if r["image_file"]:
            link = f"|link={r['url']}" if r["url"] else ""
            img = f"[[קובץ:{r['image_file']}|90px{link}]]"
        else:
            img = "—"
        vid = f"<youtube width=\"200\" height=\"120\">{r['video_id']}</youtube>" if r["video_id"] else "—"
        lines += ["|-", f"| {img} || {when} || {r['time'] or '—'} || {name} || "
                  f"{r['category']} || {r['venue']} || {r['entry']} || {vid}"]
    lines.append("|}")
    return "\n".join(lines)


def _when(r: dict) -> str:
    return f"יום {HE_WEEKDAYS[r['date'].weekday()]}, {r['date'].day}.{r['date'].month}"


def build_compact_table(rows: list) -> str:
    """Collapsed summary table for the main page — no image/video columns."""
    if not rows:
        return "''אין כרגע אירועים מתוזמנים בטווח הקרוב.''"
    lines = ['{| class="wikitable mw-collapsible mw-collapsed" style="width:100%"',
             '|+ אירועי השבועיים הקרובים — הקליקו להצגה / הסתרה',
             "! תאריך !! שעה !! אירוע !! קטגוריה !! מקום !! כניסה"]
    for r in rows:
        name = f"[{r['url']} {r['name']}]" if r["url"] else r["name"]
        lines += ["|-", f"| {_when(r)} || {r['time'] or '—'} || {name} || "
                  f"{r['category']} || {r['venue']} || {r['entry']}"]
    lines.append("|}")
    return "\n".join(lines)


def build_main_block(rows: list) -> str:
    body = (
        f"מה קורה במושבה — אירועי תרבות ובילוי בשבועיים הקרובים. "
        f"ל[[{PAGE_TITLE}|טבלה המלאה]] (עם תמונות וסרטונים).\n\n"
        f"{build_compact_table(rows)}"
    )
    return f"{AUTO_START}\n{body}\n{AUTO_END}"


def update_main_page(client, rows: list, dry_run: bool) -> None:
    page = client.get_page(MAIN_TITLE)
    if not page["exists"]:
        print("  main page missing — skipping summary", file=sys.stderr)
        return
    existing = page["wikitext"]
    block = build_main_block(rows)
    if AUTO_START in existing and AUTO_END in existing:
        new = splice_auto(existing, block)
    else:
        section = f"== {MAIN_HEADING} ==\n{block}\n\n"
        if MAIN_ANCHOR in existing:
            new = existing.replace(MAIN_ANCHOR, section + MAIN_ANCHOR, 1)
        else:
            new = existing.rstrip() + "\n\n" + section
    if dry_run:
        print("--- DRY RUN (עמוד ראשי) ---\n" + new[:1500])
        return
    if new == existing:
        print("Main page summary — no change.")
        return
    client.edit_page(MAIN_TITLE, new,
                     summary="עדכון אוטומטי: תקציר אירועי תרבות ובילוי בעמוד הראשי")


def build_auto_block(rows: list, today: dt.date, days: int) -> str:
    end = today + dt.timedelta(days=days)
    body = (
        f"'''עודכן לאחרונה:''' {he_date(today)} · מציג אירועים עד {he_date(end)}.\n\n"
        f"{build_table(rows)}\n\n"
        f"מקורות הנתונים: {ES_LABEL}; {MATNAS_LABEL}. ייתכנו אירועים נוספים שאינם מופיעים בלוחות אלה."
    )
    return f"{AUTO_START}\n{body}\n{AUTO_END}"


def initial_page(auto_block: str) -> str:
    return (
        "דף זה מרכז '''אירועי תרבות ובילוי הקרובים''' בפרדס חנה-כרכור — מופעים, "
        "הופעות, סדנאות, מסיבות ופסטיבלים. הוא '''מתעדכן אוטומטית''' מתוך לוחות האירועים "
        "המקומיים, כך שתמיד אפשר לראות מה קורה במושבה בשבועיים הקרובים.\n\n"
        f"== אירועים קרובים ==\n{auto_block}\n\n"
        "[[קטגוריה: תרבות מקומית]]\n"
    )


def splice_auto(existing: str, auto_block: str) -> str:
    s = existing.find(AUTO_START)
    e = existing.find(AUTO_END)
    if s != -1 and e != -1 and e > s:
        return existing[:s] + auto_block + existing[e + len(AUTO_END):]
    return existing.rstrip() + f"\n\n== אירועים קרובים ==\n{auto_block}\n"


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

    auto_block = build_auto_block(rows, today, args.days)
    page = client.get_page(PAGE_TITLE)
    if page["exists"]:
        new_content = splice_auto(page["wikitext"], auto_block)
    else:
        new_content = initial_page(auto_block)

    if args.dry_run:
        print(f"--- DRY RUN ---\n{new_content}")
        update_main_page(client, rows, dry_run=True)
        return
    if page["exists"] and new_content == page["wikitext"]:
        print("No change — skipping edit.")
    else:
        client.edit_page(
            PAGE_TITLE, new_content,
            summary=f"עדכון אוטומטי של אירועי תרבות ובילוי ({len(rows)} אירועים, חלון {args.days} ימים)",
        )

    # keep the compact summary on the main page in sync too
    update_main_page(client, rows, dry_run=False)


if __name__ == "__main__":
    main()
