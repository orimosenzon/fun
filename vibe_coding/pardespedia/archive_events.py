#!/usr/bin/env python3
"""Historical record of the culture board.

The board on עמוד ראשי only ever shows the next two weeks: an event that has
happened drops off it and, until now, left no trace. Pardespedia is also a
record of what happened here, so every event that passes is kept in a ledger
(`events_archive.json`) and rendered onto a wiki archive page.

Two ways in:

* `harvest_history()` reads old revisions of עמוד ראשי and parses the board out
  of each one. That recovers everything the board ever published, which is the
  only source for events that are already gone from the feeds.
* `record()` is called by update_events.py on every run with the rows it just
  built, so from now on nothing needs recovering.

Both funnel into the same ledger, keyed so that re-running is idempotent.
"""

import datetime as dt
import json
import os
import re
import sys

from wiki_client import WikiClient, API_URL

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "events_archive.json")

BOARD_PAGE = "עמוד ראשי"
ARCHIVE_PAGE = "ארכיון אירועי התרבות"

AUTO_START = "<!-- AUTO:START"
AUTO_END = "<!-- AUTO:END -->"

HE_MONTHS = ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי",
             "אוגוסט", "ספטמבר", "אוקטובר", "נובמבר", "דצמבר"]
HE_WEEKDAYS = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]


# --- ledger ----------------------------------------------------------------

def load_ledger() -> dict:
    if not os.path.exists(LEDGER):
        return {"events": {}}
    with open(LEDGER, encoding="utf-8") as fh:
        return json.load(fh)


def save_ledger(led: dict) -> None:
    with open(LEDGER, "w", encoding="utf-8") as fh:
        json.dump(led, fh, ensure_ascii=False, indent=1, sort_keys=True)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def entry_key(date_iso: str, name: str) -> str:
    """One event, one row, however many times it is seen.

    Date plus a squashed name: the same event harvested from twenty consecutive
    revisions of the main page has to land on the same key, and two genuinely
    different events on one day must not.
    """
    slug = re.sub(r"[^\w֐-׿]+", "-", _norm(name)).strip("-")[:60]
    return f"{date_iso}--{slug}"


def merge(led: dict, rec: dict) -> bool:
    """Add or enrich one event. True if the ledger changed.

    A later sighting may know things an earlier one did not (an image that was
    uploaded after the first publish, say), so filled fields win over empty
    ones. Nothing already recorded is blanked.
    """
    k = entry_key(rec["date"], rec["name"])
    cur = led["events"].get(k)
    if cur is None:
        led["events"][k] = rec
        return True
    changed = False
    for f, v in rec.items():
        if v and not cur.get(f):
            cur[f] = v
            changed = True
    return changed


# --- parsing a published board ---------------------------------------------

def _cell_link(cell: str):
    """('label', 'url') from a wiki external link, else (text, '')."""
    m = re.match(r"\s*\[(\S+)\s+([^\]]*)\]\s*$", cell)
    if m:
        return _norm(m.group(2)), m.group(1)
    return _norm(re.sub(r"</?[^>]+>", " ", cell)), ""


def parse_board(wikitext: str) -> list:
    """Every event row inside the AUTO block of one main-page revision."""
    i = wikitext.find(AUTO_START)
    j = wikitext.find(AUTO_END, i + 1) if i >= 0 else -1
    if i < 0 or j < 0:
        return []
    block = wikitext[i:j]

    out = []
    for chunk in block.split("\n|-"):
        line = chunk.strip().lstrip("\n")
        if not line.startswith("|") or "data-sort-value" not in line:
            continue
        line = line.split("\n")[0] if "\n" in line and "||" not in line.split("\n")[0] else line
        cells = [c.strip() for c in line.lstrip("|").split("||")]
        if len(cells) < 4:
            continue

        date_cell = next((c for c in cells if "data-sort-value" in c), "")
        m = re.search(r'data-sort-value="(\d{4}-\d{2}-\d{2})"', date_cell)
        if not m:
            continue
        iso = m.group(1)
        di = cells.index(date_cell)

        img = ""
        mi = re.search(r"\[\[\s*קובץ:([^\]|]+)", cells[0]) if di > 0 else None
        if mi:
            img = _norm(mi.group(1))

        rest = cells[di + 1:]
        time = _norm(rest[0]) if len(rest) > 0 else ""
        name, url = _cell_link(rest[1]) if len(rest) > 1 else ("", "")
        category = _norm(rest[2]) if len(rest) > 2 else ""
        venue = _cell_link(rest[3])[0] if len(rest) > 3 else ""
        entry = _norm(rest[4]) if len(rest) > 4 else ""

        # A venue-day row folds several events into one cell, one per line.
        # Keep the first as the row's name and drop the "ועוד N" tail.
        name = name.split("<br")[0].strip() or _norm(rest[1]).split("<br")[0]
        name = re.sub(r"^'''|'''$", "", name).strip().rstrip(":").strip()
        if not name or name == "—":
            continue

        out.append({"date": iso, "time": "" if time == "—" else time,
                    "name": name, "url": url, "category": "" if category == "—" else category,
                    "venue": "" if venue == "—" else venue,
                    "entry": "" if entry == "—" else entry, "image_file": img})
    return out


# --- sources into the ledger -----------------------------------------------

def harvest_history(client, led: dict, limit: int = 0) -> int:
    """Walk the main page's revisions and pull every board ever published."""
    revs, cont = [], {}
    while True:
        p = {"action": "query", "prop": "revisions", "titles": BOARD_PAGE,
             "rvprop": "ids|timestamp", "rvlimit": "500", "format": "json"}
        p.update(cont)
        j = client.session.get(API_URL, params=p).json()
        page = list(j["query"]["pages"].values())[0]
        revs += page.get("revisions", [])
        if "continue" not in j:
            break
        cont = j["continue"]

    if limit:
        revs = revs[:limit]
    print(f"  scanning {len(revs)} revisions of {BOARD_PAGE}", file=sys.stderr)

    added = 0
    for n, rev in enumerate(revs, 1):
        j = client.session.get(API_URL, params={
            "action": "query", "prop": "revisions", "revids": rev["revid"],
            "rvprop": "content", "rvslots": "main", "format": "json"}).json()
        try:
            txt = list(j["query"]["pages"].values())[0]["revisions"][0]["slots"]["main"]["*"]
        except (KeyError, IndexError):
            continue
        for rec in parse_board(txt):
            rec["source"] = "היסטוריית העמוד הראשי"
            if merge(led, rec):
                added += 1
        if n % 40 == 0:
            print(f"    {n}/{len(revs)} revisions, {added} new events", file=sys.stderr)
    return added


def record(rows: list, today: dt.date) -> int:
    """Called by update_events.py: keep every event the board just showed.

    Future events are recorded too. The board is the only place some of them
    are ever written down, and waiting for them to pass would mean losing any
    whose run happens to skip the day they drop off.
    """
    led = load_ledger()
    added = 0
    for r in rows:
        rec = {"date": r["date"].isoformat(), "time": r.get("time") or "",
               "name": re.sub(r"^'''|'''$", "", _norm(r.get("name"))),
               "url": r.get("url") or "", "category": r.get("category") or "",
               "venue": r.get("venue") or "", "entry": r.get("entry") or "",
               "image_file": r.get("image_file") or "", "source": "לוח האירועים"}
        if rec["name"] and merge(led, rec):
            added += 1
    if added:
        save_ledger(led)
    return added


# --- rendering -------------------------------------------------------------

def _he_date(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"יום {HE_WEEKDAYS[d.weekday()]}, {d.day} ב{HE_MONTHS[d.month - 1]}"


def build_archive(led: dict, today: dt.date) -> str:
    past = [e for e in led["events"].values() if e["date"] < today.isoformat()]
    past.sort(key=lambda e: (e["date"], e.get("time") or "99:99"))

    months = {}
    for e in past:
        months.setdefault(e["date"][:7], []).append(e)

    total = len(past)
    span = f"{_he_date(past[0]['date'])} ואילך" if past else ""

    out = [
        "__NOTOC__",
        f"'''ארכיון אירועי התרבות''' הוא התיעוד המצטבר של [[עמוד ראשי|לוח האירועים]] "
        f"של פרדס חנה-כרכור. הלוח עצמו מציג רק את השבועיים הקרובים, וכל אירוע שחלף "
        f"נשמר כאן, כדי שיישאר תיעוד של מה שהתקיים במושבה.",
        "",
        f"נכון ל{_he_date(today.isoformat())} מתועדים כאן '''{total}''' אירועים, "
        f"מ{span}." if past else "עדיין לא נאספו אירועים.",
        "",
        "''הדף נבנה אוטומטית. אין לערוך אותו ידנית: כל שינוי יידרס בהרצה הבאה.''",
        "",
    ]

    for ym in sorted(months, reverse=True):
        y, m = ym.split("-")
        evs = months[ym]
        out += [f"== {HE_MONTHS[int(m) - 1]} {y} ==",
                f"''{len(evs)} אירועים''", "",
                '<div style="overflow-x:auto;">',
                '{| class="wikitable sortable" style="width:100%"',
                '! class="unsortable" | תמונה !! תאריך !! class="unsortable" | שעה '
                '!! אירוע !! סוג !! מקום !! כניסה']
        for e in evs:
            img = f"[[קובץ:{e['image_file']}|90px]]" if e.get("image_file") else "—"
            name = f"[{e['url']} {e['name']}]" if e.get("url") else e["name"]
            out += ["|-", f'| {img} || data-sort-value="{e["date"]}" | {_he_date(e["date"])} '
                          f'|| {e.get("time") or "—"} || {name} || {e.get("category") or "—"} '
                          f'|| {e.get("venue") or "—"} || {e.get("entry") or "—"}']
        out += ["|}", "</div>", ""]

    out += ["[[קטגוריה:תרבות מקומית]]", "[[קטגוריה:תיעוד מקומי]]", "[[קטגוריה:היסטוריה]]"]
    return "\n".join(out)


def publish(client, led: dict, today: dt.date, dry_run: bool = False) -> None:
    text = build_archive(led, today)
    if dry_run:
        print(text[:3000])
        print(f"\n... [{len(text)} chars]")
        return
    cur = client.get_page(ARCHIVE_PAGE)
    if cur.get("exists") and (cur.get("wikitext") or "").strip() == text.strip():
        print("  archive unchanged", file=sys.stderr)
        return
    client.edit_page(ARCHIVE_PAGE, text, summary="עדכון ארכיון אירועי התרבות")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", action="store_true",
                    help="scan the main page's revision history for past boards")
    ap.add_argument("--limit", type=int, default=0, help="only the newest N revisions")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = dt.date.today()
    led = load_ledger()
    client = WikiClient()

    if args.harvest:
        client.login()
        added = harvest_history(client, led, args.limit)
        save_ledger(led)
        print(f"harvested {added} new events, ledger now {len(led['events'])}", file=sys.stderr)

    if not args.dry_run:
        client.login()
    publish(client, led, today, args.dry_run)


if __name__ == "__main__":
    main()
