#!/usr/bin/env python3
"""Watch the moshava's objection windows, once a day, and say when one moves.

The banner in the app already knows what is open, but only for whoever happens
to open the app, and only on that day. A window lasts two months and the first
anybody hears of it is often the week it shuts. This runs from cron, compares
today's answer with yesterday's, and writes a report only when there is news:

    נפתח         a plan now has an open window it did not have before
    עומד להיסגר   a window shuts in 7, 3, 1 or 0 days - once per threshold, not
                 every morning of the last week
    נסגר         a window that was open yesterday is not today
    הופקדה       a plan entered the deposit stage but has no date yet. It is on
                 its way to a window, and this is the earliest sign of it

Two sources, since 27/9/2026
----------------------------
Xplan says which plans are at the deposit stage and, for some, the last day.
On 27/9/2026 it had a date for five plans out of thirty-four, and at least four
of the other twenty-nine were open that day. מבא"ת knew: the sign on the plot,
the web notice, רשומות, whether it takes objections online. mavat.py reads that
page for every plan at the deposit stage and works out the window from what is
recorded and from the law; see its docstring for the rules and for why a date
is sometimes an estimate. An estimate is marked as one in the report and in
the app.

It asks the same question as `openForObjection()` in web/plan_here.js, through
the same helpers build_plans.py uses for the layer.

The map's file
--------------
Besides the report, every run writes web/data/objections.json: each plan that
is open, with its date, how that date is known, its blue line, and the ids of
the parcels it covers in web/data/parcels.json. The parcels layer paints those
glowing red. `--publish` also puts the file in the data repo, where the app
reads it from.

State lives in .watch/objections.json (not in git): the plans seen in deposit,
the deadline each had, and which reminders were already said. The first run has
nothing to compare against, so it records the current picture and writes a
report of what is open now, without calling any of it new.

    python3 watch_objections.py                  # ask both services, report
    python3 watch_objections.py --publish        # ...and publish the map's file
    python3 watch_objections.py --no-mavat       # Xplan only (מבא"ת from cache)
    python3 watch_objections.py --fixture f.json # a recorded answer instead
    python3 watch_objections.py --today 2026-09-25 --state /tmp/s.json

Prints REPORT=<path> when it wrote one, and NEWS=<n>, for the cron wrapper.
"""

import argparse
import base64
import datetime as dt
import json
import os
import subprocess
import sys

from zoneinfo import ZoneInfo

import build_plans as bp
import mavat

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, ".watch", "objections.json")
REPORTS = os.path.join(HERE, "reports", "objections")
PUBLIC = os.path.join(HERE, "web", "data", "objections.json")
PARCELS = os.path.join(HERE, "web", "data", "parcels.json")
APP = "https://orimosenzon.github.io/fun/vibe_coding/dereh_kitzur/web/index.html"
DATA_REPO = "orimosenzon/derech-kitzur-data"
DATA_PATH = "data/objections.json"
BOT = "בוטי"
BOT_EMAIL = "orimosenzon@gmail.com"      # as on the bot's earlier commits there

DEPOSIT = ["פרסום הפקדה", "בתהליך הפקדה"]
REMIND = (7, 3, 1, 0)
TZ = ZoneInfo("Asia/Jerusalem")
LOCAL = 3          # pl_by_auth_of: 3 is the local committee, 2 the district


# ---------------------------------------------------------------- the answer

def fetch():
    quoted = ",".join("'" + s + "'" for s in DEPOSIT)
    doc = bp.get(bp.XPLAN, {
        "where": f"{bp.AREA} AND internet_short_status IN ({quoted})",
        "outFields": ",".join([
            "pl_number", "pl_name", "pl_url", "internet_short_status",
            "pl_last_deposit_date", "pl_rejection_date", "pl_date_advertise",
            "pl_by_auth_of", "depositing_date",
            "quantity_delta_120", "quantity_delta_125", "quantity_delta_75",
            "quantity_delta_60", "quantity_delta_80"]),
        # The blue line too, for the map: a few hundred vertices in all, and
        # half a metre is finer than any parcel boundary it is laid against.
        "returnGeometry": "true",
        "outSR": "4326",
        "maxAllowableOffset": "0.000005",
        "f": "json",
    })
    if "error" in doc:
        raise SystemExit(f"שגיאה מהשירות: {doc['error']}")
    return doc.get("features", [])


def shut_date(at):
    """The last day for objections as a calendar date, or None.

    The service stores it as midnight UTC of that day, so the UTC date is the
    day itself. Counting days in local time instead of from time.time() keeps a
    run at 01:00 Israel time from being one day off.
    """
    stamp = at.get("pl_rejection_date") or at.get("pl_last_deposit_date")
    if not stamp:
        return None
    return dt.datetime.fromtimestamp(int(stamp) / 1000, dt.timezone.utc).date()


def enrich(features, use_mavat, mavat_dir=None, log=print):
    """{plan number: what מבא"ת says}, fetched now or read from the cache."""
    mids = {}
    for f in features:
        at = f["attributes"]
        mid = mavat.mid_of(at.get("pl_url"))
        if mid:
            mids[(at.get("pl_number") or "").strip()] = mid
    if mavat_dir:
        mavat.CACHE = mavat_dir
    got = {}
    if use_mavat and mids:
        try:
            got = mavat.fetch(sorted(set(mids.values())), log=log)
        except Exception as err:                        # noqa: BLE001 - Xplan alone still works
            log(f"מבא\"ת לא זמין: {err}")
    out = {}
    for num, mid in mids.items():
        rec = got.get(mid) or mavat.cached(mid)
        if rec:
            out[num] = rec
    return out


def picture(features, today, pages=None):
    """{plan number: what the report and the map need} for every plan in deposit."""
    pages = pages or {}
    out = {}
    for f in features:
        at = f["attributes"]
        num = (at.get("pl_number") or "").strip()
        if not num:
            continue
        shuts = shut_date(at)
        rec = pages.get(num)
        win = mavat.window(rec, today) if rec else None
        if shuts:
            # Xplan's date, and it has always matched מבא"ת's where both had one.
            is_open, exact, basis = shuts >= today, True, "המועד האחרון במאגר מינהל התכנון"
        elif win and win["state"] != "none":
            shuts, exact, basis = win["shuts"], win["exact"], win["basis"]
            is_open = win["state"] == "open"
        else:
            is_open, exact, basis = False, False, ""
        out[num] = {
            "name": (at.get("pl_name") or "").strip() or num,
            "url": at.get("pl_url") or bp.XPLAN_SITE,
            "adds": bp.adds(at),
            "units": int(float(at.get("quantity_delta_120") or 0)),
            "shuts": shuts.isoformat() if shuts else None,
            "left": (shuts - today).days if shuts else None,
            "open": is_open,
            "exact": exact,
            "basis": basis,
            # Xplan had no date: the window exists only on מבא"ת. The app says
            # so, because it is the reason nobody heard of it.
            "hidden": is_open and not shut_date(at),
            "local": at.get("pl_by_auth_of") == LOCAL,
            "advertised": bp.when(at.get("pl_date_advertise"))
                          or (win["papers"].strftime("%d/%m/%Y") if win and win["papers"] else None),
            "sign": win["sign"].isoformat() if win and win["sign"] else None,
            "address": ", ".join(rec.get("address", [])) if rec else "",
            "where": rec.get("where", "") if rec else "",
            "geometry": f.get("geometry"),
            "record": rec,
        }
    return out


# ---------------------------------------------------------------- the news

def compare(now, before, today):
    """What changed since the last run, and the state to keep for the next."""
    seen = before.get("plans", {})
    first = not seen and not before.get("ran")
    # "Open last time" is judged against the day of the last run, not today:
    # a window whose last day was yesterday was open then and is shut now.
    last_run = before.get("ran") or today.isoformat()
    news = {"opened": [], "closing": [], "closed": [], "deposited": []}
    keep = {}

    def was_open_then(old):
        if "open" in old:
            return bool(old["open"])
        return bool(old.get("shuts")) and old["shuts"] >= last_run

    for num, p in now.items():
        old = seen.get(num, {})
        said = list(old.get("said", []))
        is_open = p["open"]
        was_open = was_open_then(old)

        if is_open and not first:
            if not was_open:
                news["opened"].append(num)
            elif old.get("shuts") != p["shuts"] and p["exact"] and old.get("exact", True):
                # The date moved while the window was open: an extension. Say
                # it as a new opening so it cannot be missed, and let the
                # reminders run again against the new date. An estimate that
                # became a recorded date is not an extension.
                news["opened"].append(num)
                said = []
        if is_open and p["left"] is not None and p["left"] >= 0:
            due = [d for d in REMIND if p["left"] <= d and d not in said]
            if due and not first:
                news["closing"].append(num)
            said += due
        if not p["shuts"] and not is_open and num not in seen and not first:
            news["deposited"].append(num)
        if was_open and not is_open and not first:
            news["closed"].append(num)

        keep[num] = {"name": p["name"], "shuts": p["shuts"], "open": is_open,
                     "exact": p["exact"], "said": sorted(set(said))}

    # A plan that left the deposit stage altogether while its window was open.
    # Rare - the window normally shuts first - but it should not vanish silently.
    for num, old in seen.items():
        if num not in now and was_open_then(old):
            news["closed"].append(num)
            now[num] = {"name": old["name"], "url": bp.XPLAN_SITE, "adds": None,
                        "shuts": old["shuts"], "left": None, "open": False,
                        "exact": old.get("exact", True), "basis": "",
                        "advertised": None, "gone": True}

    state = {"ran": today.isoformat(), "plans": keep}
    return first, news, state


# ---------------------------------------------------------------- the report

def when_text(left):
    return "היום" if left == 0 else "מחר" if left == 1 else f"בעוד {left} ימים"


def heb(iso):
    return dt.date.fromisoformat(iso).strftime("%d/%m/%Y")


def line(p):
    bits = [f"**{p['name']}**"]
    if p.get("address"):
        bits.append(f"({p['address']})")
    if p.get("adds"):
        bits.append(f"מוסיפה {p['adds']}.")
    if p.get("gone"):
        bits.append("יצאה משלב ההפקדה לפני שהחלון נסגר.")
    elif p.get("open") and p["left"] is not None and p["left"] >= 0:
        if p.get("exact", True):
            bits.append(f"אפשר להתנגד עד {heb(p['shuts'])}, {when_text(p['left'])}.")
        else:
            bits.append(f"פתוחה להתנגדות. מועד משוער: {heb(p['shuts'])}, {when_text(p['left'])}.")
    elif p.get("open"):
        bits.append("פתוחה להתנגדות, מועד הסגירה לא ידוע.")
    elif p["shuts"]:
        bits.append(f"החלון נסגר ב-{heb(p['shuts'])}"
                    + ("." if p.get("exact", True) else " (משוער)."))
    if p.get("hidden"):
        bits.append("לא מופיעה כפתוחה במאגר הארצי (Xplan).")
    if p.get("open") and not p.get("exact", True) and p.get("basis"):
        bits.append(p["basis"] + ".")
    if p.get("advertised"):
        bits.append(f"פורסמה בעיתונים ב-{p['advertised']}.")
    bits.append(f"[במבא\"ת]({p['url']})")
    return "- " + " ".join(bits)


def report(first, news, now, today):
    open_now = sorted((n for n, p in now.items() if p.get("open")),
                      key=lambda n: (now[n]["left"] is None, now[n]["left"] or 0))
    out = [f"# חלונות התנגדות בפרדס חנה-כרכור, {today.strftime('%d/%m/%Y')}", ""]

    if first:
        out += ["ריצה ראשונה, אין עדיין יום קודם להשוות אליו. זו התמונה הנוכחית, "
                "ומחר ואילך יופיעו כאן רק שינויים.", ""]
    sections = [
        ("opened", "נפתח חלון התנגדות"),
        ("closing", "עומד להיסגר"),
        ("closed", "נסגר"),
        ("deposited", "הופקדה, עוד בלי תאריך לחלון"),
    ]
    for key, title in sections:
        nums = sorted(news[key], key=lambda n: (now[n]["left"] is None, now[n]["left"] or 0))
        if nums:
            out += [f"## {title}", ""] + [line(now[n]) for n in nums] + [""]

    out += [f"## פתוח עכשיו ({len(open_now)})", ""]
    out += [line(now[n]) for n in open_now] or ["אין כרגע תכנית פתוחה להתנגדות במושבה."]
    out += ["", f"[באפליקציה]({APP}) · "
            f"[שירות המפה של מינהל התכנון]({bp.XPLAN_SITE})", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- the map's file

def load_parcels(path):
    """{feature id: shapely polygon} of the moshava's parcels, or {} without them."""
    try:
        from shapely.geometry import shape
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    except (ImportError, FileNotFoundError):
        return {}
    return {f["id"]: shape(f["geometry"]) for f in doc.get("features", [])}


def blue_line(geometry):
    """An esri polygon as a shapely one, or None."""
    rings = (geometry or {}).get("rings")
    if not rings:
        return None
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = [Polygon(r) for r in rings if len(r) >= 4]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    return unary_union(polys) if polys else None


def covered(p, parcels):
    """Ids of the parcels a plan is about.

    The plan's own list of block and parcel on מבא"ת first: it is what the plan
    says it is about. A parcel listed as "חלק" is kept when the blue line really
    lies on it, since a road parcel beside the plot is often listed as partial
    for a strip of pavement. Without a list, the parcels the blue line covers
    most of; failing that, the one under its middle.
    """
    if not parcels:
        return []
    line_ = blue_line(p.get("geometry"))
    whole, partial = mavat.listed_parcels(p.get("record") or {})

    def overlap(fid):
        poly = parcels.get(fid)
        if line_ is None or poly is None:
            return None
        try:
            inter = poly.intersection(line_).area
        except Exception:                               # noqa: BLE001 - bad ring
            return None
        return inter / max(min(poly.area, line_.area), 1e-14)

    ids = []
    for g, n in whole:
        fid = g * 10000 + n
        if fid in parcels:
            ids.append(fid)
    for g, n in partial:
        fid = g * 10000 + n
        share = overlap(fid)
        if fid in parcels and (share is None or share >= 0.15):
            ids.append(fid)
    if not ids and line_ is not None:
        minx, miny, maxx, maxy = line_.bounds
        for fid, poly in parcels.items():
            bx = poly.bounds
            if bx[2] < minx or bx[0] > maxx or bx[3] < miny or bx[1] > maxy:
                continue
            inter = poly.intersection(line_).area
            if inter >= 0.5 * poly.area:
                ids.append(fid)
        if not ids:
            mid = line_.representative_point()
            ids = [fid for fid, poly in parcels.items() if poly.contains(mid)][:1]
    return sorted(set(ids))


def public(now, today, parcels):
    """The file the app paints from: only what is open, and only what it shows."""
    plans = []
    for num, p in sorted(now.items(), key=lambda kv: (kv[1]["left"] is None, kv[1]["left"] or 0)):
        if not p.get("open"):
            continue
        line_ = blue_line(p.get("geometry"))
        centre = None
        if line_ is not None:
            pt = line_.representative_point()
            centre = [round(pt.x, 6), round(pt.y, 6)]
        ids = covered(p, parcels)
        if centre is None and ids:
            pt = parcels[ids[0]].representative_point()
            centre = [round(pt.x, 6), round(pt.y, 6)]
        rings = [[[round(x, 6), round(y, 6)] for x, y, *_ in r]
                 for r in ((p.get("geometry") or {}).get("rings") or [])]
        plans.append({
            "num": num, "name": p["name"], "url": p["url"],
            "adds": p.get("adds"), "units": p.get("units", 0),
            "shuts": p["shuts"], "exact": p["exact"], "basis": p["basis"],
            "hidden": p.get("hidden", False), "local": p.get("local", False),
            "address": p.get("address", ""), "where": p.get("where", ""),
            "advertised": p.get("advertised"), "sign": p.get("sign"),
            "parcels": ids, "centre": centre, "rings": rings,
        })
    return {
        "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "day": today.isoformat(),
        "source": "מינהל התכנון: Xplan ומבא\"ת",
        "checked": len(now),
        "plans": plans,
    }


def publish(doc):
    """Put the file in the data repo, through the GitHub API as the bot.

    One file, one commit, no working copy to keep clean. Skipped when nothing
    but the timestamp changed, so the repo does not gain a commit a day.
    """
    body = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    api = f"repos/{DATA_REPO}/contents/{DATA_PATH}"
    sha = None
    try:
        got = json.loads(subprocess.run(["gh", "api", api], capture_output=True,
                                        text=True, check=True).stdout)
        sha = got.get("sha")
        old = json.loads(base64.b64decode(got.get("content", "")).decode("utf-8"))
        strip = lambda d: {k: v for k, v in d.items() if k not in ("updated",)}
        if strip(old) == strip(doc):
            return "ללא שינוי"
    except subprocess.CalledProcessError:
        pass                                            # first time: no file yet
    payload = {
        "message": f"חלונות התנגדות: {len(doc['plans'])} פתוחים ({doc['day']})",
        "content": base64.b64encode(body.encode("utf-8")).decode("ascii"),
        "committer": {"name": BOT, "email": BOT_EMAIL},
        "author": {"name": BOT, "email": BOT_EMAIL},
    }
    if sha:
        payload["sha"] = sha
    subprocess.run(["gh", "api", "-X", "PUT", api, "--input", "-"],
                   input=json.dumps(payload), capture_output=True, text=True, check=True)
    return "פורסם"


# ---------------------------------------------------------------- main

def load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}


def main():
    ap = argparse.ArgumentParser(description="מעקב אחר חלונות התנגדות לתכניות במושבה")
    ap.add_argument("--fixture", help="תשובה מוקלטת של השירות (JSON של features)")
    ap.add_argument("--mavat-dir", help="תיקייה של דפי מבא\"ת מוקלטים (<mid>.json)")
    ap.add_argument("--no-mavat", action="store_true", help="בלי לפתוח את מבא\"ת (רק מהמטמון)")
    ap.add_argument("--state", default=STATE)
    ap.add_argument("--reports", default=REPORTS)
    ap.add_argument("--public", default=PUBLIC, help="לאן לכתוב את קובץ המפה")
    ap.add_argument("--parcels", default=PARCELS)
    ap.add_argument("--publish", action="store_true", help="לפרסם את קובץ המפה בריפו הנתונים")
    ap.add_argument("--today", help="YYYY-MM-DD, ברירת מחדל: היום לפי שעון ישראל")
    args = ap.parse_args()

    today = (dt.date.fromisoformat(args.today) if args.today
             else dt.datetime.now(TZ).date())
    if args.fixture:
        doc = load(args.fixture)
        features = doc.get("features", doc if isinstance(doc, list) else [])
    else:
        features = fetch()

    # A recorded answer never opens a browser: the tests run offline.
    live_mavat = not (args.no_mavat or args.fixture or args.mavat_dir)
    pages = enrich(features, live_mavat, args.mavat_dir, log=lambda m: print(m, file=sys.stderr))

    now = picture(features, today, pages)
    first, news, state = compare(now, load(args.state), today)
    count = sum(len(v) for v in news.values())

    written = None
    if first or count:
        os.makedirs(args.reports, exist_ok=True)
        written = os.path.join(args.reports, f"{today.isoformat()}.md")
        with open(written, "w", encoding="utf-8") as handle:
            handle.write(report(first, news, now, today))
    bp.save_json(args.state, state)

    doc = public(now, today, load_parcels(args.parcels))
    bp.save_json(args.public, doc)

    opened = sum(1 for p in now.values() if p.get("open"))
    hidden = sum(1 for p in now.values() if p.get("hidden"))
    print(f"{len(now)} תכניות בשלב ההפקדה, {opened} פתוחות להתנגדות"
          + (f" ({hidden} מהן בלי תאריך ב-Xplan)" if hidden else ""))
    for key, nums in news.items():
        if nums:
            print(f"  {key}: {', '.join(nums)}")
    if args.publish:
        try:
            print(f"קובץ המפה: {publish(doc)}")
        except subprocess.CalledProcessError as err:
            print(f"פרסום נכשל: {(err.stderr or '')[:200]}", file=sys.stderr)
    print(f"NEWS={count}")
    if written:
        print(f"REPORT={written}")


if __name__ == "__main__":
    sys.exit(main())
