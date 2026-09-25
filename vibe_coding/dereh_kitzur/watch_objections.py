#!/usr/bin/env python3
"""Watch the moshava's objection windows, once a day, and say when one moves.

The banner in the app already knows what is open, but only for whoever happens
to open the app, and only on that day. A window lasts sixty days and the first
anybody hears of it is often the week it shuts. This runs from cron, compares
today's answer with yesterday's, and writes a report only when there is news:

    נפתח         a plan now has a last day for objections it did not have before
    עומד להיסגר   a window shuts in 7, 3, 1 or 0 days - once per threshold, not
                 every morning of the last week
    נסגר         a window that was open yesterday is not today
    הופקדה       a plan entered the deposit stage but has no date yet. It is on
                 its way to a window, and this is the earliest sign of it

It asks the same question as `openForObjection()` in web/plan_here.js, through
the same helpers build_plans.py uses for the layer, so the three cannot drift
apart on what "open" means.

State lives in .watch/objections.json (not in git): the plans seen in deposit,
the deadline each had, and which reminders were already said. The first run has
nothing to compare against, so it records the current picture and writes a
report of what is open now, without calling any of it new.

    python3 watch_objections.py                  # ask the service, report
    python3 watch_objections.py --fixture f.json # a recorded answer instead
    python3 watch_objections.py --today 2026-09-25 --state /tmp/s.json

Prints REPORT=<path> when it wrote one, and NEWS=<n>, for the cron wrapper.
"""

import argparse
import datetime as dt
import json
import os
import sys

from zoneinfo import ZoneInfo

import build_plans as bp

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, ".watch", "objections.json")
REPORTS = os.path.join(HERE, "reports", "objections")
APP = "https://orimosenzon.github.io/fun/vibe_coding/dereh_kitzur/web/index.html"

DEPOSIT = ["פרסום הפקדה", "בתהליך הפקדה"]
REMIND = (7, 3, 1, 0)
TZ = ZoneInfo("Asia/Jerusalem")


# ---------------------------------------------------------------- the answer

def fetch():
    quoted = ",".join("'" + s + "'" for s in DEPOSIT)
    doc = bp.get(bp.XPLAN, {
        "where": f"{bp.AREA} AND internet_short_status IN ({quoted})",
        "outFields": ",".join([
            "pl_number", "pl_name", "pl_url", "internet_short_status",
            "pl_last_deposit_date", "pl_rejection_date", "pl_date_advertise",
            "quantity_delta_120", "quantity_delta_125", "quantity_delta_75",
            "quantity_delta_60", "quantity_delta_80"]),
        "returnGeometry": "false",
        "f": "json",
    })
    if "error" in doc:
        raise SystemExit(f"שגיאה מהשירות: {doc['error']}")
    return [f["attributes"] for f in doc.get("features", [])]


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


def picture(rows, today):
    """{plan number: what the report needs about it} for every plan in deposit."""
    out = {}
    for at in rows:
        num = (at.get("pl_number") or "").strip()
        if not num:
            continue
        shuts = shut_date(at)
        out[num] = {
            "name": (at.get("pl_name") or "").strip() or num,
            "url": at.get("pl_url") or bp.XPLAN_SITE,
            "adds": bp.adds(at),
            "shuts": shuts.isoformat() if shuts else None,
            "left": (shuts - today).days if shuts else None,
            "advertised": bp.when(at.get("pl_date_advertise")),
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

    for num, p in now.items():
        old = seen.get(num, {})
        said = list(old.get("said", []))
        is_open = p["left"] is not None and p["left"] >= 0
        was_open = bool(old.get("shuts")) and old["shuts"] >= last_run

        if is_open and not first:
            if old.get("shuts") != p["shuts"] and not was_open:
                news["opened"].append(num)
            elif old.get("shuts") != p["shuts"]:
                # The date moved while the window was open: an extension. Say
                # it as a new opening so it cannot be missed, and let the
                # reminders run again against the new date.
                news["opened"].append(num)
                said = []
        if is_open:
            due = [d for d in REMIND if p["left"] <= d and d not in said]
            if due and not first:
                news["closing"].append(num)
            said += due
        if not p["shuts"] and num not in seen and not first:
            news["deposited"].append(num)
        if was_open and not is_open and not first:
            news["closed"].append(num)

        keep[num] = {"name": p["name"], "shuts": p["shuts"], "said": sorted(set(said))}

    # A plan that left the deposit stage altogether while its window was open.
    # Rare - the window normally shuts first - but it should not vanish silently.
    for num, old in seen.items():
        if num not in now and old.get("shuts") and old["shuts"] >= last_run:
            news["closed"].append(num)
            now[num] = {"name": old["name"], "url": bp.XPLAN_SITE, "adds": None,
                        "shuts": old["shuts"], "left": None, "advertised": None,
                        "gone": True}

    state = {"ran": today.isoformat(), "plans": keep}
    return first, news, state


# ---------------------------------------------------------------- the report

def when_text(left):
    return "היום" if left == 0 else "מחר" if left == 1 else f"בעוד {left} ימים"


def heb(iso):
    return dt.date.fromisoformat(iso).strftime("%d/%m/%Y")


def line(p):
    bits = [f"**{p['name']}**"]
    if p.get("adds"):
        bits.append(f"מוסיפה {p['adds']}.")
    if p.get("gone"):
        bits.append("יצאה משלב ההפקדה לפני שהחלון נסגר.")
    elif p["left"] is not None and p["left"] >= 0:
        bits.append(f"אפשר להתנגד עד {heb(p['shuts'])}, {when_text(p['left'])}.")
    elif p["shuts"]:
        bits.append(f"החלון נסגר ב-{heb(p['shuts'])}.")
    if p.get("advertised"):
        bits.append(f"פורסמה בעיתונים ב-{p['advertised']}.")
    bits.append(f"[במבא\"ת]({p['url']})")
    return "- " + " ".join(bits)


def report(first, news, now, today):
    open_now = sorted((n for n, p in now.items()
                       if p["left"] is not None and p["left"] >= 0),
                      key=lambda n: now[n]["left"])
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
    ap.add_argument("--state", default=STATE)
    ap.add_argument("--reports", default=REPORTS)
    ap.add_argument("--today", help="YYYY-MM-DD, ברירת מחדל: היום לפי שעון ישראל")
    args = ap.parse_args()

    today = (dt.date.fromisoformat(args.today) if args.today
             else dt.datetime.now(TZ).date())
    if args.fixture:
        doc = load(args.fixture)
        rows = [f["attributes"] for f in doc.get("features", doc if isinstance(doc, list) else [])]
    else:
        rows = fetch()

    now = picture(rows, today)
    first, news, state = compare(now, load(args.state), today)
    count = sum(len(v) for v in news.values())

    written = None
    if first or count:
        os.makedirs(args.reports, exist_ok=True)
        written = os.path.join(args.reports, f"{today.isoformat()}.md")
        with open(written, "w", encoding="utf-8") as handle:
            handle.write(report(first, news, now, today))
    bp.save_json(args.state, state)

    opened = sum(1 for p in now.values() if p["left"] is not None and p["left"] >= 0)
    print(f"{len(now)} תכניות בשלב ההפקדה, {opened} פתוחות להתנגדות")
    for key, nums in news.items():
        if nums:
            print(f"  {key}: {', '.join(nums)}")
    print(f"NEWS={count}")
    if written:
        print(f"REPORT={written}")


if __name__ == "__main__":
    sys.exit(main())
