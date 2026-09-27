#!/usr/bin/env python3
"""What מבא"ת knows about a plan's objection window that Xplan does not.

Why this exists
---------------
Xplan (the map service build_plans.py and the app read) carries a plan's last
day for objections in `pl_last_deposit_date`. On 27/9/2026 thirty-four plans in
the moshava were at the deposit stage and five had that date. The other
twenty-nine were not all closed. Reading each one's page on מבא"ת showed three
kinds of plan hiding behind an empty field:

  * plans of the district committee whose newspaper notice is not recorded
    yet. The sign is up on the plot, the notice is in רשומות, מבא"ת takes
    objections online (`CAN_SUBMIT_OPPN`), and Xplan says nothing. 308-1549039,
    הגליל 34, had had its sign up for eight weeks;
  * plans of the local committee (`pl_by_auth_of` 3). The local committee
    records nothing on מבא"ת but the web notice and רשומות: no newspaper, no
    sign, no deadline, and it never moves the plan on when the window shuts.
    Their windows are invisible in every national service. 308-1395748 was
    re-published on 23/8/2026 and was open that day;
  * plans long past their window whose stage nobody updated since 2023.

What the law says, and how a date is worked out from it
-------------------------------------------------------
Section 102 of the Planning and Building Law: objections within two months of
the notice, counted from the *later* of the newspaper publications, and the
institution may extend it to three. Section 1ג(ב): the web notice goes up with
the *first* newspaper publication. So, best evidence first:

    OPPN_END_DATE on מבא"ת            the date itself
    newspapers (4410) recorded        latest + 2 months + 2 days, estimated
    web notice (7890) / sign (4415)   latest + 2 months + 2 days, "not before"

The two days are what the recorded dates show: Xplan's deadline is the first
recorded newspaper day plus two months plus two, on every plan that has both
(the later paper, usually the local weekly, comes out a couple of days after).
An estimate is labelled as one all the way to the screen.

How it is read
--------------
מבא"ת's JSON (`/rest/api/SV4/1?mid=`) wants a reCAPTCHA v3 token, the invisible
kind the page mints for itself in any ordinary browser, with no challenge. So
this opens each plan's page in Chrome, the way a person would, and keeps the
answer the page itself asks for. Nothing is solved or bypassed; it is one page
per plan in the deposit stage, once a day, two seconds apart.
"""

import datetime as dt
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".watch", "mavat")
PAGE = "https://mavat.iplan.gov.il/SV4/1/{mid}/310"

DEPOSIT_STAGE = "הפקדה להתנגדויות/השגות"

# rsInternet codes (LIS_CODE) and what they are.
NEWSPAPERS = 4410        # פרסום להפקדה בעיתונים
RESHUMOT = 4400          # פרסום להפקדה ברשומות
SIGN = 4415              # פרסום נוסח ההפקדה על גבי שלט בתחום התכנית
WEB = 7890               # פרסום נוסח הודעה בדבר הפקדת תכנית באתר אינטרנט
OBJECTED = 34301008      # הוגשו התנגדויות


def mid_of(url):
    m = re.search(r"/SV4/1/(\d+)/", url or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------- reading

def fetch(mids, pause=2.0, log=print):
    """{mid: the plan's JSON as the page receives it}. Failures are left out."""
    from playwright.sync_api import sync_playwright

    out = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context()
        # The page embeds a govmap map that keeps the network busy for a minute
        # and has nothing to do with the dates.
        ctx.route(re.compile(r"govmap\.gov\.il|fonts\.(googleapis|gstatic)"),
                  lambda route: route.abort())
        page = ctx.new_page()
        for mid in mids:
            try:
                with page.expect_response(
                        lambda r: "rest/api/SV4/1?mid=" in r.url, timeout=45000) as info:
                    page.goto(PAGE.format(mid=mid), wait_until="commit", timeout=45000)
                doc = info.value.json()
                out[mid] = keep(doc)
                save(mid, out[mid])
            except Exception as err:                    # noqa: BLE001 - one plan, not the run
                log(f"  מבא\"ת {mid}: {str(err)[:90]}")
            time.sleep(pause)
        browser.close()
    return out


def keep(doc):
    """The part of the page's JSON this needs, so the cache stays small."""
    return {
        "status": doc.get("unifiedStatus"),
        "statusDate": doc.get("statusDate"),
        "openOpp": doc.get("rsOpenOpp") or {},
        "internet": [{"date": r.get("EIS_DATE"), "code": int(r.get("LIS_CODE") or 0),
                      "desc": r.get("LIS_DESC")}
                     for r in (doc.get("rsInternet") or [])],
        "blocks": [{"gush": b.get("BLOCKS"), "whole": b.get("PARCELS_WHOLE") or "",
                    "partial": b.get("PARCELS_PARTIAL") or ""}
                   for b in (doc.get("rsBlocks") or [])],
        "auth": (doc.get("planDetails") or {}).get("AUTH"),
        "address": [f"{r.get('STREET_NAME') or ''} {r.get('HOUSE_NUMBER') or ''}".strip()
                    for r in (doc.get("rsLocation") or []) if r.get("STREET_NAME")],
        "where": (doc.get("locationDesc") or "").strip(),
    }


def save(mid, rec):
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, f"{mid}.json"), "w", encoding="utf-8") as handle:
        json.dump(rec, handle, ensure_ascii=False, indent=1)


def cached(mid):
    try:
        with open(os.path.join(CACHE, f"{mid}.json"), encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------- the window

def day(text):
    """'03/08/2026' or '20260803' as a date, or None."""
    text = (text or "").strip()
    for fmt in ("%d/%m/%Y", "%Y%m%d"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def plus_two_months(d):
    """The same day two months on (31/7 -> 30/9: the month's last day wins),
    and the two days the recorded deadlines show on top of it."""
    month = d.month + 2
    year = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    last = [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return dt.date(year, month, min(d.day, last)) + dt.timedelta(days=2)


def latest(rec, code):
    days = [day(r["date"]) for r in rec.get("internet", []) if r["code"] == code]
    days = [d for d in days if d]
    return max(days) if days else None


def window(rec, today):
    """What the page says about the plan's window, as of `today`.

    {"state": "open" | "closed" | "none", "shuts": date or None,
     "exact": bool, "basis": sentence, "published": date or None, ...}
    "none" is a plan with no deposit notice at all yet.
    """
    opp = rec.get("openOpp") or {}
    end = day(opp.get("OPPN_END_DATE"))
    can = bool(opp.get("CAN_SUBMIT_OPPN"))
    in_stage = (rec.get("status") or "") == DEPOSIT_STAGE
    papers = latest(rec, NEWSPAPERS)
    web = latest(rec, WEB)
    sign = latest(rec, SIGN)
    reshumot = latest(rec, RESHUMOT)
    out = {"published": papers or web or sign, "papers": papers, "web": web,
           "sign": sign, "reshumot": reshumot, "can_submit": can,
           "objected": latest(rec, OBJECTED)}

    if end:
        # The date מבא"ת itself shows. It stays "can submit" for weeks after
        # the date, until someone moves the plan on, so the date decides.
        return dict(out, state="open" if end >= today else "closed", shuts=end,
                    exact=True, basis="המועד האחרון כפי שמופיע במבא\"ת")
    if papers:
        shuts = plus_two_months(papers)
        return dict(out, state="open" if shuts >= today and in_stage else "closed",
                    shuts=shuts, exact=False,
                    basis=f"חודשיים מהפרסום בעיתונים ב-{papers:%d/%m/%Y} (סעיף 102 לחוק); "
                          "המועד המדויק בנוסח ההודעה")
    start = max(d for d in (web, sign) if d) if (web or sign) else None
    if not start:
        return dict(out, state="none", shuts=None, exact=False, basis="")
    shuts = plus_two_months(start)
    # A district plan whose page takes objections and has no date: open, and
    # the window cannot shut before two months from the first notice. A local
    # plan's page never takes them, so the notice date is all there is.
    is_open = in_stage and (shuts >= today or can)
    what = "הפרסום באינטרנט" if start == web else "תליית השלט"
    return dict(out, state="open" if is_open else "closed", shuts=shuts, exact=False,
                basis=f"מועד הפרסום בעיתונים לא נרשם. לפי {what} ב-{start:%d/%m/%Y}, "
                      "החלון נסגר חודשיים אחרי הפרסום המאוחר בעיתונים, ולא לפני התאריך הזה")


# ---------------------------------------------------------------- parcels

def numbers(text):
    """'1 - 2, 4, 110 - 111' -> [1, 2, 4, 110, 111]."""
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 0 <= b - a <= 400:
                out.extend(range(a, b + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


def listed_parcels(rec):
    """([(gush, parcel) whole], [(gush, parcel) partial]) as the plan lists them."""
    whole, partial = [], []
    for b in rec.get("blocks", []):
        try:
            g = int(b["gush"])
        except (TypeError, ValueError):
            continue
        whole += [(g, p) for p in numbers(b["whole"])]
        partial += [(g, p) for p in numbers(b["partial"])]
    return whole, partial
