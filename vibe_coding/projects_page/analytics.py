#!/usr/bin/env python3
"""Pull page views from Google Analytics into stats.json for the landing page.

Every GA page path is credited to the card whose app lives under it (longest
prefix wins), so the page can show visits per project. stats.json is not in
git: the Pages workflow regenerates it on every deploy and on a schedule.

Credentials, first found wins:
  GA_SA_KEY                        the service-account JSON itself (CI secret)
  GOOGLE_APPLICATION_CREDENTIALS   path to that JSON
  ~/.config/vc-analytics/ga-reader.json

    python3 projects_page/analytics.py           # write stats.json
    python3 projects_page/analytics.py --print   # also print the table
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from build import DATA, ROOT, assign_anchors

PROPERTY = "562055020"  # GA4 property "Vibe coding", stream G-SJEBVTNZFY
SINCE = "2026-11-03"    # the day the tag went live
SITE = "/fun/vibe_coding/"
OUT = Path(__file__).resolve().parent / "stats.json"
RANGES = [("d7", "6daysAgo"), ("d30", "29daysAgo"), ("all", SINCE)]
KEY_FILE = Path.home() / ".config/vc-analytics/ga-reader.json"


def credentials():
    from google.oauth2 import service_account

    scopes = ["https://www.googleapis.com/auth/analytics.readonly"]
    if os.environ.get("GA_SA_KEY"):
        info = json.loads(os.environ["GA_SA_KEY"])
        return service_account.Credentials.from_service_account_info(info, scopes=scopes)
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or KEY_FILE
    return service_account.Credentials.from_service_account_file(str(path), scopes=scopes)


def run_report(session, dimensions, metrics):
    """One runReport over all RANGES. Returns rows as (dims, range index, values)."""
    body = {
        "dateRanges": [{"startDate": s, "endDate": "today", "name": n} for n, s in RANGES],
        "dimensions": [{"name": d} for d in dimensions],
        "metrics": [{"name": m} for m in metrics],
        "limit": 100000,
    }
    r = session.post(
        f"https://analyticsdata.googleapis.com/v1beta/properties/{PROPERTY}:runReport",
        json=body, timeout=60,
    )
    r.raise_for_status()
    names = [n for n, _ in RANGES]
    out = []
    for row in r.json().get("rows", []):
        dims = [d["value"] for d in row["dimensionValues"]]
        # with several date ranges GA appends the range name as a last dimension
        idx = names.index(dims.pop())
        out.append((dims, idx, [int(float(m["value"])) for m in row["metricValues"]]))
    return out


def normalize(path: str) -> str:
    """/fun/vibe_coding/fable/index.html → /fun/vibe_coding/jetbikes/fable/

    Decodes Hebrew paths, drops index.html and .html, and follows the
    top-level symlinks kept for old shared links."""
    path = unquote(path.split("?")[0])
    if path.endswith("index.html"):
        path = path[: -len("index.html")]
    if path.startswith(SITE):
        rest = path[len(SITE):]
        top, _, tail = rest.partition("/")
        link = ROOT / top
        if top and link.is_symlink():
            target = os.path.relpath(link.resolve(), ROOT)
            path = SITE + target + ("/" + tail if tail or rest.endswith("/") else "")
    return path.removesuffix(".html")


def card_prefixes(data: dict) -> dict:
    """anchor → the path its app is served under, for cards hosted here."""
    out = {}
    for s in data["sections"]:
        for p in s["projects"]:
            if p.get("dir"):
                out[p["_anchor"]] = SITE + p["dir"].strip("/") + "/"
                continue
            for l in p["links"]:
                href = unquote(l["href"])
                if href.startswith(("http:", "https:")):
                    continue
                if href.endswith("/") or href.endswith("index.html"):
                    out[p["_anchor"]] = SITE + href.rsplit("/", 1)[0] + "/"
                else:
                    out[p["_anchor"]] = (SITE + href).removesuffix(".html")
                break
    return out


def owner(path: str, prefixes: dict):
    best = None
    for anchor, pre in prefixes.items():
        hit = path.startswith(pre) if pre.endswith("/") else path == pre or path == pre + "/"
        if hit and (best is None or len(pre) > len(prefixes[best])):
            best = anchor
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", help="print views per card")
    args = ap.parse_args()

    from google.auth.transport.requests import AuthorizedSession

    data = json.loads(DATA.read_text(encoding="utf-8"))
    assign_anchors(data)
    prefixes = card_prefixes(data)
    session = AuthorizedSession(credentials())

    n = len(RANGES)
    apps, home, other = {}, [0] * n, [0] * n
    for (path,), i, (views,) in run_report(session, ["pagePath"], ["screenPageViews"]):
        path = normalize(path)
        anchor = owner(path, prefixes)
        if anchor:
            apps.setdefault(anchor, [0] * n)[i] += views
        elif path in (SITE, SITE.rstrip("/")):
            home[i] += views
        else:
            other[i] += views

    totals = {"views": [0] * n, "users": [0] * n}
    for _, i, (views, users) in run_report(session, [], ["screenPageViews", "totalUsers"]):
        totals["views"][i], totals["users"][i] = views, users

    stats = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "since": SINCE,
        "ranges": [name for name, _ in RANGES],
        "apps": dict(sorted(apps.items(), key=lambda kv: -kv[1][-1])),
        "home": home,
        "other": other,
        "totals": totals,
    }
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}: {len(apps)} apps, {totals['views'][-1]} views since {SINCE}")
    if args.print:
        for anchor, v in stats["apps"].items():
            print(f"  {anchor:28} 7d {v[0]:6}  30d {v[1]:6}  all {v[2]:6}")
        print(f"  {'(דף הפרויקטים)':28} 7d {home[0]:6}  30d {home[1]:6}  all {home[2]:6}")
        print(f"  {'(אחר)':28} 7d {other[0]:6}  30d {other[1]:6}  all {other[2]:6}")


if __name__ == "__main__":
    sys.exit(main())
