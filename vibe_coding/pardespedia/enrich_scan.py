#!/usr/bin/env python3
"""Detect NEW pardespedia articles since the last run and report what they lack.

Report-first automation: this script does NOT edit the wiki. It finds
namespace-0 articles created since the last recorded run, checks each for
missing image / missing video / thin text / missing moshava link, and writes
a Markdown report. The agentic drafting stage (cron_enrich.sh) fills in
proposed enrichment per article; this scanner just finds and triages them.

State is kept in enrich_state.json ({"last_run": "<ISO8601 UTC>"}). On the
first run (no state) it initializes to "now" so we don't dump the whole
backlog, and reports nothing.

Usage:
    python3 enrich_scan.py                 # normal cron run (updates state)
    python3 enrich_scan.py --since 2026-06-01T00:00:00Z   # override window
    python3 enrich_scan.py --no-state      # don't advance the timestamp
"""
import argparse
import datetime as dt
import json
import os
import sys

import requests

API = "https://pardespedia.info/api.php"
H = {"User-Agent": "Pardespedia-Bot/1.0 (orimosenzon@gmail.com)"}
DIR = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(DIR, "enrich_state.json")
REPORTS = os.path.join(DIR, "enrich_reports")

MOSHAVA_LINK_MARKERS = ["[[פרדס חנה", "[[כרכור]]", "[[כרכור|", "|פרדס חנה", "[[היסטוריה של פרדס"]


def now_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_last_run():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))["last_run"]
        except Exception:
            pass
    return None


def save_last_run(ts):
    json.dump({"last_run": ts}, open(STATE, "w"))


def new_pages_since(since_iso):
    """Return [(title, created_ts)] for NS0 pages newly created after since_iso."""
    out = []
    cont = None
    while True:
        params = {
            "action": "query", "list": "recentchanges",
            "rctype": "new", "rcnamespace": 0, "rcshow": "!redirect",
            "rcprop": "title|timestamp", "rclimit": 500,
            "rcend": since_iso, "rcdir": "older", "format": "json",
        }
        if cont:
            params["rccontinue"] = cont
        d = requests.get(API, params=params, headers=H, timeout=40).json()
        for c in d.get("query", {}).get("recentchanges", []):
            out.append((c["title"], c["timestamp"]))
        cont = d.get("continue", {}).get("rccontinue")
        if not cont:
            break
    return out


def inspect(title):
    """Return dict of signals for one page."""
    d = requests.get(API, params={
        "action": "query", "titles": title,
        "prop": "revisions", "rvprop": "content", "rvslots": "main",
        "format": "json",
    }, headers=H, timeout=40).json()
    p = next(iter(d["query"]["pages"].values()))
    wt = ""
    if "revisions" in p:
        wt = p["revisions"][0]["slots"]["main"]["*"]
    is_redirect = wt.lstrip().startswith("#הפניה") or wt.lstrip().upper().startswith("#REDIRECT")
    return {
        "title": title,
        "redirect": is_redirect,
        "bytes": len(wt.encode("utf-8")),
        "has_image": ("[[קובץ:" in wt) or ("[[File:" in wt),
        "has_video": "<youtube" in wt,
        "links_moshava": any(m in wt for m in MOSHAVA_LINK_MARKERS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since")
    ap.add_argument("--no-state", action="store_true")
    args = ap.parse_args()

    last_run = args.since or load_last_run()
    run_ts = now_iso()

    if last_run is None:
        # First run: initialize and report nothing (avoid backlog dump).
        if not args.no_state:
            save_last_run(run_ts)
        print(f"[enrich_scan] initialized state to {run_ts}; no report on first run.")
        print("COUNT=0")
        return 0

    pages = new_pages_since(last_run)
    rows = []
    for title, created in pages:
        try:
            info = inspect(title)
        except Exception as e:
            print(f"[enrich_scan] inspect failed for {title}: {e}", file=sys.stderr)
            continue
        if info["redirect"]:
            continue
        info["created"] = created
        rows.append(info)

    os.makedirs(REPORTS, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    path = os.path.join(REPORTS, f"enrich-{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# דוח ערכים חדשים בפרדספדיה — {stamp}\n\n")
        f.write(f"חלון: מאז `{last_run}` ועד `{run_ts}`.\n\n")
        if not rows:
            f.write("לא נמצאו ערכים חדשים בחלון הזה.\n")
        else:
            f.write(f"נמצאו **{len(rows)}** ערכים חדשים (לא כולל הפניות). ")
            f.write("סמן ✓ = קיים, ✗ = חסר. עמודת 'הצעת העשרה' תמולא בשלב האגנטי.\n\n")
            f.write("| ערך | נוצר | תווים | תמונה | סרטון | קישור למושבה |\n")
            f.write("|---|---|---|:--:|:--:|:--:|\n")
            for r in rows:
                f.write(
                    f"| [[{r['title']}]] | {r['created'][:10]} | {r['bytes']} | "
                    f"{'✓' if r['has_image'] else '✗'} | "
                    f"{'✓' if r['has_video'] else '✗'} | "
                    f"{'✓' if r['links_moshava'] else '✗'} |\n"
                )
            f.write("\n## הצעות העשרה\n\n")
            f.write("<!-- ENRICH-DRAFTS -->\n")
            f.write("*(טרם מולא — השלב האגנטי ימלא כאן הצעות טקסט, תמונות וסרטון לכל ערך.)*\n")

    print(f"[enrich_scan] {len(rows)} new articles; report: {path}")
    print(f"REPORT={path}")
    print(f"COUNT={len(rows)}")
    # advance state only after a successful report write
    if not args.no_state:
        save_last_run(run_ts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
