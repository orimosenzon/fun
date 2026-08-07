#!/usr/bin/env python3
"""Find where the em-dash habit has spread across pardespedia.

An em-dash (—) is not reachable from a Hebrew keyboard, so in Hebrew text it is
close to a signature of machine-written (or copy-pasted) prose. The bot has been
scattering them for months. This surveys the damage so the cleanup can be
planned and batched rather than guessed at.

Pulls every article's wikitext in batches of 50 and caches it under
dash_cache/ so the cleanup pass can work offline and only write back the
pages it actually changes.

    python3 dash_survey.py              # survey namespace 0, write the report
    python3 dash_survey.py --ns 0,14    # other namespaces too
"""
import argparse
import json
import os
import re

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://pardespedia.info/api.php"
H = {"User-Agent": "Pardespedia-Bot/1.0 (orimosenzon@gmail.com)"}
CACHE = os.path.join(HERE, "dash_cache")
REPORT = os.path.join(HERE, "dash_report.json")

EM = "—"   # — the one we care about
EN = "–"   # – shows up in number ranges, usually fine

# An em-dash between digits is a legitimate range (1935—1937). Everything else
# in Hebrew prose is the tell.
RANGE = re.compile(r"\d\s*[" + EM + EN + r"]\s*\d")


def fetch_all(session, ns):
    """Yield (title, wikitext) for every page in the namespace."""
    cont = {}
    while True:
        params = {
            "action": "query", "format": "json",
            "generator": "allpages", "gapnamespace": ns, "gaplimit": "50",
            "gapfilterredir": "nonredirects",
            "prop": "revisions", "rvprop": "content", "rvslots": "main",
        }
        params.update(cont)
        r = session.get(API, params=params, timeout=120)
        r.raise_for_status()
        data = r.json()
        for page in data.get("query", {}).get("pages", {}).values():
            revs = page.get("revisions")
            if not revs:
                continue
            yield page["title"], revs[0]["slots"]["main"]["*"]
        if "continue" not in data:
            break
        cont = data["continue"]


def count(text):
    """(em-dashes worth fixing, em-dashes inside number ranges)."""
    ranges = len(RANGE.findall(text))
    return text.count(EM) - len([m for m in RANGE.finditer(text) if EM in m.group()]), ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="0", help="comma-separated namespace ids")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    s = requests.Session()
    s.headers.update(H)

    rows, total_pages, total_em = [], 0, 0
    for ns in [int(x) for x in args.ns.split(",")]:
        for title, text in fetch_all(s, ns):
            total_pages += 1
            prose_em, in_range = count(text)
            safe = re.sub(r"[^\w֐-׿ -]", "_", title)[:80]
            with open(os.path.join(CACHE, safe + ".wiki"), "w", encoding="utf-8") as f:
                f.write(text)
            if prose_em:
                total_em += prose_em
                rows.append({"title": title, "ns": ns, "em": prose_em,
                             "em_in_ranges": in_range, "bytes": len(text),
                             "per_1k": round(prose_em / max(len(text), 1) * 1000, 1),
                             "cache": safe + ".wiki"})

    rows.sort(key=lambda r: -r["em"])
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"pages scanned:      {total_pages}")
    print(f"pages with em-dash: {len(rows)}")
    print(f"em-dashes in prose: {total_em}")
    print(f"\nworst 20:")
    for r in rows[:20]:
        print(f'  {r["em"]:4}  ({r["per_1k"]:4}/1k)  {r["title"]}')
    print(f"\nreport -> {REPORT}\ncache  -> {CACHE}/")


if __name__ == "__main__":
    main()
