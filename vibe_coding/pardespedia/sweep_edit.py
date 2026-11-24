#!/usr/bin/env python3
"""Safe get/put for the all-articles improvement sweep (23/11/2026).

    python3 sweep_edit.py get  "<title>" <out.wiki>
    python3 sweep_edit.py put  "<title>" <new.wiki> "<summary>"
    python3 sweep_edit.py diff "<title>" <new.wiki>

`get` saves the wikitext plus the base revision in <out.wiki>.meta.json and
prints who edited last and when. If a human edited the page in the last
RECENT_HOURS hours it prints SKIP-RECENT-HUMAN (leave that page alone).

`put` sends the full page with baserevid/basetimestamp from the .meta.json, so
an edit made by someone else after `get` fails with an edit conflict instead of
being overwritten. Refuses if the new text lost more than 30% of the old
length (a sign of an accidental rewrite) unless --allow-shrink is passed.
Every edit is appended to ~/.cache/pardespedia_sweep/edits.jsonl.
"""
import datetime as dt
import difflib
import json
import os
import sys

from wiki_client import WikiClient, API_URL

BOT = "אורי מוסנזון בוט"
RECENT_HOURS = 6
LOG_DIR = os.path.expanduser("~/.cache/pardespedia_sweep")
PREFIX = "סבב שיפור: "


def fetch(client, title):
    r = client.session.get(API_URL, params={
        "action": "query", "titles": title, "prop": "revisions",
        "rvprop": "ids|timestamp|user|content", "rvslots": "main",
        "rvlimit": 1, "format": "json", "formatversion": 2,
    })
    r.raise_for_status()
    page = r.json()["query"]["pages"][0]
    if page.get("missing"):
        return None
    rev = page["revisions"][0]
    return {"title": page["title"], "revid": rev["revid"],
            "timestamp": rev["timestamp"], "user": rev["user"],
            "text": rev["slots"]["main"]["content"]}


def cmd_get(title, out):
    c = WikiClient()
    p = fetch(c, title)
    if p is None:
        print("MISSING")
        sys.exit(2)
    with open(out, "w", encoding="utf-8") as f:
        f.write(p["text"])
    meta = {k: p[k] for k in ("title", "revid", "timestamp", "user")}
    meta["len"] = len(p["text"])
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(
        p["timestamp"].replace("Z", "+00:00"))
    hours = age.total_seconds() / 3600
    print(f"title={p['title']} revid={p['revid']} last_user={p['user']} "
          f"age_hours={hours:.1f} len={len(p['text'])}")
    if p["user"] != BOT and hours < RECENT_HOURS:
        print("SKIP-RECENT-HUMAN")


def cmd_put(title, path, summary, allow_shrink=False):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    meta_path = None
    for cand in (path + ".meta.json",):
        if os.path.exists(cand):
            meta_path = cand
    if meta_path is None:
        sys.exit("no .meta.json next to the file: run `get` into the same path first, then edit that file")
    meta = json.load(open(meta_path, encoding="utf-8"))
    if meta["title"].replace("_", " ") != title.replace("_", " "):
        sys.exit(f"meta is for {meta['title']!r}, not {title!r}")
    if not allow_shrink and len(text) < 0.7 * meta["len"]:
        sys.exit(f"refusing: new text {len(text)} chars vs old {meta['len']} (>30% shrink). "
                 "Pass --allow-shrink only if the cut is deliberate.")
    c = WikiClient()
    c.login()
    if not summary.startswith(PREFIX):
        summary = PREFIX + summary
    r = c.session.post(API_URL, data={
        "action": "edit", "title": title, "text": text, "summary": summary,
        "baserevid": meta["revid"], "basetimestamp": meta["timestamp"],
        "nocreate": "1", "token": c._csrf_token(), "format": "json",
    })
    r.raise_for_status()
    res = r.json()
    if "error" in res:
        sys.exit(f"EDIT FAILED: {res['error'].get('code')}: {res['error'].get('info')}")
    e = res.get("edit", {})
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "edits.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": dt.datetime.now().isoformat(timespec="seconds"),
                            "title": title, "summary": summary,
                            "old": e.get("oldrevid"), "new": e.get("newrevid"),
                            "nochange": "nochange" in e,
                            "delta": len(text) - meta["len"]}, ensure_ascii=False) + "\n")
    print(f"OK {e.get('result')} old={e.get('oldrevid')} new={e.get('newrevid')}"
          + (" (no change)" if "nochange" in e else ""))
    # the base is now our revision, so a second put needs a fresh get
    if "nochange" not in e:
        os.remove(meta_path)


def cmd_diff(title, path):
    meta = json.load(open(path + ".meta.json", encoding="utf-8"))
    c = WikiClient()
    p = fetch(c, title)
    new = open(path, encoding="utf-8").read().splitlines()
    for line in difflib.unified_diff(p["text"].splitlines(), new, "live", "new", lineterm="", n=1):
        print(line)


if __name__ == "__main__":
    a = sys.argv[1:]
    shrink = "--allow-shrink" in a
    a = [x for x in a if x != "--allow-shrink"]
    if len(a) >= 3 and a[0] == "get":
        cmd_get(a[1], a[2])
    elif len(a) >= 4 and a[0] == "put":
        cmd_put(a[1], a[2], a[3], shrink)
    elif len(a) >= 3 and a[0] == "diff":
        cmd_diff(a[1], a[2])
    else:
        print(__doc__)
        sys.exit(1)
