#!/usr/bin/env python3
"""Bookkeeping for the recurring all-articles sweep (every 14 days).

Ori asked (24/11/2026) that the one-off sweep of 23/11/2026 (every article
checked against the web and improved) repeat every two weeks. A full sweep is
~31 agent batches, too much for one night, so cron_sweep.sh runs daily and
does a few batches per night. This script holds the state between nights:

    python3 sweep_cron.py next          # print BATCH=<name> FILE=<json> RUN=<dir>, or NONE
    python3 sweep_cron.py done <batch>  # mark a batch finished
    python3 sweep_cron.py fail <batch>  # count a failed attempt (3 strikes -> skipped)
    python3 sweep_cron.py status        # human-readable state

`next` starts a new run when there is no active run and the last run started
at least INTERVAL_DAYS ago. A run's batches are the thematic grouping in
sweep_batches.json (the grouping of the first sweep), minus pages that no
longer exist, plus every new article in "new-N" batches. When the last batch
is done it writes SUMMARY.md for the run (the "for Ori" sections of all the
batch reports, gathered in one file).

State and run files live outside the repo, in ~/.cache/pardespedia_sweep/cron/.
"""
import datetime as dt
import json
import os
import re
import sys

from wiki_client import WikiClient

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.expanduser("~/.cache/pardespedia_sweep/cron")
STATE = os.path.join(BASE, "state.json")
GROUPS = os.path.join(HERE, "sweep_batches.json")
INTERVAL_DAYS = 14
NEW_BATCH_SIZE = 35
MAX_FAILS = 3
# The one-off sweep this repeats; the first automatic run is 14 days after it.
FIRST_SWEEP = "2026-11-23"
# Never part of a sweep: navigation pages, not articles.
SKIP = {"עמוד ראשי", "Pardespedia.info:Sidebar", "ארגז חול נוסף", "דוגמא לערך שעדיין לא קיים"}
# Interview transcripts are verbatim speech (see CLAUDE.md): never "improved".
SKIP_PREFIXES = ("תמלול ראיון",)


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"last_start": FIRST_SWEEP, "last_end": FIRST_SWEEP, "active": None}


def save_state(s):
    os.makedirs(BASE, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE)


def build_run(today, prev_date):
    with open(GROUPS, encoding="utf-8") as f:
        groups = json.load(f)
    live = {p["title"] for p in WikiClient().list_pages(redirects="nonredirects")}
    live = {t for t in live - SKIP if not t.startswith(SKIP_PREFIXES)}
    run_dir = os.path.join(BASE, "runs", today)
    os.makedirs(os.path.join(run_dir, "batches"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "reports"), exist_ok=True)
    batches, seen = [], set()
    for name, titles in groups.items():
        keep = [t for t in titles if t in live]
        seen.update(titles)
        if keep:
            batches.append((name, keep))
    new = sorted(live - seen)
    for i in range(0, len(new), NEW_BATCH_SIZE):
        batches.append((f"new-{i // NEW_BATCH_SIZE + 1}", new[i:i + NEW_BATCH_SIZE]))
    for name, titles in batches:
        with open(os.path.join(run_dir, "batches", name + ".json"), "w", encoding="utf-8") as f:
            json.dump(titles, f, ensure_ascii=False)
    return {"id": today, "dir": run_dir, "prev": prev_date,
            "pending": [b for b, _ in batches], "done": [], "skipped": [], "fails": {},
            "articles": sum(len(t) for _, t in batches), "new_articles": len(new)}


def cmd_next():
    s = load_state()
    today = dt.date.today().isoformat()
    if not s["active"]:
        age = (dt.date.today() - dt.date.fromisoformat(s["last_start"])).days
        if age < INTERVAL_DAYS:
            print(f"NONE (next run in {INTERVAL_DAYS - age} days)")
            return
        s["active"] = build_run(today, s["last_start"])
        s["last_start"] = today
        save_state(s)
        a = s["active"]
        print(f"NEW-RUN {a['id']}: {len(a['pending'])} batches, {a['articles']} articles "
              f"({a['new_articles']} not in the base grouping)")
    a = s["active"]
    if not a["pending"]:
        print("NONE (run finished)")
        return
    b = a["pending"][0]
    print(f"BATCH={b}")
    print(f"FILE={os.path.join(a['dir'], 'batches', b + '.json')}")
    print(f"REPORT={os.path.join(a['dir'], 'reports', b + '.md')}")
    print(f"WORK={os.path.join(a['dir'], 'work', b)}")
    print(f"PREV={a['prev']}")


def finish_if_empty(s):
    a = s["active"]
    if a["pending"]:
        return
    write_summary(a)
    s["last_end"] = dt.date.today().isoformat()
    s.setdefault("history", []).append(
        {k: a[k] for k in ("id", "articles", "done", "skipped")})
    s["active"] = None
    print(f"RUN-FINISHED {a['id']} -> {os.path.join(a['dir'], 'SUMMARY.md')}")


def cmd_done(batch):
    s = load_state()
    a = s["active"]
    if not a or batch not in a["pending"]:
        print(f"not pending: {batch}")
        return
    a["pending"].remove(batch)
    a["done"].append(batch)
    finish_if_empty(s)
    save_state(s)


def cmd_fail(batch):
    s = load_state()
    a = s["active"]
    if not a or batch not in a["pending"]:
        print(f"not pending: {batch}")
        return
    a["fails"][batch] = a["fails"].get(batch, 0) + 1
    a["pending"].remove(batch)
    if a["fails"][batch] >= MAX_FAILS:
        a["skipped"].append(batch)
        print(f"SKIPPED {batch} after {MAX_FAILS} failures")
    else:
        a["pending"].append(batch)  # retry later, after the others
    finish_if_empty(s)
    save_state(s)


def section(text, heading):
    m = re.search(r"^## " + re.escape(heading) + r"\s*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""


def write_summary(a):
    rep_dir = os.path.join(a["dir"], "reports")
    edits = checked = 0
    parts = {h: [] for h in ("לבירור אצל אורי", "הצעות לדפים אחרים", "רעיונות לערכים חדשים")}
    for b in a["done"]:
        path = os.path.join(rep_dir, b + ".md")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        lines = [l for l in text.splitlines() if l.startswith("* [[")]
        checked += len(lines)
        edits += sum(1 for l in lines if "אין חדש" not in l and "דולג" not in l)
        for h in parts:
            body = section(text, h)
            if body:
                parts[h].append(f"### {b}\n{body}")
    out = [f"# סבב שיפור תקופתי {a['id']}", "",
           f"אצוות: {len(a['done'])} הושלמו, {len(a['skipped'])} דולגו "
           f"({', '.join(a['skipped']) or 'אין'}). ערכים בדוחות: {checked}, מתוכם עם שינוי: {edits}.",
           f"הדוחות המלאים: `{rep_dir}/`", ""]
    for h, items in parts.items():
        out += [f"## {h}", ""] + (items or ["אין."]) + [""]
    with open(os.path.join(a["dir"], "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def cmd_status():
    s = load_state()
    print(f"last run started {s['last_start']}, ended {s['last_end']}")
    a = s["active"]
    if a:
        print(f"active run {a['id']}: done {len(a['done'])}, pending {len(a['pending'])}, "
              f"skipped {len(a['skipped'])}; dir {a['dir']}")
    else:
        age = (dt.date.today() - dt.date.fromisoformat(s["last_start"])).days
        print(f"no active run; next in {max(0, INTERVAL_DAYS - age)} days")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "next":
        cmd_next()
    elif cmd == "done":
        cmd_done(sys.argv[2])
    elif cmd == "fail":
        cmd_fail(sys.argv[2])
    else:
        cmd_status()
