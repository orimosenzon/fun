#!/usr/bin/env python3
"""Detect users with meaningful recent contributions and prepare a report for
the agentic stage to comment on their talk pages.

Unlike enrich_scan.py (report-first, human approves), this feeds a stage that
POSTS AUTOMATICALLY to live user talk pages — that was an explicit choice
(2026-07-11) despite the higher social/visibility risk of an unreviewed bot
comment, so the safety rails live here instead of in a human approval step:
  * excludes the bot's own account and the owner's personal account
  * requires >= MIN_EDITS new edits since that user was last processed (skip
    a lone trivial edit — not worth a comment)
  * per-user state (contributor_state.json) so the same edits are never
    re-praised
  * only counts article (ns0) and file (ns6) edits — not talk-page chatter

State: {"initialized_at": "<ISO8601 UTC>", "users": {"<username>": "<ISO8601
UTC of last processed edit>"}}. First run initializes state and reports
nothing, to avoid dumping praise for the entire edit history at once.

The scan window is a fixed lookback (LOOKBACK_DAYS), not a global cursor —
deliberately, so a user's edits keep accumulating across cycles (toward
MIN_EDITS) even in a cycle where they don't cross the threshold. Only a
user's OWN state timestamp advances, and only once they're actually selected
as a candidate (mirroring enrich_scan.py's accepted tradeoff: state advances
when the report is written, not after the posting stage confirms success —
if the later `claude -p` stage fails to post, this batch's edits won't be
retried automatically).

Usage:
    python3 contributor_scan.py
    python3 contributor_scan.py --no-state
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
STATE = os.path.join(DIR, "contributor_state.json")
REPORTS = os.path.join(DIR, "contributor_reports")

EXCLUDE_USERS = {"אורי מוסנזון בוט", "אורי מוסנזון"}
MIN_EDITS = 2
MAX_USERS_PER_RUN = 5
LOOKBACK_DAYS = 10  # comfortably > the 3-day cron cycle, so one missed run doesn't drop edits

# Policy (2026-07-25, Ori): the bot may only INITIATE talk-page contact with
# new contributors — encouragement/praise/gentle tips. Veteran editors are
# left alone here entirely; the bot only responds if THEY start a thread
# (that's cron_all_bot_talk.sh's job, not this script's). A user's total
# edit count (lifetime, not just this window) is the proxy for "new"; sysops
# are excluded outright regardless of count, since account age/edit count
# alone doesn't capture seniority for an operator.
NEW_USER_MAX_EDITS = 20


def now_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE))
        except Exception:
            pass
    return {"initialized_at": None, "users": {}}


def save_state(state):
    json.dump(state, open(STATE, "w"), ensure_ascii=False, indent=2)


def recent_changes_since(since_iso):
    """Return list of recentchanges entries for article/file edits after
    since_iso (a fixed lookback bound, not a resumption cursor)."""
    out = []
    cont = None
    while True:
        params = {
            "action": "query", "list": "recentchanges",
            "rcnamespace": "0|6", "rcshow": "!bot",
            "rcprop": "title|timestamp|user|comment|sizes|ids", "rclimit": 500,
            "rcend": since_iso, "rcdir": "older", "format": "json",
        }
        if cont:
            params["rccontinue"] = cont
        d = requests.get(API, params=params, headers=H, timeout=40).json()
        for c in d.get("query", {}).get("recentchanges", []):
            out.append(c)
        cont = d.get("continue", {}).get("rccontinue")
        if not cont:
            break
    return out


def filter_new_users(usernames):
    """Return the subset of usernames that qualify as 'new' — lifetime edit
    count <= NEW_USER_MAX_EDITS and not a sysop. Queries in one batched call
    (MediaWiki allows multiple ususers separated by '|')."""
    if not usernames:
        return set()
    params = {
        "action": "query", "list": "users",
        "ususers": "|".join(usernames),
        "usprop": "groups|editcount", "format": "json",
    }
    d = requests.get(API, params=params, headers=H, timeout=40).json()
    new_users = set()
    for u in d.get("query", {}).get("users", []):
        name = u.get("name")
        if not name:
            continue
        if "sysop" in u.get("groups", []):
            continue
        if u.get("editcount", 0) > NEW_USER_MAX_EDITS:
            continue
        new_users.add(name)
    return new_users


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-state", action="store_true")
    args = ap.parse_args()

    state = load_state()
    run_ts = now_iso()

    if state.get("initialized_at") is None:
        if not args.no_state:
            state["initialized_at"] = run_ts
            save_state(state)
        print(f"[contributor_scan] initialized state to {run_ts}; no report on first run.")
        print("COUNT=0")
        return 0

    lookback_iso = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    changes = recent_changes_since(lookback_iso)

    by_user = {}
    for c in changes:
        user = c.get("user", "")
        if not user or user in EXCLUDE_USERS:
            continue
        by_user.setdefault(user, []).append(c)

    users_state = state.setdefault("users", {})
    floor = state.get("initialized_at", lookback_iso)
    candidates = []
    for user, edits in by_user.items():
        last_seen = users_state.get(user, floor)
        new_edits = [e for e in edits if e["timestamp"] > last_seen]
        if len(new_edits) < MIN_EDITS:
            continue
        new_edits.sort(key=lambda e: e["timestamp"])
        candidates.append((user, new_edits))

    # Only new contributors get proactive praise/tips (see NEW_USER_MAX_EDITS
    # above) — veteran editors are dropped here entirely, not just capped.
    new_user_names = filter_new_users([user for user, _ in candidates])
    candidates = [(user, edits) for user, edits in candidates if user in new_user_names]

    # Most-active first; cap how many we touch in one run (cost + pacing).
    candidates.sort(key=lambda t: -len(t[1]))
    candidates = candidates[:MAX_USERS_PER_RUN]

    os.makedirs(REPORTS, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    path = os.path.join(REPORTS, f"contributors-{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# משתמשים לתגובה על תרומה — {stamp}\n\n")
        f.write(f"חלון סריקה (lookback קבוע): מאז `{lookback_iso}`.\n\n")
        if not candidates:
            f.write("לא נמצאו משתמשים עם מספיק תרומה חדשה בחלון הזה.\n")
        else:
            f.write(f"נמצאו **{len(candidates)}** משתמשים (סף: {MIN_EDITS}+ עריכות חדשות שטרם קיבלו תגובה).\n\n")
            for user, edits in candidates:
                f.write(f"## {user}\n")
                f.write(f"דף שיחה: `שיחת משתמש:{user}`\n\n")
                f.write("| דף | חותמת זמן | תקציר עריכה | הפרש גודל |\n")
                f.write("|---|---|---|--:|\n")
                for e in edits:
                    title = e.get("title", "")
                    ts = e.get("timestamp", "")
                    comment = (e.get("comment", "") or "").replace("|", "/").replace("\n", " ")
                    sizediff = e.get("newlen", 0) - e.get("oldlen", 0)
                    f.write(f"| [[{title}]] | {ts} | {comment} | {sizediff:+d} |\n")
                f.write("\n")

    print(f"[contributor_scan] {len(candidates)} users to comment on; report: {path}")
    print(f"REPORT={path}")
    print(f"COUNT={len(candidates)}")

    # Advance state ONLY for users actually selected this run, to the latest
    # timestamp among the edits we're about to comment on. Users who stayed
    # below MIN_EDITS are left untouched so their edits keep accumulating.
    if not args.no_state:
        for user, edits in candidates:
            latest = max(e["timestamp"] for e in edits)
            users_state[user] = latest
        save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
