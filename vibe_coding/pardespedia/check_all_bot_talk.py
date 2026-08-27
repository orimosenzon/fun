#!/usr/bin/env python3
"""Detect unanswered messages on ANY user-talk page the bot has participated in.

check_bot_talk.py only ever watched the bot's own talk page
(שיחת_משתמש:אורי מוסנזון בוט). That misses replies posted on OTHER users' talk
pages — e.g. cron_contributor_comments.sh posts appreciation/questions on a
contributor's own talk page (שיחת_משתמש:חוה אהרון), and if she replies there,
the old check never saw it, because the conversation doesn't live on the
bot's page.

Approach: scan recent changes in the User-talk namespace (ns=3) within a
lookback window, group by page title, and for each page where the bot has a
prior revision, check whether the CURRENT latest revision on that page is by
someone other than the bot. If so, it's a pending reply.

Stateless like check_bot_talk.py: "who has the last word on this page" is the
only state that matters, no separate tracking file needed.

Usage:
    python3 check_all_bot_talk.py [--hours N]   # default 26h lookback
"""
import argparse
import sys

import requests

API = "https://pardespedia.info/api.php"
H = {"User-Agent": "Pardespedia-Bot/1.0 (orimosenzon@gmail.com)"}
BOT_NAME = "אורי מוסנזון בוט"
USER_TALK_NS = 3


# Edit-summary prefixes of the bot's own maintenance edits. These touch a talk
# page without saying anything to anybody, so they must not count as "the bot
# had the last word". Keep in sync with archive_bot_talk.py's summaries.
HOUSEKEEPING_PREFIXES = ("ארכוב אוטומטי", "רענון תיבת הארכיונים")


def is_housekeeping(comment):
    return comment.strip().startswith(HOUSEKEEPING_PREFIXES)


def recent_talk_pages(hours):
    """Titles of User-talk pages touched in the last `hours`, most-recent first, deduped."""
    r = requests.get(API, params={
        "action": "query", "list": "recentchanges",
        "rcnamespace": USER_TALK_NS, "rcprop": "title|timestamp",
        "rclimit": 200, "rcshow": "!bot",  # rcshow filter is best-effort; we still de-dup below
        "format": "json",
    }, headers=H, timeout=40).json()
    changes = r["query"]["recentchanges"]
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    seen = []
    for rc in changes:
        ts = datetime.fromisoformat(rc["timestamp"].replace("Z", "+00:00"))
        if ts < cutoff:
            continue
        title = rc["title"]
        if title not in seen:
            seen.append(title)
    return seen


def page_revisions_user_check(title):
    """Return (latest_user, latest_ts, latest_revid, bot_has_prior_revision).

    Housekeeping edits by the bot are skipped when deciding who spoke last.
    archive_bot_talk.py edits the bot's own talk page daily, and such an edit
    lands *after* a question without answering it — which used to make the
    bot look like the last speaker and hid the pending question entirely.
    (That happened on 2026-08-26: a question at 08:25 was masked by an
    archiving edit 13 minutes later and never got picked up.)
    """
    r = requests.get(API, params={
        "action": "query", "titles": title,
        "prop": "revisions", "rvprop": "user|timestamp|ids|comment", "rvlimit": 50,
        "format": "json",
    }, headers=H, timeout=40).json()
    page = next(iter(r["query"]["pages"].values()))
    if "missing" in page or "revisions" not in page:
        return None, None, None, False
    revs = page["revisions"]
    bot_has_prior = any(rv["user"] == BOT_NAME for rv in revs)
    for rv in revs:                       # newest first
        if rv["user"] == BOT_NAME and is_housekeeping(rv.get("comment", "")):
            continue
        return rv["user"], rv["timestamp"], rv["revid"], bot_has_prior
    return None, None, None, bot_has_prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=26.0)
    args = ap.parse_args()

    pending = []
    for title in recent_talk_pages(args.hours):
        user, ts, revid, bot_has_prior = page_revisions_user_check(title)
        if user is None or user == BOT_NAME:
            continue
        if not bot_has_prior:
            # The bot never participated on this page — out of scope for this
            # check (check_bot_talk.py's own-page check already covers the
            # bot's own talk page even on a first-ever message).
            continue
        pending.append((title, user, ts, revid))

    if not pending:
        print("[check_all_bot_talk] no pending replies on any bot-participated talk page.")
        print("COUNT=0")
        return 0

    print(f"[check_all_bot_talk] {len(pending)} pending repl{'y' if len(pending)==1 else 'ies'}:")
    for title, user, ts, revid in pending:
        print(f"  - {title} :: last edit by '{user}' at {ts} (rev {revid})")
    print(f"COUNT={len(pending)}")
    print("PAGES=" + "|".join(t for t, *_ in pending))
    return 0


if __name__ == "__main__":
    sys.exit(main())
