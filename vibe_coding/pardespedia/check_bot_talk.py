#!/usr/bin/env python3
"""Detect unanswered sysop messages on the bot's own talk page.

Stateless by design: the bot's talk page IS the state. If the latest revision
was made by the bot account itself, the bot already had the last word and
there is nothing pending. If the latest revision was made by someone else,
we check whether that user is a sysop (action=query&list=users&usprop=groups)
before flagging it — messages from non-sysop accounts are logged but not
treated as actionable requests.

Usage:
    python3 check_bot_talk.py            # normal run, prints COUNT=/SYSOP=
"""
import sys

import requests

API = "https://pardespedia.info/api.php"
H = {"User-Agent": "Pardespedia-Bot/1.0 (orimosenzon@gmail.com)"}
TALK_PAGE = "שיחת_משתמש:אורי מוסנזון בוט"
BOT_NAME = "אורי מוסנזון בוט"


def latest_revision_user(title):
    d = requests.get(API, params={
        "action": "query", "titles": title,
        "prop": "revisions", "rvprop": "user|timestamp|ids", "rvlimit": 1,
        "format": "json",
    }, headers=H, timeout=40).json()
    page = next(iter(d["query"]["pages"].values()))
    if "missing" in page:
        return None, None, None
    rev = page["revisions"][0]
    return rev["user"], rev["timestamp"], rev["revid"]


def is_sysop(username):
    d = requests.get(API, params={
        "action": "query", "list": "users", "ususers": username,
        "usprop": "groups", "format": "json",
    }, headers=H, timeout=40).json()
    users = d["query"]["users"]
    if not users or "groups" not in users[0]:
        return False
    return "sysop" in users[0]["groups"]


def main():
    user, ts, revid = latest_revision_user(TALK_PAGE)
    if user is None:
        print("[check_bot_talk] talk page does not exist yet.")
        print("COUNT=0")
        return 0

    if user == BOT_NAME:
        print(f"[check_bot_talk] latest edit (rev {revid}, {ts}) is the bot's own — nothing pending.")
        print("COUNT=0")
        return 0

    if is_sysop(user):
        print(f"[check_bot_talk] pending sysop message from '{user}' (rev {revid}, {ts}), unanswered.")
        print(f"SYSOP_USER={user}")
        print(f"REV={revid}")
        print("COUNT=1")
    else:
        print(f"[check_bot_talk] latest edit by non-sysop '{user}' (rev {revid}, {ts}) — not auto-actioned.")
        print("COUNT=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
