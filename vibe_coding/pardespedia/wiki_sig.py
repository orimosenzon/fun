#!/usr/bin/env python3
"""Print the current time as a MediaWiki-style Hebrew signature timestamp,
matching the format used across pardespedia talk pages, e.g.:
    14:03, 11 ביולי 2026 (IDT)

Usage: python3 wiki_sig.py
"""
import datetime as dt

try:
    from zoneinfo import ZoneInfo
    now = dt.datetime.now(ZoneInfo("Asia/Jerusalem"))
except Exception:
    now = dt.datetime.now()

MONTHS = [
    "בינואר", "בפברואר", "במרץ", "באפריל", "במאי", "ביוני",
    "ביולי", "באוגוסט", "בספטמבר", "באוקטובר", "בנובמבר", "בדצמבר",
]

print(f"{now:%H:%M}, {now.day} {MONTHS[now.month - 1]} {now.year} (IDT)")
