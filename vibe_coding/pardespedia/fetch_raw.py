#!/usr/bin/env python3
"""Fetch raw wikitext of a page to stdout (no headers), for safe targeted edits.

Usage: python3 fetch_raw.py "שם הדף" > out.wiki
"""
import sys
from wiki_client import WikiClient

if len(sys.argv) < 2:
    print("usage: fetch_raw.py <title>", file=sys.stderr)
    sys.exit(1)

page = WikiClient().get_page(sys.argv[1])
if not page["exists"]:
    print(f"PAGE DOES NOT EXIST: {sys.argv[1]}", file=sys.stderr)
    sys.exit(2)
sys.stdout.write(page["wikitext"])
