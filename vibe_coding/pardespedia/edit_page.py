#!/usr/bin/env python3
"""Write a full page from a local wikitext file, or show the current one.

The project rule is get-then-write-whole-page, so the safe flow is:
fetch_raw.py > file, edit the file, then push it back with this.

Usage:
    python3 edit_page.py <title> <wikitext_file> "<edit summary>"
"""
import sys

from wiki_client import WikiClient


def main():
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    title, path, summary = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    client = WikiClient()
    client.login()
    before = client.get_page(title)
    if not before["exists"]:
        print(f"NOTE: creating new page {title}", file=sys.stderr)
    client.edit_page(title, text, summary=summary)
    print(f"{len(before['wikitext'])} -> {len(text)} chars")


if __name__ == "__main__":
    main()
