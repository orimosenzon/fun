#!/usr/bin/env python3
"""Write a full page from a local wikitext file.

The project rule is get-then-write-whole-page, so the safe flow is:
fetch_raw.py > file, edit the file, then push it back with this.

Refuses to write over an existing page unless --replace is passed. A title that
turned out to already exist means the local file was almost certainly written
without reading what is there — which is how a good article gets silently
replaced by a thinner one. (This guard exists because that happened: "מוני
ביטן" was checked for as "מוני ביתן", came back missing, and got overwritten.)

Usage:
    python3 edit_page.py <title> <wikitext_file> "<edit summary>" [--replace]
"""
import sys

from wiki_client import WikiClient


def main():
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    title, path, summary = sys.argv[1], sys.argv[2], sys.argv[3]
    replace = "--replace" in sys.argv[4:]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    client = WikiClient()
    client.login()
    before = client.get_page(title)
    if before["exists"] and not replace:
        print(f"REFUSING: '{title}' already exists ({len(before['wikitext'])} chars). "
              f"Read it first and merge, then pass --replace.", file=sys.stderr)
        sys.exit(1)
    if not before["exists"]:
        print(f"NOTE: creating new page {title}", file=sys.stderr)
    client.edit_page(title, text, summary=summary)
    print(f"{len(before['wikitext'])} -> {len(text)} chars")


if __name__ == "__main__":
    main()
