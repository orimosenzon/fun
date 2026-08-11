#!/usr/bin/env python3
"""Keep "האינדקס של הדפים" in sync with all main-namespace pages.

Idempotent: finds pages that exist but are not yet listed in the index table,
builds a row for each (description from the lead sentence + a representative
image — the page's own first image, else a generic image chosen by category),
and inserts it in the correct alphabetical position WITHOUT touching existing
rows. Running it again after new pages are created simply tops up the index.

Usage:
    python3 update_index.py            # dry run — shows what would be added
    python3 update_index.py --write    # actually edit the index page
"""
import re
import sys
from wiki_client import WikiClient

INDEX_TITLE = "האינדקס של הדפים"
DEFAULT_IMG = "תמונה כללית - השדרה בפרדס חנה כרכור.jpg"

# Keyword (matched against the page's categories + title) -> generic image.
# First match wins; order matters (most specific first).
GENERIC_IMG_RULES = [
    ("רחוב", "תמונה כללית - רחוב בפרדס חנה.jpg"),
    ("שכונ", "תמונה כללית - שכונה בפרדס חנה.jpg"),
    ("בית ספר", "תמונה כללית - בית ספר.jpg"),
    ("חינוך", "תמונה כללית - בית ספר.jpg"),
    ("שעשוע", "תמונה כללית - גן משחקים.jpg"),
    ("משחק", "תמונה כללית - גן משחקים.jpg"),
    ("בנק", "תמונה כללית - בנק.jpg"),
    ("בריאות", "תמונה כללית - קופת חולים.jpg"),
    ("קופת חולים", "תמונה כללית - קופת חולים.jpg"),
    ("מיחזור", "תמונה כללית - מיחזור.jpg"),
    ("סביבה", "תמונה כללית - מיחזור.jpg"),
    ("עלמין", "תמונה כללית - חורש בפרדס חנה.jpg"),
    ("טבע", "תמונה כללית - חורש בפרדס חנה.jpg"),
    ("תרבות", "תמונה כללית - אירוע באמפיתאטרון.jpg"),
    ("תיאטרון", "תמונה כללית - אירוע באמפיתאטרון.jpg"),
    ("מוזיק", "תמונה כללית - אירוע באמפיתאטרון.jpg"),
    ("אירוע", "תמונה כללית - אירוע באמפיתאטרון.jpg"),
]


def first_own_image(wikitext: str) -> str | None:
    m = re.search(r"\[\[(?:קובץ|File|תמונה):([^\|\]\n]+)", wikitext)
    return m.group(1).strip() if m else None


def pick_image(wikitext: str, cats: list[str], title: str) -> str:
    own = first_own_image(wikitext)
    if own:
        return own
    haystack = " ".join(cats) + " " + title
    for kw, img in GENERIC_IMG_RULES:
        if kw in haystack:
            return img
    return DEFAULT_IMG


def make_description(wikitext: str) -> str:
    """First prose sentence of the page, stripped of markup, for a table cell."""
    t = wikitext
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)        # html comments
    t = re.sub(r"\{\{.*?\}\}", " ", t, flags=re.S)        # templates
    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.S)  # refs
    t = re.sub(r"<[^>]+>", " ", t)                         # other html tags
    t = re.sub(r"\[\[(?:קובץ|File|תמונה):[^\]]*\]\]", " ", t)  # file links
    t = re.sub(r"\[\[[^\]\|]*\|([^\]]+)\]\]", r"\1", t)    # [[a|b]] -> b
    t = re.sub(r"\[\[([^\]]+)\]\]", r"\1", t)              # [[a]] -> a
    t = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", t)   # [url text] -> text
    t = t.replace("'''", "").replace("''", "")            # bold/italic
    t = t.replace("|", "／")                               # never break the cell
    # first non-empty prose line
    line = ""
    for raw in t.splitlines():
        s = raw.strip()
        if not s or s.startswith(("=", "{", "}", "!", "*", "#", "__")):
            continue
        line = s
        break
    line = re.sub(r"\s+", " ", line).strip()
    # cut at first sentence end, but not on the dots inside dates/abbreviations
    m = re.search(r"[.!?](?:\s|$)", line)
    if m and m.start() > 25:
        line = line[: m.start()].strip()
    if len(line) > 220:
        line = line[:217].rstrip() + "…"
    return line or "(דף ללא תיאור עדיין)"


def make_row(title: str, wikitext: str, cats: list[str]) -> str:
    desc = make_description(wikitext)
    img = pick_image(wikitext, cats, title)
    return f"| [[{title}]] || {desc} || [[קובץ:{img}|ממוזער|120px]]"


def norm(title: str) -> str:
    """MediaWiki title as the index and the API would both spell it."""
    return title.replace("_", " ").strip()


def parse_table(wikitext: str):
    """Return (preamble, rows, postamble).

    rows: list of (title, block) where block is the verbatim cell text of a row.
    preamble: everything up to (not including) the first row's `|-`.
    postamble: the closing `|}` and anything after it.
    """
    start = wikitext.index("{|")
    end = wikitext.index("|}", start)
    table = wikitext[start:end]
    postamble = wikitext[end:]
    # split on row separators `|-` that sit on their own line
    parts = re.split(r"\n\|-\s*\n", table)
    preamble_table = parts[0]  # `{| ...` + header
    rows = []
    for block in parts[1:]:
        block = block.rstrip("\n")
        m = re.search(r"\[\[([^\]\|]+)", block)
        title = m.group(1).strip() if m else block[:40]
        rows.append((title, block))
    preamble = wikitext[:start] + preamble_table
    return preamble, rows, postamble


def build(wikitext: str, new_rows: list[tuple[str, str]], drop: set[str] | None = None) -> str:
    preamble, rows, postamble = parse_table(wikitext)
    if drop:
        rows = [(t, b) for t, b in rows if t not in drop]
    for title, block in sorted(new_rows):
        pos = len(rows)
        for i, (t, _) in enumerate(rows):
            if t > title:
                pos = i
                break
        rows.insert(pos, (title, block))
    body = "\n|-\n".join(block for _, block in rows)
    return f"{preamble}\n|-\n{body}\n{postamble}"


def main():
    write = "--write" in sys.argv
    c = WikiClient()
    idx_wt = c.get_page(INDEX_TITLE)["wikitext"]
    _, existing_rows, _ = parse_table(idx_wt)
    indexed = {t for t, _ in existing_rows}

    # redirects are not entries: they carry no text, so they would land in the
    # index as "(דף ללא תיאור עדיין)" rows pointing at a page already listed
    all_pages = [p["title"] for p in c.list_pages(namespace=0, redirects="nonredirects")]
    live = {norm(t) for t in all_pages}
    missing = [t for t in all_pages if norm(t) not in {norm(x) for x in indexed} and t != INDEX_TITLE]

    # ...and a row whose target stopped being an article (renamed, so now a
    # redirect, or deleted) has to come back out. Adding without removing is how
    # the index accumulated 68 dead rows out of 707 by August 2026.
    stale = {t for t in indexed if norm(t) not in live and norm(t) != norm(INDEX_TITLE)}

    if not missing and not stale:
        print("האינדקס מעודכן: אין דפים חסרים ואין שורות מתות.")
        return

    if stale:
        print(f"שורות מתות להסרה ({len(stale)}):")
        for t in sorted(stale):
            print("  - " + t)

    print(f"דפים חסרים ({len(missing)}):")
    new_rows = []
    for t in missing:
        wt = c.get_page(t)["wikitext"]
        cats = c.get_categories(t)
        row = make_row(t, wt, cats)
        new_rows.append((t, row))
        print("  " + row)

    if not write:
        print("\n(dry run) הרץ עם --write כדי לכתוב לאינדקס.")
        return

    c.login()
    new_wt = build(idx_wt, new_rows, drop=stale)
    bits = []
    if new_rows:
        bits.append(f"הוספת {len(new_rows)} דפים חדשים")
    if stale:
        bits.append(f"הסרת {len(stale)} שורות שאינן מצביעות עוד על ערך")
    c.edit_page(INDEX_TITLE, new_wt, summary="עדכון אוטומטי: " + ", ".join(bits))
    print(f"\nנוספו {len(new_rows)} שורות, הוסרו {len(stale)}.")


if __name__ == "__main__":
    main()
