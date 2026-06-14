#!/usr/bin/env python3
"""Scan cached pages for internal-link candidates.

For each page, find mentions of OTHER existing page titles that appear as plain
text (not already inside a [[link]] or external link), bounded by non-Hebrew
characters. Outputs a report grouped by source page.

Usage: python3 analyze_links.py [cache_dir]
"""
import os
import re
import sys
import json

cache_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pardes_cache"
with open(os.path.join(cache_dir, "_index.json")) as f:
    index = json.load(f)

titles = [e["title"] for e in index]
file_of = {e["title"]: e["file"] for e in index}

# Pages that should never be link TARGETS (meta/utility/ambiguous).
META = {
    "Pardespedia.info:Sidebar", "ארגז חול", "ארגז החול של קלוד", "נסיון",
    "מידע", "דוגמא לערך שעדיין לא קיים", "האינדקס של הדפים",
    "דברים להוספה לויקי", "המלצות והצעות", "הסברים ועוד", "איך עורכים כאן?",
    "אתרים באינטרט", "מאגר מידע וקישורים", "דורשים בינוי שפוי",
}

HE = "א-ת"
LETTER = re.compile(f"[{HE}A-Za-z0-9]")


def mask_spans(text):
    """Return list of (start,end) spans that are already links / not plain text."""
    spans = []
    for m in re.finditer(r"\[\[.*?\]\]", text, re.S):
        spans.append((m.start(), m.end()))
    for m in re.finditer(r"\[https?://\S+.*?\]", text, re.S):
        spans.append((m.start(), m.end()))
    for m in re.finditer(r"<youtube.*?</youtube>", text, re.S):
        spans.append((m.start(), m.end()))
    for m in re.finditer(r"\[\[קטגוריה:.*?\]\]", text, re.S):
        spans.append((m.start(), m.end()))
    # section headers (== .. ==) - avoid linking inside headers for cleanliness? keep them
    return spans


def in_spans(pos, spans):
    for s, e in spans:
        if s <= pos < e:
            return True
    return False


def find_candidates(title_src, text):
    spans = mask_spans(text)
    found = {}  # target -> list of context snippets
    for t in titles:
        if t == title_src or t in META:
            continue
        if t.startswith("קטגוריה:") or ":" in t:
            continue
        # search occurrences
        start = 0
        while True:
            idx = text.find(t, start)
            if idx == -1:
                break
            start = idx + 1
            end = idx + len(t)
            # boundary check
            before = text[idx - 1] if idx > 0 else ""
            after = text[end] if end < len(text) else ""
            if LETTER.match(before) or LETTER.match(after):
                continue
            if in_spans(idx, spans):
                continue
            ctx = text[max(0, idx - 30):min(len(text), end + 30)].replace("\n", " ")
            found.setdefault(t, []).append(ctx)
    return found


report = {}
for e in index:
    src = e["title"]
    if src in META or ":" in src:
        continue
    path = os.path.join(cache_dir, e["file"])
    with open(path) as f:
        text = f.read()
    cands = find_candidates(src, text)
    if cands:
        report[src] = cands

# Print grouped
total = 0
for src in sorted(report):
    print(f"\n### {src}")
    for tgt, ctxs in sorted(report[src].items(), key=lambda x: -len(x[1])):
        total += len(ctxs)
        print(f"  → [[{tgt}]]  ×{len(ctxs)}")
        for ctx in ctxs[:2]:
            print(f"       …{ctx}…")

print(f"\n\nTOTAL candidate mentions: {total} across {len(report)} pages")
