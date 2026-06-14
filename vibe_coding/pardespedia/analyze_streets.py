#!/usr/bin/env python3
"""Find high-confidence street-link candidates.

Looks for the pattern "רחוב <Name>" / "ברחוב <Name>" in page text where a page
titled "<Name> (רחוב)" exists, and the mention is not already linked. The
explicit word רחוב disambiguates fruit/name vs. street.

Usage: python3 analyze_streets.py [cache_dir]
"""
import os
import re
import sys
import json

cache_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pardes_cache"
with open(os.path.join(cache_dir, "_index.json")) as f:
    index = json.load(f)

titles = {e["title"] for e in index}
# street base names: "X (רחוב)" -> X
streets = {}
for t in titles:
    m = re.match(r"^(.*?) \(רחוב\)$", t)
    if m:
        streets[m.group(1)] = t

HE = "א-ת"
LETTER = re.compile(f"[{HE}A-Za-z0-9\"']")


def mask_spans(text):
    spans = []
    for pat in (r"\[\[.*?\]\]", r"\[https?://\S+.*?\]", r"<youtube.*?</youtube>",
                r"==.*?==", r"\{\|.*?\|\}"):
        for m in re.finditer(pat, text, re.S):
            spans.append((m.start(), m.end()))
    return spans


def in_spans(pos, spans):
    return any(s <= pos < e for s, e in spans)


report = {}
for e in index:
    src = e["title"]
    if "(רחוב)" in src or ":" in src:
        continue
    with open(os.path.join(cache_dir, e["file"])) as f:
        text = f.read()
    spans = mask_spans(text)
    hits = {}
    for base, page in streets.items():
        if base == src:
            continue
        # match "רחוב base" with optional ב/ה prefix on רחוב
        for m in re.finditer(r"(?:ב|ה)?רחוב\s+" + re.escape(base), text):
            # position of base inside
            bpos = m.start() + m.group().find(base)
            end = bpos + len(base)
            after = text[end] if end < len(text) else ""
            if LETTER.match(after):
                continue
            if in_spans(bpos, spans):
                continue
            ctx = text[max(0, m.start() - 20):min(len(text), end + 20)].replace("\n", " ")
            hits.setdefault(page, []).append((m.group(), ctx))
            break  # first only per street per page
    if hits:
        report[src] = hits

total = 0
for src in sorted(report):
    print(f"\n### {src}")
    for page, lst in sorted(report[src].items()):
        total += len(lst)
        for matched, ctx in lst:
            print(f"  → [[{page}]]  (match: '{matched}')   …{ctx}…")
print(f"\nTOTAL: {total} street-link candidates across {len(report)} pages")
