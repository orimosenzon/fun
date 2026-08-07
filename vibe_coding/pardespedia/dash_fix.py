#!/usr/bin/env python3
"""Replace em-dashes in Hebrew wikitext with punctuation a person would type.

The em-dash is not on a Hebrew keyboard, so in Hebrew prose it reads as
machine-written. This converts the mechanical cases and leaves the rest alone
for a human to judge, rather than blanket-replacing and producing comma splices.

What it converts:
  * '''מונח''' — הסבר          ->  * '''מונח''': הסבר      (definition-style list)
  * [[ערך]] — הערה              ->  * [[ערך]], הערה         (link list annotation)
  ... — הסתייגות — ...          ->  ... (הסתייגות) ...      (paired parenthetical)
  1935 — 1937                   ->  1935–1937               (number range: en-dash)

What it deliberately leaves:
  a single mid-sentence dash, where the right answer is a comma, a colon, a full
  stop, or deleting the aside entirely. Those need reading, so --report lists
  them with context instead of guessing.

    python3 dash_fix.py --report  <file.wiki>       what is there, and what is left over
    python3 dash_fix.py --diff    <file.wiki>       show the automatic conversions
    python3 dash_fix.py --write   <file.wiki>       apply them in place
"""
import argparse
import difflib
import re
import sys

EM = "—"

# ---- mechanical conversions -------------------------------------------------

RULES = [
    # number range -> en-dash (1935—1937 -> 1935–1937)
    (re.compile(r"(?<=\d)\s*" + EM + r"\s*(?=\d)"), "–"),
    # list item: bold term then dash -> colon.  * '''X''' — y   ->   * '''X''': y
    (re.compile(r"(?m)^(\*+ .*?''')\s*" + EM + r"\s+"), r"\1: "),
    # list item: wiki/external link then dash -> comma.  * [[X]] — y  ->  * [[X]], y
    (re.compile(r"(?m)^(\*+ .*?(?:\]\]|\]))\s*" + EM + r"\s+"), r"\1, "),
    # paired parenthetical inside one line -> real parentheses
    (re.compile(r"\s" + EM + r"\s([^" + EM + r"\n]{2,90}?)\s" + EM + r"\s"), r" (\1) "),
]


def convert(text):
    for pat, sub in RULES:
        prev = None
        while prev != text:            # paired rule can fire more than once a line
            prev = text
            text = pat.sub(sub, text)
    return text


def leftovers(text):
    """(line_no, context) for every em-dash the rules did not handle."""
    out = []
    for i, line in enumerate(text.split("\n"), 1):
        for m in re.finditer(EM, line):
            s, e = max(0, m.start() - 60), min(len(line), m.end() + 60)
            out.append((i, line[s:e]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--report", action="store_true")
    g.add_argument("--diff", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()

    original = open(args.path, encoding="utf-8").read()
    fixed = convert(original)
    before, after = original.count(EM), fixed.count(EM)

    if args.report:
        print(f"{args.path}: {before} em-dashes -> {after} left after automatic rules")
        for ln, ctx in leftovers(fixed):
            print(f"  line {ln}: {ctx}")
    elif args.diff:
        for line in difflib.unified_diff(original.split("\n"), fixed.split("\n"),
                                         "before", "after", lineterm="", n=0):
            print(line)
        print(f"\n{before} -> {after}", file=sys.stderr)
    else:
        open(args.path, "w", encoding="utf-8").write(fixed)
        print(f"{args.path}: {before} -> {after}")


if __name__ == "__main__":
    main()
