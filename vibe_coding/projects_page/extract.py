#!/usr/bin/env python3
"""One-time migration: parse the hand-written index.html into projects.json.

After this runs, projects.json is the source of truth and build.py regenerates
index.html from it. This script is kept for reference / re-migration only.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "index.html"
OUT = Path(__file__).resolve().parent / "projects.json"

src = HTML.read_text(encoding="utf-8")

# ── sections ──────────────────────────────────────────────────────────────
sec_re = re.compile(
    r'<div class="section-header">\s*'
    r'<span class="section-emoji">(?P<emoji>.*?)</span>\s*'
    r'<span class="section-title">(?P<title>.*?)</span>\s*'
    r'<span class="section-count">.*?</span>\s*'
    r'</div>\s*'
    r'<div class="(?P<grid>projects|projects-mini)">',
    re.S,
)

marks = [(m.start(), m) for m in sec_re.finditer(src)]
body_end = src.index("<footer>")

sections = []
for i, (pos, m) in enumerate(marks):
    end = marks[i + 1][0] if i + 1 < len(marks) else body_end
    sections.append(
        {
            "emoji": m.group("emoji").strip(),
            "title": m.group("title").strip(),
            "grid": m.group("grid"),
            "body": src[m.end():end],
        }
    )


def split_cards(body: str):
    """Split a section body into individual .project-card blocks by brace-less
    depth counting over <div>/</div>."""
    cards = []
    for m in re.finditer(r'<div class="project-card ([^"]*)">', body):
        depth = 0
        i = m.start()
        while True:
            tag = re.compile(r"<div\b|</div>").search(body, i)
            if not tag:
                break
            depth += 1 if tag.group().startswith("<div") else -1
            i = tag.end()
            if depth == 0:
                break
        cards.append((m.group(1).strip(), body[m.end():i - len("</div>")]))
    return cards


def grab(pattern, text, default=""):
    m = re.search(pattern, text, re.S)
    return m.group(1).strip() if m else default


def parse_card(classes: str, inner: str) -> dict:
    theme = next((c for c in classes.split() if c.startswith("t-")), "t-indigo")
    small = "card-small" in classes

    title_block = grab(r'<div class="card-title">(.*?)</div>', inner)
    badges = []
    for bm in re.finditer(r'<span class="(live|wip)-badge">(.*?)</span>\s*(?:$|<span)', title_block, re.S):
        badges.append({"type": bm.group(1), "label": re.sub(r"<[^>]+>", "", bm.group(2)).strip()})
    # title text = the block minus any badge spans
    # badge spans contain a nested <span>, so strip through the *last* </span>
    title = re.sub(r"<span class=\"(live|wip)-badge\">.*</span>", "", title_block, flags=re.S)
    title = re.sub(r"<[^>]+>", "", title).strip()

    tags_block = grab(r'<div class="card-tags">(.*?)</div>\s*</div>', inner)
    tags = [t.strip() for t in re.findall(r'<span class="tag">(.*?)</span>', tags_block, re.S)]

    desc = grab(r'<div class="card-desc">(.*?)</div>', inner)
    desc = re.sub(r"\s*\n\s*", " ", desc).strip()

    links_block = grab(r'<div class="card-links">(.*?)</div>', inner)
    links = []
    for lm in re.finditer(
        r'<a\s+href="(?P<href>[^"]*)"\s+class="(?P<cls>[^"]*)"(?P<rest>[^>]*)>(?P<label>.*?)</a>',
        links_block,
        re.S,
    ):
        links.append(
            {
                "href": lm.group("href"),
                "style": lm.group("cls").replace("btn-", ""),
                "label": lm.group("label").strip(),
                "external": "target=" in lm.group("rest"),
            }
        )

    illus = grab(r'<div class="card-illus">(.*?)</div>', inner)
    illus = re.sub(r"<!--.*?-->", "", illus, flags=re.S).strip()

    card = {
        "title": title,
        "icon": grab(r'<div class="card-icon">(.*?)</div>', inner),
        "theme": theme,
        "tags": tags,
        "desc": desc,
        "links": links,
    }
    if badges:
        card["badges"] = badges
    if small:
        card["small"] = True
    if illus:
        card["illus"] = illus
    return card


data = {
    "site": {
        "title": grab(r"<h1>(.*?)</h1>", src),
        "subtitle": grab(r'<p class="header-sub">(.*?)</p>', src),
        "meta": grab(r'<p class="header-meta">(.*?)</p>', src),
    },
    "sections": [],
}
for s in sections:
    data["sections"].append(
        {
            "emoji": s["emoji"],
            "title": s["title"],
            "grid": s["grid"],
            "projects": [parse_card(c, inner) for c, inner in split_cards(s["body"])],
        }
    )

OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
total = sum(len(s["projects"]) for s in data["sections"])
print(f"wrote {OUT} — {len(data['sections'])} sections, {total} projects")
