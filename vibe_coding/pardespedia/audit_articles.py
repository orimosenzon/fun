#!/usr/bin/env python3
"""Audit every article on pardespedia and score what each one is missing.

Deterministic, read-only: pulls all of namespace 0 in bulk, measures each
article (prose length, image, video, links, categories, sources), and writes a
JSON dump plus a prioritized markdown report. Nothing is edited — the report is
the worklist for the enrichment pass.

Usage:
    python3 audit_articles.py [--out DIR]
"""
import argparse
import datetime as dt
import json
import os
import re
import sys

from wiki_client import WikiClient, API_URL

HERE = os.path.dirname(os.path.abspath(__file__))

# A page whose whole job is to point elsewhere is not "thin" — judge it apart.
INDEX_HINTS = ("רשימת", "קטגוריה:", "אינדקס", "פורטל")


def fetch_all(client, titles):
    """Bulk-fetch wikitext for many titles (API caps `titles` at 50 per call)."""
    out = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        r = client.session.get(API_URL, params={
            "action": "query", "titles": "|".join(chunk),
            "prop": "revisions", "rvprop": "content|timestamp", "rvslots": "main",
            "format": "json",
        })
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        for p in pages.values():
            if "missing" in p:
                continue
            rev = (p.get("revisions") or [{}])[0]
            out[p["title"]] = {
                "wikitext": rev.get("slots", {}).get("main", {}).get("*", ""),
                "touched": rev.get("timestamp", ""),
            }
        print(f"  fetched {min(i + 50, len(titles))}/{len(titles)}", file=sys.stderr)
    return out


def prose_chars(text):
    """Rough count of reader-facing prose: strip markup, files, categories, refs."""
    t = re.sub(r"\[\[(?:קובץ|File|Image):[^\]]*\]\]", " ", text, flags=re.S)
    t = re.sub(r"\[\[קטגוריה:[^\]]*\]\]", " ", t)
    t = re.sub(r"\{\{[^}]*\}\}", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s([^\]]*)\]", r"\1", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[=*#'|{}\[\]]", " ", t)
    return len(re.sub(r"\s+", " ", t).strip())


def audit_one(title, rec):
    text = rec["wikitext"]
    is_redirect = bool(re.match(r"\s*#(?:REDIRECT|הפניה)", text, re.I))
    images = re.findall(r"\[\[(?:קובץ|File|Image):([^\]|]+)", text)
    videos = re.findall(r"youtube|יוטיוב|\{\{#ev:", text, re.I)
    return {
        "title": title,
        "url": f"https://pardespedia.info/wiki/{title.replace(' ', '_')}",
        "redirect": is_redirect,
        "index_page": any(h in title for h in INDEX_HINTS),
        "bytes": len(text),
        "prose": prose_chars(text),
        "images": len(images),
        "image_names": images,
        "video": len(videos) > 0,
        "sections": len(re.findall(r"^==[^=]", text, re.M)),
        "internal_links": len(set(re.findall(r"\[\[(?!קובץ:|File:|קטגוריה:)([^\]|#]+)", text))),
        "external_links": len(re.findall(r"\[https?://", text)),
        "categories": re.findall(r"\[\[קטגוריה:\s*([^\]]+)\]\]", text),
        "touched": rec["touched"],
    }


def needs(a):
    """What this article is missing, most valuable first."""
    out = []
    if a["redirect"] or a["index_page"]:
        return out
    if a["images"] == 0:
        out.append("תמונה")
    if a["prose"] < 700:
        out.append("הרחבה")
    if not a["video"]:
        out.append("וידאו")
    if a["external_links"] == 0:
        out.append("מקורות")
    if not a["categories"]:
        out.append("קטגוריה")
    if a["internal_links"] < 3:
        out.append("קישורים")
    return out


def score(a):
    """Priority: thin + imageless articles that readers actually land on."""
    if a["redirect"] or a["index_page"]:
        return -1
    s = 0
    if a["images"] == 0:
        s += 40
    if a["prose"] < 300:
        s += 40
    elif a["prose"] < 700:
        s += 25
    elif a["prose"] < 1500:
        s += 10
    if a["external_links"] == 0:
        s += 15
    if not a["categories"]:
        s += 15
    if a["sections"] == 0:
        s += 10
    if not a["video"]:
        s += 5
    if a["internal_links"] < 3:
        s += 5
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "audit_reports"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    client = WikiClient()
    client.login()
    titles = [p["title"] for p in client.list_pages()]
    print(f"{len(titles)} articles in ns0", file=sys.stderr)
    raw = fetch_all(client, titles)

    audit = []
    for title, rec in raw.items():
        a = audit_one(title, rec)
        a["needs"] = needs(a)
        a["score"] = score(a)
        audit.append(a)
    audit.sort(key=lambda a: (-a["score"], a["title"]))

    stamp = dt.date.today().isoformat()
    jpath = os.path.join(args.out, f"audit-{stamp}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=1)

    real = [a for a in audit if not a["redirect"] and not a["index_page"]]
    lines = [f"# ביקורת ערכים — פרדספדיה, {stamp}", "",
             f"- ערכים בסך הכל: {len(audit)} (מתוכם {sum(1 for a in audit if a['redirect'])} הפניות, "
             f"{sum(1 for a in audit if a['index_page'])} דפי אינדקס)",
             f"- ערכי תוכן שנבדקו: {len(real)}",
             f"- **בלי תמונה: {sum(1 for a in real if a['images'] == 0)}**",
             f"- **דלים (< 700 תווי טקסט): {sum(1 for a in real if a['prose'] < 700)}**",
             f"- בלי וידאו: {sum(1 for a in real if not a['video'])}",
             f"- בלי מקורות חיצוניים: {sum(1 for a in real if a['external_links'] == 0)}",
             f"- בלי קטגוריה: {sum(1 for a in real if not a['categories'])}", "",
             "## סדר עדיפויות", "",
             "| # | ערך | ניקוד | טקסט | תמונות | חסר |", "|---|---|---|---|---|---|"]
    for i, a in enumerate(real, 1):
        lines.append(f"| {i} | [{a['title']}]({a['url']}) | {a['score']} | {a['prose']} | "
                     f"{a['images']} | {', '.join(a['needs'])} |")
    mpath = os.path.join(args.out, f"audit-{stamp}.md")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {jpath}\nwrote {mpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
