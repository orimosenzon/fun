#!/usr/bin/env python3
"""Index every free-licensed Commons file under the Pardes Hanna-Karkur tree.

Walks Category:Pardes Hanna-Karkur recursively, pulls each file's license,
author and description, and writes commons_index.json — the pool the enrichment
pass draws on when an article needs an image. Free licenses only: anything
non-free is dropped, so nothing here needs a fair-use rationale.

Usage:
    python3 commons_index.py [--out FILE]
"""
import argparse
import json
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://commons.wikimedia.org/w/api.php"
H = {"User-Agent": "Pardespedia-Bot/1.0 (orimosenzon@gmail.com)"}
ROOT = "Pardes Hanna-Karkur"

# Matched against the license name with every non-alphanumeric stripped, so
# "CC BY 4.0", "CC-BY-SA-3.0" and "cc by sa" all collapse to the same "ccby..".
FREE_HINTS = ("ccby", "cczero", "cc0", "pd", "publicdomain", "gfdl", "attribution")
NONFREE_HINTS = ("fairuse", "noncommercial", "ccbync", "nd", "copyright")


def members(cat):
    out, cont = [], None
    while True:
        p = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
             "cmlimit": 500, "cmtype": "file|subcat", "format": "json"}
        if cont:
            p["cmcontinue"] = cont
        r = requests.get(API, params=p, headers=H, timeout=30)
        r.raise_for_status()
        data = r.json()
        out += data.get("query", {}).get("categorymembers", [])
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            return out


def walk(root):
    """Depth-first over subcategories; returns {file_title: [categories]}."""
    seen_cats, files = set(), {}
    stack = [root]
    while stack:
        cat = stack.pop()
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        for m in members(cat):
            if m["title"].startswith("Category:"):
                stack.append(m["title"][len("Category:"):])
            else:
                files.setdefault(m["title"], []).append(cat)
    return files, seen_cats


def file_info(titles):
    """imageinfo + extmetadata (license, author, description) in batches of 50."""
    out = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        # POST, not GET: long Hebrew filenames blow past the URI length limit.
        r = requests.post(API, data={
            "action": "query", "titles": "|".join(chunk), "prop": "imageinfo",
            "iiprop": "url|extmetadata|size", "format": "json",
        }, headers=H, timeout=60)
        r.raise_for_status()
        for p in r.json().get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            md = ii.get("extmetadata", {}) or {}
            def g(k):
                return (md.get(k, {}) or {}).get("value", "")
            out[p["title"]] = {
                "url": ii.get("url", ""),
                "width": ii.get("width"), "height": ii.get("height"),
                "license": g("LicenseShortName"),
                "license_url": g("LicenseUrl"),
                "artist": g("Artist"),
                "credit": g("Credit"),
                "description": g("ImageDescription"),
                "date": g("DateTimeOriginal"),
            }
        print(f"  info {min(i + 50, len(titles))}/{len(titles)}", file=sys.stderr)
    return out


def strip_html(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "commons_index.json"))
    args = ap.parse_args()

    files, cats = walk(ROOT)
    print(f"{len(files)} files across {len(cats)} categories", file=sys.stderr)
    info = file_info(list(files))

    idx = []
    for title, cat_list in files.items():
        d = info.get(title, {})
        lic = re.sub(r"[^a-z0-9]", "", (d.get("license") or "").lower())
        if any(h in lic for h in NONFREE_HINTS) or not any(h in lic for h in FREE_HINTS):
            print(f"  SKIP non-free ({d.get('license')}): {title}", file=sys.stderr)
            continue
        idx.append({
            "title": title, "categories": cat_list, "url": d.get("url", ""),
            "license": d.get("license", ""), "license_url": d.get("license_url", ""),
            "artist": strip_html(d.get("artist")), "credit": strip_html(d.get("credit")),
            "description": strip_html(d.get("description")),
            "date": strip_html(d.get("date")),
            "width": d.get("width"), "height": d.get("height"),
        })
    idx.sort(key=lambda x: (x["categories"][0], x["title"]))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out} ({len(idx)} free files)", file=sys.stderr)


if __name__ == "__main__":
    main()
