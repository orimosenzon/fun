#!/usr/bin/env python3
"""Download all main-namespace pages to a local cache dir for offline analysis.

Usage: python3 cache_all.py [cache_dir]
"""
import os
import sys
import json
from wiki_client import WikiClient

cache_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pardes_cache"
os.makedirs(cache_dir, exist_ok=True)

c = WikiClient()
pages = c.list_pages(namespace=0, redirects="nonredirects")
index = []
for i, p in enumerate(pages):
    title = p["title"]
    info = c.get_page(title)
    safe = title.replace("/", "__")
    path = os.path.join(cache_dir, safe + ".wiki")
    with open(path, "w") as f:
        f.write(info["wikitext"])
    index.append({"title": title, "file": safe + ".wiki", "len": len(info["wikitext"])})
    print(f"[{i+1}/{len(pages)}] {title} ({len(info['wikitext'])} chars)")

with open(os.path.join(cache_dir, "_index.json"), "w") as f:
    json.dump(index, f, ensure_ascii=False, indent=2)
print(f"\nCached {len(index)} pages to {cache_dir}")
