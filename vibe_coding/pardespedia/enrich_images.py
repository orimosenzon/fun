#!/usr/bin/env python3
"""Place free-licensed Commons images into pardespedia articles.

Driven by a plan file: each entry names a Commons file (from commons_index.json),
the target article, the local filename, and the caption. The script downloads,
downscales, uploads with a full attribution page, and splices the thumbnail into
the article — after the lead paragraph by default, so the image sits where a
reader expects it.

Editing rule (project-wide): fetch the page, make one targeted change, write the
whole page back. Never blind-overwrite.

Usage:
    python3 enrich_images.py PLAN.json [--dry-run]

Plan entry:
    {"commons": "File:X.jpg", "article": "ערך", "filename": "שם מקומי.jpg",
     "caption": "כיתוב", "width": 320, "align": "ימין",
     "section": "== כותרת =="}       # optional: insert under this heading
"""
import argparse
import io
import json
import os
import re
import sys

import requests

from wiki_client import WikiClient, API_URL

HERE = os.path.dirname(os.path.abspath(__file__))
H = {"User-Agent": "Pardespedia-Bot/1.0 (orimosenzon@gmail.com)"}
MAX_DIM = 1600  # wiki copies stay light; the Commons original keeps full res


def load_index():
    with open(os.path.join(HERE, "commons_index.json"), encoding="utf-8") as f:
        return {x["title"]: x for x in json.load(f)}


def file_exists(client, filename):
    r = client.session.get(API_URL, params={
        "action": "query", "titles": f"קובץ:{filename}",
        "prop": "imageinfo", "format": "json",
    })
    return "missing" not in next(iter(r.json()["query"]["pages"].values()))


def upload(client, filename, meta, caption, dry_run=False):
    if file_exists(client, filename):
        print(f"  SKIP upload (exists): {filename}")
        return True
    if dry_run:
        print(f"  [dry-run] would upload {filename} from {meta['title']}")
        return True

    img = requests.get(meta["url"], headers=H, timeout=120)
    img.raise_for_status()
    data = img.content
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        if max(im.size) > MAX_DIM:
            im = im.convert("RGB")
            scale = MAX_DIM / max(im.size)
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=88)
            data = buf.getvalue()
    except ImportError:
        pass

    lic_url = meta.get("license_url") or ""
    lic = f"{meta['license']} ([{lic_url} הרישיון])" if lic_url else meta["license"]
    desc = (
        f"{caption}\n\n"
        f"מקור: [[commons:{meta['title']}|{meta['title']}]] מוויקישיתוף.\n\n"
        f"צלם/יוצר: {meta.get('artist') or 'לא צוין'}.\n\n"
        f"רישיון: {lic}.\n\n"
        "[[קטגוריה: תמונות מוויקישיתוף]]\n"
    )
    r = client.session.post(API_URL, data={
        "action": "upload", "filename": filename,
        "comment": "העלאת תמונה חופשית מוויקישיתוף להעשרת ערך",
        "text": desc, "token": client._csrf_token(),
        "ignorewarnings": "1", "format": "json",
    }, files={"file": (filename, io.BytesIO(data), "image/jpeg")})
    r.raise_for_status()
    if r.json().get("upload", {}).get("result") == "Success":
        print(f"  UPLOADED: {filename}")
        return True
    print(f"  UPLOAD FAILED: {filename} -> {r.json()}", file=sys.stderr)
    return False


def insert_image(text, entry):
    """Splice the thumbnail in. Returns (new_text, note) — note explains a skip."""
    tag = (f"[[קובץ:{entry['filename']}|ממוזער|{entry.get('align', 'ימין')}|"
           f"{entry.get('width', 320)}px|{entry['caption']}]]")
    if entry["filename"] in text:
        return text, "already placed"

    if entry.get("section"):
        head = entry["section"]
        i = text.find(head)
        if i == -1:
            return text, f"section not found: {head}"
        j = i + len(head)
        return text[:j] + "\n" + tag + text[j:], None

    # Default: straight after the lead paragraph, before the first heading.
    m = re.search(r"\n\s*\n", text)
    first_head = re.search(r"^==", text, re.M)
    if m and (not first_head or m.start() < first_head.start()):
        return text[:m.start()] + "\n\n" + tag + text[m.start():], None
    if first_head:
        return text[:first_head.start()] + tag + "\n\n" + text[first_head.start():], None
    return text.rstrip() + "\n\n" + tag + "\n", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.plan, encoding="utf-8") as f:
        plan = json.load(f)
    index = load_index()

    client = WikiClient()
    client.login()

    done = skipped = failed = 0
    for entry in plan:
        art = entry["article"]
        print(f"\n== {art}  <-  {entry['commons']}")
        meta = index.get(entry["commons"])
        if not meta:
            print(f"  NOT IN INDEX: {entry['commons']}", file=sys.stderr)
            failed += 1
            continue
        if not upload(client, entry["filename"], meta, entry["caption"], args.dry_run):
            failed += 1
            continue

        page = client.get_page(art)
        if not page["exists"]:
            print(f"  ARTICLE MISSING: {art}", file=sys.stderr)
            failed += 1
            continue
        new, note = insert_image(page["wikitext"], entry)
        if note:
            print(f"  skip insert ({note})")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  [dry-run] would insert into {art}")
            done += 1
            continue
        client.edit_page(art, new, summary=f"הוספת תמונה חופשית מוויקישיתוף: {entry['caption'][:60]}")
        done += 1

    print(f"\ninserted={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
