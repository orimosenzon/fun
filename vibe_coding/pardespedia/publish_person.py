#!/usr/bin/env python3
"""Publish a new person article from a spec file: image upload, page creation,
and a row in the people hub (דמויות בפרדס חנה-כרכור), in one go.

Written for the September 2026 batch of ~50 biographies found by sweeping
Hebrew Wikipedia for residents of the moshava. The article text itself is
written by hand into a .wiki file; this script only does the mechanics, so
every article goes through the same steps and none is left out of the hub.

Spec (JSON):
    {
      "title": "נתן גושן",
      "wiki": "articles/נתן גושן.wiki",              # wikitext, relative to spec dir or absolute
      "summary": "יצירת ערך חדש: נתן גושן — זמר-יוצר, מתושבי כרכור",
      "image": null | {
          "type": "commons", "commons": "File:X.jpg", "url": "https://upload...",
          "author": "...", "license": "CC BY-SA 4.0",
          "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
          "filename": "נתן גושן.jpg", "desc": "נתן גושן בהופעה, 2019"
      } | {
          "type": "fairuse", "local": "/path/img.jpg", "filename": "X.jpg",
          "text": "<full file-page wikitext with source + rationale>"
      },
      "hub": {"domain": "מוזיקה", "blurb": "3–6 lines for the hub table"}
    }

Usage:
    python3 publish_person.py SPEC.json [--dry-run]

Rules honoured: never overwrites an existing page (refuses), never re-uploads
an existing file (skips), edits the hub get-first and appends one row with the
next running number (the hub table is sortable; recent rows are appended, not
inserted alphabetically).
"""
import argparse
import io
import json
import os
import re
import sys

import requests

from wiki_client import WikiClient, API_URL

H = {"User-Agent": "Pardespedia-Bot/1.0 (orimosenzon@gmail.com)"}
HUB = "דמויות בפרדס חנה-כרכור"
MAX_W = 800


def page_exists(client, title):
    r = client.session.get(API_URL, params={"action": "query", "titles": title, "format": "json"})
    page = next(iter(r.json()["query"]["pages"].values()))
    return "missing" not in page


def file_exists(client, filename):
    return page_exists(client, f"קובץ:{filename}")


def downscale(data, mime):
    """Keep wiki copies light: cap width at MAX_W. Returns (bytes, mime)."""
    try:
        from PIL import Image
    except ImportError:
        return data, mime
    im = Image.open(io.BytesIO(data))
    if im.width <= MAX_W and mime != "image/png":
        return data, mime
    if im.width > MAX_W:
        im = im.resize((MAX_W, int(im.height * MAX_W / im.width)), Image.LANCZOS)
    out = io.BytesIO()
    if mime == "image/png" and im.mode in ("RGBA", "LA", "P"):
        im.save(out, format="PNG", optimize=True)
        return out.getvalue(), "image/png"
    im.convert("RGB").save(out, format="JPEG", quality=88)
    return out.getvalue(), "image/jpeg"


def upload(client, spec, dry):
    fn = spec["filename"]
    if file_exists(client, fn):
        print(f"  image: SKIP (exists) {fn}")
        return True
    if spec["type"] == "commons":
        if spec.get("local"):
            # a locally prepared copy (rotated / cropped) of the Commons file
            with open(os.path.expanduser(spec["local"]), "rb") as f:
                data = f.read()
        else:
            data = requests.get(spec["url"], headers=H, timeout=60).content
        mime = "image/png" if spec["url"].lower().endswith(".png") else "image/jpeg"
        text = (
            f"{spec['desc']}\n\n"
            f"מקור: [[commons:{spec['commons']}|{spec['commons']}]] מוויקישיתוף.\n\n"
            f"צלם/יוצר: {spec['author']}.\n\n"
            f"רישיון: {spec['license']} ([{spec['license_url']} הרישיון]).\n\n"
            f"[[קטגוריה:תמונות מוויקישיתוף]]\n[[קטגוריה:תמונות של אישים]]\n"
        )
        comment = "העלאת דיוקן חופשי מוויקישיתוף עבור ערך אישים"
    else:
        with open(os.path.expanduser(spec["local"]), "rb") as f:
            data = f.read()
        mime = "image/png" if spec["local"].lower().endswith(".png") else "image/jpeg"
        text = spec["text"].rstrip() + "\n\n[[קטגוריה:תמונות בשימוש הוגן]]\n[[קטגוריה:תמונות של אישים]]\n"
        comment = "העלאת תמונה (שימוש הוגן, רזולוציה נמוכה) עבור ערך אישים"
    data, mime = downscale(data, mime)
    if fn.lower().endswith(".png") and mime == "image/jpeg":
        print(f"  image: WARNING filename is .png but content is jpeg; rename spec filename")
    print(f"  image: {'would upload' if dry else 'uploading'} {fn} ({len(data)//1024} KB, {mime})")
    if dry:
        return True
    token = client._csrf_token()
    r = client.session.post(API_URL, data={
        "action": "upload", "filename": fn, "comment": comment, "text": text,
        "token": token, "ignorewarnings": "1", "format": "json",
    }, files={"file": (fn, io.BytesIO(data), mime)})
    r.raise_for_status()
    res = r.json()
    ok = res.get("upload", {}).get("result") == "Success"
    print(f"  image: {'UPLOADED' if ok else 'FAILED ' + json.dumps(res, ensure_ascii=False)}")
    return ok


def hub_append(client, title, domain, blurb, img_filename, dry):
    page = client.get_page(HUB)
    text = page["wikitext"]
    nums = [int(n) for n in re.findall(r"^\| (\d+) \|\|", text, flags=re.M)]
    if f"[[{title}]]" in text or f"[[{title}|" in text:
        print(f"  hub: SKIP (already listed) {title}")
        return True
    n = max(nums) + 1
    img = f"[[קובץ:{img_filename}|95px]]" if img_filename else "—"
    row = (f"|-\n| {n} || style=\"text-align:center\" | {img} || '''[[{title}]]''' || {domain} || {blurb.strip()}\n")
    idx = text.rfind("|}")
    if idx < 0:
        print("  hub: ERROR table end not found")
        return False
    new = text[:idx] + row + text[idx:]
    print(f"  hub: {'would append' if dry else 'appending'} row #{n}")
    if dry:
        return True
    client.edit_page(HUB, new, summary=f"הוספת {title} לטבלת הדמויות")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-hub", action="store_true")
    a = ap.parse_args()
    with open(a.spec, encoding="utf-8") as f:
        spec = json.load(f)
    base = os.path.dirname(os.path.abspath(a.spec))
    wpath = spec["wiki"] if os.path.isabs(spec["wiki"]) else os.path.join(base, spec["wiki"])
    with open(wpath, encoding="utf-8") as f:
        wikitext = f.read()
    title = spec["title"]
    print(f"== {title}")

    client = WikiClient()
    client.login()
    if page_exists(client, title):
        print("  page: EXISTS — refusing to overwrite. Use the get-first edit flow instead.")
        return 1

    img_fn = None
    if spec.get("image"):
        if not upload(client, spec["image"], a.dry_run):
            return 1
        img_fn = spec["image"]["filename"]
        if f"קובץ:{img_fn}" not in wikitext:
            print(f"  page: WARNING article does not reference קובץ:{img_fn}")

    # sanity: mandatory category and no literal signature
    if "[[קטגוריה: אישים בפרדס חנה-כרכור]]" not in wikitext and "[[קטגוריה:אישים בפרדס חנה-כרכור]]" not in wikitext:
        print("  page: ERROR missing persons category")
        return 1
    if "~~~~" in wikitext:
        print("  page: ERROR literal signature in article")
        return 1

    print(f"  page: {'would create' if a.dry_run else 'creating'} ({len(wikitext)} chars)")
    if not a.dry_run:
        client.edit_page(title, wikitext, summary=spec.get("summary", f"יצירת ערך חדש: {title}"), create_only=True)
        print(f"  page: CREATED https://pardespedia.info/wiki/{title.replace(' ', '_')}")

    if not a.skip_hub and spec.get("hub"):
        hub_append(client, title, spec["hub"]["domain"], spec["hub"]["blurb"], img_fn, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
