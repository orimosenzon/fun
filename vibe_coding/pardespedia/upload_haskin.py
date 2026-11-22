#!/usr/bin/env python3
"""Upload a photo from Gili Haskin's site (gilihaskin.com) with a proper file page.

Dr. Gili Haskin, a Pardes Hanna tour guide, permitted Pardespedia (22/11/2026,
in conversation with Ori Mosenzon) to use all material on his site, provided
we credit him and link to the site. Use ONLY for photos whose caption on his
site says "צילום: גילי חסקין" — archive photos ("באדיבות ארכיון המושבה" etc.)
are not his to license.

Usage:
    python3 upload_haskin.py <image_url> <target_filename> <description> <post_url> [--dry-run]

Downloads the image (EXIF-rotated, downscaled to max 1600px), uploads it, and
skips if the target file already exists.
"""
import io
import sys
import urllib.parse

import requests
from PIL import Image, ImageOps

from wiki_client import WikiClient, API_URL

UA = {"User-Agent": "Mozilla/5.0 (pardespedia bot)"}


def file_exists(client, filename):
    r = client.session.get(API_URL, params={
        "action": "query", "titles": f"קובץ:{filename}",
        "prop": "imageinfo", "format": "json",
    })
    page = next(iter(r.json()["query"]["pages"].values()))
    return "missing" not in page


def fetch_image(url):
    # URLs on his site often contain raw Hebrew; quote the path safely.
    p = urllib.parse.urlsplit(url)
    url = urllib.parse.urlunsplit(p._replace(path=urllib.parse.quote(urllib.parse.unquote(p.path))))
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content)))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail((1600, 1600))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=88)
    return out.getvalue(), img.size


def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry = "--dry-run" in sys.argv
    if len(args) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    url, filename, desc, post_url = args
    if not filename.lower().endswith((".jpg", ".jpeg")):
        filename += ".jpg"
    text = (
        "== תיאור ==\n"
        f"{desc}\n\n"
        "== מקור ==\n"
        f"צילום: ד\"ר [[גילי חסקין]]. מתוך [{post_url} מאמר באתר שלו], "
        "[https://www.gilihaskin.com www.gilihaskin.com].\n\n"
        "== רישיון ==\n"
        "גילי חסקין התיר לפרדספדיה להשתמש בחומר שבאתרו, ובכלל זה בתמונות, "
        "בתנאי שיינתן קרדיט וקישור לאתר (נובמבר 2026). שימוש מחוץ לפרדספדיה "
        "מחייב רשות ממנו.\n\n"
        "[[קטגוריה:תמונות מאתר גילי חסקין]]\n"
    )
    data, size = fetch_image(url)
    if dry:
        print(f"DRY: {filename} {size} {len(data)} bytes")
        print(text)
        return
    client = WikiClient()
    client.login()
    if file_exists(client, filename):
        print(f"SKIP (exists): {filename}")
        return
    token = client._csrf_token()
    r = client.session.post(API_URL, data={
        "action": "upload", "filename": filename,
        "comment": "תמונה מאתר גילי חסקין, באישורו",
        "text": text, "token": token, "ignorewarnings": "1", "format": "json",
    }, files={"file": (filename, io.BytesIO(data), "image/jpeg")})
    r.raise_for_status()
    res = r.json()
    if res.get("upload", {}).get("result") == "Success":
        print(f"UPLOADED: {filename}")
    else:
        print(f"FAILED: {filename} -> {res}")
        sys.exit(1)


if __name__ == "__main__":
    main()
