#!/usr/bin/env python3
"""Upload a local photo that its subject/owner gave to Pardespedia for use.

Usage:
    python3 upload_given.py <local_path> <target_filename> <description> <source_line>

<source_line> is wikitext saying who gave it and on what terms, e.g.
"התמונה נמסרה לפרדספדיה בידי גילי חסקין, בנובמבר 2026, לשימוש בערך עליו."
EXIF-rotated, downscaled to max 1200px. Skips if the target exists.
"""
import io
import os
import sys

from PIL import Image, ImageOps

from wiki_client import WikiClient, API_URL


def main():
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    path, filename, desc, source = sys.argv[1:5]
    img = ImageOps.exif_transpose(Image.open(os.path.expanduser(path))).convert("RGB")
    img.thumbnail((1200, 1200))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    text = f"== תיאור ==\n{desc}\n\n== מקור ורישיון ==\n{source}\n\n[[קטגוריה:תמונות של אישים]]\n"
    c = WikiClient()
    c.login()
    r = c.session.get(API_URL, params={"action": "query", "titles": f"קובץ:{filename}", "format": "json"})
    if "missing" not in next(iter(r.json()["query"]["pages"].values())):
        print(f"SKIP (exists): {filename}")
        return
    r = c.session.post(API_URL, data={
        "action": "upload", "filename": filename, "comment": "תמונה שנמסרה לפרדספדיה",
        "text": text, "token": c._csrf_token(), "ignorewarnings": "1", "format": "json",
    }, files={"file": (filename, io.BytesIO(buf.getvalue()), "image/jpeg")})
    res = r.json()
    print("UPLOADED" if res.get("upload", {}).get("result") == "Success" else f"FAILED {res}", filename)


if __name__ == "__main__":
    main()
