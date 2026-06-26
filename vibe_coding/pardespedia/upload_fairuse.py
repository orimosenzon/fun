#!/usr/bin/env python3
"""Upload a local image to pardespedia under a FAIR-USE rationale (not own work).

Usage:
    python3 upload_fairuse.py <local_path> <target_filename> <description_text_file>

The description file holds the full wikitext for the file page (desc, source, fair-use rationale).
Skips if the target already exists.
"""
import io
import os
import sys
import mimetypes
from wiki_client import WikiClient, API_URL


def file_exists(client, filename):
    r = client.session.get(API_URL, params={
        "action": "query", "titles": f"קובץ:{filename}",
        "prop": "imageinfo", "format": "json",
    })
    page = next(iter(r.json()["query"]["pages"].values()))
    return "missing" not in page


def main():
    local_path, filename, text_path = sys.argv[1:4]
    local_path = os.path.expanduser(local_path)
    with open(text_path, encoding="utf-8") as f:
        text = f.read()

    client = WikiClient()
    client.login()
    if file_exists(client, filename):
        print(f"SKIP (exists): {filename}")
        return

    with open(local_path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(local_path)[0] or "image/jpeg"
    token = client._csrf_token()
    r = client.session.post(API_URL, data={
        "action": "upload",
        "filename": filename,
        "comment": "העלאה בשימוש הוגן (חומר קידום של הפסטיבל, רזולוציה נמוכה)",
        "text": text,
        "token": token,
        "ignorewarnings": "1",
        "format": "json",
    }, files={"file": (filename, io.BytesIO(data), mime)})
    r.raise_for_status()
    res = r.json()
    if res.get("upload", {}).get("result") == "Success":
        print(f"UPLOADED: {filename}")
    else:
        print(f"FAILED: {filename} -> {res}")


if __name__ == "__main__":
    main()
