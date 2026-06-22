#!/usr/bin/env python3
"""Upload a local image file to pardespedia with a license/attribution page.

Usage:
    python3 upload_local.py <local_path> <target_filename> <description> <author>

Example:
    python3 upload_local.py ~/Pictures/x.jpg "תאיר ענבר.jpg" "תאיר ענבר" "אורי מוסנזון"

License is the uploader's own work (CC BY-SA 4.0). Skips if the target exists.
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
    local_path, filename, desc, author = sys.argv[1:5]
    local_path = os.path.expanduser(local_path)
    client = WikiClient()
    client.login()

    if file_exists(client, filename):
        print(f"SKIP (exists): {filename}")
        return

    with open(local_path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(local_path)[0] or "image/jpeg"

    text = (
        f"{desc}\n\n"
        f"צלם/מקור: {author}.\n\n"
        f"רישיון: יצירה עצמית של מעלה הקובץ, שניתנה לשימוש בפרדספדיה "
        f"([https://creativecommons.org/licenses/by-sa/4.0 CC BY-SA 4.0]).\n"
    )
    token = client._csrf_token()
    r = client.session.post(API_URL, data={
        "action": "upload",
        "filename": filename,
        "comment": "העלאת דיוקן עבור ערך אישים (יצירה עצמית)",
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
