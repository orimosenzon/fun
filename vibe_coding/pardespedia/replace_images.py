#!/usr/bin/env python3
"""
Replace 3 persona portraits on pardespedia with images DIFFERENT from the ones
used on the subjects' Hebrew Wikipedia pages (the originals duplicated WP's
infobox photo, which was confusing). Overwrites the existing wiki filename.
"""
import io
import requests
from wiki_client import WikiClient, API_URL

H = {"User-Agent": "Pardespedia-Bot/1.0 (orimosenzon@gmail.com)"}

REPLACEMENTS = {
    "שי אביבי.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f8/Avivi.jpg",
        "commons": "File:Avivi.jpg",
        "author": "Magister",
        "license": "ייחוס (Attribution) — רישיון חופשי המחייב ייחוס",
        "license_url": "https://commons.wikimedia.org/wiki/File:Avivi.jpg",
        "desc": "שי אביבי",
    },
    "אברהם טל.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9c/%D7%90%D7%91%D7%A8%D7%94%D7%9D_%D7%98%D7%9C_%28%D7%96%D7%9E%D7%A8%29.jpg",
        "commons": "File:אברהם טל (זמר).jpg",
        "author": "התקבל מאברהם טל (חיתוך: הידרו)",
        "license": "CC BY-SA 2.5",
        "license_url": "https://creativecommons.org/licenses/by-sa/2.5",
        "desc": "אברהם טל",
    },
    "קובי שבתאי.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8b/Yaakov_Shabtai%2C_Sept_mod.jpg",
        "commons": "File:Yaakov Shabtai, Sept mod.jpg",
        "author": "משטרת ישראל (Israel Police); חיתוך: Bo",
        "license": "CC BY-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
        "desc": "קובי שבתאי",
    },
}


def reupload(client, filename, spec):
    img = requests.get(spec["url"], headers=H)
    img.raise_for_status()
    desc = (
        f"{spec['desc']}\n\n"
        f"מקור: [[commons:{spec['commons']}|{spec['commons']}]] מוויקישיתוף.\n\n"
        f"צלם/יוצר: {spec['author']}.\n\n"
        f"רישיון: {spec['license']} ([{spec['license_url']} פרטים]).\n"
    )
    token = client._csrf_token()
    r = client.session.post(API_URL, data={
        "action": "upload",
        "filename": filename,
        "comment": "החלפה לתמונה שונה מזו שבוויקיפדיה (כדי למנוע בלבול)",
        "text": desc,
        "token": token,
        "ignorewarnings": "1",
        "format": "json",
    }, files={"file": (filename, io.BytesIO(img.content), "image/jpeg")})
    r.raise_for_status()
    res = r.json()
    if res.get("upload", {}).get("result") == "Success":
        print(f"REPLACED: {filename}")
    else:
        print(f"FAILED: {filename} -> {res}")


def main():
    client = WikiClient()
    client.login()
    for filename, spec in REPLACEMENTS.items():
        reupload(client, filename, spec)


if __name__ == "__main__":
    main()
