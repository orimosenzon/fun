#!/usr/bin/env python3
"""
Upload free-licensed images from Wikimedia Commons to pardespedia.

For each entry: downloads the Commons file, then uploads it to pardespedia
under a Hebrew filename with a proper attribution/license description page.
Idempotent-ish: skips upload if the target file already exists on the wiki.

Usage: python3 upload_commons.py
"""
import io
import requests
from wiki_client import WikiClient, API_URL

H = {"User-Agent": "Pardespedia-Bot/1.0 (orimosenzon@gmail.com)"}

# target_filename (no "קובץ:" prefix) -> source spec
IMAGES = {
    "שי אביבי.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/48/Shay_Avivi.jpg",
        "commons": "File:Shay Avivi.jpg",
        "author": "ליאור נורדמן (Lior Nordman)",
        "license": "CC BY-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
        "desc": "שי אביבי",
    },
    "חנן בן ארי.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/4c/Hanan_Ben-Ari%2C_October_2019.jpg",
        "commons": "File:Hanan Ben-Ari, October 2019.jpg",
        "author": "דולב (Dolev)",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
        "desc": "חנן בן ארי",
    },
    "אברהם טל.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/03/Avraham_Tal.jpg",
        "commons": "File:Avraham Tal.jpg",
        "author": "Moshe Nachumovich",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
        "desc": "אברהם טל",
    },
    "רמה בורשטין-שי.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b7/Rama_Burshtein.jpg",
        "commons": "File:Rama Burshtein.jpg",
        "author": "Ehud Arieli",
        "license": "CC BY-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
        "desc": "רמה בורשטין-שי",
    },
    "יצחק לאור.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/10/Yitzhak_Laor.jpg",
        "commons": "File:Yitzhak Laor.jpg",
        "author": "Alina Korn",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
        "desc": "יצחק לאור",
    },
    "קובי שבתאי.jpg": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/6d/Kobi_Shabtai_%28cropped%29.jpg",
        "commons": "File:Kobi Shabtai (cropped).jpg",
        "author": "Israel Police",
        "license": "CC BY-SA 3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0",
        "desc": "קובי שבתאי",
    },
}


def file_exists(client, filename):
    r = client.session.get(API_URL, params={
        "action": "query", "titles": f"קובץ:{filename}",
        "prop": "imageinfo", "format": "json",
    })
    page = next(iter(r.json()["query"]["pages"].values()))
    return "missing" not in page


def upload(client, filename, spec):
    if file_exists(client, filename):
        print(f"SKIP (exists): {filename}")
        return
    img = requests.get(spec["url"], headers=H)
    img.raise_for_status()
    desc = (
        f"{spec['desc']}\n\n"
        f"מקור: [[commons:{spec['commons']}|{spec['commons']}]] מוויקישיתוף.\n\n"
        f"צלם/יוצר: {spec['author']}.\n\n"
        f"רישיון: {spec['license']} ([{spec['license_url']} הרישיון]).\n"
    )
    token = client._csrf_token()
    r = client.session.post(API_URL, data={
        "action": "upload",
        "filename": filename,
        "comment": "העלאת דיוקן חופשי מוויקישיתוף עבור ערך אישים",
        "text": desc,
        "token": token,
        "ignorewarnings": "1",
        "format": "json",
    }, files={"file": (filename, io.BytesIO(img.content), "image/jpeg")})
    r.raise_for_status()
    res = r.json()
    if res.get("upload", {}).get("result") == "Success":
        print(f"UPLOADED: {filename}")
    else:
        print(f"FAILED: {filename} -> {res}")


def main():
    client = WikiClient()
    client.login()
    for filename, spec in IMAGES.items():
        upload(client, filename, spec)


if __name__ == "__main__":
    main()
