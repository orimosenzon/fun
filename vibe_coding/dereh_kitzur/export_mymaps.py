#!/usr/bin/env python3
"""Generate the KML that keeps Yoav's Google My Maps layer in step with us.

The direction used to run the other way: the map was the source and this
project read it. Since 22/8/2026 the dataset lives in its own repo, the app
writes to it directly, and the map receives a copy - because Google offers no
way at all to write into a My Maps layer from code.

    python3 export_mymaps.py
    # then, in My Maps: add a layer, Import, pick the file, delete the old layer

Importing always creates a *new* layer and cannot merge into an existing one,
so refreshing means importing the new file and deleting the previous layer.
Ten layers is the ceiling per map.

One file comes out per trail layer, because that is the shape My Maps wants
anyway: one import, one layer. A dataset with no extra layers produces the one
file it always did.
"""

import json
import re
import sys
import urllib.request

DATA = ("https://raw.githubusercontent.com/orimosenzon/"
        "derech-kitzur-data/main/data/trails.json")
IMG = "https://raw.githubusercontent.com/orimosenzon/derech-kitzur-data/main/"
APP = "https://orimosenzon.github.io/fun/vibe_coding/dereh_kitzur/"
BASE = "derech_kitzur"

MAIN = {"id": None, "name": "דרך קיצור · שבילי פרדס חנה-כרכור"}


def esc(text):
    return (str(text or "").replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def kml_colour(hex_colour):
    """CSS #rrggbb to KML aabbggrr, which reverses the channels *and* moves
    alpha to the front."""
    h = (hex_colour or "#097138").lstrip("#")
    return "ff" + h[4:6] + h[2:4] + h[0:2]


def describe(item):
    """The description is HTML that My Maps renders, carried inside XML.

    That means two rounds of escaping, and they are not the same round: the
    user's text is escaped as HTML, and the finished HTML is then escaped as
    XML. Emitting a bare <br> here is what makes the file fail to parse.
    """
    bits = []
    if item.get("note"):
        bits.append(esc(item["note"]))
    if item.get("length"):
        bits.append(f'אורך {item["length"]} מ׳')
    if item.get("origin") == "app":
        bits.append("נוסף מאפליקציית דרך קיצור")

    # Photos travel with the map rather than being left behind in the repo.
    for photo in item.get("photos", [])[:4]:
        bits.append(f'<img src="{IMG}{photo["full"]}" width="400">')
    for link in item.get("links", [])[:4]:
        bits.append(f'<a href="{esc(link["url"])}">{esc(link.get("title") or link["url"])}</a>')
    bits.append(f'<a href="{APP}">פתיחה באפליקציה</a>')
    return esc("<br>".join(bits))


def placemarks(items):
    styles, places = [], []
    seen = set()

    for seg in items:
        if seg.get("path"):
            colour = seg.get("color") or "#097138"
            sid = "s" + colour.lstrip("#")
            if sid not in seen:
                seen.add(sid)
                styles.append(f'  <Style id="{sid}"><LineStyle>'
                              f'<color>{kml_colour(colour)}</color><width>4</width>'
                              f'</LineStyle></Style>')
            coords = " ".join(f"{lng},{lat},0" for lat, lng in seg["path"])
            body = (f"<LineString><tessellate>1</tessellate>"
                    f"<coordinates>{coords}</coordinates></LineString>")
            style = f"\n    <styleUrl>#{sid}</styleUrl>"
        else:
            body = f'<Point><coordinates>{seg["lng"]},{seg["lat"]},0</coordinates></Point>'
            style = ""

        places.append(f"""  <Placemark>
    <name>{esc(seg["name"])}</name>
    <description>{describe(seg)}</description>{style}
    {body}
  </Placemark>""")

    return styles, places


def slug(name):
    """A file name a person can match to a layer at the My Maps import dialog.

    Hebrew is kept: the person doing the import is picking the file that
    matches the layer they just made, and 'derech_kitzur_2.kml' tells them
    nothing about which one that is.
    """
    clean = re.sub(r"[^\w֐-׿-]+", "_", name).strip("_")
    return clean or "layer"


def write(layer, items, updated):
    styles, places = placemarks(items)
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{esc(layer["name"])}</name>
  <description>{esc(f'נוצר מהמסד של היוזמה, עדכון אחרון {updated}')}</description>
{chr(10).join(styles)}
{chr(10).join(places)}
</Document>
</kml>
"""
    name = BASE + ".kml" if layer["id"] is None else f"{BASE}_{slug(layer['name'])}.kml"
    with open(name, "w", encoding="utf-8") as fh:
        fh.write(kml)
    print(f"  {name}: {len(places)} פריטים, {len(kml) / 1024:.0f} KB")
    return name


def main():
    with urllib.request.urlopen(DATA, timeout=60) as res:
        data = json.load(res)

    known = {l["id"] for l in data.get("layers", [])}
    # A trail pointing at a layer that no longer exists rides with the main
    # file rather than being silently dropped from every file.
    home = lambda it: it.get("layer") if it.get("layer") in known else None

    everything = data["segments"] + data["waypoints"]
    updated = data.get("updated", "")

    print("נכתבו:")
    for layer in [MAIN] + data.get("layers", []):
        items = [it for it in everything if home(it) == layer["id"]]
        if items or layer["id"] is None:
            write(layer, items, updated)

    print(f"\nסך הכול {len(data['segments'])} שבילים ו-{len(data['waypoints'])} נקודות ציון.")
    print("ב-My Maps: לכל קובץ, הוספת שכבה, ייבוא, בחירת הקובץ, ומחיקת השכבה הישנה.")


if __name__ == "__main__":
    sys.exit(main())
