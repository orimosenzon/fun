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
"""

import json
import sys
import urllib.request

DATA = ("https://raw.githubusercontent.com/orimosenzon/"
        "derech-kitzur-data/main/data/trails.json")
IMG = "https://raw.githubusercontent.com/orimosenzon/derech-kitzur-data/main/"
APP = "https://orimosenzon.github.io/fun/vibe_coding/dereh_kitzur/"
OUT = "derech_kitzur.kml"


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
    bits.append(f'<a href="{APP}">פתיחה באפליקציה</a>')
    return esc("<br>".join(bits))


def main():
    with urllib.request.urlopen(DATA, timeout=60) as res:
        data = json.load(res)

    styles, places = [], []
    seen = set()

    for seg in data["segments"]:
        colour = seg.get("color") or "#097138"
        sid = "s" + colour.lstrip("#")
        if sid not in seen:
            seen.add(sid)
            styles.append(f'  <Style id="{sid}"><LineStyle>'
                          f'<color>{kml_colour(colour)}</color><width>4</width>'
                          f'</LineStyle></Style>')
        coords = " ".join(f"{lng},{lat},0" for lat, lng in seg["path"])
        places.append(f"""  <Placemark>
    <name>{esc(seg["name"])}</name>
    <description>{describe(seg)}</description>
    <styleUrl>#{sid}</styleUrl>
    <LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString>
  </Placemark>""")

    for wp in data["waypoints"]:
        places.append(f"""  <Placemark>
    <name>{esc(wp["name"])}</name>
    <description>{describe(wp)}</description>
    <Point><coordinates>{wp["lng"]},{wp["lat"]},0</coordinates></Point>
  </Placemark>""")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>דרך קיצור · שבילי פרדס חנה-כרכור</name>
  <description>{esc(f'נוצר מהמסד של היוזמה, עדכון אחרון {data.get("updated", "")}')}</description>
{chr(10).join(styles)}
{chr(10).join(places)}
</Document>
</kml>
"""
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(kml)

    print(f"wrote {OUT}: {len(data['segments'])} שבילים, "
          f"{len(data['waypoints'])} נקודות ציון, {len(kml) / 1024:.0f} KB")
    print("ב-My Maps: הוספת שכבה, ייבוא, בחירת הקובץ, ואז מחיקת השכבה הישנה.")


if __name__ == "__main__":
    sys.exit(main())
