"""Conservation-appendix articles (Irit Rotem's request, 12/11/2026).

The master plan's conservation appendix (נספח השימור, plan 353-0138586, Dr.
Hadas Shadar, Jan 2015) lists 92 heritage sites, each with a card. This
module turns a card transcription (~/.cache/pardespedia_shimur/cards/<n>.json,
read by eye from the page images, because the PDF's text layer garbles every
digit) into a wiki article, uploads the card photo as fair use, and adds the
infobox + a "בנספח השימור" section to articles that already exist.

    python3 shimur.py template            # create/refresh תבנית:אתר לשימור
    python3 shimur.py photo <n> [<i>]     # upload a card photo
    python3 shimur.py dry <n>             # print the generated article
    python3 shimur.py create <n>          # create the article (createonly)

Nothing here overwrites a page a person wrote: new articles use createonly,
and existing ones are edited get-first by a separate, explicit call.
"""

import json
import os
import sys

from pyproj import Transformer

from wiki_client import WikiClient

CACHE = os.path.expanduser("~/.cache/pardespedia_shimur")
HUB = "נספח השימור לתוכנית המתאר"
TEMPLATE = "תבנית:אתר לשימור"
SOURCE_NOTE = ("נספח השימור לתוכנית המתאר המקומית לפרדס חנה-כרכור, תכנית מס' 353-0138586, "
               "עורכת הנספח: אדר' ד\"ר הדס שדר, ינואר 2015")

NARRATIVES = {
    "1": "נרטיב המים", "2": "נרטיב החקלאות", "3": "נרטיב המושבה",
    "4": "נרטיב המנדט הבריטי", "5": "נרטיב המבנים הביטחוניים",
    "6": "נרטיב קליטת העלייה", "7": "נרטיב המוסדות הייחודיים",
}
GRADES = {"מחמיר": "שימור עם הגבלות מחמירות", "רגיל": "שימור רגיל"}

_itm = Transformer.from_crs(2039, 4326, always_xy=True)


def latlon(x, y):
    lon, lat = _itm.transform(x, y)
    return round(lat, 6), round(lon, 6)


def map_link(card):
    if card.get("x"):
        lat, lon = latlon(card["x"], card["y"])
        return (f"[https://www.google.com/maps/search/?api=1&query={lat},{lon}&hl=iw "
                f"{lat}, {lon}]")
    return "—"


TEMPLATE_TEXT = """<includeonly>{| class="wikitable" style="float:left; clear:left; margin:0 1em 1em 0; width:290px; font-size:90%;"
|+ style="font-size:110%;" | [[נספח השימור לתוכנית המתאר|אתר לשימור]]
|-
! style="width:38%;" | מספר ברשימה
| '''{{{מספר}}}'''
|-
! דרגת שימור
| {{{דרגה}}}
|-
! נרטיב
| {{{נרטיב}}}
|-
! כתובת
| {{{כתובת}}}
|-
! גוש וחלקה
| {{{גוש}}}
|-
! הקמה
| {{{הקמה}}}
|-
! מיקום
| {{{מיקום}}}
|}[[קטגוריה:מבנים לשימור]]</includeonly><noinclude>
תבנית לערכים על אתרים מתוך [[נספח השימור לתוכנית המתאר]] של פרדס חנה-כרכור (2015).
התבנית מציגה את מספר האתר ברשימת השימור ואת דרגת השימור שלו, ומוסיפה את הערך ל[[:קטגוריה:מבנים לשימור]].

כל השדות חובה: הרחבה ParserFunctions אינה מותקנת באתר, ולכן שדה ריק יוצג כטקסט גולמי. כשאין מידע כותבים "—".

<pre>
{{אתר לשימור
| מספר   = 3.3.18
| דרגה   = שימור רגיל
| נרטיב  = נרטיב המושבה
| כתובת  = רחוב הנחילות, כרכור
| גוש    = גוש 10071, חלקה 12
| הקמה   = שנות ה-40
| מיקום  = [https://www.google.com/maps/search/?api=1&query=32.48,34.98&hl=iw 32.48, 34.98]
}}
</pre>
</noinclude>"""


def infobox(num, card):
    narr = NARRATIVES[num.split(".")[0]]
    return ("{{אתר לשימור\n"
            f"| מספר   = {card.get('numbers', num)}\n"
            f"| דרגה   = {GRADES[card['grade']]}\n"
            f"| נרטיב  = {narr}\n"
            f"| כתובת  = {card.get('address_short') or card['address']}\n"
            f"| גוש    = {card.get('gush') or '—'}\n"
            f"| הקמה   = {card.get('year_short') or '—'}\n"
            f"| מיקום  = {map_link(card)}\n"
            "}}")


def photo_title(card, i=0):
    base = card.get("photo_name") or card["name"]
    suffix = "" if i == 0 else f" {i + 1}"
    return f"{base}{suffix} - נספח השימור.jpg"


def file_page(card, num):
    return f"""== תיאור ==
{card['name']}, צילום מתוך כרטיס האתר (מס' {card.get('numbers', num)}) ב[[{HUB}]] של פרדס חנה-כרכור. הצילום נעשה בסביבות {card.get('card_date') or '2011'}, בזמן הסקר.

== מקור ==
{SOURCE_NOTE}. נספח לתוכנית המתאר המקומית, מסמך תכנוני ציבורי.

צילומי הנספח: אדריכל אמיר תומשוב והעמותה לתולדות פרדס חנה-כרכור (כך בשער הנספח; הנספח אינו מייחס כל צילום לצלם מסוים).

== רישיון ==
'''שימוש הוגן.''' הצילום מוגן בזכויות יוצרים ואינו ברישיון חופשי. הוא מובא כאן בשימוש הוגן לפי סעיף 19 לחוק זכות יוצרים, התשס"ח-2007, לצורך תיעוד אתר מורשת ציבורי בערך אנציקלופדי שאינו מסחרי, ברזולוציה נמוכה ובציון המקור. בעל זכויות שמבקש להסיר את הצילום מוזמן לכתוב ב[[שיחת משתמש:אורי מוסנזון בוט]].

[[קטגוריה:מבנים לשימור]]
"""


def upload_photo(client, card, num, i=0):
    photos = card["photo"] if isinstance(card["photo"], list) else [card["photo"]]
    path = os.path.join(CACHE, "img", photos[i])
    title = photo_title(card, i)
    token = client._csrf_token()
    with open(path, "rb") as fh:
        r = client.session.post("https://pardespedia.info/api.php", data={
            "action": "upload", "filename": title, "token": token, "format": "json",
            "comment": f"צילום מנספח השימור (אתר {card.get('numbers', num)}), שימוש הוגן",
            "text": file_page(card, num), "ignorewarnings": "0"},
            files={"file": (title, fh, "image/jpeg")}).json()
    res = r.get("upload", {}).get("result")
    if res == "Success":
        print(f"uploaded: {title}")
    elif res == "Warning" and "exists" in r["upload"].get("warnings", {}):
        print(f"exists already: {title}")
    else:
        raise RuntimeError(f"upload failed for {title}: {r}")
    return title


def load(num):
    with open(os.path.join(CACHE, "cards", f"{num}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def article_path(num):
    return os.path.join(CACHE, "articles", f"{num}.wiki")


def create(num):
    card = load(num)
    with open(article_path(num), encoding="utf-8") as fh:
        text = fh.read()
    c = WikiClient().login()
    c.edit_page(card["title"], text, summary=f"ערך חדש: אתר {card.get('numbers', num)} בנספח השימור",
                create_only=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "template":
        c = WikiClient().login()
        c.edit_page(TEMPLATE, TEMPLATE_TEXT, summary="תבנית לאתרי נספח השימור (בקשת עירית רתם)")
    elif cmd == "photo":
        num = sys.argv[2]
        i = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        upload_photo(WikiClient().login(), load(num), num, i)
    elif cmd == "infobox":
        print(infobox(sys.argv[2], load(sys.argv[2])))
    elif cmd == "create":
        create(sys.argv[2])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
