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
    base = card.get("photo_name") or card.get("name") or card["title"]
    suffix = "" if i == 0 else f" {i + 1}"
    return f"{base}{suffix} - נספח השימור.jpg"


def file_page(card, num):
    return f"""== תיאור ==
{card.get('name') or card['title']}, צילום מתוך כרטיס האתר (מס' {card.get('numbers', num)}) ב[[{HUB}]] של פרדס חנה-כרכור. הצילום נעשה בסביבות {card.get('card_date') or '2011'}, בזמן הסקר.

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


def build(num, card):
    """Assemble a full article from a card JSON that carries a hand-written
    'body' (prose, using <ref name="card" /> for the card citation)."""
    photos = card["photo"] if isinstance(card["photo"], list) else [card["photo"]]
    cap = card.get("caption") or f"{card['title'].split(' (')[0]}, צילום מתוך נספח השימור (בסביבות 2011)"
    parts = [infobox(num, card)]
    if photos and photos[0]:
        parts.append(f"[[קובץ:{photo_title(card, 0)}|ממוזער|{cap}]]")
    parts.append(card["body"].strip())
    see = ["[[נספח השימור לתוכנית המתאר]]: הרשימה המלאה של אתרי השימור"] + card.get("see_also", [])
    parts.append("== ראו גם ==\n" + "\n".join(f"* {s}" for s in see))
    page = card.get("page", "")
    cite = (f'<ref name="card">{{{{ציטוט נספח השימור}}}}, כרטיס אתר {card.get("numbers", num)}'
            + (f", עמ' {page}–{page + 1}" if page else "") + ".</ref>")
    if card.get("cite"):
        cite = f'<ref name="card">{{{{ציטוט נספח השימור}}}}, {card["cite"]}.</ref>'
    parts.append(f"== מקורות ==\n<references>\n{cite}\n</references>")
    if card.get("reading"):
        parts.append("== לקריאה נוספת ==\n" + "\n".join(f"* {r}" for r in card["reading"]))
    parts.append("\n".join(f"[[קטגוריה:{c}]]" for c in card.get("cats", ["היסטוריה"])))
    return "\n\n".join(parts) + "\n"


def publish(num):
    """Upload the card photo(s) and create the article (createonly)."""
    card = load(num)
    text = build(num, card)
    with open(article_path(num), "w", encoding="utf-8") as fh:
        fh.write(text)
    c = WikiClient().login()
    photos = card["photo"] if isinstance(card["photo"], list) else [card["photo"]]
    for i, p in enumerate(photos):
        if p:
            upload_photo(c, card, num, i)
    c.edit_page(card["title"], text,
                summary=f"ערך חדש: אתר {card.get('numbers', num)} בנספח השימור (בקשת עירית רתם)",
                create_only=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd.startswith("augment"):
        _main_existing()
        return
    if cmd == "build":
        print(build(sys.argv[2], load(sys.argv[2])))
        return
    if cmd == "publish":
        for num in sys.argv[2:]:
            publish(num)
        return
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


# ---------------------------------------------------------------- existing articles

ANCHORS = ("== ראו גם", "==ראו גם", "== מקורות", "==מקורות", "== קישורים חיצוניים",
           "==קישורים חיצוניים", "== הערות שוליים", "[[קטגוריה:")


def augment(num, card, dry=False):
    """Add the infobox and a "בנספח השימור" section to an article people wrote.
    get-first, insert-only: nothing that is there is removed or rewritten."""
    import re
    c = WikiClient()
    if not dry:
        c.login()
    title = card["title"]
    text = c.get_page(title)["wikitext"]
    if not text:
        raise RuntimeError(f"{title}: page is empty or missing")
    if "{{אתר לשימור" in text or "== בנספח השימור ==" in text:
        print(f"{title}: already augmented, skipped")
        return
    page = card["page"]
    cite = (f'<ref name="card{num}">{{{{ציטוט נספח השימור}}}}, כרטיס אתר '
            f'{card.get("numbers", num)}, עמ\' {page}–{page + 1}.</ref>')
    section = "== בנספח השימור ==\n" + card["section"].strip().replace("<ref card/>", cite, 1) \
        .replace("<ref card/>", f'<ref name="card{num}" />') + "\n\n"
    # insert before the first trailing block (see also / sources / links / categories)
    pos = min([i for i in (text.find(a) for a in ANCHORS) if i != -1] or [len(text)])
    new = text[:pos].rstrip() + "\n\n" + section + text[pos:]
    if "<references" not in new and "{{הערות שוליים" not in new:
        cpos = new.find("[[קטגוריה:")
        block = "== הערות שוליים ==\n<references />\n\n"
        new = (new[:cpos] + block + new[cpos:]) if cpos != -1 else new.rstrip() + "\n\n" + block
    if card.get("infobox", True):
        new = infobox(num, card) + "\n" + new
    elif "[[קטגוריה:מבנים לשימור]]" not in new:
        new = new.rstrip() + "\n[[קטגוריה:מבנים לשימור]]\n"
    if dry:
        print(new)
        return
    c.edit_page(title, new, summary=f"תבנית אתר לשימור ומידע מנספח השימור (אתר {card.get('numbers', num)})")


def _main_existing():
    if len(sys.argv) > 2 and sys.argv[1] in ("augment", "augment-dry"):
        for num in sys.argv[2:]:
            augment(num, load(num), dry=sys.argv[1] == "augment-dry")
        return True
    return False


if __name__ == "__main__":
    main()
