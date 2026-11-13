"""Urban-nature survey articles (follow-up to the conservation appendix, 13/11/2026).

The nature survey of Pardes Hanna-Karkur (SPNI, April 2014) comes with a card
file: 47 survey sites, each with a two-page card (borders, access, flora,
fauna, hazards, land use, recommendations) and a list of point findings. The
PDF's text layer is scrambled, so every card was read from the page images
into ~/.cache/pardespedia_teva/cards/<n>.json. The prose of each article is
hand-written in ~/.cache/pardespedia_teva/articles/<key>.json.

    python3 teva.py templates          # create the infobox + citation templates
    python3 teva.py build <key>        # print the article
    python3 teva.py publish <key>...   # upload the card aerial, create (createonly)
    python3 teva.py augment[-dry] <key>...   # add a section to an existing article
    python3 teva.py hub                # create/refresh the survey article

New articles are createonly. Existing ones are edited get-first, insert-only.
"""
import json
import os
import sys

from wiki_client import WikiClient

CACHE = os.path.expanduser("~/.cache/pardespedia_teva")
HUB = "סקר תשתיות טבע עירוני"
CARDS_URL = "https://drive.google.com/file/d/13AEsdISYosIoQe82_24P3OEV-5dTWGRc/view"
REPORT_URL = "https://drive.google.com/file/d/10qwVNDWsvKIHi_qrqb5v6fnucpXVXxo-/view"

INFOBOX = """<includeonly>{| class="wikitable" style="float:left; clear:left; margin:0 1em 1em 0; width:290px; font-size:90%;"
|+ style="font-size:110%;" | [[סקר תשתיות טבע עירוני|אתר בסקר הטבע העירוני]]
|-
! style="width:38%;" | מספר בסקר
| '''{{{מספר}}}'''
|-
! שם בסקר
| {{{שם}}}
|-
! שטח
| {{{שטח}}}
|-
! אופי השטח
| {{{אופי}}}
|-
! מיני דגל
| {{{דגל}}}
|-
! ייעוד בתוכנית המתאר
| {{{ייעוד}}}
|}[[קטגוריה:טבע עירוני]][[קטגוריה:אתרי סקר הטבע העירוני]]</includeonly><noinclude>
תבנית לערכים על אתרים מתוך [[סקר תשתיות טבע עירוני]] של פרדס חנה-כרכור (2014). היא מציגה את מספר האתר בסקר ואת עיקרי הכרטיס שלו, ומוסיפה את הערך ל[[:קטגוריה:טבע עירוני]] ול[[:קטגוריה:אתרי סקר הטבע העירוני]].

כל השדות חובה: ההרחבה ParserFunctions אינה מותקנת באתר, ולכן שדה ריק יוצג כטקסט גולמי. כשאין מידע כותבים "אין".

<pre>
{{אתר טבע עירוני
| מספר  = 31
| שם    = שמורת אלון
| שטח   = 22 דונם
| אופי  = יער פארק של אלון התבור
| דגל   = אלון התבור, צבעוני ההרים
| ייעוד = שמורת טבע, קרקע חקלאית ודרך
}}
</pre>
</noinclude>"""

CITE = ("<includeonly>[[סקר תשתיות טבע עירוני|סקר טבע עירוני פרדס חנה-כרכור]], כרטיסי האתרים, "
        "החברה להגנת הטבע (היחידה לסקרי טבע עירוני) והמועצה המקומית פרדס חנה-כרכור, 2014 "
        f"([{CARDS_URL} הכרטיסים המלאים, PDF])</includeonly><noinclude>הפניה אחידה לכרטיסי האתרים של "
        "[[סקר תשתיות טבע עירוני]], לשימוש בתוך <code>&lt;ref&gt;</code>. אחריה כותבים את מספר האתר, למשל: "
        "<code>{{ציטוט סקר טבע}}, כרטיס אתר 31.</code></noinclude>")

CAT_TEXT = ("אתרים שנסקרו ב[[סקר תשתיות טבע עירוני]] של פרדס חנה-כרכור (החברה להגנת הטבע, 2014). "
            "הרשימה המלאה של 47 אתרי הסקר נמצאת בערך על הסקר.\n\n[[קטגוריה:טבע עירוני]]")


def art(key):
    with open(os.path.join(CACHE, "articles", f"{key}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def card(n):
    with open(os.path.join(CACHE, "cards", f"{n}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def nums(a):
    return a["sites"] if "sites" in a else [a["site"]]


def cite(a, first=True):
    ns = nums(a)
    label = ("כרטיס אתר " + str(ns[0])) if len(ns) == 1 else ("כרטיסי אתרים " + " ו-".join(map(str, ns)))
    return f'<ref name="card">{{{{ציטוט סקר טבע}}}}, {label}.</ref>'


def photo_title(a):
    return f"{a['title']} - סקר הטבע העירוני.jpg"


def file_page(a):
    ns = " ו-".join(map(str, nums(a)))
    return f"""== תיאור ==
{a['title']}: תצלום אוויר עם גבול אתר הסקר (בקו אדום), מתוך כרטיס אתר {ns} ב[[{HUB}]] של פרדס חנה-כרכור. התצלום משקף את מצב השטח בזמן הסקר, 2012–2013.

== מקור ==
סקר טבע עירוני פרדס חנה-כרכור, כרטיסי האתרים. החברה להגנת הטבע (היחידה לסקרי טבע עירוני) והמועצה המקומית פרדס חנה-כרכור, 2014. מסמך ציבורי שהוכן עבור המועצה.

== רישיון ==
'''שימוש הוגן.''' התצלום מוגן בזכויות יוצרים ואינו ברישיון חופשי. הוא מובא כאן בשימוש הוגן לפי סעיף 19 לחוק זכות יוצרים, התשס"ח-2007, לצורך תיעוד אתר טבע ציבורי בערך אנציקלופדי שאינו מסחרי, ברזולוציה נמוכה ובציון המקור. בעל זכויות שמבקש להסיר את התצלום מוזמן לכתוב ב[[שיחת משתמש:אורי מוסנזון בוט]].

[[קטגוריה:טבע עירוני]]
"""


def infobox(a):
    c = card(nums(a)[0])
    area = a.get("area") or (c["area"] + " דונם")
    return ("{{אתר טבע עירוני\n"
            f"| מספר  = {' ו-'.join(map(str, nums(a)))}\n"
            f"| שם    = {a.get('survey_name') or c['name']}\n"
            f"| שטח   = {area}\n"
            f"| אופי  = {a['kind']}\n"
            f"| דגל   = {a['flag']}\n"
            f"| ייעוד = {a['zoning']}\n"
            "}}")


def build(key):
    a = art(key)
    parts = [infobox(a)]
    if a.get("photo", True):
        parts.append(f"[[קובץ:{photo_title(a)}|ממוזער|{a['title']} בתצלום אוויר מתוך כרטיס הסקר. הקו האדום הוא גבול אתר הסקר]]")
    body = a["body"].strip().replace("<ref card/>", cite(a), 1).replace("<ref card/>", '<ref name="card" />')
    g = G2023
    _unused = ('<ref name="g2023">הנחיות לטיפוח תשתיות טבע עירוני וטיפול במפגעים סביבתיים, פרדס חנה-כרכור. '
         'יחידת הטבע העירוני בחברה להגנת הטבע, 2023, פרק 3.1: האתרים הנבחרים '
         '([https://drive.google.com/file/d/162UWJlDgvutXLfDar5-yUzb35zBH9XNQ/view המסמך המלא, PDF]).</ref>')
    body = body.replace("<ref g/>", g, 1).replace("<ref g/>", '<ref name="g2023" />')
    parts.append(body)
    see = [f"[[{HUB}]]: הסקר ורשימת 47 האתרים שלו"] + a.get("see_also", [])
    parts.append("== ראו גם ==\n" + "\n".join(f"* {s}" for s in see))
    parts.append("== מקורות ==\n<references />")
    if a.get("reading"):
        parts.append("== לקריאה נוספת ==\n" + "\n".join(f"* {r}" for r in a["reading"]))
    extra = "\n".join(f"[[קטגוריה:{c}]]" for c in a.get("cats", []))
    if extra:
        parts.append(extra)
    return "\n\n".join(parts) + "\n"


def upload_photo(client, a, key):
    path = os.path.join(CACHE, "img", f"{key}.jpg")
    title = photo_title(a)
    token = client._csrf_token()
    with open(path, "rb") as fh:
        r = client.session.post("https://pardespedia.info/api.php", data={
            "action": "upload", "filename": title, "token": token, "format": "json",
            "comment": "תצלום אוויר מכרטיס האתר בסקר הטבע העירוני, שימוש הוגן",
            "text": file_page(a), "ignorewarnings": "0"},
            files={"file": (title, fh, "image/jpeg")}).json()
    res = r.get("upload", {}).get("result")
    if res == "Success":
        print(f"uploaded: {title}")
    elif res == "Warning" and ("exists" in r["upload"].get("warnings", {})
                               or "duplicate" in r["upload"].get("warnings", {})):
        print(f"exists already: {title} {list(r['upload']['warnings'])}")
    else:
        raise RuntimeError(f"upload failed for {title}: {r}")


def publish(key, client):
    a = art(key)
    text = build(key)
    if a.get("photo", True):
        upload_photo(client, a, key)
    client.edit_page(a["title"], text,
                     summary=f"ערך חדש: אתר {' ו-'.join(map(str, nums(a)))} בסקר הטבע העירוני",
                     create_only=True)
    print("created:", a["title"])


ANCHORS = ("== ראו גם", "==ראו גם", "== מקורות", "==מקורות", "== קישורים חיצוניים",
           "==קישורים חיצוניים", "== הערות שוליים", "==הערות שוליים", "{{הרחבה}}", "[[קטגוריה:")
SECTION = "== בסקר הטבע העירוני =="
G2023 = ('<ref name="g2023">הנחיות לטיפוח תשתיות טבע עירוני וטיפול במפגעים סביבתיים, פרדס חנה-כרכור. '
         'יחידת הטבע העירוני בחברה להגנת הטבע, 2023, פרק 3.1: האתרים הנבחרים '
         '([https://drive.google.com/file/d/162UWJlDgvutXLfDar5-yUzb35zBH9XNQ/view המסמך המלא, PDF]).</ref>')


def augment(key, client, dry=False):
    """Add a "בסקר הטבע העירוני" section to an article people wrote.
    get-first, insert-only: nothing that is there is removed or rewritten."""
    a = art(key)
    title = a["title"]
    text = client.get_page(title)["wikitext"]
    if not text:
        raise RuntimeError(f"{title}: page is empty or missing")
    if SECTION in text:
        print(f"{title}: already augmented, skipped")
        return
    n = nums(a)[0]
    ref = f'<ref name="teva{n}">{{{{ציטוט סקר טבע}}}}, כרטיס אתר {n}.</ref>'
    sec = a["section"].strip().replace("<ref card/>", ref, 1).replace("<ref card/>", f'<ref name="teva{n}" />')
    sec = sec.replace("<ref g/>", G2023, 1).replace("<ref g/>", '<ref name="g2023" />')
    section = SECTION + "\n" + sec + "\n\n"
    pos = min([i for i in (text.find(x) for x in ANCHORS) if i != -1] or [len(text)])
    new = text[:pos].rstrip() + "\n\n" + section + text[pos:]
    if "<references" not in new and "{{הערות שוליים" not in new:
        cpos = min([i for i in (new.find(x) for x in ("{{הרחבה}}", "[[קטגוריה:")) if i != -1] or [-1])
        block = "== הערות שוליים ==\n<references />\n\n"
        new = (new[:cpos] + block + new[cpos:]) if cpos != -1 else new.rstrip() + "\n\n" + block
    for cat in ("[[קטגוריה:טבע עירוני]]", "[[קטגוריה:אתרי סקר הטבע העירוני]]"):
        if cat not in new and "{{אתר טבע עירוני" not in new:
            new = new.rstrip() + "\n" + cat + "\n"
    if dry:
        print(new)
        return
    client.edit_page(title, new, summary=f"סעיף חדש: ממצאי סקר הטבע העירוני (אתר {n})")
    print("augmented:", title)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        print(build(sys.argv[2]))
    elif cmd == "templates":
        c = WikiClient().login()
        c.edit_page("תבנית:אתר טבע עירוני", INFOBOX, summary="תבנית לאתרי סקר הטבע העירוני")
        c.edit_page("תבנית:ציטוט סקר טבע", CITE, summary="הפניה אחידה לכרטיסי סקר הטבע העירוני")
        c.edit_page("קטגוריה:אתרי סקר הטבע העירוני", CAT_TEXT, summary="תיאור הקטגוריה")
    elif cmd == "publish":
        c = WikiClient().login()
        for k in sys.argv[2:]:
            publish(k, c)
    elif cmd in ("augment", "augment-dry"):
        c = WikiClient()
        if cmd == "augment":
            c.login()
        for k in sys.argv[2:]:
            augment(k, c, dry=cmd == "augment-dry")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
