#!/usr/bin/env python3
"""Build the [[דרך הנדיב 2026]] guide page from the festival's own API.

The festival site (2026.hanadiv.org) is an Angular app over a public JSON API:
    https://api.hanadiv.org/festival?festival=2026   -> dates, links
    https://api.hanadiv.org/event?festival=2026      -> every approved event
Rerunning picks up events added since the last run. Event submission stays open
until the festival's publicationWindowEnd, so the page grows until then.

    python3 build_hanadiv_guide.py           # -> data/hanadiv_2026.json, guide_hanadiv_2026.wiki
    python3 build_hanadiv_guide.py --cached  # rebuild from the saved json
"""
import collections
import datetime as dt
import json
import os
import sys
import urllib.parse

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "hanadiv_2026.json")
OUT = os.path.join(HERE, "guide_hanadiv_2026.wiki")
API = "https://api.hanadiv.org"
YEAR = "2026"
EVENT_URL = "https://2026.hanadiv.org/event-page/{}"
TOWN = "פרדס חנה-כרכור"
# The app's root URL redirects to web/ and drops the query, so link web/ directly.
KITZUR_URL = "https://orimosenzon.github.io/fun/vibe_coding/dereh_kitzur/web/?layers=hanadiv"
MAP_SHOT = "דרך הנדיב 2026 - שכבת האירועים בדרך קיצור.jpg"

DAYS_HE = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
MONTHS_HE = ["", "בינואר", "בפברואר", "במרץ", "באפריל", "במאי", "ביוני", "ביולי",
             "באוגוסט", "בספטמבר", "באוקטובר", "בנובמבר", "בדצמבר"]

# Street fields as typed by hosts, mapped to (street, house number or None).
ADDRESS_FIX = {
    ("מתנס הבוטנים", 54): ("הבוטנים", 54),
    ("הבוטנים 54 פרדס חנה", 0): ("הבוטנים", 54),
    ("המושב/מעלה גיבורים", 5): ("המושב", 5),
    ("ההגנה", 0): ("ההגנה פינת הצנחנים", None),
    ("מרחוב הנורית - הוראות הגעה", 111): ("הנורית", None),
    ("פומלית", 24): ("פומלית", 2),  # the place notes say פומלית 2, like the host's other event
}

# Venues with a pardespedia article, keyed by normalized address.
VENUES = {
    ("החורש", 36): "[[שבילים (בית ספר)|בית ספר שבילים]]",
    ("הבוטנים", 54): '[[מתנ"ס פרדס חנה כרכור|המתנ"ס]]',
    ("המושב", 44): "[[מרכז בנגורה|בית בנגורה]]",
    ("ההגנה פינת הצנחנים", None): '[[האולם - בית תרבות מקומי פרכו"ר|האולם]]',
    ("נעורים", 27): "[[קפה ויעל]]",
    ("האורנים", 1): "[[הוואדי]] של ה[[שיקשוק]]",
    ("הדקלים", 90): "רחבת [[בית יד לבנים|יד לבנים]]",
    ("אורוות האמנים", 1): "[[לייבלב]], [[אורוות האמנים]]",
    ("אתרוג", 8): "המרכז הקהילתי [[נווה פרדסים (שכונה)|נווה פרדסים]]",
    ("יונה", 44): "[[גן הציפורים]], [[רמז (שכונה)|שכונת רמז]]",
    ("אחוזה", 43): '"הבית 43"',
}
# Sub-stages at a venue, recognized from the host's place description.
STAGES = [
    ("אמפי פנימי", "האמפי הפנימי"),
    ("במת הג'אז", "במת הג'אז ומוסיקת עולם"),
    ("אינדיגו", "אינדיגו, מתחם הילדים"),
    ("במה בראש", "במה בראש"),
    ("חנה - בר", "[[חנה בר]]"),
    ("הלובי", "הלובי"),
]

AUDIENCE = {"לכל המשפחה": "כל המשפחה", "רק למבוגרים": "מבוגרים", "רק לילדים": "ילדים"}


def fetch():
    fest = requests.get(f"{API}/festival", params={"festival": YEAR}, timeout=30).json()
    events = requests.get(f"{API}/event", params={"festival": YEAR}, timeout=30).json()
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", encoding="utf-8") as f:
        json.dump({"fetched": dt.date.today().isoformat(), "festival": fest, "events": events},
                  f, ensure_ascii=False, indent=1)


def clean(s):
    return " ".join((s or "").split())


def esc(s):
    """Text safe inside a table cell and inside [url text]."""
    return clean(s).replace("[", "&#91;").replace("]", "&#93;").replace("|", "&#124;") \
        .replace("{{", "&#123;&#123;").replace("''", "'&#39;")


def address(e):
    key = (clean(e["street"]), e["houseNumber"])
    return ADDRESS_FIX.get(key, key)


def maps_link(street, num):
    label = f"{street} {num}" if num else street
    q = urllib.parse.quote(f"{label}, {TOWN}")
    return f"[https://www.google.com/maps/search/?api=1&query={q}&hl=iw {label}]"


def stage_of(e):
    desc = clean(e["placeDescription"])
    for needle, name in STAGES:
        if needle in desc:
            return name
    return None


def place_cell(e):
    street, num = address(e)
    venue = VENUES.get((street, num))
    if venue:
        stage = stage_of(e)
        name = f"{venue}, {stage}" if stage else venue
    elif street == "הנורית":
        name = "חוות החזון הירוק (הגעה מרחוב הנורית, לפי ההוראות באתר)"
    else:
        name = esc(e["placeDescription"])
        if len(name) > 45:
            name = name[:45].rsplit(" ", 1)[0] + "…"
    return f"{name}<br /><small>{maps_link(street, num)}</small>"


def time_cell(e):
    start = dt.datetime.strptime(e["hour"], "%H:%M")
    if e["duration"] >= 300:
        return f"{e['hour']} ואילך"
    end = start + dt.timedelta(minutes=e["duration"])
    return f"{e['hour']}–{end:%H:%M}"


def title_of(e):
    t = clean(e["title"])
    if t.startswith("מתחם טיפולים"):
        return "מתחם טיפולים לאורך כל היום"
    return t


def event_cell(e):
    title = esc(title_of(e)).replace('"הערוגה "', '"הערוגה"')  # host's stray space
    cell = f"'''[{EVENT_URL.format(e['eventID'])} {title}]'''"
    who = esc(f"{e['initiativeOwnerFirstName'] or ''} {e['initiativeOwnerLastName'] or ''}")
    if who:
        cell += f"<br /><small>{who}</small>"
    if title_of(e).startswith("מתחם טיפולים"):
        cell += "<br /><small>ההרשמה ישירות מול המטפלים, לא דרך אתר הפסטיבל</small>"
    return cell


def notes_cell(e):
    parts = [AUDIENCE.get(e["public"], e["public"])]
    if e["isAccessible"]:
        parts.append("נגיש")
    return "<small>" + ", ".join(parts) + "</small>"


def he_date(d):
    return f"{d.day} {MONTHS_HE[d.month]}"


def build():
    with open(DATA, encoding="utf-8") as f:
        blob = json.load(f)
    fest, events = blob["festival"], blob["events"]
    fetched = dt.date.fromisoformat(blob["fetched"])
    events = [e for e in events if e["status"] == 1]
    events.sort(key=lambda e: (e["date"], e["hour"], clean(e["title"])))

    start = dt.date.fromisoformat(fest["fromDate"])
    until = dt.date.fromisoformat(fest["untilDate"])
    pub_end = dt.datetime.fromisoformat(fest["publicationWindowEnd"]).date()
    days = []
    d = start
    while d <= until:
        days.append(d)
        d += dt.timedelta(days=1)
    by_day = collections.defaultdict(list)
    for e in events:
        by_day[e["date"]].append(e)

    n = len(events)
    venues = collections.Counter(address(e) for e in events)
    genres = collections.Counter(e["genre"] for e in events)
    family = sum(e["public"] != "רק למבוגרים" for e in events)
    accessible = sum(bool(e["isAccessible"]) for e in events)
    alternadiv = [e for e in events if clean(e["title"]).startswith("אלטרנדיב")]

    L = []
    w = L.append
    w("''ערך זה מתאר את מהדורת 2026 של הפסטיבל. על הפסטיבל עצמו, ראו [[פסטיבל דרך הנדיב]].''")
    w("")
    w("[[קובץ:LOGO HANADIV.png|220px|לא ממוסגר|שמאל]]")
    w("")
    w(f"'''דרך הנדיב 2026''' היא המהדורה השמינית של [[פסטיבל דרך הנדיב]], פסטיבל התרבות "
      f"הקהילתי של [[פרדס חנה-כרכור]], והיא נערכת בימים {DAYS_HE[start.weekday()]} עד "
      f"{DAYS_HE[until.weekday()]}, '''{start.day} עד {until.day} {MONTHS_HE[until.month]} {until.year}'''. "
      "תושבים פותחים בתים, חצרות, סטודיואים ומבני ציבור ומארחים בהם הופעות, סדנאות, הרצאות, "
      "הקרנות ופעילויות לילדים. כל האירועים בכניסה חופשית, והיוצרים, המארחים וצוות ההפקה "
      "פועלים בהתנדבות.")
    w("")
    w("הדף מרכז את התוכנית כפי שהיא מופיעה באתר הפסטיבל: המוקדים הגדולים, ולוח האירועים המלא לפי ימים.")
    w("")
    top = ", ".join(f"{g} ({c})" for g, c in genres.most_common(4))
    w('<div style="background:#f4f7fb; border:1px solid #ccd6e4; border-right:5px solid #21659c; '
      'border-radius:10px; padding:14px 18px; margin:1em 0;">')
    w(f"'''בשורה אחת:''' {n} אירועים ב-{len(venues)} כתובות לאורך שלושה ימים. התחומים הבולטים: {top}. "
      f"{family} מהאירועים מתאימים גם לילדים, ו-{accessible} מהם הוגדרו נגישים. "
      f"התוכנית עוד מתמלאת: אפשר להוסיף אירועים עד {he_date(pub_end)}.")
    w("</div>")
    w("")

    w("== מתי ואיפה ==")
    w('<div style="overflow-x:auto;">')
    w('{| class="wikitable" style="width:auto"')
    w("! יום !! תאריך !! אירועים")
    for d in days:
        w("|-")
        w(f"| [[#יום {DAYS_HE[d.weekday()]}, {he_date(d)}|{DAYS_HE[d.weekday()]}]] || {he_date(d)} "
          f"|| style=\"text-align:center\" | {len(by_day[d.isoformat()])}")
    w("|}")
    w("</div>")
    w("")
    w("האירועים מפוזרים ברחבי המושבה, בבתים פרטיים ובמבני ציבור. לרוב האירועים מספר המקומות "
      "מוגבל, ונרשמים אליהם מראש בעמוד האירוע באתר הפסטיבל. שם מופיעים גם התיאור המלא, הוראות "
      "ההגעה ומה כדאי להביא.")
    w("")

    w("== במפה ==")
    w(f"[[קובץ:{MAP_SHOT}|מרכז|ממוזער|900px|אירועי הפסטיבל באפליקציית [[דרך קיצור]], "
      f"כשרק שכבת דרך הנדיב דלוקה. [{KITZUR_URL} לפתיחת המפה] ]]")  # "]]]" would swallow the link
    w("")
    w(f"אפליקציית [[דרך קיצור]] מציגה את אירועי הפסטיבל כשכבה על מפת שבילי ההליכה של המושבה. "
      f"'''[{KITZUR_URL} הקישור הזה]''' פותח אותה כשרק שכבת דרך הנדיב דלוקה. לחיצה על סיכה "
      "מציגה את פרטי האירוע, וכפתור \"איך מגיעים ברגל?\" מחשב מסלול הליכה אליו דרך קיצורי הדרך.")
    w("")

    w("== המוקדים המרכזיים ==")
    w("רוב האירועים מתקיימים בבית של מארח אחד, אבל כמה מקומות מרכזים סדרה שלמה של אירועים. "
      "מי שרוצה לראות הרבה בזמן קצר יכול להתחיל מהם.")
    w("")
    w('<div style="overflow-x:auto;">')
    w('{| class="wikitable sortable" style="width:auto"')
    w("! מוקד !! כתובת !! אירועים !! מה יש שם")
    for (street, num), c in venues.most_common():
        if c < 4:
            break
        evs = [e for e in events if address(e) == (street, num)]
        name = VENUES.get((street, num)) or esc(evs[0]["placeDescription"])
        g = collections.Counter(e["genre"] for e in evs).most_common(3)
        what = ", ".join(x for x, _ in g)
        w("|-")
        w(f"| {name} || {maps_link(street, num)} || style=\"text-align:center\" | {c} || <small>{what}</small>")
    w("|}")
    w("</div>")
    w("")
    if alternadiv:
        w(f"'''אלטרנדיב''' היא סדרת הופעות של מוזיקה אלטרנטיבית באמפי הפנימי של "
          f"[[שבילים (בית ספר)|בית ספר שבילים]], {len(alternadiv)} הופעות לאורך שלושת הימים. "
          "באותו מתחם פועלות גם במת הג'אז ומוסיקת עולם ו\"אינדיגו\", מתחם הילדים.")
        w("")

    w("== לוח האירועים ==")
    w("השעות מופיעות כפי שהמארחים הזינו אותן. כל שם אירוע הוא קישור לעמוד שלו באתר הפסטיבל, "
      "וכל כתובת נפתחת בגוגל מפות. הטבלאות ניתנות למיון לפי תחום.")
    w("")
    for d in days:
        evs = by_day[d.isoformat()]
        w(f"=== יום {DAYS_HE[d.weekday()]}, {he_date(d)} ===")
        if not evs:
            w("עדיין לא פורסמו אירועים ליום הזה.")
            w("")
            continue
        w('<div style="overflow-x:auto;">')
        w('{| class="wikitable sortable" style="width:100%"')
        w("! שעה !! אירוע !! תחום !! איפה !! למי")
        for e in evs:
            w("|-")
            w(f"| style=\"white-space:nowrap\" | {time_cell(e)} || {event_cell(e)} "
              f"|| {esc(e['genre'])} || {place_cell(e)} || {notes_cell(e)}")
        w("|}")
        w("</div>")
        w("")

    w("== ראו גם ==")
    w("* [[פסטיבל דרך הנדיב]]")
    w("* [[אמנות במושבה 2026]]")
    w("* [[מדריך קהילילה לבן 2026]]")
    w("")
    w("== קישורים חיצוניים ==")
    w("* [https://2026.hanadiv.org/ אתר הפסטיבל], עם חיפוש אירועים והרשמה")
    w(f"* [{KITZUR_URL} אירועי הפסטיבל על המפה] באפליקציית דרך קיצור")
    w(f"* [{fest['facebook']} עמוד הפסטיבל בפייסבוק]")
    w(f"* [{fest['instagram']} הפסטיבל באינסטגרם]")
    w("")
    w("== מקורות ==")
    w(f"לוח האירועים נאסף מ[https://2026.hanadiv.org/ אתר הפסטיבל], נכון ל-{he_date(fetched)} {fetched.year}. "
      "הכתובות ותיאורי המקומות הם כפי שהמארחים הזינו אותם, עם תיקון של שגיאות הקלדה ברורות.")
    w("")
    w("''התוכנית עשויה להשתנות. במקרה של סתירה, אתר הפסטיבל קובע.''")
    w("")
    w("[[קטגוריה:תרבות מקומית]]")
    w("[[קטגוריה:יוזמה קהילתית]]")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"{n} events, {len(venues)} addresses -> {OUT}")


if __name__ == "__main__":
    if "--cached" not in sys.argv:
        fetch()
    build()
