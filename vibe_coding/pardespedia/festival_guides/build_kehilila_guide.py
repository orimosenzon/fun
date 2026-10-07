#!/usr/bin/env python3
"""Build the קהילילה לבן 2026 guide page from the official programme data."""
import json
import os
import re
import urllib.parse

S = os.path.dirname(os.path.abspath(__file__))
ev = json.load(open(f"{S}/events_geo.json", encoding="utf-8"))
parking = json.load(open(f"{S}/parking.json", encoding="utf-8"))
links = json.load(open(f"{S}/wikilinks.json", encoding="utf-8"))

# matches the auto-matcher got wrong: a redirect to the town, the historical
# PICA association, a record shop with a similar name
for bad in ("לבידו כרכור", "פיקאסו", 'מתחם פיק"א', "אורה"):
    links.pop(bad, None)
links["שכונת עומרים"] = "עומרים (שכונה)"
links['שכונת רמב״ם'] = 'רמב"ם (שכונה)'
links["בית הדואר - מרחב עשייה קהילתי"] = "בית הדואר"

# Names that appear inside the programme text rather than in the venue field.
# The act, the artist or the garden is often what the reader is actually
# looking for, and it has an article of its own. Curated by hand and each one
# checked in context: an automatic sweep also matched "פרדס חנה" in half the
# rows (noise) and "בן ספקטור", where the text is about chocolate and the
# article is about a musician — not verifiably the same person.
IN_TEXT = ["סירק דה זוז", "רחל בנגורה", "רחלי שלו", "גן הציפורים", "בבושקפה", "עימנו"]

MAPS = "https://www.google.com/maps/search/?api=1&query={}&hl=iw"
MAPS_LL = "https://www.google.com/maps/search/?api=1&query={},{}&hl=iw"
AREAS = ["מרכז המושבה", "כרכור", "מערב המושבה"]
MAP_FILE = {a: f"קהילילה לבן 2026 - מפת {a}.jpg" for a in AREAS}
YADIT = "חרושת 1"


def maps(addr):
    q = urllib.parse.quote(f"{addr}, פרדס חנה-כרכור")
    return MAPS.format(q)


# The wiki's spam filter blocks URL shorteners outright (forms.gle among them),
# so the programme text gets its short links swapped for the resolved targets.
SHORTLINKS = {
    "https://forms.gle/HeZDbWAb6FaCUgfbA":
        "https://docs.google.com/forms/d/e/1FAIpQLSeN-zoujZsK_eqS0Wi5VTCvf5WgJpTvouraGp1knYysI9BPow/viewform",
    "https://forms.gle/YDU6Xb8t4FTUS3ou7":
        "https://docs.google.com/forms/d/e/1FAIpQLSd3J99V2ZW8VGqPxoondqU-YlbgULwkalJJs59DYSZS3g6o8g/viewform",
}


def clean(s, skip=None):
    """Cell-safe programme text, with in-text wiki links added.

    Newlines and pipes would break the table row, several organisers wrote
    their blurb in Markdown (**bold**), and the names worth linking are often
    buried in the blurb rather than in the venue field.
    """
    s = (s or "").replace("|", "&#124;").strip()
    for short, full in SHORTLINKS.items():
        s = s.replace(short, full)
    s = re.sub(r"\*\*(.+?)\*\*", r"'''\1'''", s, flags=re.S)   # Markdown bold
    s = re.sub(r"(?<![\w'])\*(?!\s)(.+?)(?<!\s)\*(?![\w'])", r"'''\1'''", s)
    for name in IN_TEXT:
        if name == skip or f"[[{name}" in s:
            continue
        # Hebrew glues one-letter prefixes (ב ל ו מ ה ש כ) onto the next word,
        # so the link has to open after the prefix, not swallow it.
        s = re.sub(r"(?<![֐-׿])([בלומהשכ]?)" + re.escape(name) + r"(?![֐-׿\w])",
                   lambda m: m.group(1) + f"[[{name}]]", s, count=1)
    return re.sub(r"\n+", "<br />", s)


def venue(e):
    b = e["business"]
    if b not in links:
        return b
    return f"[[{b}]]" if links[b] == b else f"[[{links[b]}|{b}]]"


def hours(e):
    return f'{e["startTime"]}–{e["endTime"]}' if e.get("endTime") else e["startTime"]


def is_yadit(e):
    return YADIT in (e.get("address") or "")


def table(rows, numbered=True):
    out = ['<div style="overflow-x:auto;">',
           '{| class="wikitable sortable" style="width:100%"']
    head = "! מס' !! שעות !! המקום !! מה קורה שם !! קהל" if numbered \
        else "! שעות !! המקום !! מה קורה שם !! קהל"
    out.append(head)
    for e in rows:
        cells = []
        if numbered:
            cells.append(f'style="text-align:center; font-weight:bold" | {e.get("n", "")}')
        cells.append(f'data-sort-value="{e["startTime"]}" | {hours(e)}')
        addr = e.get("address") or ""
        place = f"'''{venue(e)}'''"
        if addr:
            place += f'<br /><small>[{maps(addr)} {addr}]</small>'
        cells.append(place)
        skip = links.get(e["business"])       # already linked in this row
        what = clean(e["eventName"], skip)
        if e.get("description"):
            what += f'<br /><small>{clean(e["description"], skip)}</small>'
        cells.append(what)
        cells.append(e.get("audience") or "—")
        out += ["|-", "| " + " || ".join(cells)]
    out += ["|}", "</div>", ""]
    return "\n".join(out)


def bullet_list(pred, limit=None):
    got = [e for e in ev if pred(e)]
    got.sort(key=lambda e: e["startTime"])
    lines = []
    for e in got[:limit] if limit else got:
        lines.append(f'* \'\'\'{venue(e)}\'\'\' ({hours(e)}): {clean(e["eventName"])}')
    return "\n".join(lines) if lines else "''אין''"


P = []
add = P.append

add('[[קובץ:קהילילה לבן 2026 - כרזה.jpg|ממוזער|280px|כרזת האירוע]]')
add("")
add("'''קהילילה לבן 2026''' הוא הלילה שבו [[פרדס חנה-כרכור]] נשארת ערה. "
    "ביום חמישי, '''27 באוגוסט 2026''', נפתחים ברחבי המושבה עשרות מוקדים בו-זמנית: "
    "חצרות של עסקים, סטודיואים של אמנים, גינות קהילתיות, שכונות שלמות שסוגרות רחוב "
    "ועושות מסיבה. אין במה מרכזית אחת ואין מסלול אחד נכון. הרעיון הוא לצאת מהבית, "
    "להסתובב, ולגלות דברים שלא ידעתם שקיימים במרחק הליכה מכם.")
add("")
add("הדף הזה הוא '''מדריך מעשי לערב''': מה קורה, מתי, איפה בדיוק, ואיפה חונים. "
    "כל מוקד מסומן במפה עם מספר, ואותו מספר מופיע בטבלה שמתחתיה.")
add("")

def mins(t, end=False):
    """Minutes past midnight; an end time before 06:00 belongs to the next day.

    Comparing these as strings is wrong, and was: max() over the raw strings
    answered "23:30" while the last venues actually close at 03:00, because
    "23:30" sorts above "03:00".
    """
    h, m = map(int, t.split(":"))
    if end and h < 6:
        h += 24
    return h * 60 + m


n_all = len(ev)
n_free_ages = sum(1 for e in ev if e.get("audience") == "לכל הגילאים")
n_18 = sum(1 for e in ev if e.get("audience") == "גילאי 18+")
n_evening = sum(1 for e in ev if mins(e["startTime"]) >= 17 * 60)
n_daytime = n_all - n_evening
n_after_midnight = sum(1 for e in ev if mins(e["endTime"], True) >= 24 * 60)
last_close = max(ev, key=lambda e: mins(e["endTime"], True))["endTime"]
add('<div style="background:#f4f7fb; border:1px solid #ccd6e4; border-right:5px solid #21659c; '
    'border-radius:10px; padding:14px 18px; margin:1em 0;">')
add("'''בשורה אחת:''' " + f"{n_all} מוקדים על פני שלושה אזורים. "
    f"{n_evening} מהם נפתחים מ-17:00 ואילך, ורוב הפתיחות מרוכזות בשעות 18:00 ו-20:00. "
    f"{n_after_midnight} מוקדים ממשיכים גם אחרי חצות, והאחרונים נסגרים ב-{last_close} לפנות בוקר. "
    f"הכניסה לרוב המוקדים חופשית, {n_free_ages} מתאימים לכל הגילאים "
    f"ו-{n_18} מיועדים לגילאי 18 ומעלה.")
add("")
add(f"''(רק {n_daytime} עסקים פותחים כבר במהלך היום, עם מבצעים לכבוד הערב. "
    "הלב של האירוע מתחיל עם רדת החשכה.)''")
add("</div>")
add("")

add("== איך להשתמש בדף הזה ==")
add("* '''לפי אזור.''' אם אתם יודעים איפה תהיו, גללו לאזור שלכם, הסתכלו במפה ובחרו מספר.")
add("* '''לפי מה בא לכם.''' אם אתם יודעים מה בא לכם (מסיבה, שוק, סדנה, משהו עם הילדים), "
    "יש רשימות מקוצרות בהמשך.")
add("* '''לפי שעה.''' כל טבלה ניתנת למיון. לחיצה על החץ שליד '''שעות''' מסדרת את המוקדים "
    "מהמוקדם למאוחר, ולחיצה על '''קהל''' מקבצת את מה שמתאים לילדים ואת מה שלא.")
add("* '''כתובת.''' כל כתובת בטבלאות היא קישור שנפתח ישירות בגוגל מפות.")
add("")

add("== המפות והמוקדים ==")
add("")

for area in AREAS:
    rows = sorted((e for e in ev if e["area"] == area and not is_yadit(e)),
                  key=lambda e: (e.get("n") or 999, e["startTime"]))
    if not rows:
        continue
    add(f"=== {area} ===")
    add(f'[[קובץ:{MAP_FILE[area]}|מרכז|ממוזער|900px|'
        f'{area}: המספרים במפה תואמים למספרים בטבלה. האות P מסמנת חניון קהל רשמי.]]')
    add("")
    add(table(rows))

yadit = [e for e in ev if is_yadit(e)]
if yadit:
    add("=== מתחם הידית (חרושת 1) ===")
    add("במתחם אחד, ברחוב חרושת 1, פועלים באותו ערב חמישה מוקדים שונים. "
        "מי שמגיע לשם מגיע למעשה לחמישה דברים בכתובת אחת, ולכן הם מרוכזים כאן יחד. "
        f"[{maps('חרושת 1')} למיקום המתחם בגוגל מפות].")
    add("")
    add(table(sorted(yadit, key=lambda e: e["startTime"]), numbered=False))

add("== לפי מה בא לכם ==")
add("")
add("=== מסיבות רחוב שכונתיות ===")
add("שש שכונות סוגרות רחוב, כל אחת באופי משלה. אלה בדרך כלל האירועים הכי משפחתיים ולא צריך "
    "להזמין שום דבר מראש, פשוט מגיעים.")
add(bullet_list(lambda e: "שכונת" in e["business"] or "שכונה" in e.get("eventName", "")))
add("")
add("=== שווקים וירידי לילה ===")
add(bullet_list(lambda e: any("שוק" in t for t in (e.get("tags") or []))))
add("")
add("=== סדנאות ===")
add("רובן דורשות הרשמה מראש או מספר מקומות מוגבל. כדאי לבדוק מול המקום לפני שיוצאים.")
add(bullet_list(lambda e: any("סדנה" in t for t in (e.get("tags") or []))))
add("")
add("=== עם ילדים ===")
add(bullet_list(lambda e: e.get("audience") == "ילדות וילדים"))
add("")
add("=== רק למבוגרים (18+) ===")
add(bullet_list(lambda e: e.get("audience") == "גילאי 18+"))
add("")

add("== חניה ==")
add("מארגני האירוע סימנו '''שבעה חניוני קהל''' ברחבי המושבה. הם מסומנים באות P במפות שלמעלה. "
    "בערב כזה מרכז המושבה נסגר בפועל לתנועה איטית, ולכן שווה להחנות באחד מהם וללכת ברגל, "
    "או פשוט להגיע מלכתחילה ברגל או באופניים.")
add("")
add('<div style="overflow-x:auto;">')
add('{| class="wikitable" style="width:100%"')
add("! חניון !! למפה")
for p in sorted(parking, key=lambda p: p["name"]):
    add("|-")
    add(f'| {p["name"]} || [{MAPS_LL.format(p["lat"], p["lon"])} פתחו ניווט]')
add("|}")
add("</div>")
add("")

add("== כמה דברים ששווה לדעת ==")
add("* '''זה לא פסטיבל עם שער.''' אין כניסה מרכזית, אין צמיד ואין תוכנייה. כל מוקד עומד בפני עצמו.")
add("* '''רוב המוקדים חינם''', אבל בעסקים שמוכרים אוכל ושתייה משלמים על מה שקונים, וחלקם מציעים "
    "מבצעים מיוחדים לערב הזה בלבד.")
add("* '''השעות אינן אחידות.''' יש מוקדים שנסגרים ב-22:00 ויש שממשיכים עד אחרי 23:00. "
    "אם יש משהו שאתם ממש רוצים, לכו אליו קודם.")
add("* '''נגישות''' באירועים שמתקיימים בעסקים פרטיים משתנה מעסק לעסק. "
    "בשאלות נגישות אפשר לפנות למחלקת השירות לתושב במועצה, בטלפון 077-9779749.")
add("* '''קחו בקבוק מים.''' סוף אוגוסט, והרבה מהמוקדים בחצרות ובחוץ.")
add("")

add("== רוצים לקרוא עוד? ==")
add("הרבה מהמקומות שתעברו בהם הערב הם מקומות עם סיפור. בפרדספדיה, הוויקי הקהילתי של המושבה, "
    "יש עליהם ערכים שנכתבו בידי תושבים:")
add("")
seen = set()
for t in sorted(set(links.values()) | set(IN_TEXT)):
    seen.add(t)
    add(f"* [[{t}]]")
add("")
add("ואם משהו כאן חסר, שגוי או שאתם יודעים עליו יותר, '''אתם מוזמנים לתקן ולהוסיף בעצמכם'''. "
    "אין צורך בידע טכני, וההסבר נמצא בדף [[איך עורכים כאן?]].")
add("")

add("== ראו גם ==")
add("* [[קהילילה לבן]], הערך על האירוע עצמו")
add("* [[פרדס חנה-כרכור]]")
add("* [[דרך קיצור]], שבילי הליכה שמקצרים את הדרך בין השכונות")
add("")
add("== מקורות ==")
add("תוכנית האירוע והמיקומים לקוחים מ[https://kehilayla.netlify.app/ אתר האירוע הרשמי] "
    "וממפת האירועים הרשמית שלו, נכון ל-27 באוגוסט 2026. "
    "המפות בדף זה הופקו לפרדספדיה על בסיס נתוני "
    "[https://www.openstreetmap.org/copyright OpenStreetMap] (רישיון ODbL).")
add("")
add("''התוכנית עשויה להשתנות. במקרה של סתירה, האתר הרשמי קובע.''")
add("")
add("[[קטגוריה:תרבות מקומית]]")
add("[[קטגוריה:פעילות קהילתית]]")

out = "\n".join(P)
open(f"{S}/guide.wiki", "w", encoding="utf-8").write(out)
print(f"{len(out)} chars, {n_all} events, {len(seen)} wiki links")
