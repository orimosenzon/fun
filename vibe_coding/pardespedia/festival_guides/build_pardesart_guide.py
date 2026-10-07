#!/usr/bin/env python3
"""Build the pardespedia guide page for אמנות במושבה 2026 from the scraped JSON.

Inputs (produced by scrape_pardesart.py and scrape_events.py):
  pa/artists.json, pa/foods.json, pa/events.json, pa/street_centroids.json
Output: guide_pardesart.wiki
"""
import json
import os
import re
import urllib.parse

S = os.path.dirname(os.path.abspath(__file__))
artists = json.load(open(f"{S}/data/artists.json", encoding="utf-8"))
foods = json.load(open(f"{S}/data/foods.json", encoding="utf-8"))
events = json.load(open(f"{S}/data/events.json", encoding="utf-8"))
cent = json.load(open(f"{S}/data/street_centroids.json", encoding="utf-8"))

# ---------------------------------------------------------------- wiki links
# Auto-matching on the entry name gets the obvious ones; these are the cases
# where the festival's label and the wiki title differ. Deliberately NOT linked:
# "שרון גנדלמן – Shuzi Jewelry" (the wiki's שוזי is a restaurant, not the
# jewellery brand), "ליאת אליעז – המיוחדת" (the wiki's מייוחדת is מיי אפרת's
# gallery at אורוות האמנים) and "זיוה שמחי – Vardi" (the wiki's ורדי sits at a
# different address) — same-name traps, not the same business.
LINKS = {
    "גליה שמחאי – KWALA Paper Art": "Kwala Paper Art",
    "קארין קויפמן – קריניטי": "KaRiniTi",
    "האוורד פוקס": "הווארד פוקס",
    "אפרת ועמית – דודסון רוקחות טבעית": "דודסון רוקחות טבעית",
    "אפרת רון – נגיעות בחומר": "נגיעות בחומר",
    "נוגה ספקטור – אמנות בפסיפס": "נוגה ספקטור",
    "שירה פרידמן – שוקדת": "שוקדת",
    "רחל בנגורה – בית בנגורה": "רחל בנגורה",
    "BLOOMS": "בלומס",
    "Guy's burgertruck": "גיא בורגר",
    "בבושקפה עגלת קפה": "בבושקפה",
    "בבושקפה בית הדואר": 'בבושקפה "בית הדואר"',
    "נוילנד – דברים טובים מאוד": "נוילנד (בית קפה)",
    "דליקטו – ג'לאטו איטלקי אמיתי": "דליקטו",
    "קבב ואחיו – אוכל רחוב בגבוה": "קבב ואחיו",
    "קפה בחורשה – מאפים טריים, בראנצ'ים, פסטורלי בסטייל": "קפה בחורשה",
    "קפה ויעל – קפה שכונה": "קפה ויעל",
    "רוברטה וינצ'י – מסעדה איטלקית מקומית": "רוברטה וינצ'י",
    "נוני ופורטונה – קפה משובח ופינוקים": "נוני ופורטונה",
    "רות מורן ואוהד – נוני ופורטונה": "נוני ופורטונה",
    "שושנה – פיתה שיפודים – השוק הישן": "שושנה",
}

# hubs: exact address -> (display title, wiki article for the venue or None)
HUBS = [
    ("חרושת 1", "מתחם הידית", "מתחם הידית"),
    ("האורנים 40", "מתחם האורנים", None),
    ("קדמה 8ב", "מתחם הקשת", None),
    ("הבוטנים 25", "מתחם התורמוסים", None),
    ("הדרים 77א", "מתחם הדרים Dream", None),
    ("האורנים 12", "השוק הישן", "השוק הישן"),
    ("ביל״ו 10", "בית תרבות נרבתא", "נרבתא"),
    ("הבוטנים 7", "הבוטנים 7", None),
    ("שאננים 9", "מתחם שאננים", None),
    ("הדקלים 131", "הדקלים 131", None),
    ("המזמור 11", "המזמור 11", None),
]
HUB_BY_ADDR = {addr: (title, art) for addr, title, art in HUBS}

AREAS = [("מרכז המושבה", 34.975, 34.988), ("כרכור", 34.988, 99.0),
         ("מערב ודרום המושבה", 0.0, 34.975)]
# streets OSM has no geometry for, placed from the festival's own route names
AREA_OVERRIDE = {"חרושת": "מרכז המושבה", "שאננים": "כרכור",
                 'דרך פיק"א': "מערב ודרום המושבה", "חוגלה": "מערב ודרום המושבה"}


def street_of(label):
    """'מתחם הקשת, קדמה 8ב' -> 'קדמה'; keeps multi-word names like 'דרך הבנים'."""
    s = re.sub(r"^מתחם [^,]+,\s*", "", label)
    s = re.sub(r"\s*,.*$", "", s)
    return re.sub(r"\s+\d.*$", "", s).strip()


def area_of(label):
    st = street_of(label)
    if st in AREA_OVERRIDE:
        return AREA_OVERRIDE[st]
    last = st.split()[-1] if st else ""
    for name in (st, "ה" + st, st.lstrip("ה"), last, "ה" + last, last.lstrip("ה")):
        if name in cent:
            lon = cent[name][1]
            for area, lo, hi in AREAS:
                if lo <= lon < hi:
                    return area
    return "אחר"


def gmaps(addr):
    q = urllib.parse.quote(f"{addr}, פרדס חנה-כרכור")
    return f"https://www.google.com/maps/search/?api=1&query={q}&hl=iw"


def name_cell(r):
    art = LINKS.get(r["name"])
    disp = r["name"].replace("–", "-")
    return f"'''[[{art}|{disp}]]'''" if art else f"'''{disp}'''"


def blurb(s, n=230):
    s = re.sub(r"\s*\n+\s*", " ", s).strip()
    # the source text uses em dashes; Hebrew wiki text here does not
    s = s.replace(" — ", ", ").replace("—", "-")
    if len(s) <= n:
        return s
    cut = s[:n]
    p = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    return (cut[:p + 1] if p > n * 0.5 else cut.rsplit(" ", 1)[0] + "…")


def badge(r):
    bits = []
    acc = r.get("accessibility", "")
    if acc == "כניסה נגישה":
        bits.append("נגיש")
    elif acc == "לא נגיש":
        bits.append("לא נגיש")
    elif acc:
        bits.append("נגישות בתיאום")
    if r.get("shabbat"):
        bits.append("סגור בשבת")
    return "<br />".join(bits)


# pardespedia's spam blacklist rejects these outright, and one edit carrying a
# single blocked link fails the whole page. Dropped rather than fought.
BLOCKED = ("linktr.ee", "share.google", "jewelry.com")


def contact(r):
    bits = []
    if r.get("phone"):
        bits.append(r["phone"])
    for key, label in (("website", "אתר"), ("instagram", "אינסטגרם"), ("facebook", "פייסבוק")):
        url = r["links"].get(key)
        if url and not any(b in url for b in BLOCKED):
            bits.append(f"[{url} {label}]")
    return "<br />".join(f"<small>{b}</small>" for b in bits)


def table(rows, show_address):
    out = ['<div style="overflow-x:auto;">',
           '{| class="wikitable sortable" style="width:100%"']
    head = "! מי !! תחום !!"
    if show_address:
        head += " כתובת !!"
    head += " על מה מדובר !! טוב לדעת !! קשר"
    out.append(head)
    for r in rows:
        cats = ", ".join(r["categories"][:2]) if r["categories"] else "קולינריה"
        out.append("|-")
        cells = [name_cell(r), cats]
        if show_address:
            cells.append(f"[{gmaps(r['address_label'])} {r['address_label']}]")
        desc = blurb(r["description"])
        cells += [f"<small>{desc}</small>" if desc else "", badge(r), contact(r)]
        out.append("| " + " || ".join(cells))
    out += ["|}", "</div>", ""]
    return out


# ------------------------------------------------------------------- assemble
L = []
A = L.append

n_art, n_food = len(artists), len(foods)
free = sum(1 for e in events if "ללא עלות" in e["price"] or "חופשי" in e["price"])
no_reg = sum(1 for e in events if "לא נדרשת" in e["registration"] or "חופשי" in e["registration"])

A("''ערך זה מתאר את מהדורת 2026 של הפסטיבל. על הפסטיבל עצמו, ראו [[אמנות במושבה]].''")
A("")
A("[[קובץ:אמנות במושבה 2026 - המפה הדיגיטלית.jpg|ממוזער|260px|המפה הדיגיטלית של "
  "הפסטיבל. כל סמן הוא סטודיו או מתחם פתוח]]")
A("")
A("'''אמנות במושבה 2026''' היא המהדורה ה-28 של פסטיבל האמנות, העיצוב והקראפט של "
  "[[פרדס חנה-כרכור]], והיא נערכת בימים חמישי עד שבת, '''3 עד 5 בספטמבר 2026'''. "
  "במשך שלושה ימים פותחים אמניות ואמנים מקומיים את הסטודיואים שבהם הם עובדים, "
  "ומוכרים עבודות מקוריות ישירות מהיוצר. הכניסה לכל הסטודיואים ולמתחמים חופשית.")
A("")
A("הדף הזה מרכז את התוכנית המלאה: מי מציג ואיפה, ארבע התערוכות, לוח האירועים "
  "לפי ימים, ומה פתוח לאכול בין לבין.")
A("")
A('<div style="background:#f4f7fb; border:1px solid #ccd6e4; border-right:5px solid #21659c; '
  'border-radius:10px; padding:14px 18px; margin:1em 0;">')
A(f"'''בשורה אחת:''' {n_art} אמניות ואמנים פותחים סטודיו. {len(HUBS)} כתובות הן מתחמים "
  f"שבכל אחד מהם בין ארבעה לאחד עשר יוצרים, ולצדם ארבע תערוכות "
  f"ו-{len(events)} אירועים לאורך שלושת הימים. {free} מהאירועים ללא עלות, "
  f"ו-{no_reg} מהם בלי הרשמה מראש. בנוסף פתוחים {n_food} עסקי אוכל.")
A("</div>")
A("")

A("== מתי ואיפה ==")
A('<div style="overflow-x:auto;">')
A('{| class="wikitable" style="width:auto"')
A("! יום !! תאריך !! שעות")
for d, dt, hrs in (("חמישי", "3 בספטמבר", "10:00–20:00"),
                   ("שישי", "4 בספטמבר", "10:00–17:00"),
                   ("שבת", "5 בספטמבר", "10:00–20:00")):
    A("|-")
    A(f"| {d} || {dt} || {hrs}")
A("|}")
A("</div>")
A("")
A("הכניסה לסטודיואים, למתחמים ולתערוכות חופשית. חלק מהאירועים כרוכים בתשלום סמלי "
  "ובהרשמה מראש, והפירוט מופיע בטבלאות שלמטה. שעות הפתיחה של סטודיו מסוים עשויות "
  "להיות צרות יותר משעות הפסטיבל, ובמיוחד בשבת: 14 מהמשתתפים סימנו שאינם פותחים בשבת.")
A("")

A("== מה זה הפסטיבל ==")
A("הפסטיבל התחיל ב-1999 כמיזם קהילתי של האמנית '''אסנת דן''' ז\"ל, שאספה 21 אמניות "
  "ואמנים מהמושבה סביב רעיון אחד: לפתוח את הבתים והסטודיואים ולפגוש קהל דרך העבודה "
  "עצמה. האירוע הראשון נקרא \"בתים פתוחים\" והתקיים במרץ 1999. בגרעין המייסד היו גם "
  "[[אילנה פלדה|אילנה]] ו[[הנס פלדה]] ז\"ל. מאז גדל הפסטיבל משנה לשנה, והוא מופעל "
  "בידי עמותת אמנות פרדס חנה-כרכור.")
A("")
A("מהדורת 2026 מתקיימת תחת הכותרת '''Slow Art''', הזמנה להאט ולשהות מול היצירה במקום "
  "לעבור עליה ברפרוף. לפי המארגנים, המשתתפים הם אמניות ואמנים תושבי המושבה, ולצדם "
  "כמה אמניות שפונו מבתיהן במהלך המלחמה והצטרפו לקהילה המקומית.")
A("")
A("''הערה על המועד:'' באתר העמותה נכתב שהפסטיבל נערך מדי שנה כשבועיים לפני פסח, "
  "ואכן מהדורות קודמות התקיימו באפריל. מהדורת 2026 נערכת בספטמבר.")
A("")

A("== איך לקרוא את הדף ==")
A("* '''לפי מתחם.''' רוב המוקדים אינם סטודיו בודד אלא מתחם שבו יושבים כמה יוצרים "
  "בכתובת אחת. הסעיף הבא מסודר לפי מתחמים, מהגדול לקטן, וזו הדרך היעילה ביותר "
  "לראות הרבה בזמן קצר.")
A("* '''לפי אזור.''' מי שאינו במתחם מופיע בסעיף שאחריו, מחולק למרכז המושבה, לכרכור "
  "ולמערב ולדרום.")
A("* '''לפי תחום.''' כל טבלה ניתנת למיון. לחיצה על החץ שליד '''תחום''' מקבצת ציור, "
  "קרמיקה, צורפות וכן הלאה.")
A("* '''כתובת.''' כל כתובת היא קישור שנפתח בגוגל מפות.")
A("* '''טוב לדעת.''' העמודה מציינת נגישות ומי שאינו פותח בשבת, לפי הצהרת המשתתפים.")
A("")

# ------------------------------------------------------------------- מתחמים
everyone = artists + foods
in_hub = set()
placement = {p["address"]: p for p in
             json.load(open(f"{S}/hub_placement.json", encoding="utf-8"))}


def heading(addr, title):
    """Section title for a hub; the map legend anchors to exactly this string."""
    pin = placement.get(addr, {}).get("pin", "")
    base = title if title == addr else f"{title} ({addr})"
    return f"{pin}. {base}" if pin else base

A("== המתחמים ==")
A("אחת עשרה כתובות מרכזות בין ארבעה לאחד עשר משתתפים כל אחת. יחד הן מכסות יותר "
  "ממחצית מהמשתתפים בפסטיבל, וזו הדרך היעילה ביותר לראות הרבה בזמן קצר.")
A("")
A("[[קובץ:אמנות במושבה 2026 - מפת המתחמים.jpg|מרכז|ממוזער|900px|אחד עשר המתחמים. "
  "המספרים תואמים לטבלה שמתחת. '''סמן מלא''' הוא מיקום מדויק; '''סמן חלול''' הוא "
  "הרחוב הנכון, כשהמיקום המדויק לאורכו משוער.]]")
A("")
A('<div style="overflow-x:auto;">')
A('{| class="wikitable sortable" style="width:auto"')
A("! מס' !! מתחם !! כתובת !! משתתפים !! דיוק הסימון")
HOW_HE = {"exact": "מדויק", "fit": "משוער על הרחוב", "anchor": "משוער על הרחוב",
          "street": "מרכז הרחוב"}
for addr, title, art in HUBS:
    n = len([r for r in artists + foods if r["address_label"].strip().endswith(addr)])
    p = placement.get(addr, {})
    A("|-")
    A(f"| style=\"text-align:center; font-weight:bold\" | {p.get('pin', '')} || "
      f"[[#{heading(addr, title)}|{title}]] || [{gmaps(addr)} {addr}] || "
      f"style=\"text-align:center\" | {n} || {HOW_HE.get(p.get('how'), '')}")
A("|}")
A("</div>")
A("")
for addr, title, art in HUBS:
    grp = [r for r in everyone if r["address_label"].strip().endswith(addr)]
    for r in grp:
        in_hub.add(r["url"])
    A(f"=== {heading(addr, title)} ===")
    if art:
        A(f"''ערך מורחב: [[{art}]]''")
    A(f"{len(grp)} משתתפים. [{gmaps(addr)} למיקום בגוגל מפות].")
    A("")
    L.extend(table(sorted(grp, key=lambda r: r["name"]), show_address=False))

# --------------------------------------------------------------- לפי אזור
rest = [r for r in everyone if r["url"] not in in_hub]
A("== סטודיואים נוספים, לפי אזור ==")
A(f"{len(rest)} משתתפים נוספים פזורים ברחבי המושבה. החלוקה לאזורים כאן נגזרה "
  "ממיקום הרחובות ולא מהמסלולים הרשמיים של הפסטיבל, שמסומנים בארבעה צבעים במפה "
  "הדיגיטלית שלו.")
A("")
for area in ("מרכז המושבה", "כרכור", "מערב ודרום המושבה", "אחר"):
    heading = "כתובת לא צוינה באתר" if area == "אחר" else area
    grp = [r for r in rest if area_of(r["address_label"]) == area]
    if not grp:
        continue
    A(f"=== {heading} ===")
    L.extend(table(sorted(grp, key=lambda r: r["address_label"]), show_address=True))

# ------------------------------------------------------------------ תערוכות
A("== ארבע התערוכות ==")
A("לצד הסטודיואים הפתוחים מוצגות בפסטיבל ארבע תערוכות אצורות. הכניסה לכולן חופשית.")
A("")
EXH = [
 ("קווי אחיזה", "תערוכה קבוצתית", "בית לִים, מרכז לתרבות חומרית", "הדרים 126", None,
  "התערוכה בוחנת את הקו ככוח בונה כוונה, בצורה וברעיון. דרך חומרים, מחוות ופעולות "
  "נחשפים מבנים עדינים, גלויים וסמויים, שמייצרים חיבור, תמיכה ועמידות ברגעי שבר. "
  "היא שואלת מהם החוטים הקושרים קהילות זו לזו מעבר להבדלים, ואילו קווים מסמנים "
  "ביטחון, חיבור או מרחק.",
  "ברק רובין",
  "אילנה אביב, יעל אגוזי, יורם אפק, ענת בינג, בן בר, מינדה גלאון, מרגו גראן, "
  "אבנר זינגר, אורלי חלק, שלומית חפר ודנה בהרב, עמי ליבוביץ, פריאל עזר, דרור פרבדה, "
  "דוד רוזנברג, גבריאל רענן, מתן שמאי, ליאת שרון",
  ["אירוע פתיחה: חמישי 3.9 בשעה 18:00"],
  "חמישי 10:00–20:00, שישי 10:00–17:00, שבת 10:00–20:00", "052-3344208"),
 ("מגורשת", "תערוכת יחיד של דורון כפרי דהן", "הסולם, בית ספר לאמנות", "האורנים 104",
  "הסולם (בית ספר לאמנות)",
  "עבודתה הפיסולית של דורון כפרי-דהן מחיה מיתוסים עבריים וקדם-מקראיים, מתוך חקירת "
  "הדחקתם של המיתוסים הנשיים הקדומים מהתרבות. באמצעות פסליה היא מפרקת את תודעת "
  "ההשתקה, ומנסחת אתוס פלסטי עברי חדש.",
  "מיכל קרסני", None,
  ["אירוע פתיחה: חמישי 3.9 בשעה 18:30", "מפגש שיח אמן: שישי 4.9 בשעה 11:00",
   "שיח גלריה: שבת 5.9 בשעה 11:00"],
  "חמישי 16:00–21:00, שישי 09:00–15:00, שבת 10:00–18:00", "050-4313077"),
 ("ערות", "תערוכה קבוצתית", "בית תרבות נרבתא", "ביל\"ו 10", "נרבתא",
  "התערוכה מתבוננת במשבר חברתי ובאי-ודאות מנקודת מבט נשית, ובוחנת כיצד נשים חוות "
  "ומעבדות שבר דרך ציור כמרחב של שהייה והתבוננות. חלק מהעבודות נוצרו בשנתיים "
  "האחרונות וחלקן משקפות משברי עבר.",
  "עליזה אשכנזי ומירי ספקטור צ׳רניץ",
  "מיכל גבע, מירי ספקטור צ׳רניץ, מרגו גראן, עליזה אשכנזי, ענת אור מגל",
  ["אירוע פתיחה: חמישי 3.9 בשעה 18:30", "מפגש שיח אמניות: שבת 5.9 בשעה 11:00"],
  "חמישי 10:00–20:00, שישי 10:00–17:00, שבת 10:00–20:00", None),
 ("בכורות", "תערוכה קבוצתית", "הפרדס, מרכז צעירים", "המושב 48", "הפרדס - מרכז צעירים",
  "התערוכה מביאה לקדמת הבמה את הדור הצעיר של אמני ואמניות המושבה, בוגרי מגמות אמנות "
  "ויוצרים אוטודידקטים. העבודות נעות בין חופש נעורים לאחריות אישית וקולקטיבית, ובין "
  "קיום בבועה פרטית להתגבשות תודעה חברתית.",
  "איתמר גנדלמן",
  "גורי אמיתי, גילי בן יוסף, עלי ברדה, רוני פולוביאן, חן פוקס, זיו פיקלר, דריה רותם",
  ["אירוע פתיחה: חמישי 3.9 בשעה 19:00"],
  "חמישי 19:00–22:30, שישי 10:00–14:00", None),
]
for name, kind, venue, addr, art, desc, curator, participants, extra, hours, tel in EXH:
    A(f"=== {name} ===")
    vlink = f"[[{art}|{venue}]]" if art else venue
    A(f"'''{kind}''', {vlink}, [{gmaps(addr)} {addr}].")
    A("")
    A(desc)
    A("")
    if participants:
        A(f"'''משתתפים:''' {participants}")
        A("")
    A(f"'''אוצרוּת:''' {curator}")
    A("")
    for x in extra:
        A(f"* {x}")
    A(f"* שעות פתיחה: {hours}")
    if tel:
        A(f"* טלפון: {tel}")
    A("")

# ------------------------------------------------------------------- אירועים
A("== לוח האירועים ==")
A(f"{len(events)} אירועים לאורך שלושת הימים: שיחות אמן בסטודיו, הדגמות, הרצאות, "
  "הופעות ומופעים. חלקם דורשים הרשמה מראש, וחלקם בתשלום. הטבלאות ניתנות למיון "
  "לפי שעה, סוג או עלות.")
A("")
for day, label in (("חמישי", "חמישי 3 בספטמבר"), ("שישי", "שישי 4 בספטמבר"),
                   ("שבת", "שבת 5 בספטמבר")):
    grp = [e for e in events if e["day"] == day]
    A(f"=== {label} ===")
    A(f"{len(grp)} אירועים.")
    A("")
    A('<div style="overflow-x:auto;">')
    A('{| class="wikitable sortable" style="width:100%"')
    A("! שעה !! מי !! סוג !! מה !! איפה !! עלות !! הרשמה")
    for e in sorted(grp, key=lambda x: x["time"]):
        start = e["time"].split("–")[0]
        venue = e["venue"]
        vcell = f"[{gmaps(venue)} {venue}]" if venue else ""
        price = e["price"] or "לא צוין"
        A("|-")
        A(f'| data-sort-value="{start}" | {e["time"]} || \'\'\'{e["artist"]}\'\'\' || '
          f'{e["kind"]} || <small>{blurb(e["description"], 180)}</small> || '
          f'<small>{vcell}</small> || {price} || <small>{e["registration"]}'
          f'{("<br />" + e["phone"]) if e["phone"] else ""}</small>')
    A("|}")
    A("</div>")
    A("")

# ----------------------------------------------------------------- קולינריה
A("== איפה אוכלים ==")
A(f"{len(foods)} עסקי אוכל שותפים לפסטיבל. חלקם יושבים בתוך המתחמים ומופיעים גם "
  "בטבלאות שלמעלה.")
A("")
L.extend(table(sorted(foods, key=lambda r: r["address_label"]), show_address=True))

# ------------------------------------------------- אילנה והנס פלדה
A("== לזכרם של אילנה והנס פלדה ==")
A("מהדורת 2026 מוקדשת לזכרם של [[אילנה פלדה|אילנה]] ו[[הנס פלדה]], שנפטרו השנה. "
  "השניים היו בגרעין המייסד של הפסטיבל ב-1999, והנס היה גם ממקימי [[תיאטרון הידית]]. "
  "שני מוקדים בפסטיבל מוקדשים להם.")
A("")
A("=== סיור \"מקום שמור\" ===")
A("[[מקום שמור]] הוא פרויקט תיעוד המבנים, הנופים והסיפורים של המושבה, שיזמה אילנה "
  "הרשנברג פלדה. במהלך הפסטיבל יתקיים סיור בעקבות האתרים שסומנו בפרויקט, בהובלת "
  "'''יואב טריפון''' ו'''יעל גתי'''.")
A("* מועדים: חמישי 3.9 בשעה 18:00, שישי 4.9 בשעה 09:00")
A("* משך: כשעה וחצי, בהליכה")
A("* נקודת יציאה: מרכז המושבה. המיקום המדויק נמסר לנרשמים")
A("* ההשתתפות ללא עלות, בהרשמה מראש")
A("")
A("=== קיר השראה בוואדי ===")
A("[[שיקשוק|השיקשוק]], שוק היד השנייה שנולד ב[[הוואדי|ואדי]] של המושבה ביוזמת בני "
  "הזוג פלדה לפני יותר מעשרים שנה, הוא היום חלק מחיי התרבות המקומיים. במהלך ימי "
  "הפסטיבל יוקם בוואדי קיר השראה לזכרם, ובו תמונות ורגעים מתחנות בדרכם. המבקרים "
  "מוזמנים להוסיף מילה אישית. לצד הקיר תעמוד סלסילת בועות סבון, מחווה להנס, שנהג "
  "למלא את הוואדי בבועות.")
A("")

# ------------------------------------------------------------------- מקורות
A("== קישורים חיצוניים ==")
A("* [https://www.pardesart.co.il/ אתר הפסטיבל הרשמי]")
A("* [https://www.pardesart.co.il/map/ המפה הדיגיטלית של הפסטיבל], ובה ארבעת "
  "המסלולים המסומנים בצבעים")
A("* [https://www.pardesart.co.il/plan/ תכנון ביקור]")
A("")
A("== מקורות ==")
A("רשימת המשתתפים, התערוכות ולוח האירועים נאספו מ[https://www.pardesart.co.il/ "
  "אתר הפסטיבל הרשמי] של עמותת אמנות פרדס חנה-כרכור, נכון ל-29 באוגוסט 2026. "
  "החלוקה לאזורים נגזרה ממיקומי הרחובות ב[https://www.openstreetmap.org/copyright "
  "OpenStreetMap] (רישיון ODbL) ואינה חלוקה רשמית של הפסטיבל.")
A("")
A("''התוכנית עשויה להשתנות. במקרה של סתירה, האתר הרשמי קובע.''")
A("")
A("[[קטגוריה:תרבות מקומית]]")
A("[[קטגוריה:אמנות]]")

out = "\n".join(L).rstrip() + "\n"
# belt and braces: no em dash survives into Hebrew wiki text, including inside
# names the artists themselves spell that way
out = out.replace(" — ", ", ").replace("—", "-")
open(f"{S}/guide_pardesart.wiki", "w", encoding="utf-8").write(out)
print(f"wrote guide_pardesart.wiki  {len(out):,} chars  {out.count(chr(10)):,} lines")
assert "—" not in out, "em dash found"
