#!/usr/bin/env python3
"""Taxonomy round 2: give the 'facilities' categories non-overlapping jobs.

Three problems this fixes, all found by reading the categories' actual members:

1. All 33 playground articles carried BOTH [[קטגוריה:גני משחקים]] and its parent
   [[קטגוריה:מתקנים ציבוריים]]. The parent was 83% playgrounds, so nothing else
   in it was visible. A page belongs in the most specific category that fits —
   the parent tag is dropped.

2. 'מוסדות ביישוב' (8 members) was a second label on articles that already sat
   in a better category. Only 'בית עינת' would have been orphaned by removing
   it, and it is plainly a מרכז קהילתי. The category is retired.

3. After yesterday's split created 'מרחבים קהילתיים' ("places the community
   meets"), the מתנ"ס and the halls list belonged there, not under a generic
   'facilities' heading. Moving them leaves each category a real job:

     מתקנים ציבוריים   — infrastructure the authority provides
     מרחבים קהילתיים   — places the community meets
     מרחבי פנאי וזיכרון — outdoor leisure and memorial sites

Idempotent: re-running makes no edits once the wiki matches the target state.
"""
import re
import sys

from wiki_client import WikiClient, API_URL

FACILITIES = "מתקנים ציבוריים"
COMMUNITY = "מרחבים קהילתיים"
INSTITUTIONS = "מוסדות ביישוב"
SENIORS_OLD = "אזרחים ותיקים"
SENIORS_NEW = "מסגרות לגיל השלישי"

# title -> (categories to remove, categories to add)
MOVES = {
    # community-meeting places misfiled under the generic facilities heading
    'מתנ"ס פרדס חנה כרכור': ([FACILITIES], [COMMUNITY]),
    "אולמות ומרחבי פעילות": ([FACILITIES], [COMMUNITY]),
    # 'מוסדות ביישוב' members: each keeps the category that actually describes it
    "בית עינת": ([INSTITUTIONS], [COMMUNITY]),
    "בית האיכר": ([INSTITUTIONS], [COMMUNITY]),
    "בית הראשונים": ([INSTITUTIONS], []),
    "בית יד לבנים": ([INSTITUTIONS], []),
    "המרכז המסחרי בכיכר הנדיב": ([INSTITUTIONS], []),
    "המרכז לגישור קהילתי פרדס חנה-כרכור": ([INSTITUTIONS], []),
    "מחנה 80": ([INSTITUTIONS], []),
    "קטגוריה:שרותי דת": ([INSTITUTIONS], [FACILITIES]),
}

SCOPES = {
    f"קטגוריה:{FACILITIES}": """<!-- SCOPE:START — הגדרת תחום הקטגוריה -->
קטגוריה זו מאגדת '''מתקנים ותשתיות ציבוריות''' בפרדס חנה-כרכור — מה שהרשות מעמידה \
לרשות התושבים: גני משחקים, ספריות, גינות כלבים ובתי עלמין.

'''מה שייך לכאן:''' המתקן עצמו, ובעיקר דרך קטגוריות המשנה שלו.

'''מה לא שייך לכאן:''' מקום שהקהילה נפגשת ופועלת בו — מרכז קהילתי, אולם או מועדון — \
שייך ל[[:קטגוריה:מרחבים קהילתיים]]. גנים, פארקים ואתרי הנצחה תחת כיפת השמיים שייכים \
ל[[:קטגוריה:מרחבי פנאי וזיכרון בקהילה]]. ערך שכבר נמצא בקטגוריית משנה כאן \
(למשל [[:קטגוריה:גני משחקים]]) '''אינו''' מסווג גם בקטגוריית האם.
<!-- SCOPE:END -->""",
    f"קטגוריה:{SENIORS_NEW}": f"""<!-- SCOPE:START — הגדרת תחום הקטגוריה -->
קטגוריה זו מאגדת '''מקומות ומסגרות''' בפרדס חנה-כרכור הפועלים עבור בני הגיל השלישי — \
מרכזי גמלאים, מועדונים וחוגים המיועדים לאזרחים ותיקים.

'''מה שייך לכאן:''' המקום או המסגרת.

'''מה לא שייך לכאן:''' אנשים. ערך על תושב ותיק שייך \
ל[[:קטגוריה:אישים בפרדס חנה-כרכור]]. קטגוריה זו החליפה את הקטגוריה הקודמת \
"{SENIORS_OLD}", ששמה רמז לקטגוריה של אנשים בעוד שכל חבריה היו מקומות.
<!-- SCOPE:END -->""",
}


def members(client, cat):
    out, cont = [], None
    while True:
        p = {"action": "query", "list": "categorymembers",
             "cmtitle": f"קטגוריה:{cat}", "cmlimit": 500, "format": "json"}
        if cont:
            p["cmcontinue"] = cont
        r = client.session.get(API_URL, params=p).json()
        out += [m["title"] for m in r["query"]["categorymembers"]]
        cont = r.get("continue", {}).get("cmcontinue")
        if not cont:
            return out


def retag(text, remove, add):
    """Drop `remove` category tags, append any of `add` not already present."""
    for cat in remove:
        text = re.sub(r"\n?\[\[קטגוריה:\s*" + re.escape(cat) + r"\s*\]\]", "", text)
    for cat in add:
        if not re.search(r"\[\[קטגוריה:\s*" + re.escape(cat) + r"\s*\]\]", text):
            text = text.rstrip() + f"\n[[קטגוריה:{cat}]]"
    return text.rstrip() + "\n"


def apply(client, title, remove, add, summary, dry):
    page = client.get_page(title)
    new = retag(page["wikitext"], remove, add)
    if new.strip() == page["wikitext"].strip():
        return False
    if dry:
        print(f"  would edit: {title}")
    else:
        client.edit_page(title, new, summary=summary)
    return True


def main():
    dry = "--write" not in sys.argv
    client = WikiClient()
    client.login()

    edits = 0

    # 1. playgrounds: drop the redundant parent tag
    gardens = [t for t in members(client, "גני משחקים") if not t.startswith("קטגוריה:")]
    print(f"playgrounds: {len(gardens)}")
    for t in gardens:
        edits += apply(client, t, [FACILITIES], [],
                       "טקסונומיה: הסרת סיווג כפול — הערך כבר בקטגוריית המשנה "
                       "'גני משחקים', ולכן אינו מסווג גם בקטגוריית האם", dry)

    # 2 + 3. targeted moves
    for t, (rm, add) in MOVES.items():
        edits += apply(client, t, rm, add,
                       "טקסונומיה: מיזוג 'מוסדות ביישוב' והפרדת מתקנים ציבוריים "
                       "ממרחבים קהילתיים", dry)

    # 4. seniors: rename the category on every member
    for t in members(client, SENIORS_OLD):
        edits += apply(client, t, [SENIORS_OLD], [SENIORS_NEW],
                       f"טקסונומיה: '{SENIORS_OLD}' נקראה על שם אנשים אך כל חבריה "
                       f"מקומות — שונתה ל'{SENIORS_NEW}'", dry)

    # 5. scope pages for the surviving categories
    for title, scope in SCOPES.items():
        try:
            cur = client.get_page(title)["wikitext"]
        except Exception:
            cur = ""
        new = (re.sub(r"<!-- SCOPE:START.*?<!-- SCOPE:END -->", scope, cur,
                      flags=re.S) if "SCOPE:START" in cur
               else (scope + "\n" + cur).strip() + "\n")
        if new.strip() != cur.strip():
            if dry:
                print(f"  would write scope: {title}")
            else:
                client.edit_page(title, new, summary="הגדרת תחום הקטגוריה")
            edits += 1

    print(f"\n{edits} page(s) {'would be ' if dry else ''}edited.")
    if dry:
        print("(dry run) הרץ עם --write כדי לבצע.")


if __name__ == "__main__":
    main()
