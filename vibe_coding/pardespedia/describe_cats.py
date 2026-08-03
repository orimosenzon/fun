#!/usr/bin/env python3
"""Give the community/education categories a written scope.

The duplicates merged by merge_cats.py were not an accident: three of the four
community categories had no category page at all, so an editor filing a new
article had nothing to read and simply invented a name. A merge without a
written scope just buys time until the next editor guesses differently.

Each page below states what belongs, what does not, and where the neighbours
are — the "where does this go" question answered in the one place someone
already is when they ask it.

Existing text is preserved: the scope block is prepended, and a page that
already carries the marker is updated in place rather than accumulating copies.
"""
import re
import sys

from wiki_client import WikiClient

MARK_START = "<!-- SCOPE:START — הגדרת תחום הקטגוריה -->"
MARK_END = "<!-- SCOPE:END -->"

# the community scheme, shown on every member of the cluster
NAV = (
    "{| class=\"wikitable\" style=\"background:#f9f6f0; font-size:95%;\"\n"
    "|+ שלושת הצירים של קטגוריות הקהילה\n"
    "! ציר !! קטגוריה !! מה נכנס\n"
    "|-\n| '''מי''' || [[:קטגוריה:קהילות]] || קבוצות של אנשים — קהילה על בסיס מקום, זהות או תחום עניין\n"
    "|-\n| '''מה''' || [[:קטגוריה:יוזמה קהילתית]] || יוזמות, מיזמים ופעילויות שתושבים מפעילים\n"
    "|-\n| '''איפה''' || [[:קטגוריה:מרחבים קהילתיים]] || מקומות פיזיים שהקהילה נפגשת בהם\n"
    "|}"
)

PAGES = {
    "קהילות": (
        "קטגוריה זו מאגדת '''קבוצות של אנשים''' בפרדס חנה-כרכור — קהילות על בסיס מקום מגורים, "
        "זהות, אמונה או תחום עניין משותף.\n\n"
        "'''מה שייך לכאן:''' הקהילה עצמה — האנשים והקשר ביניהם.\n\n"
        "'''מה לא שייך לכאן:''' מיזם או פעילות שהקהילה מפעילה שייכים ל[[:קטגוריה:יוזמה קהילתית]]; "
        "המקום שבו היא נפגשת שייך ל[[:קטגוריה:מרחבים קהילתיים]]. ערך יכול כמובן להשתייך ליותר מאחת.",
        NAV,
    ),
    "יוזמה קהילתית": (
        "קטגוריה זו מאגדת '''יוזמות, מיזמים ופעילויות''' שתושבי פרדס חנה-כרכור מקימים ומפעילים — "
        "מגמ\"ח שכונתי ועד פסטיבל שנתי.\n\n"
        "'''מה שייך לכאן:''' הדבר שנעשה — היוזמה, המיזם, האירוע החוזר.\n\n"
        "'''מה לא שייך לכאן:''' הקבוצה שמאחורי היוזמה שייכת ל[[:קטגוריה:קהילות]]; "
        "המבנה או המגרש שבו היא מתקיימת שייך ל[[:קטגוריה:מרחבים קהילתיים]].\n\n"
        "''הערה: הקטגוריות \"פעילות קהילתית\" ו\"שרותי קהילה\" אוחדו לכאן — הן היו שם אחר לאותו דבר.''",
        NAV,
    ),
    "מרחבים קהילתיים": (
        "קטגוריה זו מאגדת '''מקומות פיזיים''' בפרדס חנה-כרכור שהקהילה נפגשת ופועלת בהם — "
        "מרכזים קהילתיים, אולמות, מועדונים ומרחבי פעילות.\n\n"
        "'''מה שייך לכאן:''' המקום עצמו.\n\n"
        "'''מה לא שייך לכאן:''' הפעילות שמתקיימת בו שייכת ל[[:קטגוריה:יוזמה קהילתית]]; "
        "הקבוצה שמשתמשת בו שייכת ל[[:קטגוריה:קהילות]]. גנים, פארקים ואתרי הנצחה תחת כיפת השמיים — "
        "ראו [[:קטגוריה:מרחבי פנאי וזיכרון בקהילה]].",
        NAV,
    ),
    "מרחבי פנאי וזיכרון בקהילה": (
        "קטגוריה זו מאגדת '''מרחבים פתוחים''' בפרדס חנה-כרכור — גנים, פארקים, שבילים ואתרי הנצחה.\n\n"
        "'''מה שייך לכאן:''' מרחב תחת כיפת השמיים, בין אם הוא משמש לפנאי ובין אם להנצחה.\n\n"
        "'''מה לא שייך לכאן:''' מבנים ואולמות שייכים ל[[:קטגוריה:מרחבים קהילתיים]]; "
        "מתקני משחק שייכים ל[[:קטגוריה:גני משחקים]].",
        None,
    ),
    "חינוך": (
        "קטגוריה זו מאגדת את '''כלל הנושאים הנוגעים לחינוך''' בפרדס חנה-כרכור — מדיניות, "
        "מאבקים ושינויים במערכת החינוך המקומית.\n\n"
        "'''מוסדות החינוך עצמם''' — בתי ספר, גני ילדים ומסגרות לימוד — מרוכזים בתת-הקטגוריה "
        "[[:קטגוריה:מוסדות חינוך]].",
        None,
    ),
}

# category -> parent to add (subcategory relations)
PARENTS = {"מוסדות חינוך": "חינוך"}


def splice(text, block):
    """Put the scope block at the top, replacing an earlier one if present."""
    marked = f"{MARK_START}\n{block}\n{MARK_END}"
    if MARK_START in text and MARK_END in text:
        s = text.index(MARK_START)
        e = text.index(MARK_END) + len(MARK_END)
        return text[:s] + marked + text[e:]
    return marked + ("\n\n" + text.lstrip() if text.strip() else "\n")


def main():
    dry = "--dry-run" in sys.argv
    c = WikiClient()
    c.login()

    for cat, (body, nav) in PAGES.items():
        title = f"קטגוריה:{cat}"
        page = c.get_page(title)
        current = page["wikitext"] if page["exists"] else ""
        block = body + ("\n\n" + nav if nav else "")
        new = splice(current, block)
        if new == current:
            print(f"  = unchanged: {title}")
            continue
        print(f"  {'would write' if dry else 'write'}: {title} "
              f"({'new' if not page['exists'] else 'update'})")
        if not dry:
            c.edit_page(title, new, summary="הגדרת תחום הקטגוריה וקישור לקטגוריות האחיות")

    for cat, parent in PARENTS.items():
        title = f"קטגוריה:{cat}"
        page = c.get_page(title)
        text = page["wikitext"] if page["exists"] else ""
        if re.search(r"\[\[\s*(?:קטגוריה|Category)\s*:\s*" + re.escape(parent), text):
            print(f"  = already under {parent}: {title}")
            continue
        new = text.rstrip() + f"\n\n[[קטגוריה:{parent}]]\n"
        print(f"  {'would parent' if dry else 'parent'}: {title} → {parent}")
        if not dry:
            c.edit_page(title, new,
                        summary=f"סידור היררכיה: תת-קטגוריה של [[:קטגוריה:{parent}]]")


if __name__ == "__main__":
    main()
