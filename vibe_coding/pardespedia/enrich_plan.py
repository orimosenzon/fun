#!/usr/bin/env python3
"""Turn the raw audit into a concrete per-article enrichment plan.

audit_articles.py measures what each article is missing; this decides what to
*do* about it. Recommendations are family-aware, because "add an image" means
something different for a street stub (no free photo will ever exist — it needs
a name etymology instead) than for a landmark (Commons probably already has
one). Where a free Commons file matches, the plan names the file.

Usage:
    python3 enrich_plan.py [--audit FILE] [--out FILE]
"""
import argparse
import datetime as dt
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# Commons categories whose photos map onto a known Hebrew article. Anything not
# listed here still shows up in the plan, just without a named candidate file.
COMMONS_TARGETS = {
    "Pardes Hanna cemetery": "בית העלמין פרדס חנה",
    "Pardes Hanna military cemetery": "בית העלמין פרדס חנה",
    "Nahal memorial": 'אנדרטת הנח"ל',
    "Caesarea-Pardes Hanna train station": "תחנת הרכבת קיסריה–פרדס חנה",
    "Remez Junction": "רמז (שכונה)",
    "Pardes Hanna old commercial center": "השוק הישן",
    "Historical images of Pardes Hanna-Karkur": "היסטוריה של פרדס חנה-כרכור",
    "Pardes Hanna Immigrant camp": "מחנה העולים בפרדס חנה",
    "Agriculture in Pardes Hanna-Karkur": "היסטוריה של פרדס חנה-כרכור",
    "Water towers in Pardes Hanna-Karkur": "מגדל המים",
    "Beit Harishonim (Pardes Hanna)": "בית הראשונים",
    "Beit Yad laBanim (Pardes Hanna)": "בית יד לבנים",
    "Pardes Hanna-Karkur town hall": "המועצה המקומית פרדס חנה-כרכור",
    "HaNasi square": "בית האיכר",
    "Schools in Pardes Hanna-Karkur": "אלונים (בית ספר)",
    "Karkur Primary School": "יסודי כרכור",
    "Pardes Hanna Agricultural High School": "בית הספר החקלאי פרדס חנה",
    "Racheli Shalev": "רחלי שלו",
    "Hanan Ben-Ari": "חנן בן ארי",
    "Ziv Avtalion": "זיו אבטליון",
    "David Tzadka": "דוד צדקה",
    "Nava Boker": "נאוה בוקר",
    "Gily Itskovitch": "גילי איצקוביץ'",
    "Tamara Avner": "תמרה אבנר",
    "Liat Levi-Kopelman": "ליאת לוי-קופלמן",
    "Eldad Bar Kochva": "אלדד בר כוכבא",
}

# Commons photo sets with no article at all — each is a creation opportunity.
NO_ARTICLE_YET = {
    "Dov Ben-Meir": "דב בן מאיר",
    "Hana Rado": "חנה רדו",
    "Moni Bitan": "מוני ביתן",
    "Sima Vaknin-Gil": "סימה ואקנין-גיל",
    "Beit Arzi (Pardes Hanna-Karkur)": "בית ארזי",
    "Kene Meir center": "מרכז קנה מאיר",
    "Camp Dotan": "מחנה דותן (מחנה 80)",
    "Sculptures in Pardes Hanna-Karkur": "יד לשילוב (פסל)",
    "Synagogues in Pardes Hanna-Karkur": "בית הכנסת בכרכור",
    "Demonstrations and protests in Pardes Hanna-Karkur": "מחאות והפגנות במושבה",
}


def family(title, cats):
    cats = [c.strip() for c in cats]
    if title.endswith("(רחוב)") or " (רחוב)" in title:
        return "רחוב"
    if title.startswith("גן ") or "גני משחקים" in cats:
        return "גן משחקים"
    if "בתי קפה" in cats or "מסעדות" in cats or "גלידריות" in cats:
        return "בית קפה/מסעדה"
    if any("אישים" in c or "סופרים" in c or "רבנים" in c for c in cats):
        return "אישים"
    if "מוסדות חינוך" in cats or "גני ילדים" in cats:
        return "חינוך"
    if "היסטוריה" in cats or "בתי עלמין" in cats or "הנצחה" in cats:
        return "היסטוריה/הנצחה"
    if "קופות חולים" in cats or "בריאות" in cats:
        return "בריאות"
    if "עסקים ותעשייה בפרדס חנה-כרכור" in cats or "בנקים" in cats:
        return "עסק"
    if "דף מיזם" in cats or "עזרה" in cats:
        return "דף מיזם"
    if "שכונות" in cats:
        return "שכונה"
    return "כללי"


# What each family actually needs, in priority order. The first entry is the
# single highest-value action — the one to do if only one thing gets done.
FAMILY_PLAYBOOK = {
    "רחוב": [
        "אטימולוגיה: על שם מי/מה נקרא הרחוב, ומתי ניתן השם",
        "הקשר שכונתי: לאיזו שכונה שייך, ומה הסכמה שלפיה נבחרו שמות הרחובות בה",
        "מה יש ברחוב: מוסדות, עסקים, גנים ומבנים בולטים — עם קישורים פנימיים",
        "תמונה: אין ולא תהיה בוויקישיתוף — נדרש צילום מקומי",
    ],
    "גן משחקים": [
        "תמונה: נדרש צילום מקומי (אין בוויקישיתוף)",
        "מה יש בגן: מתקנים, הצללה, גיל היעד, נגישות",
        "קישור לשכונה ולרחוב שבהם הגן נמצא",
    ],
    "בית קפה/מסעדה": [
        "לֵיד ראוי: מה אופי המקום, לא רק \"הוא בית קפה ב...\"",
        "פרטים מעשיים: כתובת, שעות פתיחה, טלפון, כשרות",
        "תמונה: מהפייסבוק/אינסטגרם של העסק בשימוש הוגן, או צילום מקומי",
        "הסיפור: מי הבעלים, ממתי פועל, מה המנה/המשקה המזוהה",
    ],
    "אישים": [
        "וידאו: הופעה/ראיון ביוטיוב — הכי רלוונטי למוזיקאים ואמנים",
        "דיוקן: בוויקישיתוף אם יש, אחרת שימוש הוגן עם מקור",
        "הקשר מקומי: מה הקשר שלו/שלה דווקא לפרדס חנה-כרכור",
        "מקורות: קישור לוויקיפדיה, לאתר האישי ולכתבות",
    ],
    "חינוך": [
        "נתונים: שנת הקמה, מספר תלמידים, שכבות גיל, זרם חינוכי",
        "תמונה: כמה מוסדות מופיעים בוויקישיתוף (ראו התאמות)",
        "ייחוד: מה מבדיל את המוסד — תוכנית לימודים, מסורת, אנשים",
        "קישור לאתר המוסד ולדף המועצה",
    ],
    "היסטוריה/הנצחה": [
        "תמונה היסטורית מוויקישיתוף — יש מאגר גדול ונחלת הכלל",
        "מקורות: ארכיון המדינה, בית הראשונים, עיתונות היסטורית",
        "הרחבה: תאריכים, שמות, מי הקים ולמה",
    ],
    "בריאות": [
        "פרטים מעשיים: שעות קבלה, טלפון, אילו שירותים ניתנים",
        "נגישות והגעה",
        "תמונה: צילום חזית מקומי",
    ],
    "עסק": [
        "מה העסק עושה בפועל, וממתי",
        "פרטים מעשיים: כתובת, שעות, אתר",
        "תמונה: שימוש הוגן מהעסק או צילום מקומי",
    ],
    "שכונה": [
        "גבולות ומיקום, ומספר תושבים משוער",
        "היסטוריה: מתי נבנתה ולמה נקראת כך",
        "מה יש בה: גנים, בתי כנסת, מוסדות — עם קישורים",
        "תמונה: צילום מקומי",
    ],
    "דף מיזם": ["דף פנימי — לא נדרשת העשרה אנציקלופדית"],
    "כללי": [
        "הרחבה לפי הנושא, עם מקורות חיצוניים",
        "תמונה מתאימה",
        "קישורים פנימיים לערכים קשורים",
    ],
}


def load_latest_audit(path=None):
    if path:
        return json.load(open(path, encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(HERE, "audit_reports", "audit-*.json")))
    return json.load(open(files[-1], encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit")
    ap.add_argument("--out", default=os.path.join(HERE, "audit_reports"))
    args = ap.parse_args()

    audit = load_latest_audit(args.audit)
    commons = json.load(open(os.path.join(HERE, "commons_index.json"), encoding="utf-8"))

    # article -> free Commons files that plausibly illustrate it
    matches = {}
    for f in commons:
        for cat in f["categories"]:
            target = COMMONS_TARGETS.get(cat)
            if target:
                matches.setdefault(target, []).append(f["title"])

    real = [a for a in audit if not a["redirect"] and not a["index_page"]]
    for a in real:
        a["family"] = family(a["title"], a["categories"])
        a["commons_candidates"] = matches.get(a["title"], [])

    by_family = {}
    for a in real:
        by_family.setdefault(a["family"], []).append(a)

    stamp = dt.date.today().isoformat()
    L = [f"# תוכנית העשרה פר-ערך — פרדספדיה, {stamp}", "",
         "נבנה אוטומטית מ-`audit_articles.py` + `commons_index.py`. לכל משפחת ערכים "
         "יש \"פלייבוק\" — סדר הפעולות שמחזיר הכי הרבה ערך — ואחריו רשימת הערכים "
         "עם מה שחסר בכל אחד.", ""]

    L += ["## סיכום מספרי", "",
          f"- ערכי תוכן: {len(real)}",
          f"- בלי תמונה: {sum(1 for a in real if a['images'] == 0)}",
          f"- דלים (< 700 תווים): {sum(1 for a in real if a['prose'] < 700)}",
          f"- בלי וידאו: {sum(1 for a in real if not a['video'])}",
          f"- בלי מקורות: {sum(1 for a in real if a['external_links'] == 0)}",
          f"- ערכים שיש להם תמונה חופשית מוכנה בוויקישיתוף: "
          f"{sum(1 for a in real if a['commons_candidates'] and a['images'] == 0)}", ""]

    L += ["## ערכים שחסרים לגמרי — ויש להם כבר תמונות חופשיות", "",
          "בוויקישיתוף יש אוספי תמונות שלמים על נושאים שאין עליהם ערך בפרדספדיה. "
          "כל שורה כאן היא ערך שאפשר להקים עם תמונה מוכנה:", ""]
    for cat, title in sorted(NO_ARTICLE_YET.items(), key=lambda kv: kv[1]):
        n = sum(1 for f in commons if cat in f["categories"])
        L.append(f"- **{title}** — {n} תמונות חופשיות בקטגוריה `{cat}`")
    L.append("")

    order = ["היסטוריה/הנצחה", "אישים", "חינוך", "בית קפה/מסעדה", "עסק",
             "בריאות", "שכונה", "רחוב", "גן משחקים", "כללי", "דף מיזם"]
    for fam in order:
        arts = by_family.get(fam, [])
        if not arts:
            continue
        L += [f"## {fam} ({len(arts)} ערכים)", "", "**סדר פעולות מומלץ:**", ""]
        L += [f"{i}. {step}" for i, step in enumerate(FAMILY_PLAYBOOK[fam], 1)]
        L += ["", "| ערך | טקסט | תמונות | חסר | תמונה חופשית זמינה |",
              "|---|---|---|---|---|"]
        for a in sorted(arts, key=lambda x: (-x["score"], x["title"])):
            cand = a["commons_candidates"]
            cand_s = f"{len(cand)} ✔" if cand else "—"
            L.append(f"| [{a['title']}]({a['url']}) | {a['prose']} | {a['images']} | "
                     f"{', '.join(a['needs']) or '—'} | {cand_s} |")
        L.append("")

    path = os.path.join(args.out, f"enrich-plan-{stamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    jpath = os.path.join(args.out, f"enrich-plan-{stamp}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(real, f, ensure_ascii=False, indent=1)
    print(f"wrote {path}\nwrote {jpath}")


if __name__ == "__main__":
    main()
