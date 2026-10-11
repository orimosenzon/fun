"""כל ספרי התקציב ← קובץ לכל שנה + קובץ סדרות לגרף לאורך השנים.

  python3 tools/build_years.py          קורא את כל הספרים מחדש (כמה דקות) ובונה
  python3 tools/build_years.py --fast   משתמש בקבצי השנים שכבר נקראו

פלט ב-web/data/:
  budget_<שנה>.json  העץ של השנה, עם מפתח קבוע ("key") לכל תחום ומחלקה
  years.json         רשימת השנים, הקישור לספר של כל אחת והערות
  series.json        לכל צומת: התקציב בכל שנה, והביצוע הצפוי כפי שמופיע בספר
                     של השנה שאחריה

ההתאמה בין שנים: תחומים ומחלקות לפי שם, עם טבלת כינויים לשמות שהשתנו. הטבלה
נבנתה לפי סמל הסעיף במבנה האחיד של משרד הפנים (ספרות 2–4 בקוד הכרטיס), שנשאר
קבוע גם כשהשם משתנה. כרטיסים מותאמים לפי הקוד המלא.
"""
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'web', 'data')
SRC = os.path.join(ROOT, 'src')

UP = 'https://www.pardes-hanna-karkur.muni.il/wp-content/uploads/'
BOOKS = {y: f'{UP}2025/02/תקציב-המועצה-לשנת-{y}.pdf' for y in (2013, 2014, 2015, 2016, 2019, 2020, 2021,
                                                              2022, 2023, 2024)}
BOOKS[2018] = f'{UP}2025/02/תקציב-המועצה-לשנת-2018-מפורט.pdf'
BOOKS[2025] = f'{UP}2025/02/תקציב-במועצה-לשנת-2025.pdf'
BOOKS[2026] = f'{UP}2026/01/1_תקציב-2026-ea99798703c0aba343738025777216ff.pdf'
# ל-2017 יש באתר רק דף סיכום. המספרים לקוחים מעמודת "תקציב 2017" בספר של 2018.
DERIVED = {2017: 2018}

NOTES = {
    2017: 'ספר התקציב המפורט של 2017 לא נמצא באתר המועצה. המספרים הם עמודת "תקציב 2017" בספר של 2018.',
    2022: 'בספר של 2022 ההכנסות גבוהות ב-57 אלף ₪ מההוצאות. זה כך בספר עצמו.',
}

# שמות שהשתנו ← השם העדכני (אותו סמל סעיף במבנה האחיד)
DOMAIN_ALIAS = {
    'בטחון': 'חירום וביטחון',
    'חגיגות וארועים': 'תרבות, אומנות וספורט',
}
DEPT_ALIAS = {
    'ראש הרשות המקומית וסגניו': 'ראש הרשות המקומית וסגניה',
    'הסברה ויחסי ציבור': 'דוברות והסברה',
    'הוצאות מימון': 'מימון',
    'מינהל שרותים מוניציפאליים': 'מינהל שפ"ע',
    'מינהל שמירה וביטחון': 'מחלקת חירום וביטחון',
    'חגיגות וארועים בישוב': 'תרבות,אומנות וספורט',
    'בטוח תלמידים': 'ביטוח תלמידים',
    'פעולות תרבות': 'פעולות בלתי פורמלי',
    'רווחת הפרט והמשפחה': 'פרט ומשפחה',
    'שרותים לילד ולנוער': 'ילד ונוער',
    'שרותים לזקן': 'אזרחים ותיקים',
    'שרותים למפגר': 'מוגבלויות',
    'בחירות': 'בחירות לרשויות',
    'מעונות ופעוטונים': 'חינוך קדם יסודי',
}


def parse(y):
    out = os.path.join(DATA, f'budget_{y}.json')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'parse_budget.py'),
                        os.path.join(SRC, f'budget_{y}.pdf'), str(y), out], capture_output=True, text=True)
    return y, (r.stdout + r.stderr).strip()


def derive(src, year):
    """שנה שיש לה רק עמודת "שנה קודמת" בספר של השנה שאחריה."""
    d = copy.deepcopy(src)
    d['year'] = year
    for dom in d['domains']:
        for dp in dom['depts']:
            for it in dp['items']:
                it['p'], it['b_prev'], it['f_prev'] = it['b_prev'], 0.0, 0.0
                it['details'] = ''
            dp['items'] = [i for i in dp['items'] if i['p']]
        dom['depts'] = [dp for dp in dom['depts'] if dp['items']]
    return d


def add_keys(d):
    for dom in d['domains']:
        dom['key'] = DOMAIN_ALIAS.get(dom['name'], dom['name'])
        for dp in dom['depts']:
            dp['key'] = DEPT_ALIAS.get(dp['name'], dp['name'])


def nodes_of(d, side):
    """(מפתח צומת, ערך, ערך ביצוע צפוי לשנה הקודמת) לכל כרטיס, בכל הרמות שמעליו."""
    sign = 1 if side == 'expense' else -1
    s = side[0]
    for dom in d['domains']:
        for dp in dom['depts']:
            for it in dp['items']:
                if it['section'] != side:
                    continue
                v, f = max(sign * it['p'], 0), max(sign * it['f_prev'], 0)
                for k in (f'{s}', f'{s}|D|{dom["key"]}', f'{s}|P|{dp["key"]}', f'{s}|I|{it["code"]}'):
                    yield k, v, f


def main():
    years = sorted(BOOKS)
    if '--fast' not in sys.argv:
        with ThreadPoolExecutor(5) as ex:
            for y, log in ex.map(parse, years):
                print(f'== {y}\n{log}')
    books = {y: json.load(open(os.path.join(DATA, f'budget_{y}.json'))) for y in years}
    for y, src in DERIVED.items():
        books[y] = derive(books[src], y)
        json.dump(books[y], open(os.path.join(DATA, f'budget_{y}.json'), 'w'), ensure_ascii=False)
    years = sorted(books)

    series, names = {}, {}
    for y in years:
        d = books[y]
        add_keys(d)
        json.dump(d, open(os.path.join(DATA, f'budget_{y}.json'), 'w'), ensure_ascii=False)
        for side in ('expense', 'income'):
            for k, v, f in nodes_of(d, side):
                e = series.setdefault(k, {'v': {}, 'f': {}})
                e['v'][y] = e['v'].get(y, 0) + v
                if f and y - 1 in books:
                    # הביצוע הצפוי של שנה y-1, כפי שהגזברות העריכה אותו בדצמבר של אותה שנה
                    e['f'][y - 1] = e['f'].get(y - 1, 0) + f
        for dom in d['domains']:
            names[dom['key']] = dom['name']
            for dp in dom['depts']:
                names[dp['key']] = dp['name']
                for it in dp['items']:
                    names[it['code']] = it['name']
    out = {'years': years, 'nodes': {}}
    for k, e in series.items():
        v = [round(e['v'][y]) if y in e['v'] else None for y in years]
        f = [round(e['f'][y]) if y in e['f'] else None for y in years]
        if not any(v):
            continue
        out['nodes'][k] = {'v': v, **({'f': f} if any(f) else {})}
    json.dump(out, open(os.path.join(DATA, 'series.json'), 'w'), ensure_ascii=False, separators=(',', ':'))
    meta = [{'year': y, 'source': BOOKS.get(DERIVED.get(y, y)), 'note': NOTES.get(y, ''),
             'derived_from': DERIVED.get(y)} for y in years]
    json.dump(meta, open(os.path.join(DATA, 'years.json'), 'w'), ensure_ascii=False, indent=1)
    size = os.path.getsize(os.path.join(DATA, 'series.json'))
    print(f'{len(years)} שנים, {len(out["nodes"])} סדרות, series.json {size // 1024}KB')
    for y in years:
        e = out['nodes']['e']['v'][years.index(y)]
        print(f'  {y}: הוצאות {e / 1e6:6.1f} מיליון')


if __name__ == '__main__':
    main()
