"""תב"רים וקרנות הרשות ← web/data/tabar.json.

התקציב הבלתי רגיל (תב"ר) הוא תקציב נפרד לכל פרויקט: בנייה, סלילה, שיפוץ, תכנון, ציוד. הוא לא מופיע בספר
התקציב השנתי ולכן לא בעוגה. המקורות:

- הדוחות הכספיים המבוקרים של משרד הפנים ב-data.gov.il (dataset local-authorities), לכל הרשויות:
  "טופס 3" (תקבולים ותשלומים בתב"רים), "נתונים לטופס 3" (פירוט מקורות המימון), "נספח 1 לטופס 3"
  (תב"רים לפי תחום), "ביאור 5" (התנועה בקרנות הרשות), וטופס 2 (הביצוע בתקציב הרגיל, להשוואה).
  לכל שנה Y לוקחים את הדוח של Y ("שנה נוכחית"), ואם אין, את הדוח של Y+1 ("שנה קודמת"). אין במאגר דוחות
  של 2020 ו-2021, ולכן 2020 חסרה.
- 2025: תמצית הדוחות הכספיים (אתר המועצה, 8/2026) לסכומים, והדוח הרבעוני של רבעון 4 2025 (לא מבוקר) לפירוט
  המקורות. מוזנים ידנית מ-src/tabar/ (סריקות).
- אישורי תב"רים במליאה: src/tabar/approvals.json (מהפרוטוקולים), אם קיים.
- רשויות דומות ואוכלוסייה: web/data/benchmark.json (אותה קבוצה כמו ב"מה מקבלים תמורת הכסף").

python3 tools/build_tabar.py          # משתמש במטמון ב-src/tabar/audited
python3 tools/build_tabar.py --fetch  # מוריד מחדש
"""
import json
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, 'src', 'tabar')
CACHE = os.path.join(SRC, 'audited')
DATA = os.path.join(ROOT, 'web', 'data')
UA = {'User-Agent': 'Mozilla/5.0'}
API = 'https://data.gov.il/api/3/action/datastore_search'
HOME = 'פרדס חנה-כרכור'

AUDITED = {2024: '6e153ddd-d4d4-4450-a78c-20f594a7986a', 2023: '5825080f-905c-40f7-9f53-8ea1cf8ed394',
           2022: 'e5ff9ad0-6db2-4660-a94e-4499fce9475d', 2019: 'f3f39a40-8404-4179-bcee-616ab2433631',
           2018: '1618b871-9b3d-4bc5-acc0-c1fbe939a3b1'}
YEARS = [2017, 2018, 2019, 2021, 2022, 2023, 2024]

UP = 'https://www.pardes-hanna-karkur.muni.il/wp-content/uploads/'
DOCS = {
    'summary_2025': UP + '2026/08/1_תמצית-הדוחות-הכספיים-לש-נת-2025-89344a0beb95f8fd2cc07fc9985724a2.pdf',
    'q4_2025': UP + '2026/05/דוח-רבעוני-4-2025-חתום.pdf',
    'q4_2024': UP + '2025/06/1_דוח-כספי-רבעוני-ינואר-דצמבר-2024.pdf',
    'audit_levy': UP + '2025/02/דוח-ביקורת-בנושא-היטלי-השבחה.pdf',
    'levy_law': UP + '2026/07/1_חוק-עזר-היטלי-פיתוח-מעו-דכן-72ace375c5658e41a06b179813b44a77.pdf',
}
PLANS = [  # תוכניות הפיתוח השנתיות שבאתר המועצה. אומדן כולל באלפי ש"ח, כפי שמופיע בשורת הסיכום
    {'year': 2014, 'url': UP + '2025/04/תכנית-עבודה-ותקציבי-פיתוח-לשנת-2014.pdf', 'total': None},
    {'year': 2016, 'url': UP + '2025/04/תכנית-עבודה-ותקציב-פיתוח-לשנת-2016.pdf', 'total': 60420},
    {'year': 2017, 'url': UP + '2025/04/תכנית-עבודה-ותקציבי-פיתוח-לשנת-2017.pdf', 'total': 74652},
    {'year': 2018, 'url': UP + '2025/04/תכנית-עבודה-ותקציבי-פיתוח-לשנת-2018.pdf', 'total': 95379},
    {'year': 2020, 'url': UP + '2025/04/תוכנית-פיתוח-2020-סופי-מליאה.pdf', 'total': None},
    {'year': 2021, 'url': UP + '2025/04/תוכנית-פיתוח-2021.pdf', 'total': None},
]

# 2025: עד שיתפרסם הדוח המבוקר. אלפי ש"ח.
Y2025 = {
    # תמצית הדוחות הכספיים לשנת 2025, עמ' 41: "נתוני ביצוע התקציב הבלתי רגיל" ו"נתוני ביצוע התקציב הרגיל"
    'receipts': 87149, 'payments': 87925, 'open_balance': 44286, 'close_balance': 43510,
    'regular_payments': 405499, 'fund_balance': 13426, 'fund_balance_prev': 9547,
    # דוח רבעוני 4/2025 (לא מבוקר), טופס 3. סה"כ התקבולים שם 83,507, נמוך מהתמצית
    'q4': {'receipts': 83507, 'payments': 84284, 'funds': 44109, 'loans': 21000, 'gov': 7404, 'other': 10955,
           'owners': 8, 'regular': 31,
           'ministries': {'משרד החינוך': 6144, 'משרד הבינוי והשיכון': 1044, 'משרד התחבורה': 702,
                          'משרד הפנים': 137, 'משרדים אחרים': -623}},
    'loans_outstanding': 26738, 'loans_outstanding_prev': 8143,
}

DROP = {'סהכ תברים', 'סהכ תברים נטו', 'תברים שנסתיימו', 'לא רגילים'}
MINISTRIES = ['משרד החינוך', 'משרד התחבורה', 'משרד הפנים', 'המשרד לאיכות הסביבה', 'משרד השיכון', 'משרד הרווחה',
              'המשרד לבטחון פנים', 'משרד הבטחון', 'משרדים אחרים']


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'["״\']', '', str(s or ''))).strip()


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch(rid, **filters):
    out, off = [], 0
    while True:
        p = {'resource_id': rid, 'limit': 32000, 'offset': off}
        p.update(filters)
        d = requests.get(API, headers=UA, params=p, timeout=300).json()['result']
        out += d['records']
        off += len(d['records'])
        if off >= d['total'] or not d['records']:
            return out


def slim(r):
    return {'a': norm(r['שם_רשות']), 's': norm(r.get('גליון') or r.get('גיליון')), 'r': norm(r['שורה']),
            'c': norm(r['עמודה']), 'y': int(r['שנת_נתונים']), 'v': num(r['ערך'])}


def load(refetch):
    """{שנת דוח: {'home': [...], 'all': [...]}}. home = כל הגיליונות של פרדס חנה; all = טופס 3 לכל הרשויות."""
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for rep, rid in AUDITED.items():
        path = os.path.join(CACHE, f'slim_{rep}.json')
        if os.path.exists(path) and not refetch:
            out[rep] = json.load(open(path))
            continue
        home = [slim(r) for r in fetch(rid, q='פרדס חנה') if 'פרדס' in str(r.get('שם_רשות'))]
        sheet = 'גליון' if rep >= 2023 else 'גיליון'
        allr = []
        for s in ('טופס 3', 'נתונים לטופס 3'):
            allr += [slim(r) for r in fetch(rid, filters=json.dumps({sheet: s}, ensure_ascii=False))]
        out[rep] = {'home': home, 'all': allr}
        json.dump(out[rep], open(path, 'w'), ensure_ascii=False)
        print(rep, len(home), len(allr), flush=True)
    return out


def pick(recs, sheet, row, col_re, year):
    for r in recs:
        if r['s'] == sheet and r['r'] == row and r['y'] == year and re.search(col_re, r['c']) and r['v'] is not None:
            return r['v']
    return None


def year_value(rep_data, sheet, row, year, col_now='^שנה נוכחית$', col_prev='^שנה קודמת$', key='home', recs=None):
    """הערך של שנה מהדוח שלה, ואם אין, מהדוח של השנה שאחריה."""
    if year in rep_data:
        v = pick(recs or rep_data[year][key], sheet, row, col_now, year)
        if v is not None:
            return v
    if year + 1 in rep_data and not recs:
        return pick(rep_data[year + 1][key], sheet, row, col_prev, year)
    return None


def home_year(R, y):
    g = lambda sheet, row, **k: year_value(R, sheet, norm(row), y, **k)
    rec = {
        'year': y,
        'receipts': g('טופס 3', 'סהכ תקבולים'),
        'payments': g('טופס 3', 'סהכ תשלומים'),
        'works': g('טופס 3', 'עבודות שבוצעו במשך השנה'),
        'balance': g('טופס 3', 'יתרות זמניות נטו לסוף השנה'),
        'sources': {
            'funds': g('נתונים לטופס 3', 'העברה מקרנות הרשות'),
            'gov': g('טופס 3', 'השתתפות הממשלה'),
            'loans': g('טופס 3', 'מלוות שנתקבלו'),
            'other': g('טופס 3', 'השתתפות מוסדות'),
            'owners': g('טופס 3', 'השתתפות בעלים'),
            'regular': g('טופס 3', 'השתתפות התקציב הרגיל'),
        },
        'ministries': {m: v for m in MINISTRIES if (v := g('נתונים לטופס 3', m))},
    }
    # "מקורות עצמיים אחרים" בטופס 3 = ההעברה מהקרנות (בודקים שזה מתאים)
    own = g('טופס 3', 'מקורות עצמיים אחרים')
    if rec['sources']['funds'] is None:
        rec['sources']['funds'] = own
    # התקציב הרגיל בפועל, להשוואת גודל
    reg = None
    for row in ('פעולות רגילות - סהכ תשלומים', 'סהכ כללי - תשלומים'):
        reg = reg or year_value(R, 'טופס 2', norm(row), y, col_now='ביצוע.*נוכחית', col_prev='ביצוע.*קודמת')
    rec['regular_payments'] = reg
    known = sum(v or 0 for v in rec['sources'].values())
    if rec['receipts'] and rec['receipts'] - known > 1:
        rec['sources']['rest'] = rec['receipts'] - known
    # לפי תחום (נספח 1 לטופס 3)
    rep, now = (y, True) if y in R else (y + 1, False)
    col_pay = 'ביצוע שנה נוכחית הוצאות' if now else 'ביצוע שנה קודמת הוצאות'
    dom = {}
    for r in R[rep]['home']:
        if r['s'] == 'נספח 1 לטופס 3' and r['c'] == col_pay and r['y'] == y and r['r'] not in DROP and r['v']:
            dom[r['r']] = r['v']
    # בדוח של השנה שאחריה, עמודת "שנה קודמת" כאן היא ביצוע מצטבר ולא שנתי, ולכן רק מהדוח של אותה שנה
    rec['domains'] = dict(sorted(dom.items(), key=lambda kv: -kv[1])) if now else {}
    if now:  # מצב התב"רים הפתוחים בסוף השנה, רק מהדוח של אותה שנה
        a = {}
        for r in R[y]['home']:
            if r['s'] == 'נספח 1 לטופס 3' and r['y'] == y and r['r'] not in DROP and r['v']:
                a.setdefault(r['r'], {})[r['c']] = r['v']
        rec['portfolio'] = {k: {'approved': v.get('התקציב המאושר'), 'spent': v.get('ביצוע מצטבר הוצאות'),
                                'received': v.get('ביצוע מצטבר הכנסות'), 'open': v.get('מספר התברים פתוחים'),
                                'count': v.get('סהכ מספר התברים')} for k, v in a.items()}
        rec['count_open'] = pick(R[y]['home'], 'נספח 1 לטופס 3', 'סהכ תברים', '^מספר התברים פתוחים$', y)
        rec['count'] = pick(R[y]['home'], 'נספח 1 לטופס 3', 'סהכ תברים', '^סהכ מספר התברים$', y)
        rec['approved_total'] = pick(R[y]['home'], 'נספח 1 לטופס 3', 'סהכ תברים', '^התקציב המאושר$', y)
    # קרנות הרשות (ביאור 5)
    fund = {}
    if now:
        rows = [r for r in R[y]['home'] if r['s'] == 'ביאור 5' and r['y'] == y]
        named = {'קרן הטל השבחה': 'השבחה', 'קרן ביוב (****)': 'ביוב', 'קרן ממכירת נכסים (****)': 'מכירת נכסים',
                 'קרן ממכירת מקרקעין (****)': 'מכירת מקרקעין', 'קרן ניקוז (****)': 'ניקוז',
                 'קרנות פיתוח (****)': 'פיתוח', 'קרן עודפי תברים סופיים (****)': 'עודפי תב"רים',
                 'סהכ תנועת הקרנות בשנה נוכחית': 'total'}
        for r in rows:
            k = named.get(r['r'])
            if not k:
                k = 'אחרות' if r['r'].startswith('קרנות אחר') else None
            if not k or r['v'] is None:
                continue
            c = {'יתרה לתחילת שנה': 'open', 'תקבולים בשנת הדוח': 'in', 'העברות לתברים': 'to_tabar',
                 'העברה לתקציב הרגיל': 'to_regular', 'יתרה לסוף השנה': 'close'}.get(r['c'])
            if c:
                fund.setdefault(k, {})
                fund[k][c] = fund[k].get(c, 0) + r['v']
        lev = {}
        for r in rows:
            if r['c'] == 'הרכב התקבולים בקרן היטל השבחה' and r['r'] in ('היטל השבחה', 'היטל חלף השבחה'):
                lev[r['r']] = r['v']
        if lev:
            fund['levy_mix'] = lev
    else:
        rows = [r for r in R[y + 1]['home'] if r['s'] == 'ביאור 5' and r['r'] == 'סהכ תנועת הקרנות - שנה קודמת'
                and r['y'] == y]
        t = {}
        for r in rows:
            c = {'יתרה לתחילת שנה': 'open', 'תקבולים בשנת הדוח': 'in', 'העברות לתברים': 'to_tabar',
                 'העברה לתקציב הרגיל': 'to_regular', 'יתרה לסוף השנה': 'close'}.get(r['c'])
            if c and r['v'] is not None:
                t[c] = r['v']
        if t:
            fund['total'] = t
    rec['funds'] = fund
    # היטל השבחה: חיובים וגבייה (נספח 2 לטופס 1)
    if now:
        hs = {}
        for r in R[y]['home']:
            if r['s'] == 'נספח 2 לטופס 1' and r['r'] in ('הטלי השבחה', 'הטל השבחה') and r['y'] == y:
                c = {'סהכ חיובים נטו': 'charged', 'סהכ גביות בשנת הדוח': 'collected',
                     'יתרה לסוף השנה': 'debt'}.get(r['c'])
                if c and r['v'] is not None:
                    hs[c] = r['v']
        if hs:
            rec['levy'] = hs
    return rec


def peers_table(R, bench):
    """לכל שנה ולכל רשות: תשלומי תב"רים לתושב, וחלק הקרנות בתקבולים."""
    pop = {(r['y'], norm(r['n']).replace('-', ' ')): r['pop'] for r in bench['rows']}
    names = {norm(n).replace('-', ' '): n for n in bench['peers'] + [HOME]}
    out = []
    for y in YEARS:
        vals = {}
        for rep in (y, y + 1):
            if rep not in R:
                continue
            col = '^שנה נוכחית$' if rep == y else '^שנה קודמת$'
            for r in R[rep]['all']:
                if r['y'] != y or not re.search(col, r['c']) or r['v'] is None:
                    continue
                key = {'סהכ תשלומים': 'pay', 'סהכ תקבולים': 'rec', 'העברה מקרנות הרשות': 'funds'}.get(r['r'])
                if key and (r['s'] == 'טופס 3' or key == 'funds'):
                    vals.setdefault(r['a'], {}).setdefault(key, r['v'])
        for a, v in vals.items():
            k = a.replace('-', ' ')
            if k not in names or not v.get('pay'):
                continue
            p = pop.get((y, k)) or pop.get((y - 1, k))
            if not p:
                continue
            out.append({'y': y, 'n': names[k], 'pay_res': round(v['pay'] * 1000 / p),
                        'funds_share': round(100 * v['funds'] / v['rec'], 1) if v.get('funds') and v.get('rec') else None,
                        'funds_res': round(v['funds'] * 1000 / p) if v.get('funds') else None})
    return out


def main():
    R = load('--fetch' in sys.argv)
    bench = json.load(open(os.path.join(DATA, 'benchmark.json')))
    years = [home_year(R, y) for y in YEARS]
    q = Y2025['q4']
    years.append({
        'year': 2025, 'receipts': Y2025['receipts'], 'payments': Y2025['payments'], 'balance': Y2025['close_balance'],
        'regular_payments': Y2025['regular_payments'],
        'sources': {'funds': q['funds'], 'gov': q['gov'], 'loans': q['loans'], 'other': q['other'],
                    'owners': q['owners'], 'regular': q['regular']},
        'sources_note': 'פירוט המקורות מהדוח הרבעוני של רבעון 4 (לא מבוקר), שבו סך התקבולים 83.5 מיליון. '
                        'הסכומים הכוללים מתמצית הדוחות הכספיים.',
        'ministries': q['ministries'], 'domains': {}, 'funds': {'total': {'close': Y2025['fund_balance']}},
        'provisional': True,
    })
    peers = peers_table(R, bench)
    approvals = []
    ap = os.path.join(SRC, 'approvals.json')
    if os.path.exists(ap):
        approvals = json.load(open(ap))
    out = {
        'home': HOME, 'peers': bench['peers'], 'peer_rule': bench['peer_rule'], 'years': years, 'peer_rows': peers,
        'loans': {'2024': Y2025['loans_outstanding_prev'], '2025': Y2025['loans_outstanding']},
        'plans': PLANS, 'docs': DOCS, 'approvals': approvals,
        'sources': {'audited': 'https://data.gov.il/dataset/local-authorities'},
    }
    json.dump(out, open(os.path.join(DATA, 'tabar.json'), 'w'), ensure_ascii=False, separators=(',', ':'))
    for r in years:
        s = r['sources']
        print(r['year'], 'תקבולים', r['receipts'], 'תשלומים', r['payments'], 'רגיל', r['regular_payments'],
              '| קרנות', s['funds'], 'ממשלה', s['gov'], 'מלוות', s['loans'], 'מוסדות', s['other'],
              '| תחומים', list(r['domains'].items())[:3], '| קרנות', r['funds'].get('total'))
    print(len(peers), 'שורות השוואה;', len(approvals), 'אישורים')


if __name__ == '__main__':
    main()
