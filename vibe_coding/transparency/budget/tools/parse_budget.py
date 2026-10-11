"""ספר התקציב של המועצה (PDF מהגזברות) → עץ JSON לאפליקציה.

הספר הוא דוח "תקציב מול ביצוע במבנה מאזן": עמוד לכל מחלקה, עם שורות סיכום
(סה"כ) ושורות כרטיס (קוד בן 10 ספרות). העמודות במיקום x קבוע בכל עמוד, ולכן
קוראים מילים עם קואורדינטות ב-pdfplumber ומשייכים כל מספר לעמודה לפי הקצה
הימני שלו. מיקומי העמודות נלקחים מכותרת העמוד, כי הם זזים בין שנים.

מבנה העמודים:
  "10", "20"...      עמוד פרק (רק שורת סה"כ)        ← לא מוצג, רק לבדיקת סכומים
  "חינוך"            עמוד תחום (רק שורת סה"כ)       ← רמה 1 בעוגה
  "חינוך קדם יסודי"  עמוד מחלקה (סה"כ + כרטיסים)    ← רמה 2
  עמוד שמתחיל בכרטיס הוא המשך של המחלקה הקודמת.

שימוש: python tools/parse_budget.py src/budget_2026.pdf 2026 data/budget_2026.json
"""
import json
import re
import sys

import pdfplumber

HEB = re.compile(r'[֐-׿]')
NUM = re.compile(r'^-?[\d,]+\.\d\d$')
CODE = re.compile(r'^\d{10}$')


def fix(t):
    """pdfplumber מחזיר עברית הפוכה. מספרים ותאריכים נשארים כמו שהם."""
    return t[::-1] if HEB.search(t) else t


def rows_of(words):
    words = sorted(words, key=lambda w: (round(w['top']), -w['x0']))
    rows = []
    for w in words:
        if rows and abs(rows[-1][0]['top'] - w['top']) < 3:
            rows[-1].append(w)
        else:
            rows.append([w])
    return rows


def columns(words):
    """מיקום העמודות לפי הכותרת. None אם זה לא עמוד דוח רגיל."""
    cols = {}
    for w in words:
        t = fix(w['text'])
        if 'הצעת_תקציב' in t:
            cols['p'] = w['x1']
        elif 'תחזית_ביצוע' in t:
            cols['f_prev'] = w['x1']
        elif 'הפרש' in t:
            cols['diff'] = w['x1']
        elif t.startswith('תקציב_'):
            cols['b_prev'] = w['x1']
        elif t == 'משרות':
            cols['jobs'] = w['x1']
            cols['_details_edge'] = w['x0'] - 2
        elif t == 'כרטיס':
            cols['_code_x0'] = w['x0'] - 25
    return cols if {'p', 'f_prev', 'b_prev', 'jobs'} <= cols.keys() else None


def parse_row(row, cols):
    r = {'nums': {}, 'name': [], 'details': [], 'code': None, 'total': False}
    numeric = {c: x for c, x in cols.items() if not c.startswith('_')}
    num_edge = max(numeric.values()) + 8
    signs, nums = [], []
    for w in sorted(row, key=lambda w: -w['x1']):
        t = w['text']
        if t in ('סה"כ', 'כ"הס') and w['x0'] > cols['_code_x0']:
            r['total'] = True
        elif CODE.match(t) and w['x0'] > cols['_code_x0']:
            r['code'] = t
        elif NUM.match(t) and w['x1'] < num_edge:
            c = min(numeric, key=lambda c: abs(numeric[c] - w['x1']))
            if abs(numeric[c] - w['x1']) < 14:
                nums.append((c, w['x1'], float(t.replace(',', ''))))
        elif t in ('ז', 'ח') and w['x1'] < num_edge:
            signs.append((w['x0'], t))
        elif w['x1'] < cols['_details_edge']:
            r['details'].append(fix(t))
        else:
            name = fix(t)
            if len(name) > 1 and name[-1] in 'זח' and w['x1'] < num_edge + 40 and any(
                    0 <= w['x0'] - x1 < 7 for _, x1, _ in nums):
                # שם ארוך שנדבק לסימן של המספר שלידו ("במועדוניותז")
                signs.append((w['x0'], name[-1]))
                name = name[:-1]
            r['name'].append(name)
    for c, x1, v in nums:
        # ח = חובה (הוצאה, חיובי), ז = זכות (הכנסה, שלילי). הסימן צמוד מימין למספר.
        s = next((s for sx, s in signs if 0 <= sx - x1 < 7), None)
        r['nums'][c] = -v if s == 'ז' else v
        if s is None and v and c != 'jobs':
            r.setdefault('unsigned', []).append(c)
    if 'diff' in r['nums']:
        # עמודת השנה הקודמת היא היחידה ששמות גולשים אליה. ההפרש אמין, אז ממנו.
        r['nums']['b_prev'] = r['nums'].get('p', 0) - r['nums']['diff']
    if 'b_prev' in r.get('unsigned', []):
        # הסימן של עמודת השנה הקודמת נדבק לשם ("הרשותח", "משותפיזם").
        # את הערך משחזרים מעמודת ההפרש, ואת האות מורידים מהשם.
        if 'diff' in r['nums']:
            r['nums']['b_prev'] = r['nums'].get('p', 0) - r['nums']['diff']
        for k in range(len(r['name']) - 1, -1, -1):
            w = r['name'][k]
            if len(w) > 2 and w[-1] in 'זח':
                r['name'][k] = w[:-1]
                break
            if len(w) > 3 and w[-2] in 'זח' and w[-1] in 'םןץףך':
                r['name'][k] = w[:-2] + w[-1]
                break
    r['name'] = ' '.join(r['name']).strip()
    r['details'] = ' '.join(r['details']).strip()
    return r


def read_pages(path):
    pdf = pdfplumber.open(path)
    out = []
    for pi, page in enumerate(pdf.pages, start=1):
        cols = columns(page.extract_words())
        if not cols:
            continue
        # שמות ארוכים גולשים לעמודת השנה הקודמת, והאותיות והספרות מתערבבות למילה
        # אחת ("וישיבחו 0ת0.000,52"). באזור העמודות המספריות מפרידים את התווים
        # לשני זרמים, ספרות מול כל השאר, וקוראים מילים מכל זרם בנפרד.
        band = cols['b_prev'] + 8
        numeric = lambda o: o.get('object_type') == 'char' and o['x0'] < band and o['text'] in '0123456789,.-'
        # גם אותיות הסימן (ז/ח) שבתוך העמודות יוצאות לזרם משלהן, אחרת הן נדבקות לשם
        sign = lambda o: (o.get('object_type') == 'char' and o['text'] in 'זח' and o['x0'] < band
                          and any(abs(o['x0'] - (cols[c] - 6)) < 4 for c in ('b_prev', 'f_prev', 'p', 'diff')))
        rest = lambda o: o.get('object_type') != 'char' or not (numeric(o) or sign(o))
        words = (page.filter(rest).extract_words(x_tolerance=8) + page.filter(numeric).extract_words()
                 + page.filter(sign).extract_words())
        body, started = [], False
        for raw in rows_of(words):
            r = parse_row(raw, cols)
            if r['total'] or r['code']:
                started = True
            if not started or 'מתוך' in r['name']:
                continue
            if not r['total'] and not r['code'] and not r['nums']:
                # המשך של שם או פרטים שנשברו לשורה נוספת
                if body:
                    if r['name']:
                        body[-1]['name'] += ' ' + r['name']
                    if r['details']:
                        body[-1]['details'] += ' ' + r['details']
                continue
            body.append(r)
        if body:
            out.append({'page': pi, 'rows': body})
    return out


SHORT_WORDS = {'של', 'על', 'עם', 'או', 'גן', 'אם', 'כל', 'לא', 'בי', 'מח', 'קב', 'ו', 'דת', 'חי'}


def join_fragments(name):
    """שם שגלש לעמודת הסכום נקטע סביב אות הסימן ("עסק ים", "כ רכור"). מחברים
    שברים של אות או שתיים: שבר שנגמר באות סופית הולך למילה שלפניו, אחר - לזו שאחריו."""
    ws = name.split()
    out = []
    i = 0
    while i < len(ws):
        w = ws[i]
        frag = len(w) <= 2 and HEB.fullmatch(w[0]) and all(HEB.match(c) for c in w) and w not in SHORT_WORDS
        last = i == len(ws) - 1
        if frag and (w[-1] in 'םןץףך' or last) and out and HEB.match(out[-1][-1]):
            out[-1] += w
        elif frag and i + 1 < len(ws) and HEB.match(ws[i + 1][0]):
            ws[i + 1] = w + ws[i + 1]
        else:
            out.append(w)
        i += 1
    return ' '.join(out)


def clean(name):
    # שאריות של מספרים שהתערבבו בשם (מספר רכב ליד עמודת הסכום)
    name = ' '.join(t for t in name.split() if not re.search(r'\d[\d,]*\.\d|\d,\d', t))
    name = re.sub(r'\s+', ' ', name).strip()
    name = join_fragments(name)
    name = name.replace("''", '"').replace('``', '"')
    return re.sub(r'^\d+\s+', '', name)


def build(pages, year):
    """עמודים → רשימת תחומים, כל אחד עם מחלקות וכרטיסים."""
    chapters, domains, dept = [], [], None
    for pg in pages:
        rows = pg['rows']
        first = rows[0]
        if first['code']:  # עמוד המשך
            target = dept
        else:
            label = clean(first['name'])
            has_items = any(r['code'] for r in rows)
            if re.fullmatch(r'\d+', first['name'].strip()):
                chapters.append({'name': first['name'].strip(), 'net': first['nums'].get('p', 0),
                                 'start': len(domains)})
                continue
            if not has_items and not any(r['total'] and r['name'] in ('1 הכנסות', '2 הוצאות') for r in rows):
                domains.append({'name': label, 'net': first['nums'].get('p', 0), 'page': pg['page'],
                                'depts': []})
                dept = None
                continue
            if not domains:
                raise ValueError(f'מחלקה בלי תחום בעמוד {pg["page"]}')
            dept = {'name': label, 'net': first['nums'].get('p', 0), 'page': pg['page'], 'items': []}
            domains[-1]['depts'].append(dept)
            target = dept
        # בעמוד המשך ממשיכים מהסעיף (הכנסות/הוצאות, פעולות/שכר) של העמוד הקודם
        section, sub = target.get('_state', (None, None))
        for r in rows:
            if r['total']:
                n = r['name']
                if n.endswith('הכנסות'):
                    section = 'income'
                elif n.endswith('הוצאות'):
                    section = 'expense'
                elif n in ('פעולות', 'שכר'):
                    sub = n
                continue
            if not r['code']:
                continue
            target['items'].append({
                'code': r['code'], 'name': clean(r['name']), 'details': r['details'],
                'section': section, 'kind': sub,
                'p': r['nums'].get('p', 0.0), 'b_prev': r['nums'].get('b_prev', 0.0),
                'f_prev': r['nums'].get('f_prev', 0.0), 'jobs': r['nums'].get('jobs'),
            })
        target['_state'] = (section, sub)
    return chapters, domains


def check(chapters, domains):
    """הסכומים בספר הם הבדיקה: כל תחום = סכום המחלקות, כל מחלקה = סכום הכרטיסים."""
    bad = []
    for d in domains:
        for dp in d['depts']:
            s = sum(i['p'] for i in dp['items'])
            if abs(s - dp['net']) > 1:
                bad.append(f'מחלקה {dp["name"]} (עמוד {dp["page"]}): {s:,.0f} ≠ {dp["net"]:,.0f}')
        s = sum(dp['net'] for dp in d['depts'])
        if abs(s - d['net']) > 1:
            bad.append(f'תחום {d["name"]}: {s:,.0f} ≠ {d["net"]:,.0f}')
    ends = [c['start'] for c in chapters[1:]] + [len(domains)]
    for c, e in zip(chapters, ends):
        s = sum(d['net'] for d in domains[c['start']:e])
        if abs(s - c['net']) > 1:
            bad.append(f'פרק {c["name"]}: {s:,.0f} ≠ {c["net"]:,.0f}')
    return bad


if __name__ == '__main__':
    src, year, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    chapters, domains = build(read_pages(src), year)
    for d in domains:
        for dp in d['depts']:
            dp.pop('_state', None)
    problems = check(chapters, domains)
    for p in problems:
        print('⚠', p)
    exp = sum(i['p'] for d in domains for dp in d['depts'] for i in dp['items'] if i['section'] == 'expense')
    inc = -sum(i['p'] for d in domains for dp in d['depts'] for i in dp['items'] if i['section'] == 'income')
    n = sum(len(dp['items']) for d in domains for dp in d['depts'])
    print(f'{year}: {len(domains)} תחומים, {sum(len(d["depts"]) for d in domains)} מחלקות, {n} כרטיסים')
    print(f'  הוצאות {exp:,.0f}  הכנסות {inc:,.0f}  בעיות בסכומים: {len(problems)}')
    json.dump({'year': year, 'domains': domains}, open(dst, 'w'), ensure_ascii=False, indent=1)
