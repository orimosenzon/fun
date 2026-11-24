"""ספר התקציב של עיריית ירושלים (PDF, 2023 ואילך) → רשימת סעיפים.

כל קובץ "תקציב" (t) של מינהל בנוי מעמודים כאלה:

    עיריית ירושלים
    התקציב הרגיל לשנת 2026
    פירוט סעיפי ההוצאה          (או ההכנסה)
    <שם האגף>                    ← כותרת העמוד
    מספר סעיף <שם היחידה> תקציב 2026 תקציב 2025     ← שורת כותרת של יחידה
    תקן כ"א / שעתיים            ← משרות
    8210-111-2  משכורת ...  1,904,746  1,920,217
    ____________
    סה"כ שכר ומשכורת ...
    סה"כ יחידה ...             ← בדיקה עצמית
    סה"כ אגף ...

הטקסט של pdfplumber יוצא בסדר חזותי (עברית הפוכה), ולכן עובדים עם מילים
וקואורדינטות. המספרים מיושרים לימין, והקצה הימני שלהם זהה לקצה הימני של
המילה "תקציב" בכותרת העמודה. כך יודעים לאיזו שנה שייך מספר בודד בשורה:
סעיף שבוטל מופיע רק בעמודת השנה הקודמת, וסעיף חדש רק בעמודת השנה.

שימוש לבדיקה: python3 tools/parse_pdf.py <file.pdf> [שנה]
"""
import re
import sys

import pdfplumber

HEB = re.compile(r'[֐-׿]')
NUM = re.compile(r'^-?\(?\d{1,3}(,\d{3})*(\.\d+)?\)?-?$')
CODE = re.compile(r'^(\d{4})-(\d{3})-(\d)$')
YEAR = re.compile(r'^(19|20)\d\d$')


def fix(t):
    return t[::-1] if HEB.search(t) else t


def num(t):
    neg = t.startswith('-') or t.endswith('-') or t.startswith('(')
    v = float(t.strip('-()').replace(',', ''))
    return -v if neg else v


BAD_PAGES = []   # עמודים שלא נקראו (גופן פגום)

# תווים שבורים בגופן של הספר
GLYPH = str.maketrans({'ð': 'נ'})


def lines_of(words):
    words = sorted(words, key=lambda w: (round(w['top']), -w['x1']))
    rows = []
    for w in words:
        w = dict(w, text=w['text'].translate(GLYPH))
        if rows and abs(rows[-1][0]['top'] - w['top']) < 3:
            rows[-1].append(w)
        else:
            rows.append([w])
    out = []
    for r in rows:
        r.sort(key=lambda w: -w['x1'])   # מימין לשמאל, סדר הקריאה
        # pdfplumber חותך אות סופית למילה נפרדת ("יחיד ה"), בלי רווח ביניהן: מאחדים
        merged = [r[0]]
        for w in r[1:]:
            a = merged[-1]
            if HEB.search(a['text']) and HEB.search(w['text']) and a['x0'] - w['x1'] < 0.6:
                merged[-1] = dict(a, text=w['text'] + a['text'], x0=w['x0'])
            else:
                merged.append(w)
        out.append(merged)
    return out


def text_of(ws):
    return ' '.join(fix(w['text']) for w in ws)


def header_columns(row):
    """שורת כותרת: [(שנה, קצה ימני)] לכל עמודת "תקציב <שנה>"."""
    cols = []
    for w in row:
        if fix(w['text']) == 'תקציב':
            # השנה נמצאת משמאל למילה "תקציב" (מספר אחריה בקריאה)
            yrs = [y for y in row if YEAR.match(y['text']) and y['x1'] <= w['x0'] + 2 and w['x0'] - y['x1'] < 15]
            if yrs:
                y = max(yrs, key=lambda y: y['x1'])
                cols.append((int(y['text']), w['x1']))
    return cols


def is_header(row):
    t = [fix(w['text']) for w in row]
    return 'סעיף' in t and 'מספר' in t and t.count('תקציב') >= 1


def assign(row_nums, cols):
    """מספרים בשורה → {שנה: ערך} לפי הקצה הימני הקרוב."""
    out = {}
    for w in row_nums:
        yr, edge = min(cols, key=lambda c: abs(c[1] - w['x1']))
        if abs(edge - w['x1']) < 25:
            out[yr] = num(w['text'])
    return out


def parse_file(path, year, side_hint=None):
    """מחזיר (items, checks, warnings).

    item: code, name, side, p, b_prev, agaf, unit, unit_code, page, jobs
    checks: [(level, name, page, expected, got)] לכל שורת סה"כ יחידה/אגף
    """
    items, checks, warns = [], [], []
    agaf = unit = None
    side = side_hint
    cols = None
    unit_items, agaf_items = [], []
    jobs = {}
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            try:
                rows = lines_of(page.extract_words(keep_blank_chars=False, use_text_flow=False))
            except Exception:
                BAD_PAGES.append((path, pno))
                warns.append(f'{path}:{pno}: העמוד לא נקרא')
                continue
            expect_title = False
            # עמוד של דברי הסבר (בקובץ המאוחד "תקציב ודברי הסבר") נקרא ב-parse_details
            if page_kind(rows) == 'd':
                continue
            for r in rows:
                flat = text_of(r).replace(' ', '')
                if flat.startswith('עירייתירושלים') or flat.startswith('התקציבהרגיל'):
                    continue
                if flat.startswith('פירוטסעיפי'):
                    side = 'income' if 'הכנס' in flat else 'expense'
                    expect_title = True
                    continue
                if expect_title and not is_header(r):
                    agaf = re.sub(r'\s+', ' ', text_of(r)).strip()
                    expect_title = False
                    continue
                expect_title = False
                if is_header(r):
                    c = header_columns(r)
                    if c:
                        cols = c
                    # שם היחידה: המילים בין "מספר סעיף" לבין "תקציב"
                    names = [w for w in r if HEB.search(w['text']) and fix(w['text']) not in ('מספר', 'סעיף', 'תקציב')
                             and min(cc[1] for cc in cols) + 5 < w['x1'] < max(x['x0'] for x in r if fix(x['text']) == 'סעיף')]
                    unit = text_of(names) or unit
                    jobs = {}
                    continue
                if cols is None:
                    continue
                nums = [w for w in r if NUM.match(w['text']) and w['x1'] < max(c[1] for c in cols) + 25]
                codes = [w for w in r if CODE.match(w['text'])]
                txt = [w for w in r if w not in nums and w not in codes and not re.fullmatch(r'_+', w['text'])]
                label = text_of(txt).strip()
                vals = assign(nums, cols)
                cur, prev = vals.get(year, 0.0), vals.get(year - 1, 0.0)
                if codes:
                    m = CODE.match(codes[0]['text'])
                    code = ''.join(m.groups())
                    it = dict(code=code, name=label.strip(' -'), side=side, p=cur, b_prev=prev,
                              agaf=agaf, unit=unit, unit_code=m.group(1), page=pno)
                    items.append(it)
                    unit_items.append(it)
                    agaf_items.append(it)
                    if len(nums) > len(vals):
                        warns.append(f'{path}:{pno}: מספר מחוץ לעמודות: {label}')
                elif label.startswith('תקן כ') or label.startswith('שעתיים'):
                    jobs[label] = vals.get(year, 0.0)
                elif label.startswith('סה"כ') or label.startswith("סה''כ") or label.startswith('סה”כ'):
                    lvl = label.replace('סה"כ', '').strip()
                    if lvl.startswith('יחיד'):
                        got = sum(i['p'] for i in unit_items)
                        checks.append(('unit', unit, pno, cur, got))
                        if unit_items:
                            unit_items[0]['jobs_unit'] = jobs.get('תקן כ"א') or jobs.get("תקן כ''א")
                        unit_items = []
                    elif lvl.startswith('אגף') or lvl.startswith('מינהל') or lvl.startswith('כללי'):
                        got = sum(i['p'] for i in agaf_items)
                        checks.append((lvl.split()[0], agaf, pno, cur, got))
                        agaf_items = []
    return items, checks, warns


def page_kind(rows):
    for r in rows[:5]:
        flat = text_of(r).replace(' ', '')
        if flat.startswith('פירוטסעיפי'):
            return 't'
        if flat.startswith('דבריהסבר'):
            return 'd'
    return None


AMOUNT_TAIL = re.compile(r'\s*-?\s*[\d,]+(\.\d+)?\s*ש"ח\s*\.?$')


def parse_details(path):
    """דברי ההסבר: {קוד: טקסט}. כל סעיף פותח בקוד, שם וסכום, וההסבר בשורות שאחריו
    עד הסעיף הבא, קו מפריד, או כותרת של קבוצת סעיפים ("... - 1,295,000,000 ש"ח")."""
    out = {}
    cur = None
    kind = None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                rows = lines_of(page.extract_words())
            except Exception:   # גופן פגום בעמוד (קרה ב-2023)
                BAD_PAGES.append((path, page.page_number))
                cur = None
                continue
            k = page_kind(rows)
            if k:
                kind = k
            if kind != 'd':
                cur = None
                continue
            skip_title = False
            for i, r in enumerate(rows):
                t = re.sub(r'\s+', ' ', text_of(r)).strip()
                flat = t.replace(' ', '')
                if i < 4 and (flat.startswith('עירייתירושלים') or flat.startswith('התקציבהרגיל')):
                    continue
                if i < 4 and flat.startswith('דבריהסבר'):
                    skip_title = True
                    continue
                if skip_title:
                    skip_title = False
                    continue   # שם האגף בראש העמוד
                codes = [w for w in r if CODE.match(w['text'])]
                if codes:
                    cur = ''.join(CODE.match(codes[0]['text']).groups())
                    out.setdefault(cur, [])
                    continue
                if re.fullmatch(r'[-_=]{4,}', flat) or AMOUNT_TAIL.search(t) or flat.startswith('סה"כ'):
                    cur = None
                    continue
                if cur and t and not re.fullmatch(r'\d+', t):
                    out[cur].append(t)
    return {c: fix_punct(' '.join(v)) for c, v in out.items() if v}


def fix_punct(t):
    """בטקסט המחולץ נקודה ופסיק נצמדים לתחילת המילה הבאה (".2024", " ."): מחזירים למקום."""
    t = re.sub(r'(^|\s)\.(\d[\d.,%]*)(?=\s|$)', r'\1\2.', t)
    t = re.sub(r'\s+([.,:;])', r'\1', t)
    return t.strip()


def main():
    path = sys.argv[1]
    year = int(sys.argv[2]) if len(sys.argv) > 2 else int(re.search(r'20\d\d', path).group())
    items, checks, warns = parse_file(path, year)
    bad = [c for c in checks if abs(c[3] - c[4]) > 1]
    print(f'{len(items)} סעיפים, {len(checks)} בדיקות סה"כ, {len(bad)} לא מתאימות')
    for c in bad[:20]:
        print('  ✗', c)
    for w in warns[:10]:
        print('  !', w)
    tot = sum(i['p'] for i in items)
    print(f'סה"כ {year}: {tot:,.0f}   {year - 1}: {sum(i["b_prev"] for i in items):,.0f}')
    for it in items[:3]:
        print('  ', it)
    det = parse_details(path)
    print(f'דברי הסבר: {len(det)} סעיפים')
    for c, t in list(det.items())[:6]:
        print('  ', c, t[:160])


if __name__ == '__main__':
    main()
