"""קובץ האקסל של עיריית ירושלים (2017–2022, כל קובץ מכסה שנתיים) → רשימת סעיפים.

בכל קובץ ארבעה גיליונות: הוצאה/הכנסה לפי פרק, והוצאה/הכנסה לפי יחידה ארגונית.
הגיליונות הארגוניים מכילים את אותם סעיפים עם מינהל ויחידה, ולכן קוראים רק אותם.
סדר העמודות משתנה בין השנים, ולכן קוראים לפי שם העמודה בשורת הכותרת.

עמודת השנה הקודמת היא "תקציב מעודכן", כלומר אחרי השינויים שנעשו במהלך אותה שנה.

שימוש לבדיקה: python3 tools/parse_xlsx.py <file.xlsx>
"""
import re
import sys

import openpyxl


def norm_code(v):
    s = re.sub(r'\D', '', str(v))
    return s.zfill(8) if s else None


def parse_xlsx(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    items, year = [], None
    for ws in wb.worksheets:
        title = ws.title
        if 'יחידה ארגונית' not in title:
            continue
        side = 'expense' if 'הוצאה' in title else 'income'
        rows = ws.iter_rows(values_only=True)
        head = None
        for r in rows:
            cells = [re.sub(r'\s+', ' ', str(c)).strip() if c is not None else '' for c in r]
            if head is None:
                if any('סעיף' in c for c in cells) and any(c.startswith('תקציב') for c in cells):
                    head = cells
                    col = {}
                    for i, c in enumerate(head):
                        if c.startswith("מס' סעיף") or c == 'מס סעיף' or c.startswith('מספר סעיף'):
                            col['code'] = i
                        elif c == 'שם הסעיף':
                            col['name'] = i
                        elif c == 'מינהל':
                            col['minhal'] = i
                        elif c == 'יחידה':
                            col['unit'] = i
                        elif c.startswith('סוג סעיף'):
                            col['kind'] = i
                        elif re.match(r'תקציב (מעודכן )?\d{4}', c):
                            col.setdefault('years', []).append((int(re.search(r'\d{4}', c).group()), i))
                    # השנה הגבוהה היא השנה של הקובץ, השנייה היא השנה הקודמת
                    ys = sorted(col.pop('years'))
                    year, col['cur'] = ys[-1]
                    col['prev'] = ys[0][1]
                continue
            code = norm_code(r[col['code']]) if r[col['code']] is not None else None
            if not code:
                continue
            f = lambda k: r[col[k]] if k in col and r[col[k]] is not None else None
            n = lambda v: float(v) if isinstance(v, (int, float)) else float(str(v).replace(',', '').strip() or 0) if re.search(r'\d', str(v or '')) else 0.0
            items.append(dict(
                code=code, name=re.sub(r'\s+', ' ', str(f('name') or '')).strip(), side=side,
                p=n(f('cur')), b_prev=n(f('prev')),
                minhal=str(f('minhal') or '').strip(), agaf=str(f('unit') or '').strip(),
                unit=None, unit_code=code[:4], page=None,
                kind_src=str(f('kind') or '').strip() or None,
            ))
    return year, items


def main():
    year, items = parse_xlsx(sys.argv[1])
    for side in ('expense', 'income'):
        xs = [i for i in items if i['side'] == side]
        print(year, side, len(xs), f"{sum(i['p'] for i in xs):,.0f}", f"prev {sum(i['b_prev'] for i in xs):,.0f}")
    print(items[0])


if __name__ == '__main__':
    main()
