"""תקציב המדינה: מקובץ התקציב של משרד האוצר לקבצים של האפליקציה.

מקור: src/national-budget.csv, הקובץ של משרד האוצר ברמת תקנה, 1997 עד השנה הנוכחית,
כפי שמפתח התקציב (obudget.org) מפרסם אותו:
  https://next.obudget.org/datapackages/budget/national/original/data/national-budget.csv
כל שורה היא תקנה × מיון כלכלי × סוג תקציב (מקורי / מאושר / ביצוע). הסכומים באלפי ₪.

פלט ב-web/data/:
  years.json              השנים, אילו שלבים יש לכל אחת, וסיכומים
  budget_<שנה>.json        העץ של השנה, הוצאות והכנסות. לכל עלה [מקורי, מאושר, ביצוע]
  series/top.json         כל צומת עד רמת המשרד לאורך השנים
  series/<צד>_<סעיף>.json  הצמתים שבתוך משרד אחד (תחום, תכנית, תקנה) לאורך השנים

מה נכלל (כך שהסכום שווה לסך התקציב שהממשלה מפרסמת):
  - הוצאות: התקציב הרגיל, תקציב הפיתוח, החזרי החוב והרזרבה. בלי המפעלים העסקיים (בתי חולים
    ממשלתיים, רשות מקרקעי ישראל ועוד), שמממנים את עצמם מההכנסות שלהם ומוצגים בנפרד בספר.
  - הכנסות: סעיף 00 (מסים, אגרות, הכנסות אחרות ומלוות), ועוד ההכנסות שהמשרדים גובים בעצמם
    (שורות "הכנסה" בתוך סעיפי המשרדים), שבתקציב מקזזות את ההוצאה.
  - בסעיף 00 יש "העברה מחלק ב'" ו"העברה לחלק א'": תנועה פנימית שמתקזזת לאפס, ולא נספרת.

python3 tools/build_budget.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src' / 'national-budget.csv'
OUT = ROOT / 'web' / 'data'
PHASES = {'מקורי': 0, 'מאושר': 1, 'ביצוע': 2}
TYPES = {'1', '2', '6', '7'}        # רגיל, פיתוח, החזר חובות, רזרבות. 5 = מפעלים עסקיים, 4 = הכנסות המדינה
INTERNAL = {'0049', '0095'}           # העברה מחלק ב' / לחלק א'
SALARY = '001'


def num(s):
    try:
        return float(s) if s else 0.0
    except ValueError:
        return 0.0


def clean(s):
    return ' '.join((s or '').split())


def main():
    # leaves[(year, side, item_code)] = {'path': [...], 'v': [o,r,e], 's': [...], 'j': [...], ...}
    leaves = {}
    names = {}                      # (year, key) -> שם (השם של אותה שנה)
    phases_of = defaultdict(set)
    for r in csv.DictReader(open(SRC, encoding='utf-8')):
        y = int(r['שנה'])
        ph = PHASES.get(r['סוג תקציב'])
        if ph is None:
            continue
        if not r['קוד תקנה'].strip():
            continue
        # עד 2019 האפס המוביל נשמט (סעיף 4 במקום 04), ולכן משלימים לאורך הקבוע
        sec, typ, dirn = r['קוד סעיף'].strip().zfill(2), r['קוד סוג סעיף'].strip(), r['הוצאה/הכנסה']
        net = num(r['הוצאה נטו'])
        area, prog, item = (r['קוד תחום'].strip().zfill(4), r['קוד תכנית'].strip().zfill(6),
                             r['קוד תקנה'].strip().zfill(8))
        if sec == '00':
            if dirn != 'הכנסה' or area in INTERNAL:
                continue
            side, val = 'income', net
            path = [('S', r['קוד רמה 2'].strip(), clean(r['שם רמה 2']))]
        elif typ in TYPES:
            if dirn == 'הוצאה':
                side, val = 'expense', net
                path = [('D', r['קוד רמה 1'].strip(), clean(r['שם רמה 1'])),
                        ('S', r['קוד רמה 2'].strip(), clean(r['שם רמה 2']))]
            else:
                # הכנסה של משרד: בספר היא שלילית (מקזזת הוצאה)
                side, val = 'income', -net
                path = [('S', 'M', 'הכנסות שהמשרדים גובים')]
        else:
            continue
        # לסעיפי תקציב הפיתוח יש לרוב אותו שם כמו למשרד ("חינוך" ליד "משרד החינוך")
        pname = clean(r['שם סעיף']) + (': בינוי ופיתוח' if typ == '2' else '')
        path += [('P', sec, pname), ('A', area, clean(r['שם תחום'])),
                 ('G', prog, clean(r['שם תכנית']))]
        phases_of[y].add(ph)
        k = (y, side, item)
        leaf = leaves.get(k)
        if leaf is None:
            leaf = leaves[k] = {'path': path, 'name': clean(r['שם תקנה']), 'v': [0.0] * 3, 's': [0.0] * 3,
                                'j': [0.0] * 3, 'c': [0.0] * 3, 'm': [0.0] * 3, 'econ': defaultdict(float)}
        leaf['v'][ph] += val
        if r['קוד מיון רמה 1'].strip() == SALARY:
            leaf['s'][ph] += val
        leaf['j'][ph] += num(r['שיא כח אדם'])
        leaf['c'][ph] += num(r['הרשאה להתחייב'])
        leaf['m'][ph] += num(r['הוצאה מותנית בהכנסה'])
        leaf['econ'][clean(r['שם מיון רמה 1'])] += abs(val)
        for lvl, code, name in path:
            names[(y, side, lvl + code)] = name

    years = sorted(phases_of)
    by = defaultdict(list)
    for (y, s, item), leaf in leaves.items():
        by[(y, s)].append((item, leaf))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'series').mkdir(exist_ok=True)
    rnd = lambda xs: [round(x) for x in xs]

    # עצים לפי שנה
    meta = []
    series = defaultdict(lambda: defaultdict(lambda: [[None] * len(years) for _ in range(3)]))
    for yi, y in enumerate(years):
        book = {'year': y, 'phases': sorted(phases_of[y])}
        for side in ('expense', 'income'):
            root = {'c': {}}
            for item, leaf in by[(y, side)]:
                if not any(abs(v) >= 0.5 for v in leaf['v']):
                    continue
                n = root
                keys = []
                for lvl, code, _ in leaf['path']:
                    key = lvl + code
                    keys.append(key)
                    n = n['c'].setdefault(key, {'c': {}})
                ikey = 'I' + item
                kind = max(leaf['econ'].items(), key=lambda kv: kv[1])[0] if leaf['econ'] else ''
                out = {'v': rnd(leaf['v'])}
                if any(leaf['s']): out['s'] = rnd(leaf['s'])
                if any(leaf['j']): out['j'] = [round(x, 1) for x in leaf['j']]
                if any(leaf['c']): out['h'] = rnd(leaf['c'])
                if any(leaf['m']): out['m'] = rnd(leaf['m'])
                if kind: out['t'] = kind
                out['n'] = leaf['name']
                n['c'][ikey] = out
                # סדרות: העלה ומעליו
                sec = [k for k in keys if k.startswith('P')][0][1:]
                for key in ['ROOT'] + keys + [ikey]:
                    # קודי תחום/תכנית/תקנה מתחילים בקוד הסעיף, אז המפתח ייחודי גם בלי הסעיף
                    fname = 'top' if key == 'ROOT' or key[0] in 'DSP' else f'{side[0]}_{sec}'
                    sk = f'{side[0]}|{key}'
                    arr = series[fname][sk]
                    for ph in range(3):
                        if ph in phases_of[y]:
                            arr[ph][yi] = (arr[ph][yi] or 0) + leaf['v'][ph]

            def finish(node, key):
                if 'c' not in node:
                    return node
                kids = [finish(v, k) | {'k': k} for k, v in node['c'].items()]
                return {'n': names.get((y, side, key), key), 'k': key, 'c': kids}
            book[side] = [finish(v, k) for k, v in root['c'].items()]
        (OUT / f'budget_{y}.json').write_text(json.dumps(book, ensure_ascii=False, separators=(',', ':')))
        tot = lambda side, ph: sum(l['v'][ph] for _, l in by[(y, side)])
        meta.append({'year': y, 'phases': sorted(phases_of[y]),
                     'expense': [round(tot('expense', p)) if p in phases_of[y] else None for p in range(3)],
                     'income': [round(tot('income', p)) if p in phases_of[y] else None for p in range(3)]})

    # שמות לאורך השנים: השם העדכני ביותר של כל מפתח, לסדרות
    latest = {}
    for (y, side, key), name in sorted(names.items()):
        latest[(side, key)] = name
    for fname, d in series.items():
        out = {'years': years, 'nodes': {}}
        for sk, arr in d.items():
            # [השנה הראשונה, מקורי, מאושר, ביצוע], בלי השנים הריקות בקצוות
            have = [i for i in range(len(years)) if any(a[i] is not None for a in arr)]
            lo, hi = have[0], have[-1] + 1
            out['nodes'][sk] = [years[lo]] + [[None if v is None else round(v) for v in a[lo:hi]] for a in arr]
        (OUT / 'series' / f'{fname}.json').write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')))
    (OUT / 'years.json').write_text(json.dumps(meta, ensure_ascii=False, indent=0))
    for m in meta[-6:]:
        print(m['year'], 'phases', m['phases'], 'expense', [x and round(x / 1e6, 1) for x in m['expense']],
              'income', [x and round(x / 1e6, 1) for x in m['income']])


if __name__ == '__main__':
    main()
