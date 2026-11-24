"""כל ספרי התקציב של ירושלים → web/data/.

  2017–2022: קובצי האקסל (parse_xlsx). הקובץ של 2017 נותן גם את 2016 (עמודת "תקציב מעודכן").
  2023–2026: ספר ה-PDF, קובץ "תקציב" לכל מינהל (parse_pdf), ודברי ההסבר מקובץ "דברי הסבר" הכולל.

פלט:
  web/data/years.json          השנים, המקור של כל אחת, הסכומים והערות
  web/data/budget_<שנה>.json   כל הסעיפים של השנה
  web/data/series.json         כל צומת (בשתי החלוקות) לאורך השנים

העמסות: עלויות מרכזיות (מחשוב, מוסך, ביטוח, קריית העירייה, אבטחה, דלק, טלפונים) נרשמות
בספר פעמיים: כ"העמסה" בכל יחידה שמשתמשת בהן, וכ"הפחתת העמסה" שלילית במקום אחד. הן מתקזזות
לאפס, ולכן מוציאים את שני הצדדים: העלות נשארת פעם אחת, ביחידה שמספקת את השירות.

שימוש: python3 tools/build_years.py [--fast]   (--fast: משתמש במטמון של הקריאה ב-src/cache/)
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import parse_pdf  # noqa: E402
from parse_pdf import parse_file, parse_details  # noqa: E402
from parse_xlsx import parse_xlsx  # noqa: E402
from download import local_name  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT, CACHE = ROOT / 'src', ROOT / 'web' / 'data', ROOT / 'src' / 'cache'
XLSX_YEARS = range(2017, 2023)
FETCHED = '2026-11-24'   # מתי ירדו הקבצים מאתר העירייה (לעדכן כשמורידים מחדש)
NOTES = {
    2025: 'בספר המפורט של 2025 ההוצאות מסתכמות ב-9,721.2 מיליון ₪, 1.2 מיליון יותר מהסכום הרשמי (9,720 מיליון). '
          'כל ההפרש באגף פדגוגיה במינהל חינוך: 67.7 מיליון בספר, 66.5 ב"עיקרי התקציב". '
          'עמוד אחד בספר (רכש ואספקה, 5.4 מיליון) צויר כקווים ולא כטקסט, ותומלל ידנית.',
}
CHECKS = defaultdict(lambda: [0, 0])   # שנה → [שורות סה"כ שנבדקו, לא מתאימות]
FALLBACK = []   # (שנה, צד) שנקראו מהקבצים לפי מינהל
PDF_YEARS = range(2023, 2027)

# שמות הפרקים (שתי הספרות הראשונות של הסמל). ירושלים משתמשת בגרסה משלה של המבנה האחיד:
# 79 הוא תכנון ולא חקלאות, 80 בתי ספר על-יסודיים, 86 נוער ו-87 ספורט. השמות נקבעו לפי
# היחידות שבכל פרק בפועל (הבונה מדפיס אותן לכל שנה, כדי לתפוס שינוי).
CHAPTERS = {
    'expense': {
        '61': 'מינהל כללי', '62': 'מינהל כספי', '63': 'מימון ופירעון הלוואות',
        '70': 'הנהלת מינהל התפעול', '71': 'ניקיון ואיסוף אשפה', '72': 'חירום וביטחון',
        '73': 'תחבורה ותנועה', '74': 'גנים, תאורה ותחזוקת העיר',
        '75': 'אירועים וחגיגות', '76': 'הנחות בארנונה והשתתפויות', '77': 'פיתוח כלכלי ורישוי עסקים',
        '78': 'פיקוח ואכיפה', '79': 'תכנון, בנייה ואיכות הסביבה',
        '80': 'בתי ספר על-יסודיים', '81': 'חינוך', '82': 'תרבות', '83': 'בריאות הציבור',
        '84': 'רווחה', '85': 'דת', '86': 'נוער, צעירים וקליטה', '87': 'ספורט',
        '91': 'מים', '93': 'נכסים ומבני ציבור', '94': 'רכש ומחסנים', '95': 'מוסך וצי הרכב',
        '96': 'ביטוח', '97': 'ביוב', '99': 'פנסיה ופיצויים',
    },
    'income': {
        '11': 'ארנונה', '12': 'אגרות והכנסות כלליות', '13': 'הכנסות מימון',
        '19': 'מענקים כלליים מהמדינה', '20': 'מינהל התפעול', '21': 'ניקיון ואיסוף אשפה',
        '22': 'חירום וביטחון', '23': 'תחבורה, תכנון ורישוי', '24': 'גנים, תאורה ותחזוקת העיר',
        '25': 'אירועים וחגיגות', '26': 'השתתפויות ואחרות', '27': 'פיתוח כלכלי',
        '28': 'פיקוח, קנסות ובית המשפט לעניינים מקומיים', '29': 'תכנון ואיכות הסביבה',
        '30': 'בתי ספר על-יסודיים', '31': 'חינוך', '32': 'תרבות', '33': 'בריאות הציבור',
        '34': 'רווחה', '35': 'דת', '36': 'נוער, צעירים וקליטה', '37': 'ספורט',
        '41': 'מים', '43': 'נכסים', '44': 'חניה, רכש ושונות', '45': 'מוסך וצי הרכב',
        '51': 'תקבולים בלתי רגילים', '59': 'מענק ארנונה',
    },
}


def chapter_name(side, code):
    return CHAPTERS[side].get(code[:2], f'פרק {code[:2]}')


def is_loading(name):
    # "מחשוב עירוני - העמסה", "קריית העירייה - הפחתת העמסה", ובשנים הישנות גם "טלפונים - הפחתה"
    n = name.replace(' ', '')
    return 'העמסה' in n or n.endswith('-הפחתה') or n.endswith('הפחתה')


def kind_of(side, code):
    # סמלי סוג 110–199 הם שכר במבנה האחיד
    return 'שכר' if side == 'expense' and code[4] == '1' else None


def cached(key, fn, fast):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'{key}.json'
    if fast and f.exists():
        return json.loads(f.read_text())
    v = fn()
    f.write_text(json.dumps(v, ensure_ascii=False))
    return v


def media_url(manifest_url):
    return manifest_url.split('#')[0]


def year_files(man, y):
    """מהמניפסט: לכל צד, [(מינהל, קובץ תקציב, url)] ואת הקובץ הכולל של דברי ההסבר."""
    out = {'expense': [], 'income': [], 'details': {}, 'totals': {}}
    for t in man[str(y)]['tables']:
        title = t['title']
        if 'בלתי רגיל' in title or 'רגיל' not in title:
            continue
        side = 'expense' if 'הוצאות' in title else 'income'
        head = t['head']
        for r in t['rows']:
            name = r[0]['text']
            links = {}
            for ci, c in enumerate(r):
                col = head[ci] if ci < len(head) else ''
                for l in c['links']:
                    kind = 'm' if ('ודברי' in col or '+' in col) else 'd' if 'דברי' in col else 't' if col.startswith('תקציב') else None
                    if kind is None:   # כותרת שבורה (2024 הכנסות): לפי שם הקובץ
                        fn = l.split('/')[-1]
                        kind = 'm' if '_m' in fn or '-m' in fn else 'd' if '_d' in fn or '-d' in fn else 't'
                    links.setdefault(kind, media_url(l))
            if name.startswith('סך הכל'):
                out['totals'][side] = links
                if 'd' in links:
                    out['details'][side] = links['d']
                continue
            src = links.get('t') or links.get('m')
            if src:
                out[side].append((name, src))
            if links.get('d') or links.get('m'):
                out.setdefault('details_files', []).append(links.get('d') or links.get('m'))
    return out


def pdf_year(man, y, fast):
    """הסעיפים נקראים מהקובץ הכולל של כל צד (הספר המלא), והקבצים לפי מינהל משמשים רק
    לשיוך כל סעיף למינהל. כך טעות קישור באתר (ב-2024 "מינהל תכנון ותשתיות" בהכנסות
    מקשר לקובץ של מינהל התפעול) או קובץ מינהל חסר לא משנים את הסכומים."""
    files = year_files(man, y)
    items, problems = [], []

    def read(path):
        def run():
            its, checks, _ = parse_file(str(path), y)
            bad = [c for c in checks if abs(c[3] - c[4]) > 1]
            return {'items': its, 'checks': len(checks), 'bad': bad}
        return cached(f'{y}_{path.stem}', run, fast)

    for side in ('expense', 'income'):
        tot = files['totals'].get(side, {})
        turl = tot.get('t') or tot.get('m')
        code2minhal, urls = {}, Counter(u for _, u in files[side])
        orphan_minhal = [m for m, u in files[side] if urls[u] > 1]
        for minhal, url in files[side]:
            if urls[url] > 1 and orphan_minhal.index(minhal) > 0:
                continue   # קישור כפול באתר: רק הראשון באמת שייך לקובץ
            for it in read(SRC / str(y) / local_name(url))['items']:
                code2minhal.setdefault(it['code'], minhal)
        r = read(SRC / str(y) / local_name(turl)) if turl else {'items': [], 'bad': ['אין קובץ']}
        if len(r['items']) > 100 and len(r['bad']) <= 3 and all(i['side'] == side for i in r['items']):
            missing = orphan_minhal[1:]
            for it in r['items']:
                m = code2minhal.get(it['code'])
                if m is None:
                    m = missing[0] if len(missing) == 1 else 'לא משויך למינהל'
                it.update(minhal=m, src=turl)
                items.append(it)
            CHECKS[y][0] += r['checks']
            CHECKS[y][1] += len(r['bad'])
            if r['bad']:
                problems.append(f'{y} {side}: {len(r["bad"])} שורות סה"כ לא מתאימות בספר: {r["bad"][:3]}')
        else:
            # הקובץ הכולל לא קריא (ב-2023 ו-2025 הגופן מקודד): הקבצים לפי מינהל
            problems.append(f'{y} {side}: הקובץ הכולל לא נקרא, קוראים לפי מינהל')
            FALLBACK.append((y, side))
            for minhal, url in files[side]:
                if urls[url] > 1 and orphan_minhal.index(minhal) > 0:
                    problems.append(f'{y} {side}: {minhal} מקשר לקובץ של מינהל אחר באתר, ואין לו נתונים')
                    continue
                rr = read(SRC / str(y) / local_name(url))
                CHECKS[y][0] += rr['checks']
                CHECKS[y][1] += len(rr['bad'])
                if rr['bad']:
                    problems.append(f'{y} {minhal}: {len(rr["bad"])} שורות סה"כ לא מתאימות: {rr["bad"][:2]}')
                for it in rr['items']:
                    it.update(minhal=minhal, src=url)
                    items.append(it)
        unassigned = sum(1 for it in items if it['side'] == side and it['minhal'] == 'לא משויך למינהל')
        if unassigned:
            problems.append(f'{y} {side}: {unassigned} סעיפים בלי מינהל')
    # דברי ההסבר: מהקבצים לפי מינהל (הקבצים הכוללים של 2023 ו-2025 בגופן מקודד),
    # ומהקובץ הכולל רק למה שחסר
    details = {}
    for url in dict.fromkeys(files.get('details_files', [])):
        path = SRC / str(y) / local_name(url)
        if path.exists():
            details.update(cached(f'{y}_{path.stem}_details', lambda: parse_details(str(path)), fast))
    for side, url in files['details'].items():
        path = SRC / str(y) / local_name(url)
        for k, v in cached(f'{y}_{path.stem}_details', lambda: parse_details(str(path)), fast).items():
            details.setdefault(k, v)
    for it in items:
        it['details'] = details.get(it['code'])
    srcs = {s: files['totals'].get(s, {}).get('t') or files['totals'].get(s, {}).get('m') for s in ('expense', 'income')}
    return items, problems, srcs


def main():
    fast = '--fast' in sys.argv
    man = json.loads((SRC / 'manifest.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    years = {}   # שנה → {'items': [...], 'meta': {...}}

    # ---------- אקסל ----------
    for y in XLSX_YEARS:
        x = next(p for p in (SRC / str(y)).iterdir() if '.xlsx' in p.name)
        url = next(l['href'] for l in man[str(y)]['loose'] if '.xlsx' in l['href'])
        fy, its = parse_xlsx(x)
        assert fy == y, (fy, y)
        for it in its:
            it['src'] = media_url(url)
            it['details'] = None
        years[y] = {'items': its, 'meta': {'year': y, 'source': media_url(url), 'format': 'xlsx'}}
    # 2016: רק התקציב המעודכן, מתוך הקובץ של 2017
    its16 = [dict(it, p=it['b_prev'], b_prev=0.0) for it in years[2017]['items'] if it['b_prev']]
    years[2016] = {'items': its16, 'meta': {'year': 2016, 'source': years[2017]['meta']['source'], 'format': 'xlsx',
                                            'derived_from': 2017,
                                            'note': 'ל-2016 אין ספר באתר העירייה. המספרים הם "התקציב המעודכן" של 2016, מתוך הקובץ של 2017, ולכן אין השוואה לשנה הקודמת.'}}

    # ---------- PDF ----------
    problems = []
    for y in PDF_YEARS:
        its, pr, srcs = pdf_year(man, y, fast)
        problems += pr
        years[y] = {'items': its, 'meta': {'year': y, 'source': srcs['expense'], 'source_income': srcs['income'], 'format': 'pdf'}}

    # ---------- השלמות ידניות (עמודים שהטקסט בהם מצויר כקווים) ----------
    for f in sorted((SRC / 'manual').glob('*.json')):
        man_ = json.loads(f.read_text())
        y = man_['year']
        have = {i['code'] for i in years[y]['items']}
        agaf_sum = [0, 0]
        for u in man_['units']:
            s = [sum(r[2] for r in u['items']), sum(r[3] for r in u['items'])]
            assert s == u['total'], (f.name, u['unit'], s, u['total'])
            agaf_sum = [agaf_sum[0] + s[0], agaf_sum[1] + s[1]]
            for k, (c, n, p, b) in enumerate(u['items']):
                code = c.replace('-', '')
                if code in have:
                    continue
                years[y]['items'].append(dict(
                    code=code, name=n, side=man_['side'], p=float(p), b_prev=float(b), minhal=man_['minhal'],
                    agaf=man_['agaf'], unit=u['unit'], unit_code=code[:4], page=man_['page'], src=man_['src'],
                    details=None, manual=True, jobs_unit=u['jobs'] if k == 0 else None))
        assert agaf_sum == man_['agaf_total'], (f.name, agaf_sum)
        years[y]['meta'].setdefault('manual', []).append(man_['_about'])

    # ---------- שמות יחידות לפי קוד ----------
    # ב-PDF לכל יחידה (4 ספרות) יש שם. באקסל יש רק את היחידה הארגונית (האגף), ולכן
    # לוקחים את השם מהשנה הקרובה שבה הוא מופיע, ואם אין, את שם האגף.
    unit_names = defaultdict(dict)
    for y in PDF_YEARS:
        for it in years[y]['items']:
            if it.get('unit'):
                unit_names[it['unit_code']][y] = it['unit']

    def unit_name(code, y, fallback):
        ns = unit_names.get(code)
        if not ns:
            return fallback
        return ns.get(y) or ns[min(ns, key=lambda k: abs(k - y))]

    # ---------- ניקוי וכתיבה ----------
    allyears = sorted(years)
    series = defaultdict(lambda: {'v': [None] * len(allyears), 'f': [None] * len(allyears)})
    meta_out = []
    for yi, y in enumerate(allyears):
        its = years[y]['items']
        meta = years[y]['meta']
        loads = [i for i in its if is_loading(i['name'])]
        load_net = {s: sum(i['p'] for i in loads if i['side'] == s) for s in ('expense', 'income')}
        load_gross = sum(i['p'] for i in loads if i['p'] > 0 and i['side'] == 'expense')
        keep = [i for i in its if not is_loading(i['name'])]
        tot = {s: sum(i['p'] for i in its if i['side'] == s) for s in ('expense', 'income')}
        shown = {s: sum(i['p'] for i in keep if i['side'] == s) for s in ('expense', 'income')}
        if any(abs(v) > 1 for v in load_net.values()):
            problems.append(f'{y}: ההעמסות לא מתקזזות, פער של {-load_net["expense"]:,.0f} ₪ בהוצאות המוצגות')
        neg = [i for i in keep if i['p'] < 0]
        if neg:
            problems.append(f'{y}: {len(neg)} סעיפים שליליים: ' + ', '.join(f'{i["name"]} {i["p"]:,.0f}' for i in neg[:5]))

        rows = []
        for it in keep:
            s = it['side']
            code = it['code']
            row = {
                'c': code, 'n': it['name'], 's': s[0], 'p': round(it['p']), 'b': round(it['b_prev']),
                'm': it.get('minhal') or '', 'a': it.get('agaf') or '',
                'u': unit_name(it['unit_code'], y, it.get('agaf') or ''),
            }
            if it.get('page'):
                row['pg'] = it['page']
                row['src'] = it['src']
            if it.get('details'):
                row['d'] = it['details']
            if it.get('manual'):
                row['x'] = 1   # תומלל ידנית
            if it.get('jobs_unit'):
                row['j'] = it['jobs_unit']
            if kind_of(s, code):
                row['k'] = 'שכר'
            rows.append(row)
            # סדרות: שתי החלוקות. מפתחות קבועים כדי שקישור ישרוד מעבר בין שנים.
            sk = s[0]
            for key in (sk, f'{sk}|F|{code[:2]}', f'{sk}|U|{code[:4]}', f'{sk}|I|{code}',
                        f'{sk}|M|{row["m"]}', f'{sk}|A|{row["m"]}|{row["a"]}'):
                e = series[key]
                e['v'][yi] = (e['v'][yi] or 0) + it['p']
                # f: התקציב המעודכן של השנה הקודמת, כפי שנרשם בספר של השנה הזו
                if yi > 0 and it['b_prev'] and not meta.get('derived_from'):
                    e['f'][yi - 1] = (e['f'][yi - 1] or 0) + it['b_prev']

        chapters = {s: {c: chapter_name(s, c) for c in sorted({r['c'][:2] for r in rows if r['s'] == s[0]})}
                    for s in ('expense', 'income')}
        data = {'year': y, 'chapters': chapters, 'items': rows}
        (OUT / f'budget_{y}.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
        meta.update(total_expense=round(tot['expense']), total_income=round(tot['income']),
                    shown_expense=round(shown['expense']), shown_income=round(shown['income']),
                    from_minhal_files=[s for (yy, s) in FALLBACK if yy == y],
                    loadings=round(load_gross), items=len(rows),
                    details=sum(1 for r in rows if r.get('d')))
        meta.setdefault('note', NOTES.get(y, ''))
        meta['load_gap'] = round(-load_net['expense'])
        meta['checks'] = CHECKS[y] if y in CHECKS else None
        meta['fetched'] = FETCHED
        meta.setdefault('derived_from', None)
        meta_out.append(meta)
        print(f'{y}  {meta["format"]:4}  הוצאות {tot["expense"]/1e6:>9,.1f}M (בעוגה {shown["expense"]/1e6:>9,.1f})  הכנסות {tot["income"]/1e6:>9,.1f}M  '
              f'העמסות {load_gross/1e6:6,.1f}M  סעיפים {len(rows):5}  עם הסבר {meta["details"]:5}')

    # פרקים לא מוכרים, ומה יש בכל פרק (לבדיקה ידנית של השמות)
    unknown = Counter()
    for y in allyears:
        for r in json.loads((OUT / f'budget_{y}.json').read_text())['items']:
            side = 'expense' if r['s'] == 'e' else 'income'
            if r['c'][:2] not in CHAPTERS[side]:
                unknown[(side, r['c'][:2], r['u'])] += 1
    for k, v in unknown.most_common(30):
        problems.append(f'פרק בלי שם: {k} ×{v}')

    (OUT / 'years.json').write_text(json.dumps(meta_out, ensure_ascii=False, indent=1))
    ser = {'years': allyears, 'nodes': {k: {kk: [None if x is None else round(x) for x in vv] for kk, vv in v.items()
                                            if any(x is not None for x in vv)} for k, v in series.items()}}
    (OUT / 'series.json').write_text(json.dumps(ser, ensure_ascii=False, separators=(',', ':')))
    print(f'series: {len(ser["nodes"])} צמתים')
    for p in problems:
        print('  !', p)
    for f, pg in parse_pdf.BAD_PAGES:
        print('  ! עמוד לא נקרא:', Path(f).name, pg)


if __name__ == '__main__':
    main()
