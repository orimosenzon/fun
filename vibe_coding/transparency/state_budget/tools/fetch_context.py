"""נתוני הקשר לתקציב המדינה: מדד המחירים, אוכלוסייה, והשוואה בין-לאומית.

פלט:
  web/data/context.json   מדד ואוכלוסייה לכל שנה (לתיקון לאינפלציה ולשקל לתושב)
  web/data/oecd.json      הוצאות הממשלה הכללית לפי ייעוד (COFOG) כאחוז מהתמ"ג, לכל מדינה

מקורות:
  - מדד המחירים לצרכן, ממוצע שנתי: הבנק העולמי (FP.CPI.TOTL) עד השנה האחרונה שלו, והשנה
    הנוכחית מהלמ"ס (api.cbs.gov.il, מדד 120010) לפי ממוצע החודשים שפורסמו.
  - אוכלוסייה: הבנק העולמי (SP.POP.TOTL). שנה שעוד אין לה נתון מוערכת לפי קצב הגידול האחרון.
  - COFOG: OECD, טבלה 11 (DSD_NASEC10@DF_TABLE11), הממשלה הכללית (S13), וחלוקה בתמ"ג
    במטבע מקומי מטבלה 1 (DSD_NAMAIN10@DF_TABLE1_EXPENDITURE).

python3 tools/fetch_context.py
"""
import csv
import io
import json
import statistics
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'web' / 'data'
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36',
      'Accept': 'application/json'}
FIRST, LAST = 1997, 2026


def get(url, timeout=300):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8')


def worldbank(indicator):
    d = json.loads(get(f'https://api.worldbank.org/v2/country/ISR/indicator/{indicator}'
                       f'?format=json&date={FIRST - 1}:{LAST}&per_page=200'))
    return {int(r['date']): r['value'] for r in d[1] if r['value'] is not None}


def cbs_cpi_months(year):
    """חודשי השנה במדד של הלמ"ס, לפי הבסיס הנוכחי"""
    d = json.loads(get(f'https://api.cbs.gov.il/index/data/price?id=120010&format=json&download=false'
                       f'&startPeriod=01-{year}&endPeriod=12-{year}&PageSize=100'))
    return [m['currBase']['value'] for m in d['month'][0]['date'] if m['year'] == year]


def context():
    cpi = worldbank('FP.CPI.TOTL')
    pop = worldbank('SP.POP.TOTL')
    last_cpi = max(cpi)
    notes = []
    for y in range(last_cpi + 1, LAST + 1):
        # המדד של הלמ"ס בבסיס אחד לשתי השנים, אז היחס בין הממוצעים הוא השינוי
        prev, cur = cbs_cpi_months(y - 1), cbs_cpi_months(y)
        if not cur:
            break
        cpi[y] = cpi[y - 1] * statistics.mean(cur) / statistics.mean(prev)
        notes.append(f'מדד {y}: ממוצע {len(cur)} החודשים שפורסמו')
    last_pop = max(pop)
    growth = pop[last_pop] / pop[last_pop - 1]
    for y in range(last_pop + 1, LAST + 1):
        pop[y] = pop[y - 1] * growth
        notes.append(f'אוכלוסייה {y}: הערכה לפי קצב הגידול של {last_pop}')
    years = list(range(FIRST, LAST + 1))
    return {
        'years': years,
        'cpi': [round(cpi[y], 3) for y in years],
        'pop': [round(pop[y]) for y in years],
        'notes': notes,
    }


COFOG = {
    'GF01': 'שירותים ציבוריים כלליים (כולל ריבית)', 'GF02': 'ביטחון', 'GF03': 'סדר ציבורי ובטיחות',
    'GF04': 'ענייני כלכלה (תחבורה, אנרגיה, חקלאות)', 'GF05': 'הגנת הסביבה', 'GF06': 'דיור ושירותים לקהילה',
    'GF07': 'בריאות', 'GF08': 'תרבות, ספורט ודת', 'GF09': 'חינוך', 'GF10': 'רווחה וביטחון סוציאלי',
}
COUNTRIES = {
    'ISR': 'ישראל', 'AUS': 'אוסטרליה', 'AUT': 'אוסטריה', 'BEL': 'בלגיה', 'CHE': 'שווייץ', 'CHL': "צ'ילה",
    'COL': 'קולומביה', 'CRI': 'קוסטה ריקה', 'CZE': "צ'כיה", 'DEU': 'גרמניה', 'DNK': 'דנמרק', 'ESP': 'ספרד',
    'EST': 'אסטוניה', 'FIN': 'פינלנד', 'FRA': 'צרפת', 'GBR': 'בריטניה', 'GRC': 'יוון', 'HUN': 'הונגריה',
    'IRL': 'אירלנד', 'ISL': 'איסלנד', 'ITA': 'איטליה', 'JPN': 'יפן', 'KOR': 'קוריאה', 'LTU': 'ליטא',
    'LUX': 'לוקסמבורג', 'LVA': 'לטביה', 'NLD': 'הולנד', 'NOR': 'נורווגיה', 'POL': 'פולין', 'PRT': 'פורטוגל',
    'SVK': 'סלובקיה', 'SVN': 'סלובניה', 'SWE': 'שוודיה', 'USA': 'ארה"ב',
}  # חברות OECD בלבד (בלי איחודים ובלי מדינות שאינן חברות)


def oecd_csv(url):
    return list(csv.DictReader(io.StringIO(get(url, timeout=600))))


def oecd():
    funcs = '+'.join(COFOG)
    rows = oecd_csv('https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NASEC10@DF_TABLE11,/'
                    f'A..S13._Z.D.OTE._Z.{funcs}.XDC.S.V.N.T1100?startPeriod=2005&format=csvfile')
    gdp_rows = oecd_csv('https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1_EXPENDITURE,/'
                        'A..S1.S1.B1GQ._Z._Z._Z.XDC.V.N.T0102?startPeriod=2005&format=csvfile')
    gdp = {}
    for r in gdp_rows:
        if r['OBS_VALUE']:
            gdp[(r['REF_AREA'], int(r['TIME_PERIOD']))] = float(r['OBS_VALUE']) * 10 ** int(r.get('UNIT_MULT') or 0)
    exp = {}
    for r in rows:
        if r['REF_AREA'] not in COUNTRIES or not r['OBS_VALUE']:
            continue
        v = float(r['OBS_VALUE']) * 10 ** int(r.get('UNIT_MULT') or 0)
        exp[(r['REF_AREA'], int(r['TIME_PERIOD']), r['EXPENDITURE'])] = v
    years = sorted({k[1] for k in exp})
    data = {}
    for c in COUNTRIES:
        per = {}
        for y in years:
            g = gdp.get((c, y))
            vals = {f: exp.get((c, y, f)) for f in COFOG}
            if not g or any(v is None for v in vals.values()):
                continue
            per[y] = {f: round(v / g * 100, 2) for f, v in vals.items()}
        if per:
            data[c] = per
    return {'functions': COFOG, 'countries': {c: n for c, n in COUNTRIES.items() if c in data}, 'data': data}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = context()
    (OUT / 'context.json').write_text(json.dumps(ctx, ensure_ascii=False))
    print('context:', ctx['notes'])
    o = oecd()
    (OUT / 'oecd.json').write_text(json.dumps(o, ensure_ascii=False, separators=(',', ':')))
    isr = o['data'].get('ISR', {})
    print('oecd countries:', len(o['data']), 'israel years:', sorted(isr)[-3:] if isr else None)
