#!/usr/bin/env python3
"""Build the tree-survey layer: web/data/trees.json.

Where the data comes from
-------------------------
Between October 2018 and July 2019 almost two thousand trees in the moshava's
public space were surveyed one by one: species, height, trunk, crown, age,
health, what each one needs (pruning, support, tying, water), what it stands
next to, and an Israeli-grid coordinate. Pardespedia publishes the whole table
on one page, "סקר העצים בפרדס חנה-כרכור", and that page is the input here. It
is read off the wiki's API every time, so a correction made on the wiki reaches
the map with the next build.

What is left out
----------------
The survey's "שם מדווח" column - who walked up to the tree - names private
people, and this repo is public. The layer keeps the date and drops the name.

What the layer is not
---------------------
A picture of the trees as they were in 2018-2019. Some have since been cut
down, and a tree that was sick then may be dead or recovered now. The app says
so on the layer and on every tree, and nothing here should be read as today's
state. That gap is exactly what tree guardians would close.

Positions
---------
The survey's x/y are ITM (Israeli Transverse Mercator, EPSG:2039), rounded to
the metre on the wiki. 1,924 trees sit on only ~1,700 distinct points, because
a row of trees was often recorded at one spot - up to eight on the same metre.
Those are fanned out on a small ring so that every one stays tappable, and are
flagged `spread`, which the app shows as an approximate position.

Re-run after the wiki page changes:   python3 build_trees.py
"""
import collections
import datetime
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request

from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'web', 'data', 'trees.json')
API = 'https://pardespedia.info/api.php'
PAGE = 'סקר העצים בפרדס חנה-כרכור'
PAGE_URL = 'https://pardespedia.info/index.php?title=' + urllib.parse.quote(PAGE.replace(' ', '_'))

# The layer's default colouring: how the tree was doing. Worst last, so the
# legend reads from fine to bad. Keys are the survey's own words.
HEALTH = [
    ('בריא', '#2e7d32'),
    ('חולה', '#ef6c00'),
    ('מנוון', '#ad1457'),
    ('חלול', '#8d6e63'),
    ('לא תועד', '#9e9e9e'),
]

# yes/no columns, as the survey names them, and the short word the app uses.
NEEDS = [('נדרש גיזום', 'גיזום'), ('נדרשת תמיכה', 'תמיכה'), ('נדרשת קשירה', 'קשירה'),
         ('דרושה השקיה', 'השקיה'), ('דרוש חיפוי', 'חיפוי')]
NEAR = [('קרבה למדרכה', 'מדרכה'), ('קרבה לכביש', 'כביש'), ('קרבה לחניה', 'חניה'),
        ('קרבה לחומה', 'חומה'), ('קרבה לביוב', 'ביוב')]

SPREAD_M = 3.0          # radius of the ring a stack of trees is fanned out on

# "34 עצים", "83 אורנים", "16 כאלו": a number followed by trees in the plural.
ROW = re.compile(r'(\d{1,3})\s*(?:עצים|עצי|ברושים|אורנים|דקלים|סיגלונים|פיקוסים|'
                 r'אקליפטוסים|איקליפטוסים|אלונים|כאלו|כאלה|שיטים|ברכיכיטונים|וושינגטוניות)')


def fetch_page():
    q = urllib.parse.urlencode({
        'action': 'query', 'prop': 'revisions', 'rvprop': 'content|timestamp|ids',
        'rvslots': 'main', 'format': 'json', 'formatversion': 2, 'titles': PAGE,
    })
    req = urllib.request.Request(f'{API}?{q}', headers={'User-Agent': 'derech-kitzur build_trees.py'})
    with urllib.request.urlopen(req, timeout=60) as r:
        page = json.load(r)['query']['pages'][0]
    rev = page['revisions'][0]
    return rev['slots']['main']['content'], rev['timestamp'], rev['revid']


def parse_table(text):
    body = text[text.index('{|'):text.rindex('|}')]
    header = re.search(r'^! (.*)$', body, re.M).group(1)
    cols = [h.strip() for h in header.split('!!')]
    rows = []
    for chunk in body.split('\n|-\n')[1:]:
        if not chunk.startswith('|') or chunk.startswith('|}'):
            continue
        cells = [c.strip() for c in chunk[1:].split('||')]
        if len(cells) != len(cols):
            print(f'  שורה עם {len(cells)} תאים במקום {len(cols)}, מדלג: {chunk[:60]!r}', file=sys.stderr)
            continue
        rows.append(dict(zip(cols, cells)))
    return rows


def num(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return int(v) if v == int(v) else round(v, 1)


def clean(s):
    return re.sub(r'\s+', ' ', s or '').strip()


def survey_date(s):
    """'21/7/2019' -> '2019-07-21'."""
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})$', s or '')
    if not m:
        return None
    d, mo, y = map(int, m.groups())
    try:
        return datetime.date(y, mo, d).isoformat()
    except ValueError:
        return None


def main():
    text, stamp, revid = fetch_page()
    rows = parse_table(text)
    print(f'{len(rows)} עצים בדף (גרסה {revid}, {stamp})')

    to_wgs = Transformer.from_crs('EPSG:2039', 'EPSG:4326', always_xy=True)
    stacks = collections.defaultdict(list)
    for r in rows:
        if r.get('x') and r.get('y'):
            stacks[(r['x'], r['y'])].append(r)

    trees = []
    for (x, y), stack in stacks.items():
        x, y = float(x), float(y)
        for k, r in enumerate(stack):
            dx = dy = 0.0
            if len(stack) > 1:
                a = 2 * math.pi * k / len(stack)
                dx, dy = SPREAD_M * math.cos(a), SPREAD_M * math.sin(a)
            lng, lat = to_wgs.transform(x + dx, y + dy)
            health = clean(r['מצב בריאות העץ']) or 'לא תועד'
            species = clean(r['מין העץ']) or 'מין לא ידוע'
            t = {
                'num': clean(r['מספר העץ']) or None,
                'h': num(r['גובה העץ (מטר)']),
                'd': num(r['קוטר הגזע (ס"מ)']),
                'c': num(r['היקף נוף (מטר)']),
                'age': clean(r['גיל העץ']) or None,
                'health': health,
                'needs': [w for col, w in NEEDS if r.get(col) == 'כן'],
                'near': [w for col, w in NEAR if r.get(col) == 'כן'],
                'follow': clean(r['למעקב']) or None,
                'protect': clean(r['סוג הגנה נדרש']) or None,
                'date': survey_date(r['תאריך דיווח']),
            }
            # Five decimals is a metre, which is what the wiki rounded to anyway.
            geo = {'lat': round(lat, 5), 'lng': round(lng, 5), 'source': 'survey'}
            if len(stack) > 1:
                geo['spread'] = True
            note = clean(r['הערות'])
            # A row of the survey that stands for a whole avenue or grove: "שדרת
            # ברושים לאורך רחוב רבי עקיבא. 34 עצים". The pin is one tree's worth
            # of map, and the card has to say it is more.
            many = ROW.search(note)
            if many:
                t['many'] = int(many.group(1))
            # Every tree carries the page as its link; the app fills it in from
            # `source`, which keeps a URL out of each of 1,924 records.
            trees.append({
                'id': f'tree-{r["OBJECTID"]}',
                'name': species,
                'group': health,
                'geo': geo,
                'note': note,
                'tree': {k: v for k, v in t.items() if v not in (None, [])},
            })

    trees.sort(key=lambda p: int(p['id'].split('-')[1]))
    lats = [p['geo']['lat'] for p in trees]
    lngs = [p['geo']['lng'] for p in trees]
    doc = {
        'version': 1,
        'updated': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'source': PAGE_URL,
        'revision': revid,
        'name': 'עצים: סקר 2018–2019',
        'groups': [{'name': n, 'color': c} for n, c in HEALTH
                   if any(p['group'] == n for p in trees)],
        'bounds': [[min(lngs), min(lats)], [max(lngs), max(lats)]],
        'places': trees,
        'stats': {
            'places': len(trees),
            'species': len({p['name'] for p in trees}),
            'spread': sum(1 for p in trees if p['geo'].get('spread')),
            'many': sum(1 for p in trees if p['tree'].get('many')),
            'from': min(p['tree']['date'] for p in trees if p['tree'].get('date')),
            'until': max(p['tree']['date'] for p in trees if p['tree'].get('date')),
        },
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, separators=(',', ':'))
    s = doc['stats']
    print(f'{s["places"]} עצים, {s["species"]} מינים, {s["spread"]} פוזרו מערימות, '
          f'{s["from"]} עד {s["until"]} · {os.path.getsize(OUT) // 1024}KB')
    print('  ', collections.Counter(p['group'] for p in trees).most_common())


if __name__ == '__main__':
    main()
