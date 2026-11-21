#!/usr/bin/env python3
"""The schematic neighbourhood maps of [[רשימת שכונות]].

Everything is drawn from three files in hood_maps/:
    hoods.json   the neighbourhoods as geographic polygons (hood_extract.py makes
                 it from Irit Rotem's overlays on the official map)
    roads.json   main streets and the railway, from OpenStreetMap
    and the POIS / ROAD_LABELS tables below.

The map is north-up and true to scale, with the main streets and a handful of
landmarks, so a reader can tie the coloured shapes to the place they know.

    python3 hood_maps.py render            # hood_maps/out/*.png  (no network)
    python3 hood_maps.py wikitext          # the clickable block for the list page
    python3 hood_maps.py upload [name...]  # upload the PNGs (all by default)

Needs rsvg-convert and the Rubik font.
"""

import argparse
import html
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, 'hood_maps')
OUT = os.path.join(DIR, 'out')
FILE_PREFIX = 'מפה סכמטית - '
ALL = 'כל השכונות'

LAT0, LON0 = 32.47, 34.975
MX, MY = 93800.0, 110900.0
# the frame, in degrees: the neighbourhoods plus enough around them to orient
SOUTH, NORTH, WEST, EAST = 32.4530, 32.4915, 34.9440, 35.0085
W = 2000                                   # px of the full map

PALETTE = {
    'מרכז המושבה': '#a9d3de', 'יפה נוף': '#a8cf9a', 'ממשלתי גאולה': '#ecd9a0',
    'נווה פרדסים': '#d6bfe6', 'נווה מרחב': '#b3bbe8', 'רמב"ם': '#e9a3d0',
    'רמז': '#9fd4b8', 'נווה אשר': '#d9b3a3', 'נווה התות ואלה': '#a9bfd6',
    'עומרים': '#c2c7a0', 'תל שלום': '#c9a8dc', 'חלומות כרכור': '#f0e08c',
    'מרכז כרכור': '#f3c08e', 'יובלים וקורן': '#a3dfc0', 'מזרח כרכור': '#f0a8a0',
    'הנחל': '#d9d2c8',
}
# label position nudges (metres east, north) and line breaks, where the centroid is poor
LABEL = {
    'מרכז המושבה': (-150, 150, 'מרכז המושבה'), 'ממשלתי גאולה': (-120, -20, 'ממשלתי|גאולה'),
    'נווה התות ואלה': (40, -20, 'נווה התות|ואלה'), 'חלומות כרכור': (0, -30, 'חלומות|כרכור'),
    'יובלים וקורן': (130, 230, 'יובלים וקורן'), 'נווה פרדסים': (0, 0, 'נווה|פרדסים'),
    'מרכז כרכור': (40, 105, 'מרכז כרכור'), 'מזרח כרכור': (0, 30, 'מזרח כרכור'),
    'נווה מרחב': (0, -60, 'נווה מרחב'), 'רמב"ם': (0, 0, 'רמב"ם'),
    'עומרים': (20, 40, 'עומרים'), 'תל שלום': (-60, -150, 'תל שלום'),
    'נווה אשר': (0, -40, 'נווה אשר'),
}

# landmarks: (label, lat, lon, where the label goes: l/r/t/b)
POIS = [
    ('תחנת הרכבת', 32.48526, 34.95417, 'l'),
    ('אנדרטת הנח"ל', 32.48084, 34.96202, 't'),
    ('ביג', 32.48854, 34.97010, 'l'),
    ('בית העלמין', 32.47949, 34.98065, 'l'),
    ('בית הספר החקלאי', 32.47376, 34.98167, 'b'),
    ('המתנ"ס', 32.47241, 34.96993, 'b'),
    ('האמפיתאטרון', 32.47515, 34.96938, 'l'),
    ('בית המועצה', 32.47484, 34.97523, 'b'),
    ('כיכר הנדיב', 32.47601, 34.97644, 't'),
    ('גן הגיבורים', 32.47339, 34.99179, 'r'),
    ('מרכז יובלים', 32.48113, 34.98760, 'r'),
    ('צומת כרכור', 32.46515, 34.99392, 'l'),
]
# quiet labels for what lies around the neighbourhoods
AREAS = [
    ('אזור התעשייה', 32.4890, 34.9790), ('אזור תעסוקה|אילנות', 32.4835, 34.9585),
    ('מחנה 80', 32.4633, 34.9862), ('משמרות', 32.4893, 34.9843),
    ('כפר פינס', 32.4842, 35.0030), ('גן השומרון', 32.4636, 34.9985),
    ('עין שמר', 32.4625, 35.0062),
]
# street names: (text, OSM names, lat, lon) of where to write it; the angle follows the street
ROAD_LABELS = [
    ('דרך הנדיב', ('דרך הנדיב',), 32.4838, 34.9713), ('דרך הנדיב', ('דרך הנדיב',), 32.4640, 34.9850),
    ('דרך הבנים', ('דרך הבנים', 'הבנים'), 32.4768, 34.9850), ('דרך הבנים', ('דרך הבנים', 'הבנים'), 32.4749, 34.9970),
    ('דרך הים', ('דרך הים', 'הים'), 32.4803, 34.9665),
    ('דרך למרחב', ('למרחב',), 32.4731, 34.9590),
    ('דרך פיק"א', ('פיק"א',), 32.4610, 34.9672), ('דרך פיק"א', ('פיק"א',), 32.4700, 34.9712),
    ('המייסדים', ('המייסדים',), 32.4795, 34.9912),
    ('אחוזה', ('אחוזה',), 32.4712, 34.9860),
    ('הדרים', ('הדרים',), 32.4760, 34.9640),
    ('דרך עירון', ('עירון',), 32.4667, 34.9918),
    ('חלוצים', ('חלוצים',), 32.4662, 34.9740),
    ('כביש 65', ('65', 'ואדי עארה'), 32.4600, 34.9905),
]

SS = W / ((EAST - WEST) * MX)              # px per metre
H = round((NORTH - SOUTH) * MY * SS)


def px(lat, lon):
    return ((lon - WEST) * MX * SS, (NORTH - lat) * MY * SS)


def path(ring, close=True):
    pts = [px(a, b) for a, b in ring]
    return 'M' + ' L'.join('%.1f %.1f' % p for p in pts) + (' Z' if close else '')


def smooth(ring, rounds=2):
    """Chaikin corner cutting: the traced outlines are jagged at pixel scale."""
    for _ in range(rounds):
        out = []
        for i, a in enumerate(ring):
            b = ring[(i + 1) % len(ring)]
            out += [[a[0] * .75 + b[0] * .25, a[1] * .75 + b[1] * .25],
                    [a[0] * .25 + b[0] * .75, a[1] * .25 + b[1] * .75]]
        ring = out
    return ring


def centroid(ring):
    a = cx = cy = 0.0
    for i, p in enumerate(ring):
        q = ring[(i + 1) % len(ring)]
        f = p[1] * q[0] - q[1] * p[0]
        a += f; cx += (p[1] + q[1]) * f; cy += (p[0] + q[0]) * f
    return cy / (3 * a), cx / (3 * a)      # lat, lon


def text(x, y, s, size, fill, weight=500, halo='#ffffff', halo_w=None, rot=0, anchor='middle',
         opacity=1, spacing=0):
    lines = s.split('|')
    y0 = y - (len(lines) - 1) * size * 0.55
    tr = ' transform="rotate(%.1f %.1f %.1f)"' % (rot, x, y) if rot else ''
    out = []
    for i, ln in enumerate(lines):
        common = ('x="%.1f" y="%.1f" font-family="Rubik" font-size="%d" font-weight="%d" '
                  'text-anchor="%s" direction="rtl" letter-spacing="%s"%s'
                  % (x, y0 + i * size * 1.1 + size * 0.35, size, weight, anchor, spacing, tr))
        ln = html.escape(ln)
        if halo:
            out.append('<text %s fill="none" stroke="%s" stroke-width="%.1f" stroke-linejoin="round" '
                       'opacity="%.2f">%s</text>' % (common, halo, halo_w or size * 0.28, opacity * .9, ln))
        out.append('<text %s fill="%s" opacity="%.2f">%s</text>' % (common, fill, opacity, ln))
    return '\n'.join(out)


def street_angle(ways, names, lat, lon):
    """Direction of the named street nearest to a point, as an upright text angle."""
    best = None
    for w in ways:
        if w['name'] not in names and w['ref'] not in names:
            continue
        g = w['g']
        for a, b in zip(g, g[1:]):
            m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            d = math.hypot((m[0] - lat) * MY, (m[1] - lon) * MX)
            if best is None or d < best[0]:
                best = (d, a, b, m)
    if best is None:
        return 0, lat, lon
    _, a, b, m = best
    (x1, y1), (x2, y2) = px(*a), px(*b)
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if ang > 90: ang -= 180
    if ang < -90: ang += 180
    return ang, m[0], m[1]


def build_svg(hoods, ways, focus=None, part=None):
    """The whole map; with `focus`, one neighbourhood is lit and the rest go quiet."""
    k = 1.0
    o = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (W, H, W, H),
         '<rect width="100%" height="100%" fill="#f4f2ec"/>']

    def roads(kind, width, colour, opacity=1, dash=None):
        d = ' '.join(path(w['g'], False) for w in ways if w['k'] == kind)
        o.append('<path d="%s" fill="none" stroke="%s" stroke-width="%.1f" stroke-linecap="round" '
                 'stroke-linejoin="round" opacity="%.2f"%s/>'
                 % (d, colour, width * k, opacity, ' stroke-dasharray="%s"' % dash if dash else ''))

    # streets outside the neighbourhoods, faint, so the town does not float in a void
    roads('minor', 2.2, '#e3dfd5')
    for h in hoods:
        lit = focus is None or h['name'] == focus
        fill = PALETTE[h['name']] if focus is None else ('#c96a5e' if lit and not part else '#dcd8cf')
        for r in h['rings']:
            o.append('<path d="%s" fill="%s" stroke="#ffffff" stroke-width="5" stroke-linejoin="round"/>'
                     % (path(smooth(r)), fill))
        if focus and lit and part:
            for p in h['parts']:
                if p['name'] == part:
                    for r in p['rings']:
                        o.append('<path d="%s" fill="#c96a5e" stroke="#ffffff" stroke-width="5" '
                                 'stroke-linejoin="round"/>' % path(smooth(r)))
        elif focus is None and h.get('parts'):
            pass
    # streets: residential as a faint texture, the main ones clear
    roads('minor', 1.8, '#ffffff', 0.55)
    roads('rail', 5, '#6b6b6b'); roads('rail', 2.6, '#f4f2ec', dash='12 12')
    roads('main', 8.5, '#8e8778', 0.55); roads('major', 13, '#8e8778', 0.7)
    roads('main', 6, '#ffffff'); roads('major', 9.5, '#fff1b8')

    quiet = focus is not None
    for name, names, lat, lon in ROAD_LABELS:
        ang, la, lo = street_angle(ways, names, lat, lon)
        x, y = px(la, lo)
        o.append(text(x, y - 13, name, 21, '#5d574b', 500, rot=ang, halo_w=5, opacity=.95))
    for name, lat, lon in AREAS:
        x, y = px(lat, lon)
        o.append(text(x, y, name, 24, '#9a9485', 400, halo=None, spacing='1'))
    for name, lat, lon, side in POIS:
        x, y = px(lat, lon)
        o.append('<circle cx="%.1f" cy="%.1f" r="9" fill="#2f3b45" stroke="#ffffff" stroke-width="3.5"/>' % (x, y))
        dx, dy, anchor = {'l': (-17, 0, 'start'), 'r': (17, 0, 'end'),
                          't': (0, -24, 'middle'), 'b': (0, 26, 'middle')}[side]
        # in RTL text "start" is the right edge, so a label left of the dot is anchored at "start"
        o.append(text(x + dx, y + dy, name, 23, '#2f3b45', 500, halo_w=6, anchor=anchor))
    for h in hoods:
        if quiet and h['name'] != focus:
            continue
        dx, dy, label = LABEL.get(h['name'], (0, 0, h['name']))
        if quiet and part:
            p = next(p for p in h['parts'] if p['name'] == part)
            lat, lon = centroid(max(p['rings'], key=len)); label = part
        else:
            lat, lon = centroid(max(h['rings'], key=len))
            lat += dy / MY; lon += dx / MX
        x, y = px(lat, lon)
        size = 46 if quiet else max(28, min(40, round(24 + 14 * h['area_km2'])))
        o.append(text(x, y, label, size, '#1f2a33', 700, halo_w=size * .22))
    # north arrow and scale bar
    nx, ny = W - 70, 70
    o.append('<path d="M%d %d l-13 34 l13 -8 l13 8 z" fill="#5d574b"/>' % (nx, ny - 20))
    o.append(text(nx, ny + 34, 'צפון', 20, '#5d574b', 500, halo=None))
    bar = 500 * SS
    o.append('<path d="M60 %d h%.1f" stroke="#5d574b" stroke-width="5"/>' % (H - 50, bar))
    o.append(text(60 + bar / 2, H - 76, '500 מטר', 20, '#5d574b', 500, halo=None))
    o.append(text(W - 20, H - 22, 'גבולות השכונות: לפי מפת השכונות של המושבה · רחובות: © תורמי OpenStreetMap',
                  17, '#8a8478', 400, halo=None, anchor='start'))
    o.append('</svg>')
    return '\n'.join(o)


def load():
    hoods = json.load(open(os.path.join(DIR, 'hoods.json'), encoding='utf-8'))['hoods']
    ways = json.load(open(os.path.join(DIR, 'roads.json'), encoding='utf-8'))['ways']
    return hoods, ways


def targets(hoods):
    """(file label, focus, part) for every map."""
    out = [(ALL, None, None)]
    for h in hoods:
        if h.get('parts'):
            out.append((h['name'], h['name'], None))
            out += [(p['name'], h['name'], p['name']) for p in h['parts']]
        else:
            out.append((h['name'], h['name'], None))
    return out


def render(names=None):
    hoods, ways = load()
    os.makedirs(OUT, exist_ok=True)
    done = []
    for label, focus, part in targets(hoods):
        if names and label not in names:
            continue
        svg = os.path.join(OUT, label + '.svg')
        png = os.path.join(OUT, FILE_PREFIX + label + '.png')
        open(svg, 'w', encoding='utf-8').write(build_svg(hoods, ways, focus, part))
        width = W if label == ALL else 1000
        subprocess.run(['rsvg-convert', '-w', str(width), svg, '-o', png], check=True)
        os.remove(svg)
        done.append((label, png))
    return done


def clip(ring):
    pts = [px(a, b) for a, b in ring]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    l, t, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    poly = ', '.join('%.1f%% %.1f%%' % ((x - l) / w * 100, (y - t) / h * 100) for x, y in pts)
    return ('left:%.2f%%; top:%.2f%%; width:%.2f%%; height:%.2f%%; clip-path:polygon(%s);'
            % (l / W * 100, t / H * 100, w / W * 100, h / H * 100, poly))


def wikitext(exists=lambda title: True):
    hoods, _ = load()
    o = ['<div style="text-align:center;"><div class="hoodmap" style="position:relative; display:inline-block; '
         'line-height:0; font-size:0; direction:ltr;">[[קובץ:%s%s.png|1000px|קישור=]]' % (FILE_PREFIX, ALL)]
    for h in hoods:
        for z in (h.get('parts') or [h]):
            name = z['name'].replace('"', '&quot;')
            art = z['article']
            for r in z['rings']:
                span = ('<span class="hoodhit" title="%s" style="display:block; width:100%%; height:100%%;">'
                        '&nbsp;</span>' % name)
                if art and exists(art):
                    o.append('<div class="hood" data-hood="%s" style="position:absolute; %s">[[%s|%s]]</div>'
                             % (name, clip(r), art, span))
                else:
                    o.append('<div class="hood hood-nolink" data-hood="%s" style="position:absolute; %s">%s</div>'
                             % (name, clip(r), span))
    o.append('</div></div>')
    return '\n'.join(o)


def upload(names=None):
    from wiki_client import WikiClient, API_URL
    client = WikiClient(); client.login()
    for label, png in render(names):
        fn = FILE_PREFIX + label + '.png'
        if label == ALL:
            desc = ('מפה סכמטית של שכונות פרדס חנה-כרכור, עם הרחובות הראשיים ונקודות ציון מוכרות.')
        else:
            desc = 'מיקום %s (באדום) בתוך פרדס חנה-כרכור, על גבי מפה סכמטית של השכונות.' % label
        desc += ('\n\nגבולות השכונות לפי מפת השכונות של המושבה, כפי שהם מוצגים ב[[רשימת שכונות]]. '
                 'הרחובות מתוך OpenStreetMap (© תורמי OpenStreetMap, רישיון ODbL). '
                 'נוצרה אוטומטית על ידי הבוט (hood_maps.py).\n\n[[קטגוריה:שכונות]]\n')
        r = client.session.post(API_URL, data={
            'action': 'upload', 'filename': fn, 'comment': 'מפה סכמטית מעודכנת: חלוקת השכונות החדשה, עם רחובות ראשיים ונקודות ציון',
            'text': desc, 'token': client._csrf_token(), 'ignorewarnings': '1', 'format': 'json'},
            files={'file': (fn, open(png, 'rb').read(), 'image/png')}).json()
        print(label, '->', r.get('upload', {}).get('result') or r.get('error'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['render', 'wikitext', 'upload'])
    ap.add_argument('names', nargs='*')
    a = ap.parse_args()
    if a.cmd == 'render':
        for label, png in render(a.names):
            print(label, '->', png)
    elif a.cmd == 'wikitext':
        print(wikitext())
    else:
        upload(a.names)


if __name__ == '__main__':
    sys.exit(main())
