"""בונה את data.js לדף חלוקת השכונות.

קלט (raw/):
  boundary.json   גבול המועצה מ-OSM (relation 1392849)
  stat2022.geojson  אזורים סטטיסטיים, מפקד 2022 (למ"ס, דרך ArcGIS Online)
  buildings.json  מבנים מ-OSM
  streets.json    רחובות ומסילה מ-OSM
  barriers.json   כל הדרכים, ואדיות, חורשות ופארקים מ-OSM (לגבולות הבלוקים)

שיטה: רשת תאים של CELL מטר ברשת ישראל החדשה (EPSG:2039). אוכלוסיית כל אזור
סטטיסטי מפוזרת על המבנים שבתוכו לפי שטח טביעת הרגל (מיפוי דסימטרי), ואז
מנורמלת לאומדן העדכני. השטח מחולק לבלוקים לפי גבולות טבעיים (ראו למטה).
"""
import json, math, os, collections
import pyproj
from shapely.geometry import shape, LineString, Polygon, box, mapping
from shapely.ops import transform, polygonize, unary_union
from shapely.strtree import STRtree

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')
CELL = 100.0
POP_NOW = 47194  # למ"ס דרך אתר ביטוח לאומי, 01/10/2026

to_itm = pyproj.Transformer.from_crs(4326, 2039, always_xy=True).transform
to_wgs = pyproj.Transformer.from_crs(2039, 4326, always_xy=True).transform

def load(name):
    with open(os.path.join(RAW, name), encoding='utf-8') as f:
        return json.load(f)

# --- גבול
rel = load('boundary.json')['elements'][0]
lines = [LineString([(p['lon'], p['lat']) for p in m['geometry']])
         for m in rel['members'] if m['type'] == 'way' and m['role'] == 'outer']
bnd_wgs = max(polygonize(unary_union(lines)), key=lambda g: g.area)
bnd = transform(to_itm, bnd_wgs)

# --- אזורים סטטיסטיים
stats = []
for f in load('stat2022.geojson')['features']:
    p = f['properties']
    g = transform(to_itm, shape(f['geometry'])).buffer(0).intersection(bnd)
    stats.append(dict(id=p['STAT_2022'], geom=g, pop=p['pop_approx'] or 0,
                      func=p['Main_Function_Txt'], kids=p['age0_19_pcnt'],
                      old=p['age65_pcnt'], acad=p['AcadmCert_pcnt'],
                      wage=p['employeesAnnual_medWage']))
stat_tree = STRtree([s['geom'] for s in stats])

# --- מבנים: רק כאלה שסביר שהם מגורים
NONRES = {'industrial', 'retail', 'greenhouse', 'school', 'train_station',
          'sports_centre', 'toilets', 'construction', 'no', 'synagogue',
          'commercial', 'warehouse', 'shed', 'garage', 'garages', 'barn',
          'farm_auxiliary', 'roof', 'service', 'public', 'kindergarten',
          'cowshed', 'stable', 'hangar', 'hospital', 'civic', 'office'}
buildings = []
for e in load('buildings.json')['elements']:
    t = e.get('tags', {})
    if t.get('building') in NONRES or 'amenity' in t or 'shop' in t:
        continue
    pts = [(p['lon'], p['lat']) for p in e.get('geometry', [])]
    if len(pts) < 4:
        continue
    g = transform(to_itm, Polygon(pts)).buffer(0)
    a = g.area
    if a < 35 or a > 2500:      # מחסנים קטנים / לולים ומבני משק גדולים
        continue
    try:
        lv = float(t.get('building:levels', 1))
    except ValueError:
        lv = 1
    w = min(a, 400) * max(1, min(lv, 12))
    c = g.centroid
    if not bnd.contains(c):
        continue
    buildings.append((c, w))

# שיוך מבנים לאזור סטטיסטי
stat_w = collections.defaultdict(float)
b_stat = []
for c, w in buildings:
    idx = [i for i in stat_tree.query(c) if stats[i]['geom'].contains(c)]
    si = idx[0] if idx else None
    b_stat.append(si)
    if si is not None:
        stat_w[si] += w

# --- בלוקים: השטח מחולק לפי גבולות טבעיים (רחובות, ואדיות, שולי חורשות, מסילה).
# כל שכונה היא אוסף של בלוקים שלמים, ולכן כל גבול בין שכונות עובר לאורך אחד מהם.
# לכל סוג גבול יש "מחיר" למטר: כביש ראשי, ואדי וחורשה זולים, סמטה יקרה,
# כך שהאלגוריתם מעדיף גבולות בולטים.
STRONG, MID, STREET, LANE, PATH = 1.0, 1.6, 3.0, 6.0, 9.0
MIN_BLOCK = 2500  # מ"ר. בלוק קטן מזה מתמזג בשכן
HW = {'motorway': STRONG, 'trunk': STRONG, 'primary': STRONG, 'secondary': STRONG,
      'trunk_link': STRONG, 'primary_link': STRONG, 'secondary_link': STRONG,
      'tertiary': MID, 'tertiary_link': MID,
      'residential': STREET, 'unclassified': STREET, 'living_street': STREET, 'pedestrian': STREET,
      'track': LANE,
      # שבילים ודרכי שירות: גבול יקר מאוד, רק כדי שבלוק גדול יוכל להתפצל כשאין ברירה
      'service': PATH, 'footway': PATH, 'path': PATH, 'cycleway': PATH, 'steps': PATH}
KIND = {STRONG: 'ראשי', MID: 'מאסף', STREET: 'רחוב', LANE: 'סמטה / דרך עפר', PATH: 'שביל'}

barrier_lines = []  # (LineString ITM, weight, kind)
def add_line(pts, w, kind):
    if len(pts) < 2:
        return
    g = transform(to_itm, LineString(pts))
    if g.length > 0:
        barrier_lines.append((g, w, kind))

for e in load('barriers.json')['elements']:
    t = e.get('tags', {})
    if e['type'] == 'way':
        pts = [(p['lon'], p['lat']) for p in e.get('geometry', [])]
        if 'highway' in t:
            if t['highway'] in HW:
                w = HW[t['highway']]
                if w == STREET and not (t.get('name') or t.get('name:he')):
                    w = LANE
                add_line(pts, w, 'road')
        elif t.get('waterway') in ('stream', 'river', 'drain', 'ditch', 'canal'):
            add_line(pts, STRONG, 'wadi')
        elif t.get('natural') == 'wood' or t.get('landuse') == 'forest':
            add_line(pts, STRONG, 'wood')
        elif t.get('leisure') == 'park':
            add_line(pts, MID, 'park')
    else:  # relation: כל קווי המתאר
        kind = 'wood'
        for m in e.get('members', []):
            if m['type'] == 'way' and m.get('geometry'):
                add_line([(p['lon'], p['lat']) for p in m['geometry']], STRONG, kind)
for e in load('streets.json')['elements']:
    if e.get('tags', {}).get('railway') == 'rail':
        add_line([(p['lon'], p['lat']) for p in e['geometry']], STRONG, 'rail')

kept_lines = [g for g, w, k in barrier_lines]
noded = unary_union([g.intersection(bnd.buffer(5)) for g in kept_lines] + [bnd.exterior])
faces = [f for f in polygonize(noded) if bnd.contains(f.representative_point())]
faces = [f.intersection(bnd).buffer(0) for f in faces]
faces = [f for f in faces if f.area > 1]
print('faces', len(faces))

# מיזוג בלוקים קטנים בשכן עם הגבול המשותף הארוך ביותר
from shapely.geometry import Polygon as _P
while True:
    faces.sort(key=lambda f: f.area)
    if faces[0].area >= MIN_BLOCK:
        break
    tree = STRtree(faces)
    merged = False
    used = set()
    out = list(faces)
    for i, f in enumerate(faces):
        if f.area >= MIN_BLOCK or i in used:
            continue
        best, bl = None, 0
        for q in tree.query(f.buffer(0.5)):
            if q == i or q in used:
                continue
            l = f.boundary.intersection(faces[q].boundary).length
            if l > bl:
                best, bl = q, l
        if best is None or bl < 0.5:
            continue
        out[best] = unary_union([faces[best].buffer(0.01), f.buffer(0.01)]).buffer(-0.01)
        out[i] = None
        used.add(i); used.add(best)
        merged = True
    faces = [f for f in out if f is not None]
    if not merged:
        break
# מצולע אחד לכל בלוק (מיזוג יכול להשאיר MultiPolygon זעיר)
faces = [max(getattr(f, 'geoms', [f]), key=lambda g: g.area) if f.geom_type != 'Polygon' else f for f in faces]
print('blocks', len(faces), 'small', sum(f.area < MIN_BLOCK for f in faces))

blocks = [dict(geom=f, area=f.area, pop=0.0, streets=collections.defaultdict(float)) for f in faces]
btree = STRtree(faces)

# אוכלוסייה: כל מבנה לבלוק שמכיל אותו
miss = 0
for (c, w), si in zip(buildings, b_stat):
    if si is None or stat_w[si] == 0:
        continue
    hit = [q for q in btree.query(c) if faces[q].contains(c)]
    if not hit:
        hit = [min(btree.query(c.buffer(50)) or range(len(faces)), key=lambda q: faces[q].distance(c))]
        miss += 1
    blocks[hit[0]]['pop'] += stats[si]['pop'] * w / stat_w[si]
tot22 = sum(b['pop'] for b in blocks)
scale = POP_NOW / tot22
for b in blocks:
    b['pop'] *= scale
print(f'pop2022 in blocks={tot22:.0f} scale={scale:.3f} buildings on edges={miss}')

# שטח מיושב בכל בלוק: איחוד של רדיוס 30 מ' סביב מבני המגורים. בבלוק חקלאי גדול
# עם כמה משקים נספר רק החלק שסביב הבתים, לא הפרדס כולו
SETTLE_R = 30
settled = unary_union([c.buffer(SETTLE_R, 8) for c, w in buildings])
for b in blocks:
    b['settled'] = b['geom'].intersection(settled).area

# אזור סטטיסטי דומיננטי
for b in blocks:
    best, bestA = None, 0
    for si in stat_tree.query(b['geom']):
        a = stats[si]['geom'].intersection(b['geom']).area
        if a > bestA:
            best, bestA = si, a
    b['stat'] = stats[best]['id'] if best is not None else 0
    b['edge'] = b['geom'].boundary.intersection(bnd.exterior.buffer(1)).length > 5

# שכנויות: הגבול המשותף ומחירו
ltree = STRtree(kept_lines)
def edge_cost(line):
    cost, kinds = 0.0, collections.defaultdict(float)
    for seg in getattr(line, 'geoms', [line]):
        n = max(1, int(seg.length // 10))
        for q in range(n):
            p = seg.interpolate((q + 0.5) / n, normalized=True)
            w, kd = PATH, None
            for li in ltree.query(p.buffer(2)):
                if kept_lines[li].distance(p) < 2 and barrier_lines[li][1] <= w:
                    w, kd = barrier_lines[li][1], barrier_lines[li][2]
            ln = seg.length / n
            cost += w * ln
            kinds[KIND[w] if kd in (None, 'road') else {'wadi': 'ואדי', 'wood': 'חורשה', 'park': 'פארק', 'rail': 'מסילה'}[kd]] += ln
    return cost, kinds

adj = []
for a in range(len(blocks)):
    ga = blocks[a]['geom']
    for b in btree.query(ga.buffer(0.5)):
        if b <= a:
            continue
        # סובלנות של מטר: מיזוג בלוקים קטנים מזיז קודקודים בשברירי מטר
        line = blocks[b]['geom'].boundary.intersection(ga.buffer(1.0))
        line = unary_union([g for g in getattr(line, 'geoms', [line]) if g.geom_type in ('LineString', 'MultiLineString')]) if not line.is_empty else line
        if line.is_empty or line.length < 3:
            continue
        cost, kinds = edge_cost(line)
        kind = max(kinds.items(), key=lambda kv: kv[1])[0]
        adj.append((int(a), int(b), line, cost, kind))
print('adj', len(adj))

# --- שמות רחובות לבלוקים: רחוב שעובר לאורך הבלוק או בתוכו
street_names, street_idx = [], {}
for e in load('streets.json')['elements']:
    t = e.get('tags', {})
    pts = [(p['lon'], p['lat']) for p in e.get('geometry', [])]
    if len(pts) < 2 or t.get('railway') == 'rail':
        continue
    if t.get('highway') in ('footway', 'path', 'cycleway', 'track', 'steps', 'service'):
        continue
    name = t.get('name:he') or t.get('name')
    line = transform(to_itm, LineString(pts)).intersection(bnd)
    for seg in getattr(line, 'geoms', [line]):
        if seg.is_empty or seg.geom_type != 'LineString':
            continue
        n = max(2, int(seg.length // 25))
        for q in range(n):
            p = seg.interpolate((q + 0.5) / n, normalized=True)
            hits = [bi for bi in btree.query(p.buffer(15)) if faces[bi].distance(p) < 15]
            for bi in hits:
                blocks[bi]['streets'][name] += seg.length / n / len(hits)
for b in blocks:
    out = []
    for name, ln in sorted(b['streets'].items(), key=lambda kv: -kv[1])[:3]:
        if name not in street_idx:
            street_idx[name] = len(street_names)
            street_names.append(name)
        out.append([street_idx[name], round(ln)])
    b['st'] = out

def ring(g, tol=2):
    # מצולע עם חורים: [חיצוני, חור, חור...]. בלוק חקלאי יכול להקיף שכונה שלמה
    g = transform(to_wgs, g.simplify(tol))
    rr = lambda r: [[round(y, 5), round(x, 5)] for x, y in r.coords]
    return [rr(g.exterior)] + [rr(h) for h in g.interiors]
def lines(g, tol=2):
    g = transform(to_wgs, g.simplify(tol))
    out = []
    for s in getattr(g, 'geoms', [g]):
        if s.geom_type == 'LineString':
            out.append([[round(y, 5), round(x, 5)] for x, y in s.coords])
    return out
def simp(g, tol=8):
    g = transform(to_wgs, g.simplify(tol))
    polys = getattr(g, 'geoms', [g])
    return [[[round(y, 5), round(x, 5)] for x, y in p.exterior.coords] for p in polys if p.area > 0]

KINDS = sorted({k for *_, k in adj})
data = dict(
    source=dict(pop_now=POP_NOW, pop_2022=round(sum(s['pop'] for s in stats)),
                area_km2=round(bnd.area / 1e6, 2), blocks=len(blocks)),
    # בלוק: [x, y (מרכז ב-ITM), שטח, תושבים, אזור סטטיסטי, נוגע בגבול המועצה, רחובות, מצולע, שטח מיושב]
    blocks=[[round(b['geom'].centroid.x), round(b['geom'].centroid.y), round(b['area']), round(b['pop'], 1),
             b['stat'], 1 if b['edge'] else 0, b['st'], ring(b['geom']), round(b['settled'])] for b in blocks],
    # שכנות: [a, b, אורך במטרים, מחיר, סוג הגבול, קו]
    adj=[[a, b, round(line.length), round(cost), KINDS.index(kind), lines(line)] for a, b, line, cost, kind in adj],
    kinds=KINDS,
    streets=street_names,
    boundary=simp(bnd),
    rail=[[[round(y, 5), round(x, 5)] for x, y in transform(to_wgs, g).coords] for g, w, k in barrier_lines if k == 'rail'],
    stats=[dict(id=s['id'], pop=round(s['pop'] * scale), pop22=round(s['pop']),
                func=s['func'], area=round(s['geom'].area),
                kids=s['kids'], old=s['old'], acad=s['acad'], wage=s['wage'],
                poly=simp(s['geom'])) for s in stats],
)
with open(os.path.join(HERE, 'data.js'), 'w', encoding='utf-8') as f:
    f.write('// נוצר ע"י build_data.py. לא לערוך ידנית.\nwindow.DATA = ')
    json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    f.write(';\n')
print('kinds', KINDS, 'streets', len(street_names), 'size', os.path.getsize(os.path.join(HERE, 'data.js')))
