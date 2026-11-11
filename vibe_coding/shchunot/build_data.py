"""בונה את data.js לדף חלוקת השכונות.

קלט (raw/):
  boundary.json   גבול המועצה מ-OSM (relation 1392849)
  stat2022.geojson  אזורים סטטיסטיים, מפקד 2022 (למ"ס, דרך ArcGIS Online)
  buildings.json  מבנים מ-OSM
  streets.json    רחובות ומסילה מ-OSM

שיטה: רשת תאים של CELL מטר ברשת ישראל החדשה (EPSG:2039). אוכלוסיית כל אזור
סטטיסטי מפוזרת על המבנים שבתוכו לפי שטח טביעת הרגל (מיפוי דסימטרי), ואז
מנורמלת לאומדן העדכני.
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

# --- רשת
minx, miny, maxx, maxy = bnd.bounds
x0 = math.floor(minx / CELL) * CELL
y0 = math.floor(miny / CELL) * CELL
nx = int(math.ceil((maxx - x0) / CELL))
ny = int(math.ceil((maxy - y0) / CELL))

cells = {}
for j in range(ny):
    for i in range(nx):
        sq = box(x0 + i * CELL, y0 + j * CELL, x0 + (i + 1) * CELL, y0 + (j + 1) * CELL)
        if not sq.intersects(bnd):
            continue
        part = sq.intersection(bnd)
        if part.area < 0.08 * CELL * CELL:
            continue
        # אזור סטטיסטי דומיננטי (לתצוגה)
        best, bestA = None, 0
        for si in stat_tree.query(part):
            a = stats[si]['geom'].intersection(part).area
            if a > bestA:
                best, bestA = si, a
        cells[(i, j)] = dict(i=i, j=j, area=part.area, pop=0.0, stat=best,
                             streets=collections.defaultdict(float))

def cell_of(x, y):
    return (int((x - x0) // CELL), int((y - y0) // CELL))

for (c, w), si in zip(buildings, b_stat):
    if si is None or stat_w[si] == 0:
        continue
    k = cell_of(c.x, c.y)
    if k in cells:
        cells[k]['pop'] += stats[si]['pop'] * w / stat_w[si]

# אזורים עם אוכלוסייה בלי מבנים ממופים: פיזור לפי שטח (לא אמור לקרות)
for si, s in enumerate(stats):
    if s['pop'] and not stat_w[si]:
        print('אזור בלי מבנים:', s['id'])

tot22 = sum(c['pop'] for c in cells.values())
scale = POP_NOW / tot22
for c in cells.values():
    c['pop'] *= scale
print(f'cells={len(cells)} pop2022 in cells={tot22:.0f} scale={scale:.3f}')

# --- רחובות לתאים (לשמות)
street_names = []
street_idx = {}
rail = []
for e in load('streets.json')['elements']:
    t = e.get('tags', {})
    pts = [(p['lon'], p['lat']) for p in e.get('geometry', [])]
    if len(pts) < 2:
        continue
    if t.get('railway') == 'rail':
        rail.append([[round(p[1], 5), round(p[0], 5)] for p in pts])
        continue
    if t.get('highway') in ('footway', 'path', 'cycleway', 'track', 'steps', 'service'):
        continue
    name = t.get('name:he') or t.get('name')
    line = transform(to_itm, LineString(pts)).intersection(bnd)
    if line.is_empty:
        continue
    for seg in getattr(line, 'geoms', [line]):
        n = max(2, int(seg.length // 25))
        for q in range(n):
            p = seg.interpolate((q + 0.5) / n, normalized=True)
            k = cell_of(p.x, p.y)
            if k in cells:
                cells[k]['streets'][name] += seg.length / n

for c in cells.values():
    top = sorted(c['streets'].items(), key=lambda kv: -kv[1])[:3]
    out = []
    for name, ln in top:
        if name not in street_idx:
            street_idx[name] = len(street_names)
            street_names.append(name)
        out.append([street_idx[name], round(ln)])
    c['st'] = out

# --- התאמה אפינית ITM -> latlon (מספיק מדויק בקנה המידה הזה)
import numpy as np
samples = [(x0 + a * nx * CELL, y0 + b * ny * CELL) for a in (0, .5, 1) for b in (0, .5, 1)]
A = np.array([[x, y, 1] for x, y in samples])
ll = np.array([to_wgs(x, y) for x, y in samples])  # lon, lat
coef_lon, *_ = np.linalg.lstsq(A, ll[:, 0], rcond=None)
coef_lat, *_ = np.linalg.lstsq(A, ll[:, 1], rcond=None)

def simp(g, tol=8):
    g = transform(to_wgs, g.simplify(tol))
    polys = getattr(g, 'geoms', [g])
    return [[[round(y, 5), round(x, 5)] for x, y in p.exterior.coords] for p in polys if p.area > 0]

data = dict(
    source=dict(pop_now=POP_NOW, pop_2022=round(sum(s['pop'] for s in stats)),
                area_km2=round(bnd.area / 1e6, 2), cell=CELL),
    grid=dict(x0=x0, y0=y0, nx=nx, ny=ny, lon=list(coef_lon), lat=list(coef_lat)),
    cells=[[c['i'], c['j'], round(c['area']), round(c['pop'], 1),
            stats[c['stat']]['id'] if c['stat'] is not None else 0, c['st']]
           for c in sorted(cells.values(), key=lambda c: (c['j'], c['i']))],
    streets=street_names,
    boundary=simp(bnd),
    rail=rail,
    stats=[dict(id=s['id'], pop=round(s['pop'] * scale), pop22=round(s['pop']),
                func=s['func'], area=round(s['geom'].area),
                kids=s['kids'], old=s['old'], acad=s['acad'], wage=s['wage'],
                poly=simp(s['geom'])) for s in stats],
)
with open(os.path.join(HERE, 'data.js'), 'w', encoding='utf-8') as f:
    f.write('// נוצר ע"י build_data.py. לא לערוך ידנית.\nwindow.DATA = ')
    json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    f.write(';\n')
print('streets', len(street_names), 'size', os.path.getsize(os.path.join(HERE, 'data.js')))
