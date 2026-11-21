#!/usr/bin/env python3
"""Turn Irit Rotem's neighbourhood overlays into geographic polygons.

Irit draws each neighbourhood as a coloured overlay on a photo of the official
"מפת המושבה", one JPEG per neighbourhood, all on the same 3588x2948 photo. The
per-pixel median of the set is therefore the bare photo, and the difference
from it is that neighbourhood's area.

The official map turns out to be true to scale (rotated ~12.6 degrees), so an
affine fit on ten junctions places it on real coordinates to within ~10 m.
That lets the schematic map be drawn north-up with real OSM streets.

Output: hood_maps/hoods.json  (the single source for hood_maps.py)

Needs numpy, pillow, scipy, opencv (not in the system python):
    python3 -m venv /tmp/hv && /tmp/hv/bin/pip install numpy pillow scipy opencv-python-headless
    /tmp/hv/bin/python hood_extract.py <dir with Irit's jpgs> [old_list_page.wiki]

The optional wikitext of the previous [[רשימת שכונות]] is used to compare the
old polygons with the new ones and to keep יובלים and קורן apart inside the
neighbourhood Irit now draws as one.
"""

import json
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'hood_maps', 'hoods.json')

SC = 2                      # work at half resolution
FULL = (3588, 2948)

# file stem in Irit's folder -> (display name, article, district)
HOODS = [
    ('מרכז המושבה',   'מרכז המושבה',   'מרכז המושבה',                 'פרדס חנה'),
    ('יפה נוף',       'יפה נוף',       'יפה נוף (שכונה)',             'פרדס חנה'),
    ('ממשלתי גאולה',  'ממשלתי גאולה',  'גאולה (שכונה)',               'פרדס חנה'),
    ('נווה פרדסים',   'נווה פרדסים',   'נווה פרדסים (שכונה)',         'פרדס חנה'),
    ('נווה מרחב',     'נווה מרחב',     'נווה מרחב (שכונה)',           'פרדס חנה'),
    ('רמבם',          'רמב"ם',         'רמב"ם (שכונה)',               'פרדס חנה'),
    ('רמז',           'רמז',           'רמז (שכונה)',                 'פרדס חנה'),
    ('נווה אשר',      'נווה אשר',      'נווה אשר (שכונה)',            'פרדס חנה'),
    ('נווה תות אלה',  'נווה התות ואלה', 'נווה התות (שכונה)',          'פרדס חנה'),
    ('עומרים',        'עומרים',        'עומרים (שכונה)',              'כרכור'),
    ('תל שלום',       'תל שלום',       'עממי, תל שלום נווה אפריים',   'כרכור'),
    ('חלומות כרכור',  'חלומות כרכור',  'חלומות כרכור (שכונה)',        'כרכור'),
    ('מרכז כרכור',    'מרכז כרכור',    'מרכז כרכור',                  'כרכור'),
    ('יובלים קורן',   'יובלים וקורן',  None,                          'כרכור'),
    ('מזרח כרכור',    'מזרח כרכור',    'מזרח כרכור',                  'כרכור'),
    ('הנחל',          'הנחל',          'הנחל (שכונה)',                'כרכור'),
]
# a neighbourhood drawn as one but covered by two articles: split by the old polygons
SPLIT = {'יובלים וקורן': [('יובלים', 'יובלים (שכונה)'), ('קורן', 'קורן (שכונה)')]}

# junctions: half-res pixel on the photo <-> (lat, lon) from OSM
CONTROL = [
    ((1277, 675), (32.47555, 34.99076)),     # דרך הבנים x המייסדים
    ((1290, 792), (32.47220, 34.99024)),     # המייסדים x אחוזה
    ((710, 593), (32.4812389, 34.9730083)),  # דרך הנדיב x דרך הים
    ((367, 725), (32.4796749, 34.9611257)),  # דרך הים x הדרים
    ((589, 943), (32.4723683, 34.9671972)),  # דרך למרחב x הדרים
    ((664, 1162), (32.4661, 34.96785)),      # דרך פיק"א x עציון
    ((1317, 939), (32.4679, 34.98998)),      # המייסדים x דרך עירון
    ((1000, 692), (32.47684, 34.98175)),     # דרך הבנים x דרך משמרות
    ((155, 1007), (32.47326, 34.95226)),     # דרך למרחב x שד' הציונות
    ((1459, 1020), (32.46515, 34.99392)),    # צומת כרכור
]
LAT0, LON0 = 32.47, 34.975
MX, MY = 93800.0, 110900.0   # metres per degree here


def fit():
    a = np.array([[(g[1] - LON0) * MX, (g[0] - LAT0) * MY, 1] for _, g in CONTROL])
    b = np.array([p for p, _ in CONTROL], float)
    m, *_ = np.linalg.lstsq(a, b, rcond=None)        # [x y 1] @ m = pixel
    err = np.hypot(*(a @ m - b).T)
    full = np.vstack([m.T, [0, 0, 1]])
    return np.linalg.inv(full), err


def to_geo(inv, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ inv.T
    return [[round(LAT0 + y / MY, 6), round(LON0 + x / MX, 6)] for x, y, _ in p]


def load(folder, stem):
    im = Image.open(os.path.join(folder, stem + '.jpg')).convert('RGB')
    assert im.size == FULL, (stem, im.size)
    return np.asarray(im.resize((FULL[0] // SC, FULL[1] // SC), Image.LANCZOS)).astype(np.int16)


def raw_mask(img, base):
    m = np.abs(img - base).sum(axis=2) > 40
    m = ndi.binary_opening(m, iterations=2)
    m = ndi.binary_closing(m, iterations=6)
    m = ndi.binary_fill_holes(m)
    lab, k = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, k + 1))
    return lab == (1 + int(np.argmax(sizes)))


def old_polygons(text):
    out = {}
    for m in re.finditer(r'data-hood="([^"]+)" style="([^"]+)"', text):
        style = m.group(2)
        box = {k: float(v) / 100 for k, v in re.findall(r'(left|top|width|height):([\d.]+)%', style)}
        poly = re.search(r'clip-path:polygon\(([^)]+)\)', style)
        if not poly or len(box) < 4:
            continue
        pts = []
        for pair in poly.group(1).split(','):
            x, y = [float(v.strip('% ')) / 100 for v in pair.strip().split()]
            pts.append(((box['left'] + x * box['width']) * FULL[0] / SC,
                        (box['top'] + y * box['height']) * FULL[1] / SC))
        out[m.group(1).replace('&quot;', '"')] = np.array(pts)
    return out


def poly_mask(pts, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def rings(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = max(cv2.contourArea(c) for c in cs)
    out = []
    for c in cs:
        if cv2.contourArea(c) < 0.05 * big:
            continue
        c = cv2.approxPolyDP(c, 1.3, True)[:, 0, :]
        out.append(c)
    return out


def main():
    folder = sys.argv[1]
    old = old_polygons(open(sys.argv[2], encoding='utf-8').read()) if len(sys.argv) > 2 else {}
    inv, err = fit()
    print('georeference: max error %.0f px, mean %.1f px (1 px = %.1f m)' % (err.max(), err.mean(), 1 / 0.325))

    imgs = [load(folder, h[0]) for h in HOODS]
    base = np.median(np.stack(imgs), axis=0).astype(np.int16)
    raw = [raw_mask(im, base) for im in imgs]
    shape = raw[0].shape

    # one label per pixel: smooth each mask, let the strongest win, and let the
    # neighbourhoods grow into the street-wide gaps between them
    union = ndi.binary_closing(np.any(raw, axis=0), iterations=9)
    union = ndi.binary_fill_holes(union)
    soft = np.stack([ndi.gaussian_filter(m.astype(np.float32), 3.0) for m in raw])
    dist = np.stack([ndi.distance_transform_edt(~m) for m in raw])
    label = np.where(soft.max(axis=0) > 0.5, soft.argmax(axis=0), dist.argmin(axis=0))
    label = np.where(union, label, -1)

    px_m2 = (1 / 0.325) ** 2
    hoods = []
    for i, (stem, name, article, district) in enumerate(HOODS):
        m = label == i
        m = ndi.binary_opening(m, iterations=2)
        entry = {'name': name, 'article': article, 'district': district,
                 'area_km2': round(float(m.sum()) * px_m2 / 1e6, 3),
                 'rings': [to_geo(inv, r) for r in rings(m)]}
        if name in SPLIT:
            subs = SPLIT[name]
            d = np.stack([ndi.distance_transform_edt(~poly_mask(old[s], shape)) for s, _ in subs])
            near = d.argmin(axis=0)
            entry['parts'] = [{'name': s, 'article': a,
                               'rings': [to_geo(inv, r) for r in rings(m & (near == j))]}
                              for j, (s, a) in enumerate(subs)]
        hoods.append(entry)
        # compare with the previous polygons
        best = max(((float((m & poly_mask(p, shape)).sum()) / float((m | poly_mask(p, shape)).sum()), n)
                    for n, p in old.items()), default=(0, None))
        print('%-16s %.2f km2   closest old: %s (IoU %.2f)' % (name, entry['area_km2'], best[1], best[0]))

    # who touches whom
    grown = [ndi.binary_dilation(label == i, iterations=5) for i in range(len(HOODS))]
    for i, h in enumerate(hoods):
        h['neighbours'] = [hoods[j]['name'] for j in range(len(HOODS))
                           if j != i and (grown[i] & grown[j]).sum() > 60]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({'source': 'שכבות השכונות של עירית רותם על מפת המושבה הרשמית',
               'hoods': hoods}, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('->', OUT)


if __name__ == '__main__':
    main()
