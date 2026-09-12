#!/usr/bin/env python3
"""Render "מפה סכמטית - <שכונה>.png" for neighbourhoods that lack one.

The clickable map on [[רשימת שכונות]] carries every neighbourhood's outline as
a CSS clip-path polygon, in percentages of a positioned div. That is the only
place the polygons survive (the polygons.json of 19/8/2026 was never
committed), so this reads them back from the wiki page.

The per-neighbourhood maps all share one grey base with a single region
painted red. The base itself was never uploaded, but the per-pixel median of
the nine existing maps *is* the base — every pixel is grey in at least five
of them. The red fill and dark outline are sampled from an existing map so a
new one is indistinguishable in style.

Usage:
    python3 hood_maps.py --list                 # polygons found, with/without a map
    python3 hood_maps.py "קורן" "הנחל" ...      # render + upload those
    python3 hood_maps.py --all --dry-run        # render everything missing, no upload
"""

import argparse
import io
import os
import re
import sys

import numpy as np
import requests
from PIL import Image, ImageDraw

from wiki_client import WikiClient, API_URL

LIST_PAGE = 'רשימת שכונות'
FILE_PREFIX = 'מפה סכמטית - '
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hood_maps')

FILL = (160, 112, 112)          # the dusty red of the existing maps
OUTLINE = (70, 45, 45)
OUTLINE_W = 3


def polygons_from_page(text: str) -> dict:
    """name -> list of (x, y) in fractions of the whole image."""
    out = {}
    for m in re.finditer(
            r'<div class="hood[^"]*" data-hood="([^"]+)" style="([^"]+)"', text):
        name = m.group(1).replace('&quot;', '"')
        style = m.group(2)
        box = {k: float(v) / 100 for k, v in
               re.findall(r'(left|top|width|height):([\d.]+)%', style)}
        poly = re.search(r'clip-path:polygon\(([^)]+)\)', style)
        if not poly or len(box) < 4:
            continue
        pts = []
        for pair in poly.group(1).split(','):
            x, y = [float(v.strip('% ')) / 100 for v in pair.strip().split()]
            pts.append((box['left'] + x * box['width'],
                        box['top'] + y * box['height']))
        out[name] = pts
    return out


def file_url(client, name: str):
    r = client.session.get(API_URL, params={
        'action': 'query', 'titles': 'קובץ:' + name,
        'prop': 'imageinfo', 'iiprop': 'url', 'format': 'json'}).json()
    page = next(iter(r['query']['pages'].values()))
    return (page.get('imageinfo') or [{}])[0].get('url')


def existing_maps(client, names) -> dict:
    """name -> PIL image, for neighbourhoods that already have a map."""
    found = {}
    for n in names:
        url = file_url(client, FILE_PREFIX + n + '.png')
        if url:
            found[n] = Image.open(io.BytesIO(
                requests.get(url, timeout=60).content)).convert('RGBA')
    return found


def grey_base(maps: dict) -> Image.Image:
    size = next(iter(maps.values())).size
    stack = np.stack([np.asarray(im.resize(size)) for im in maps.values()])
    return Image.fromarray(np.median(stack, axis=0).astype(np.uint8))


def render(base: Image.Image, poly) -> Image.Image:
    im = base.copy()
    w, h = im.size
    pts = [(x * w, y * h) for x, y in poly]
    d = ImageDraw.Draw(im)
    d.polygon(pts, fill=FILL + (255,))
    d.line(pts + [pts[0]], fill=OUTLINE + (255,), width=OUTLINE_W, joint='curve')
    return im


def upload(client, name: str, im: Image.Image) -> str:
    fn = FILE_PREFIX + name + '.png'
    buf = io.BytesIO()
    im.save(buf, 'PNG', optimize=True)
    desc = ('מיקום שכונת %s (באדום) בתוך השטח הבנוי של פרדס חנה-כרכור, '
            'על גבי מפה סכמטית של השכונות.\n\n'
            'המפה נגזרה מגבולות השכונות שבמפת המושבה הרשמית, כפי שסומנו '
            'ב[[רשימת שכונות]]. נוצרה אוטומטית על ידי הבוט.\n\n'
            '[[קטגוריה:שכונות]]\n' % name)
    r = client.session.post(API_URL, data={
        'action': 'upload', 'filename': fn,
        'comment': 'מפה סכמטית לשכונת %s' % name,
        'text': desc, 'token': client._csrf_token(),
        'ignorewarnings': '1', 'format': 'json'},
        files={'file': (fn, buf.getvalue(), 'image/png')})
    j = r.json()
    return j.get('upload', {}).get('result') or str(j.get('error'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*')
    ap.add_argument('--all', action='store_true', help='every polygon without a map')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='render to hood_maps/ only')
    args = ap.parse_args()

    client = WikiClient()
    polys = polygons_from_page(client.get_page(LIST_PAGE)['wikitext'])
    have = existing_maps(client, polys)
    if args.list:
        for n in polys:
            print('%s  %s' % ('✓' if n in have else '✗', n))
        return 0

    targets = [n for n in polys if n not in have] if args.all else args.names
    missing = [n for n in targets if n not in polys]
    if missing:
        sys.exit('אין מצולע במפה עבור: %s' % ', '.join(missing))
    if not targets:
        print('אין מה לרנדר.')
        return 0
    if len(have) < 3:
        sys.exit('צריך לפחות שלוש מפות קיימות כדי לשחזר את הבסיס')

    base = grey_base(have)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.dry_run:
        client.login()
    for n in targets:
        im = render(base, polys[n])
        path = os.path.join(OUT_DIR, FILE_PREFIX + n + '.png')
        im.save(path)
        print('%s -> %s%s' % (n, path, '' if args.dry_run else
                                 '  upload: ' + upload(client, n, im)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
