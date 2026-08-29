#!/usr/bin/env python3
"""אמנות במושבה - the festival's own map, turned into a layer of this one.

`אמנות במושבה` is Pardes Hanna-Karkur's art, design and craft festival. Its
site publishes a map of everywhere you can walk to during the festival - the
artists opening their studios, the exhibitions, the food, the venues - and that
map is a WordPress plugin whose whole dataset is base64'd into the page. So this
reads it rather than a feed, because there is no feed.

What comes out is `data/art2026.json`, shaped exactly like `places.json`: the
app already knows how to draw a layer of places, and a festival is one more of
those. See `layers.js`.

Photos are left as absolute URLs on the festival's own site rather than copied
into the data repo. `Store.asset()` passes an absolute URL straight through,
which is the same thing the pardespedia places do with the wiki's images. The
festival owns these pictures; pointing at them keeps them theirs, and keeps a
one-off event out of a repo that is meant to outlast it.

    python3 build_art.py            # rebuild data/art2026.json
    python3 build_art.py --no-net   # from the cache, without touching the site

The artist pages are fetched once each and cached under .cache/art/, so a rerun
costs nothing and the site sees one visit per artist rather than one per run.
"""

import base64
import collections
import html
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

# The pile-of-pins problem is not new here, and neither is the answer to it.
from build_places import spread

MAP_URL = 'https://www.pardesart.co.il/map/'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web', 'data', 'art2026.json')
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'art')

UA = 'derech-kitzur/1.0 (+https://orimosenzon.github.io/fun/vibe_coding/dereh_kitzur/)'
PAUSE = 0.4                      # between artist pages; this is somebody's server

# The four categories the festival sorts its map into, with the colours this app
# draws them in. Their own map uses one pin shape per category and no colour at
# all, so these are ours - picked to stay apart from the trail greens and from
# the purple the pardespedia places already own.
GROUPS = [
    {'name': 'אמנים', 'color': '#c2185b'},
    {'name': 'תערוכות', 'color': '#00838f'},
    {'name': 'אוכל', 'color': '#ef6c00'},
    {'name': 'עסקים', 'color': '#5d4037'},
]

# Everything on wp-content that is furniture rather than content.
CHROME = re.compile(r'cropped-fav|/logo|favicon|placeholder|sprite|icon[-_]', re.I)

# WordPress writes its resized copies as name-800x600.ext next to the original,
# and `-scaled` is what it calls the copy it makes when an upload is bigger than
# it wants to serve. Neither is a different picture.
SIZED = re.compile(r'-(\d{2,4})x(\d{2,4})(?=\.[a-z]{3,4}$)')
SCALED = re.compile(r'-scaled(?=\.[a-z]{3,4}$)')

# A photo on one artist's page is that artist's work. The same photo on twenty
# of them is the sponsors' strip down the side of the theme, and no amount of
# pattern-matching on filenames tells the two apart - the ads are named in
# Hebrew after the businesses that bought them, exactly like the art is named
# after the artists. How often it recurs does tell them apart.
SHARED_BY = 3


def get(url, cache_key=None):
    """Fetch a page, remembering it. Returns None rather than raising: one
    artist page that will not load should cost that artist's gallery, not the
    whole run."""
    path = os.path.join(CACHE, cache_key) if cache_key else None
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            return fh.read()
    if NO_NET:
        return None
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            body = res.read().decode('utf-8', 'replace')
    except Exception as err:                       # noqa: BLE001 - report and go on
        print(f'  ! {url}: {err}', file=sys.stderr)
        return None
    if path:
        os.makedirs(CACHE, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(body)
    time.sleep(PAUSE)
    return body


def slug(url):
    """A filename for the cache, from the artist's own URL."""
    name = urllib.parse.unquote(url.rstrip('/').rsplit('/', 1)[-1])
    name = unicodedata.normalize('NFC', name)
    return re.sub(r'[^\w֐-׿-]', '_', name)[:80] + '.html'


# Invisible direction controls: marks, embeddings, overrides and isolates.
# Whoever typed the phone numbers into the festival's admin pasted them out of
# something that wrapped each one in an override, so "054-8830630" arrives as
# U+202D 054-8830630 U+202C. They are invisible and they are not inert: an
# unbalanced override leaks into everything drawn after it, and this text lands
# in a Hebrew page next to Latin URLs and phone numbers. Out they go, and the
# page's own `dir` decides.
BIDI = re.compile('[‎‏‪-‮⁦-⁩]')


def text(raw):
    """The wiki-ish prose out of a WordPress field: tags gone, entities
    resolved, the runs of whitespace WordPress leaves behind collapsed."""
    if not raw:
        return ''
    s = re.sub(r'<br\s*/?>|</p>|</div>', '\n', raw, flags=re.I)
    s = re.sub(r'<[^>]+>', '', s)
    s = html.unescape(s)
    s = BIDI.sub('', s).replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r'[ \t\xa0]+', ' ', s)
    s = re.sub(r'\n\s*\n\s*\n+', '\n\n', s)
    return s.strip()


def base_of(url):
    """The original behind one of WordPress's resized copies."""
    return SCALED.sub('', SIZED.sub('', url))


def key_of(url):
    """What makes two URLs the same photograph. The theme serves most pictures
    as both .jpg and .webp, so the extension is part of the rendition rather
    than part of the identity."""
    return base_of(url).rsplit('.', 1)[0]


def urls_on(page):
    """Every uploaded picture a page points at, in the order it names them."""
    if not page:
        return []
    found = re.findall(r'https://www\.pardesart\.co\.il/wp-content/uploads/[^"\'\s\\)]+?'
                       r'\.(?:jpe?g|png|webp)', page)
    out = []
    for url in found:
        if not CHROME.search(url) and url not in out:
            out.append(url)
    return out


def pick(variants):
    """One photo out of its renditions: a mid-size copy for the strip in the
    detail pane, the biggest for the lightbox.

    Preference goes to webp over jpeg at the same size, since the theme emits
    both and the webp is a third of the weight over a phone connection."""
    def width(url):
        m = SIZED.search(url)
        return int(m.group(1)) if m else 10000        # no -WxH means the original

    def rank(url):
        return (width(url), url.endswith('.webp'))

    ordered = sorted(variants, key=rank)
    full = ordered[-1]
    mid = next((v for v in ordered if width(v) >= 400), full)
    return {'thumb': mid, 'full': full}


def photos_from(page, common, lead=None):
    """The pictures that belong to this place, lead image first.

    `lead` is the featured image the map itself carries, which is the one the
    festival chose to represent the place - so it opens the strip even though
    the page lists it somewhere in the middle.

    The lead is also the one image the shared-across-pages rule must not touch.
    A featured image is exactly what the theme reprints in every "more artists"
    strip, so the pictures the festival picked as most representative are the
    ones that recur most - and dropping them would leave each place illustrated
    by everything except its own portrait."""
    lead_key = key_of(lead) if lead else None
    groups = {}
    for url in ([lead] if lead else []) + urls_on(page):
        key = key_of(url)
        if key in common and key != lead_key:
            continue
        groups.setdefault(key, set()).add(url)

    return [pick(v) for v in groups.values()]


def links_for(fields):
    """Everywhere this place asks to be followed to.

    The festival's own page is not among them: it goes in `url`, which the
    detail pane already renders at the head of the list as the place's source.
    Waze is left out too - the app has its own navigation, and that one routes
    through the shortcuts, which is the entire point of putting this layer
    here."""
    out = []
    site = text(fields.get('%website%'))
    if site:
        out.append({'url': site, 'title': 'האתר'})
    fb = text(fields.get('%facebook_page%'))
    if fb:
        out.append({'url': fb, 'title': 'פייסבוק'})
    ig = text(fields.get('%instagram_facebook%'))
    if ig:
        out.append({'url': ig, 'title': 'אינסטגרם'})
    return out


def note_for(fields):
    """What the festival wrote about this place.

    `post_content` is the full text and `post_excerpt` is WordPress's own
    truncation of it, ending in an ellipsis, so the whole thing comes first.
    The kitchens are filled in on a different form from the studios and get no
    body text at all - one line under `food_sub` is everything the festival
    says about a restaurant, and one line beats none."""
    for field in ('post_content', 'post_excerpt', '%food_sub%'):
        body = text(fields.get(field))
        if body:
            return body
    return ''


def craft_of(fields):
    """The medium, as the festival's own taxonomy has it: אמנות | ציור.

    Two taxonomies, coarse and fine, and the festival prints both together at
    the top of every artist's page - so both, in that order, and neither twice
    when a craft is its own heading."""
    parts = []
    for field in ('taxonomy=artist_maincrafts', 'taxonomy=artist_crafts'):
        value = text(fields.get(field))
        if value and value not in parts:
            parts.append(value)
    return ' | '.join(parts)


def featured(snippet):
    """The picture the festival hangs on a pin. It arrives as a snippet of HTML
    rather than a URL, because the plugin renders it straight into the popup."""
    src = re.search(r'src=[\'"]([^\'"]+)[\'"]', snippet or '')
    if not src:
        return None
    url = html.unescape(src.group(1))
    return None if CHROME.search(url) else url


def build():
    page = get(MAP_URL, 'map.html')
    if not page:
        raise SystemExit('לא הצלחתי להוריד את דף המפה.')
    blob = re.search(r'window\.wpgmp\.mapdata2\s*=\s*"([A-Za-z0-9+/=]+)"', page)
    if not blob:
        raise SystemExit('מבנה הדף השתנה: לא נמצא wpgmp.mapdata2.')
    data = json.loads(base64.b64decode(blob.group(1)).decode('utf-8'))

    # Every page first, then the pictures. Which images are this artist's work
    # and which are the theme's sponsor strip can only be told by looking across
    # all of them at once, so the pages are collected before any are read.
    pages = {}
    for raw in data['places']:
        link = ((raw['location'].get('extra_fields') or {}).get('post_link') or '').strip()
        if link and link not in pages:
            pages[link] = get(link, slug(link))

    common = {key for key, n in collections.Counter(
        key_of(url) for page in pages.values() for url in urls_on(page)
    ).items() if n >= SHARED_BY}
    print(f'{len(pages)} דפים, {len(common)} תמונות משותפות שאינן תוכן')

    places = []
    seen_ids = set()
    for raw in data['places']:
        fields = raw['location'].get('extra_fields', {}) or {}
        title = text(raw['title'])
        groups = [c['name'] for c in raw.get('categories', [])]
        group = next((g['name'] for g in GROUPS if g['name'] in groups), 'אמנים')

        post_link = text(fields.get('post_link'))
        # Two shapes in one file: a place with a page behind it, and a pin an
        # organiser typed straight into the map. They carry different fields.
        if raw.get('source') == 'post':
            address = text(fields.get('%address%'))
            # Half the festival happens inside two shared compounds, and the
            # compound's name is the half of the address a visitor navigates by.
            where = text(fields.get('%address_desc%'))
            if where and where not in address:
                address = f'{where}, {address}' if address else where
            phone = text(fields.get('%phone%'))
            note = note_for(fields)
            lead = featured(fields.get('post_featured_image'))
            photos = photos_from(pages.get(post_link), common, lead)
        else:
            address = text(fields.get('maddress'))
            phone = text(fields.get('mphone'))
            note = text(fields.get('mdesc'))
            # A hand-typed pin has no page behind it, so its popup image is all
            # there is.
            photos = photos_from(None, common, featured(raw['location'].get('marker_image')))

        # `%number%` looks like a house number and is not one - it is the order
        # the festival lists people in, and appears nowhere on its own site. It
        # stays out rather than being shown to somebody standing in a street
        # looking for a door.

        ident = 'art-' + re.sub(r'[^\w֐-׿-]', '_', title)[:60]
        while ident in seen_ids:
            ident += '_'
        seen_ids.add(ident)

        place = {
            'id': ident,
            'name': title,
            'group': group,
            'cats': [],
            'note': note,
            'url': post_link or MAP_URL,
            'links': links_for(fields),
            'photos': photos,
            'geo': {
                'lat': round(float(raw['location']['lat']), 6),
                'lng': round(float(raw['location']['lng']), 6),
                # The festival placed these pins itself, on its own map, so they
                # are not this app's to re-guess or to drag about.
                'source': 'festival',
            },
        }
        craft = craft_of(fields)
        if craft:
            place['craft'] = craft
        if address:
            place['address'] = address
        if phone:
            place['phone'] = phone
        acc = text(fields.get('%acc%'))
        if acc:
            place['access'] = acc
        # An empty string here means "not open Saturday"; the field only says
        # anything when it says yes.
        sat = text(fields.get('%saurday%'))
        if sat:
            place['saturday'] = sat
        places.append(place)

    # Half the festival happens in two shared yards, so a third of these pins
    # land on somebody else's exact coordinates - ten studios in one compound
    # get the compound's point. Their own map hides that behind a spiderfier;
    # this one puts a pile onto a ring around itself, the same way the several
    # dozen businesses sharing a pardespedia address are handled.
    spread(places)

    places.sort(key=lambda p: (p['group'], p['name']))
    doc = {
        'version': 1,
        'updated': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source': MAP_URL,
        'name': 'אמנות במושבה 2026',
        'groups': GROUPS,
        'places': places,
        'stats': {
            'places': len(places),
            'photos': sum(len(p['photos']) for p in places),
        },
    }
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
        fh.write('\n')

    by_group = {}
    for p in places:
        by_group[p['group']] = by_group.get(p['group'], 0) + 1
    print(f'{OUT}: {len(places)} מקומות, {doc["stats"]["photos"]} תמונות')
    for name, n in sorted(by_group.items(), key=lambda kv: -kv[1]):
        print(f'  {name}: {n}')
    missing = [p['name'] for p in places if not p['photos']]
    if missing:
        print(f'  בלי תמונה: {len(missing)} ({", ".join(missing[:5])}…)')
    empty = [p['name'] for p in places if not p['note']]
    if empty:
        print(f'  בלי טקסט: {len(empty)} ({", ".join(empty[:5])}…)')


NO_NET = '--no-net' in sys.argv

if __name__ == '__main__':
    build()
