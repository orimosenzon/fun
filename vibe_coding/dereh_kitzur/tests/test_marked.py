"""The marked trails layer: every marked hiking trail in Israel, in a real Chrome.

Drawn as a grid from web/data/marked.json (build_marked.py), like the parcels.
This checks:
  * nothing is downloaded until the layer is switched on;
  * switched on, the trails are drawn, white casing and colour both, under
    the shortcuts, and שביל ישראל among them;
  * a tap on a trail opens it in the pane with a name, a length and links,
    lights it, and names the other trails at the same spot;
  * a link to a trail (?sel=marked-<id>) opens it again;
  * switching the layer off hides it.
Uses ?local, since the data repo copy is what --publish writes.
"""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8773
URL = f'http://127.0.0.1:{PORT}/index.html?local'
INT = 282071                     # שביל ישראל

doc = json.load(open(os.path.join(WEB, 'data', 'marked.json'), encoding='utf-8'))
trail = next(f for f in doc['features'] if f['properties']['id'] == INT)
line = trail['geometry']['coordinates']
line = max(line, key=len) if trail['geometry']['type'] == 'MultiLineString' else line
# A point on the Israel Trail on the Carmel, mid-segment so the tap is on the line.
a, b = min(zip(line, line[1:]), key=lambda s: abs(s[0][1] - 32.70) + abs(s[0][0] - 35.0))
SPOT = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]

fails = []
def check(name, ok, detail=''):
    print(('  ok   ' if ok else '  FAIL ') + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
  with sync_playwright() as p:
     b = p.chromium.launch(channel='chrome', args=['--use-gl=angle', '--use-angle=gl'])
     ctx = b.new_context(viewport={'width': 1200, 'height': 900}, locale='he-IL')
     ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); } catch (e) {}")
     pg = ctx.new_page()
     errs, asked = [], []
     pg.on('pageerror', lambda e: errs.append('pageerror: ' + str(e)))
     pg.on('console', lambda m: errs.append('console: ' + m.text) if m.type == 'error' else None)
     pg.on('request', lambda r: asked.append(r.url) if 'marked.json' in r.url else None)
     pg.goto(URL)
     pg.wait_for_function('() => typeof Layers !== "undefined" && typeof Marked !== "undefined"'
                          ' && typeof map !== "undefined" && !!map.getSource("src-trails")', timeout=40000)
     pg.wait_for_timeout(2500)
     pg.evaluate('''() => { document.querySelectorAll('.sheet').forEach((x) => { x.hidden = true; });
       if (map.getTerrain()) map.setTerrain(null); }''')

     lay = pg.evaluate('() => { const l = Layers.byId("marked"); return l && {on: l.on, grid: l.grid, cat: l.category}; }')
     check('the layer exists, off, with a grid file', lay and not lay['on'] and lay['grid'], lay)
     check('nothing is downloaded while it is off', not asked, asked)

     # The whole north, from far out.
     pg.evaluate('() => { map.jumpTo({center: [35.2, 32.8], zoom: 9, pitch: 0, bearing: 0}); Layers.turnOn("marked"); }')
     pg.wait_for_function('() => map.getSource("grd-marked") && map.isSourceLoaded("grd-marked")', timeout=40000)
     pg.wait_for_timeout(3000)
     far = pg.evaluate('''() => {
       const f = map.queryRenderedFeatures({ layers: ['pgl-marked', 'pgd-marked'] });
       const ids = new Set(f.map((x) => x.properties.id));
       const colours = new Set(f.map((x) => x.properties.m));
       const ids2 = map.getStyle().layers.map((l) => l.id);
       return { drawn: ids.size, int: ids.has(282071), colours: [...colours],
                casing: map.queryRenderedFeatures({ layers: ['pgp-marked'] }).length,
                grid: ids2.indexOf('pgl-marked'), trail: ids2.indexOf('ln-trails') };
     }''')
     print('  at z9 over the north:', {k: v for k, v in far.items() if k != 'colours'}, far['colours'])
     check('downloaded once', len(asked) == 1, asked)
     check('hundreds of trails drawn', far['drawn'] > 300, far)
     check('the Israel Trail is among them', far['int'], far)
     check('all four committee colours', all(c in far['colours'] for c in ('red', 'blue', 'green', 'black')), far)
     check('with their white casing', far['casing'] > 0, far)
     check('drawn under the shortcuts', 0 <= far['grid'] < far['trail'], far)
     pg.screenshot(path=os.path.join(OUT, 'shot_marked_north.png'))

     # Close in on the Carmel and tap the Israel Trail.
     pg.evaluate(f'() => map.jumpTo({{center: {SPOT}, zoom: 15}})')
     pg.wait_for_timeout(3000)
     pg.screenshot(path=os.path.join(OUT, 'shot_marked_carmel.png'))
     at = pg.evaluate(f'''() => {{ const p = map.project({SPOT}); const r = map.getCanvas().getBoundingClientRect();
       return {{ x: p.x + r.left, y: p.y + r.top }}; }}''')
     pg.mouse.click(at['x'], at['y'])
     pg.wait_for_timeout(1500)
     got = pg.evaluate('''() => {
       const d = document.getElementById('detail-view');
       const l = Layers.byId('marked');
       return { open: !d.hidden, title: (d.querySelector('h2') || {}).textContent,
                text: d.textContent.replace(/\\s+/g, ' ').slice(0, 900),
                seg: l.segments.length && l.segments[0].id,
                links: [...d.querySelectorAll('a.act-link')].map((a) => a.href),
                lit: map.queryRenderedFeatures({ layers: ['ln-marked'] }).length,
                url: location.search };
     }''')
     print('  tapped:', got['title'], '|', got['text'][:400])
     check('the pane opens on the Israel Trail', got['open'] and got['seg'] == f'marked-{INT}', got)
     check('it is named', 'שביל ישראל' in (got['title'] or ''), got['title'])
     check('with its length in km', 'ק"מ' in got['text'], got['text'][:200])
     check('and how it is marked on the ground', 'לבן, כחול וכתום' in got['text'], got['text'][:300])
     check('links to Israel Hiking and OSM',
           any('israelhiking' in u for u in got['links']) and any('openstreetmap.org/relation' in u for u in got['links']), got['links'])
     check('the trail is lit on the map', got['lit'] > 0, got)
     check('the address names it', f'marked-{INT}' in got['url'], got['url'])
     pg.screenshot(path=os.path.join(OUT, 'shot_marked_tap.png'))

     # A link opens it again.
     link = pg.evaluate('() => location.href')
     pg2 = ctx.new_page()
     pg2.on('pageerror', lambda e: errs.append('pageerror(link): ' + str(e)))
     pg2.goto(link)
     pg2.wait_for_function(f'() => typeof Layers !== "undefined" && Layers.byId("marked")'
                           f' && Layers.byId("marked").segments.length'
                           f' && !document.getElementById("detail-view").hidden', timeout=40000)
     t2 = pg2.evaluate('() => document.querySelector("#detail-view h2").textContent')
     check('a link opens the same trail', 'שביל ישראל' in t2, t2)
     pg2.close()

     # Off.
     pg.evaluate('() => { Layers.byId("marked").on = false; Layers.applyVisibility(); }')
     pg.wait_for_timeout(800)
     vis = pg.evaluate('() => [map.getLayoutProperty("pgl-marked", "visibility"), map.getLayoutProperty("pgd-marked", "visibility")]')
     check('switching it off hides it', vis == ["none", "none"], vis)

     # Under ?local the trail photos 404: they live in the data repo only.
     errs = [e for e in errs if 'status of 404' not in e]
     check('no page errors', not errs, errs)
     b.close()
finally:
  srv.terminate()

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
