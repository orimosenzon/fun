"""The ownership layer (בעלות קרקע), in a real Chrome.

web/data/ownership.json (build_ownership.py) is the parcels with the land
registry's kind of owner on each. This checks:
  * nothing is downloaded until the layer is switched on;
  * switched on, the whole moshava is coloured, and קק"ל's forest is drawn;
  * the key lists the kinds of owner and the forest;
  * a tap on a parcel of the council's opens it as the council's, with the
    link to order an extract;
  * with the parcels on as well, the parcel's card says who owns it;
  * switching it off hides it.
"""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8773
URL = f'http://127.0.0.1:{PORT}/index.html'
SPOT = [34.970436, 32.475224]      # gush 10102 parcel 17, the council's, by the כיכר
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
     pg.on('request', lambda r: asked.append(r.url) if 'ownership.json' in r.url else None)
     pg.goto(URL)
     pg.wait_for_function('() => typeof Layers !== "undefined" && typeof Ownership !== "undefined"'
                          ' && typeof map !== "undefined" && !!map.getSource("src-trails")', timeout=40000)
     pg.wait_for_timeout(2500)
     pg.evaluate('''() => { document.querySelectorAll('.sheet').forEach((x) => { x.hidden = true; });
       if (map.getTerrain()) map.setTerrain(null); }''')

     lay = pg.evaluate('() => { const l = Layers.byId("ownership"); return l && {on: l.on, grid: l.grid, name: l.name}; }')
     check('the layer exists, off, named בעלות קרקע', lay and not lay['on'] and lay['name'] == 'בעלות קרקע', lay)
     check('nothing is downloaded while it is off', not asked, asked)

     pg.evaluate('() => { map.jumpTo({center: [34.975, 32.475], zoom: 13.2, pitch: 0, bearing: 0}); Layers.turnOn("ownership"); }')
     pg.wait_for_function('() => map.getSource("grd-ownership") && map.isSourceLoaded("grd-ownership")', timeout=30000)
     pg.wait_for_timeout(2500)
     far = pg.evaluate('''() => {
       const fs = map.queryRenderedFeatures({ layers: ['pgf-ownership'] });
       const ids = new Set(fs.map((f) => f.id)), kinds = {};
       fs.forEach((f) => { kinds[f.properties.o] = (kinds[f.properties.o] || 0) + 1; });
       return { drawn: ids.size, kinds, forest: map.queryRenderedFeatures({ layers: ['pgd-ownership'] }).length };
     }''')
     print('  at z13.2:', far)
     check('downloaded once', len(asked) == 1, asked)
     check('the whole moshava is coloured', far['drawn'] > 8500, far)
     check('the council, the state and private ground are all there',
           all(k in far['kinds'] for k in 'mip'), far['kinds'])
     check('leased state ground is its own colour', far['kinds'].get('l', 0) > 1000, far['kinds'])
     check('קק"ל\'s forest is drawn', far['forest'] > 0, far)

     pg.evaluate("() => Layers.setLegendOpen(true)")
     pg.wait_for_timeout(300)
     key = pg.evaluate('() => document.getElementById("legend-body").textContent.replace(/\\s+/g, " ")')
     check('the key names the council, מקרקעי ישראל and the forest',
           'המועצה המקומית' in key and 'מקרקעי ישראל' in key and 'יער בניהול קק"ל' in key, key[:200])
     # The key's patches are the map's colours at the map's strength (Ori,
     # 21/10/2026: they were the colours at full strength, and did not match).
     tint = pg.evaluate("""() => [...document.querySelectorAll('#legend-body .lg-dot.area.tint')]
       .map((e) => getComputedStyle(e).backgroundColor)""")
     check('the key has a tinted patch per kind, leased among them',
           len(tint) == 7 and 'בחכירה לדורות' in key, tint)
     pg.screenshot(path=os.path.join(OUT, 'shot_ownership_far.png'))

     pg.evaluate(f'() => map.jumpTo({{center: {SPOT}, zoom: 17}})')
     pg.wait_for_timeout(2500)
     at = pg.evaluate(f'''() => {{ const p = map.project({SPOT}); const r = map.getCanvas().getBoundingClientRect();
       const under = map.queryRenderedFeatures(p, {{ layers: ['pgf-ownership'] }})[0];
       return {{ x: p.x + r.left, y: p.y + r.top, g: under && under.properties.g, p: under && under.properties.p,
                o: under && under.properties.o }}; }}''')
     print('  under the spot:', at)
     check('the spot is gush 10102 parcel 17, the council\'s', at['g'] == 10102 and at['p'] == 17 and at['o'] == 'm', at)
     pg.mouse.click(at['x'], at['y'])
     pg.wait_for_timeout(1500)
     got = pg.evaluate('''() => {
       const d = document.getElementById('detail-view');
       const l = Layers.byId('ownership');
       return { sel: selectedId, pane: d && !d.hidden, text: d ? d.textContent.replace(/\\s+/g, ' ') : '',
                items: l.waypoints.length, url: l.waypoints[0] && l.waypoints[0].url };
     }''')
     print('  after the tap:', json.dumps(got, ensure_ascii=False)[:300])
     check('the tap selects that parcel', got['sel'] == 'ownership-101020017', got['sel'])
     check('the pane says it is the council\'s', got['pane'] and 'המועצה המקומית' in got['text'], got['text'][:120])
     check('and links to ordering an extract', 'land_registration_extract' in (got['url'] or ''), got['url'])
     pg.screenshot(path=os.path.join(OUT, 'shot_ownership_tap.png'))

     # With the parcels on, the parcel's own card carries the owner.
     pg.evaluate('() => Layers.turnOn("parcels")')
     pg.wait_for_function('() => map.getSource("grd-parcels") && map.isSourceLoaded("grd-parcels")', timeout=30000)
     pg.wait_for_timeout(1500)
     pg.mouse.click(at['x'], at['y'])
     pg.wait_for_timeout(1500)
     card = pg.evaluate('() => ({ sel: selectedId, text: document.getElementById("detail-view").textContent.replace(/\\s+/g, " ") })')
     check('with the parcels on, the parcel card names the owner',
           card['sel'] == 'parcel-10102-17' and 'הבעלות בחלקה לפי הטאבו: המועצה המקומית' in card['text'], card['text'][:160])

     pg.evaluate('() => { Layers.byId("ownership").on = false; Layers.applyVisibility(); }')
     pg.wait_for_timeout(800)
     off = pg.evaluate('() => map.getLayoutProperty("pgf-ownership", "visibility")')
     check('switching it off hides it', off == 'none', off)

     check('no page errors', not errs, errs[:3])
     b.close()
finally:
    srv.terminate()

print()
print(f'{len(fails)} failed' if fails else 'all passed')
for f in fails: print('  - ' + f)
sys.exit(1 if fails else 0)
