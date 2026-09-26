"""The parcels layer: every parcel in the moshava, at every zoom, in a real Chrome.

Since 25/9/2026 the parcels are a grid drawn from web/data/parcels.json
(build_parcels.py), not items fetched from govmap per viewport. This checks:
  * nothing is downloaded until the layer is switched on;
  * switched on far out, the whole moshava is covered;
  * the grid sits under the shortcuts, and the numbers appear close in;
  * a tap anywhere names the parcel it landed in, opens it in the pane and
    lights it - and the list does not fill with nine thousand rows;
  * switching it off hides the grid.
"""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8772
URL = f'http://127.0.0.1:{PORT}/index.html'
SPOT = [34.97507, 32.47411]        # the open space: gush 10102, parcel 110
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
     pg.on('request', lambda r: asked.append(r.url) if 'parcels.json' in r.url else None)
     pg.goto(URL)
     pg.wait_for_function('() => typeof Layers !== "undefined" && typeof Parcels !== "undefined"'
                          ' && typeof map !== "undefined" && !!map.getSource("src-trails")', timeout=40000)
     pg.wait_for_timeout(2500)
     pg.evaluate('''() => { document.querySelectorAll('.sheet').forEach((x) => { x.hidden = true; });
       if (map.getTerrain()) map.setTerrain(null); }''')

     lay = pg.evaluate('() => { const l = Layers.byId("parcels"); return l && {on: l.on, n: l.waypoints.length, grid: l.grid}; }')
     check('the layer exists, off, with a grid file', lay and not lay['on'] and lay['grid'], lay)
     check('nothing is downloaded while it is off', not asked and not pg.evaluate('() => !!map.getSource("grd-parcels")'), asked)

     # Far out: the whole moshava.
     pg.evaluate(f'() => {{ map.jumpTo({{center: [34.975, 32.475], zoom: 13.2, pitch: 0, bearing: 0}}); Layers.turnOn("parcels"); }}')
     pg.wait_for_function('() => map.getSource("grd-parcels") && map.isSourceLoaded("grd-parcels")', timeout=30000)
     pg.wait_for_timeout(2500)
     far = pg.evaluate('''() => {
       const seen = new Set(map.queryRenderedFeatures({ layers: ['pgl-parcels'] }).map((f) => f.id));
       return { drawn: seen.size, vis: map.getLayoutProperty('pgl-parcels', 'visibility') };
     }''')
     print('  at z13.2:', far)
     check('downloaded once', len(asked) == 1, asked)
     check('the whole moshava is drawn far out', far['drawn'] > 8500, far)
     check('and it is visible', far['vis'] != 'none', far)

     order = pg.evaluate('''() => { const ids = map.getStyle().layers.map((l) => l.id);
       return { grid: ids.indexOf('pgl-parcels'), trail: ids.indexOf('ln-trails') }; }''')
     check('drawn under the shortcuts', 0 <= order['grid'] < order['trail'], order)

     # Close in: numbers, and a tap names the parcel.
     pg.evaluate(f'() => map.jumpTo({{center: {SPOT}, zoom: 17.5}})')
     pg.wait_for_timeout(3000)
     nums = pg.evaluate('() => map.queryRenderedFeatures({ layers: ["pgn-parcels"] }).length')
     check('parcel numbers show close in', nums > 5, nums)

     at = pg.evaluate(f'''() => {{ const p = map.project({SPOT}); const r = map.getCanvas().getBoundingClientRect();
       const under = map.queryRenderedFeatures(p, {{ layers: ['pgf-parcels'] }})[0];
       return {{ x: p.x + r.left, y: p.y + r.top, g: under && under.properties.g, p: under && under.properties.p }}; }}''')
     print('  under the spot:', at)
     check('the spot is gush 10102 parcel 110', at['g'] == 10102 and at['p'] == 110, at)
     pg.mouse.click(at['x'], at['y'])
     pg.wait_for_timeout(1500)
     got = pg.evaluate('''() => {
       const d = document.getElementById('detail-view');
       const l = Layers.byId('parcels');
       return { sel: typeof selectedId !== 'undefined' ? selectedId : null, pane: d && !d.hidden,
                text: d ? d.textContent.slice(0, 200) : '', items: l.waypoints.length,
                shape: l.waypoints[0] && l.waypoints[0].shape[0].length,
                lit: !!map.getLayer('eg-parcels') };
     }''')
     print('  after the tap:', json.dumps(got, ensure_ascii=False)[:300])
     check('the tap selects that parcel', got['sel'] == 'parcel-10102-110', got['sel'])
     check('the pane opens on it', got['pane'] and 'חלקה 110' in got['text'], got['text'][:80])
     check('its whole outline, from the file', got['shape'] and got['shape'] >= 4, got['shape'])
     check('it is lit on the map', got['lit'])
     check('the layer holds one item, not nine thousand', got['items'] == 1, got['items'])

     # Another tap replaces it.
     pg.mouse.click(at['x'] + 120, at['y'] + 90)
     pg.wait_for_timeout(1500)
     after = pg.evaluate('() => ({ sel: selectedId, items: Layers.byId("parcels").waypoints.length })')
     check('a second tap moves to the next parcel', after['items'] <= 1 and after['sel'] != 'parcel-10102-110', after)
     pg.screenshot(path=os.path.join(OUT, 'shot_parcels.png'))

     # In flight (26/9/2026): the grid stays, white, with its numbers, and
     # everything is put back on landing. The balloon, low: the first visit's
     # craft. queryRenderedFeatures counts few lines at that pitch - they are
     # there in the shot - so any at all is the test.
     pg.click('#explore')
     pg.wait_for_function('() => Explore.debug().flying', timeout=20000)
     pg.evaluate('() => document.getElementById("fly-intro").click()')
     pg.wait_for_timeout(4000)
     fly = pg.evaluate('''() => ({
       vis: ['pgl-parcels', 'pgn-parcels', 'ln-trails'].map((id) => map.getLayoutProperty(id, 'visibility')),
       lines: map.queryRenderedFeatures({ layers: ['pgl-parcels'] }).length,
       nums: map.queryRenderedFeatures({ layers: ['pgn-parcels'] }).length,
       line: map.getPaintProperty('pgl-parcels', 'line-color'),
       text: map.getPaintProperty('pgn-parcels', 'text-color'),
       zoom: map.getZoom() })''')
     print('  in flight:', fly)
     check('in flight the grid is drawn', fly['vis'][0] != 'none' and fly['lines'] > 0, fly)
     check('in flight the shortcuts are still hidden', fly['vis'][2] == 'none', fly['vis'])
     check('in flight the numbers show', fly['vis'][1] != 'none' and fly['nums'] > 20, fly)
     check('white over the satellite', fly['line'] == '#ffffff' and fly['text'] == '#ffffff', fly)
     pg.screenshot(path=os.path.join(OUT, 'shot_parcels_flight.png'))
     pg.keyboard.press('Escape')
     pg.wait_for_timeout(3000)
     back = pg.evaluate('''() => ({ vis: map.getLayoutProperty('ln-trails', 'visibility'),
       minzoom: map.getLayer('pgn-parcels').minzoom })''')
     check('landing puts the shortcuts back', back['vis'] != 'none', back)
     check('and the numbers back to close in only', back['minzoom'] == 16.5, back)

     # Off hides the grid.
     pg.evaluate('() => Layers.toggle ? Layers.toggle("parcels") : (Layers.byId("parcels").on = false, Layers.applyVisibility && Layers.applyVisibility())')
     pg.wait_for_timeout(800)
     off = pg.evaluate('() => map.getLayoutProperty("pgl-parcels", "visibility")')
     check('switching it off hides the grid', off == 'none', off)

     check('no page errors', not errs, errs[:3])
     b.close()
finally:
    srv.terminate()

print()
print(f'{len(fails)} failed' if fails else 'all passed')
for f in fails: print('  - ' + f)
sys.exit(1 if fails else 0)
