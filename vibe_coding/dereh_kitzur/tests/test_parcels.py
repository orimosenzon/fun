"""The parcels layer: empty on arrival, fills from govmap as you zoom in."""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8771
URL = f'http://127.0.0.1:{PORT}/index.html'
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
     pg = b.new_page(viewport={'width': 1200, 'height': 900})
     errs = []
     pg.on('pageerror', lambda e: errs.append('pageerror: ' + str(e)))
     pg.on('console', lambda m: errs.append('console: ' + m.text) if m.type == 'error' else None)
     pg.goto(URL)
     pg.wait_for_function('() => typeof Layers !== "undefined" && typeof Parcels !== "undefined" && typeof map !== "undefined" && !!map.getSource("src-trails")', timeout=40000)
     pg.wait_for_timeout(2500)
     pg.evaluate('''() => { document.querySelectorAll('.sheet[aria-modal]').forEach(x => x.style.display='none');
       if (map.getTerrain()) map.setTerrain(null); }''')

     lay = pg.evaluate('() => { const l = Layers.byId("parcels"); return l && {on: l.on, n: l.waypoints.length, cat: l.category, minzoom: l.minzoom}; }')
     print('layer on arrival:', json.dumps(lay, ensure_ascii=False))
     check('the layer exists and is empty', lay and lay['n'] == 0, lay)
     check('it is off by default', lay and not lay['on'], lay)

     # far out: switching it on must fetch nothing
     pg.evaluate('() => { map.jumpTo({center: [34.9700, 32.4740], zoom: 14, pitch: 0}); Layers.turnOn("parcels"); }')
     pg.wait_for_timeout(2500)
     far = pg.evaluate('() => ({n: Parcels.count(), src: !!map.getSource("shp-parcels")})')
     print('at z14:', far)
     check('nothing is fetched far out', far['n'] == 0, far)

     # close in: it fills
     pg.evaluate('() => map.jumpTo({center: [34.97507, 32.47411], zoom: 17, pitch: 0})')
     pg.wait_for_timeout(6000)
     near = pg.evaluate('''() => {
       const l = Layers.byId('parcels');
       return {n: Parcels.count(), items: l.waypoints.length,
               src: !!map.getSource('shp-parcels'),
               fill: !!map.getLayer('fl-parcels'), edge: !!map.getLayer('eg-parcels'),
               vis: map.getLayer('fl-parcels') ? map.getLayoutProperty('fl-parcels','visibility') : null,
               sample: l.waypoints.slice(0,3).map(w => ({name: w.name, pts: w.shape[0].length}))};
     }''')
     print('at z17:', json.dumps(near, ensure_ascii=False))
     check('parcels arrive', near['n'] > 20, near['n'])
     check('they reach the layer', near['items'] == near['n'], near)
     check('the shape source is built', near['src'] and near['fill'], near)
     check('and is visible', near['vis'] != 'none', near['vis'])

     # they are drawn under the trails
     under = pg.evaluate('''() => {
       const ids = map.getStyle().layers.map(l => l.id);
       return {parcel: ids.indexOf('eg-parcels'), trail: ids.indexOf('ln-trails')};
     }''')
     print('draw order:', under)
     check('the edge layer exists', under['parcel'] >= 0, under)
     check('drawn under the shortcuts', 0 <= under['parcel'] < under['trail'], under)

     # a second small pan must not re-ask
     n1 = pg.evaluate('() => Parcels.count()')
     pg.evaluate('() => map.panBy([20, 20])')
     pg.wait_for_timeout(2500)
     check('a small pan does not re-ask', pg.evaluate('() => Parcels.count()') == n1)

     # one of them is selectable and says something useful
     got = pg.evaluate('''() => {
       const l = Layers.byId('parcels');
       const w = l.waypoints.find(x => /10102/.test(x.name));
       if (!w) return null;
       select(w.id, false);
       const d = document.getElementById('detail-view');
       return {name: w.name, note: w.note.slice(0, 60), pane: !d.hidden,
               head: (d.querySelector('h2,h3')||{}).textContent};
     }''')
     print('a parcel:', json.dumps(got, ensure_ascii=False))
     check('a parcel opens in the pane', got and got['pane'], got)

     check('no page errors', not errs, errs[:3])
     pg.screenshot(path=os.path.join(OUT, 'shot_parcels.png'))
     b.close()

finally:
    srv.terminate()

print()
print(f'{len(fails)} failed' if fails else 'all passed')
for f in fails: print('  - ' + f)
sys.exit(1 if fails else 0)
