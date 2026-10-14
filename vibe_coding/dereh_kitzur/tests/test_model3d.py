"""The third background: the moshava as a 3D model (model3d.js), 14/10/2026.

What it checks, at phone and desktop width:
  - the globe button cycles streets -> satellite -> 3D, and the model's
    buildings, roofs, trees and walls are on the map, with terrain off
  - the buildings sit above every line and below every label, after the app
    has added its own layers (a trail behind a house must not show through)
  - the angle slider tilts the map, past the 3D button's 58 to 85
  - the person stands you on a street, not inside a house, at eye height
  - a tap walks you, and a tap on a house stops you short of its wall
  - the way back up lets the map be dragged again
  - leaving the 3D background puts the 80-degree ceiling and the terrain back

    python3 tests/test_model3d.py

Needs Playwright with Google Chrome. Takes about two minutes.
"""
import os, subprocess, sys, time
from playwright.sync_api import sync_playwright
OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8796
fails = []
def check(n, ok, d=''):
    print(('  ok   ' if ok else '  FAIL ') + n + ('' if ok else '   ' + str(d)))
    if not ok: fails.append(n)

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)

def settle(pg, t=30000):
    try:
        pg.wait_for_function('() => __map.loaded() && __map.areTilesLoaded()', timeout=t)
    except Exception:
        pass
    pg.wait_for_timeout(800)

try:
  with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', args=['--use-gl=angle', '--use-angle=gl'])
    for label, vp in (('phone', {'width': 390, 'height': 844}), ('desktop', {'width': 1280, 'height': 860})):
        print(f'\n{label}:')
        phone = vp['width'] < 500
        ctx = b.new_context(viewport=vp, locale='he-IL', has_touch=phone, is_mobile=phone)
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1','yes'); } catch(e){}")
        pg = ctx.new_page()
        errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        # The old moshava around the water tower, a little zoomed out.
        pg.goto(f'http://127.0.0.1:{PORT}/index.html?map=32.4741,34.9700,16.5,30,58&panel=off')
        pg.wait_for_function('() => window.__map && __map.isStyleLoaded() && typeof Model3D !== "undefined"',
                             timeout=60000)
        settle(pg)

        # streets -> satellite -> 3D
        pg.click('#basemap')
        pg.wait_for_timeout(1500)
        pg.click('#basemap')
        pg.wait_for_function('() => __map.getLayer("m3d-roofs")', timeout=60000)
        settle(pg)
        st = pg.evaluate('''() => ({
            id: BASEMAPS[baseIndex].id, terrain: !!__map.getTerrain(), max: __map.getMaxPitch(),
            layers: ['m3d-houses','m3d-roofs','m3d-trees','m3d-walls'].every((l) => __map.getLayer(l)),
            ctl: !document.querySelector('.m3d-ctl').hidden,
            url: location.search })''')
        check(f'{label}: the third press is the 3D model', st['id'] == '3d' and st['layers'], st)
        check(f'{label}: flat ground under it, and an 85-degree ceiling',
              not st['terrain'] and st['max'] == 85, st)
        check(f'{label}: the slider and the person are shown', st['ctl'], st)
        check(f'{label}: the link remembers it', 'bg=3d' in st['url'], st['url'])

        # Buildings above lines and fills, below labels and circles.
        order = pg.evaluate('''() => {
            const L = __map.getStyle().layers, ex = ['m3d-walls','m3d-trees','m3d-houses','m3d-roofs'];
            const idx = L.map((l, i) => ex.includes(l.id) ? i : -1).filter((i) => i >= 0);
            const lo = Math.min(...idx), hi = Math.max(...idx);
            const top = (l) => ['symbol', 'circle', 'heatmap'].includes(l.type);
            return { lineAbove: L.filter((l, i) => i > hi && !top(l)).map((l) => l.id),
                     labelBelow: L.filter((l, i) => i < lo && top(l)).map((l) => l.id) }; }''')
        check(f'{label}: no line or fill drawn over the buildings', not order['lineAbove'], order)
        check(f'{label}: no label or pin hidden under them', not order['labelBelow'], order)
        # And again after the app adds a layer of its own at the top.
        pg.evaluate('''() => { __map.addSource('t-x', {type:'geojson', data:{type:'FeatureCollection',features:[]}});
                               __map.addLayer({id:'t-x', type:'line', source:'t-x'}); }''')
        pg.wait_for_timeout(500)
        later = pg.evaluate('''() => { const L = __map.getStyle().layers.map((l) => l.id);
                                       return L.indexOf('t-x') < L.indexOf('m3d-walls'); }''')
        check(f'{label}: a line added later is put back under the buildings', later)

        # The slider: a drag from the thumb to the bottom of the track.
        tr = pg.evaluate('''() => { const t = document.querySelector('.m3d-track').getBoundingClientRect(),
                                           h = document.querySelector('.m3d-thumb').getBoundingClientRect();
                                    return [t.left + t.width / 2, h.top + h.height / 2, t.bottom]; }''')
        pg.mouse.move(tr[0], tr[1]); pg.mouse.down()
        pg.mouse.move(tr[0], tr[2] + 5, steps=8); pg.mouse.up()
        pg.wait_for_timeout(600)
        pitch = pg.evaluate('__map.getPitch()')
        check(f'{label}: dragging the slider down tilts to the horizon, past 80', pitch > 80, pitch)

        # The person.
        pg.click('.m3d-walk')
        pg.wait_for_timeout(2600)
        settle(pg)
        s = pg.evaluate('''() => { const s = Model3D._state();
            return { s, inside: !!(s && Model3D._inside(s.lng, s.lat)), zoom: __map.getZoom(),
                     pitch: __map.getPitch(), exit: !document.querySelector('.m3d-exit').hidden,
                     drag: __map.dragPan.isEnabled() }; }''')
        check(f'{label}: the person stands you in the street', bool(s['s']) and s['exit'], s)
        check(f'{label}: not inside a house', not s['inside'], s)
        check(f'{label}: at eye height (the zoom of a camera 1.7 m up)', s['zoom'] > 21, s['zoom'])
        check(f'{label}: the map no longer pans under the finger', not s['drag'], s)

        # Drag to look around.
        brg0 = s['s']['brg']
        mid = (vp['width'] // 2, vp['height'] // 3)
        pg.mouse.move(*mid); pg.mouse.down(); pg.mouse.move(mid[0] - 150, mid[1], steps=10); pg.mouse.up()
        pg.wait_for_timeout(300)
        brg1 = pg.evaluate('Model3D._state().brg')
        turned = ((brg1 - brg0 + 540) % 360) - 180
        check(f'{label}: dragging left turns you right', turned > 5, turned)

        # A tap on the ground ahead walks you there.
        a = pg.evaluate('Model3D._state()')
        pg.mouse.click(vp['width'] // 2, int(vp['height'] * 0.62) if not phone else int(vp['height'] * 0.5))
        pg.wait_for_timeout(2500)
        b2 = pg.evaluate('Model3D._state()')
        moved = ((b2['lat'] - a['lat']) * 111320, (b2['lng'] - a['lng']) * 111320 * 0.8436)
        dist = (moved[0] ** 2 + moved[1] ** 2) ** 0.5
        check(f'{label}: a tap walks you forward', dist > 1, dist)

        # The walls stop you: walk straight at the nearest house with the keys.
        blocked = pg.evaluate('''() => Model3D._walkInto()''')
        check(f'{label}: walking into a house stops at its wall', blocked['ok'], blocked)

        # Back up.
        pg.click('.m3d-exit')
        pg.wait_for_timeout(1600)
        up = pg.evaluate('''() => ({ street: Model3D.inStreet(), drag: __map.dragPan.isEnabled(),
                                     zoom: __map.getZoom() })''')
        check(f'{label}: the way back up gives the map back', not up['street'] and up['drag'] and up['zoom'] <= 18.01, up)

        # Off the model: the ceiling and the terrain come back.
        pg.click('#basemap')
        pg.wait_for_timeout(3000)
        off = pg.evaluate('''() => ({ id: BASEMAPS[baseIndex].id, max: __map.getMaxPitch(),
                                      pitch: __map.getPitch(), terrain: !!__map.getTerrain(),
                                      ctl: document.querySelector('.m3d-ctl').hidden })''')
        check(f'{label}: back to streets, with the ceiling at 80 and the terrain on',
              off['id'] == 'streets' and off['max'] == 80 and off['pitch'] <= 80 and off['terrain'] and off['ctl'], off)
        check(f'{label}: no page errors', not errs, errs[:2])
        ctx.close()
    b.close()
finally:
    srv.terminate()
print()
print(f'{len(fails)} failed' if fails else 'all passed')
for f in fails: print('  - ' + f)
sys.exit(1 if fails else 0)
