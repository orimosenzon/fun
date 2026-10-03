"""Walking routes through the shortcuts (route.js), in a real Chrome.

The geolocation is pinned inside the moshava, beside the end of הדקלים - גפן,
so "from where I am" is a known point. Checks the engine on a pair of points
where the shortcut must win, then the whole flow through real clicks: the card,
a tap for the destination, the bar, the pane, the link, walking it, closing it,
and "מסלול הליכה לכאן" from a place's own page.

    python3 tests/test_route.py
"""
import json, os, subprocess, sys, time
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8771
URL = f'http://127.0.0.1:{PORT}/index.html'
HOME = {'latitude': 32.47830, 'longitude': 34.96950}   # just past one end of p11
OTHER = {'lat': 32.47920, 'lng': 34.97160}               # just past the other end

fails = []
def check(name, ok, detail=''):
    print(('  ok   ' if ok else '  FAIL ') + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

READY = '() => typeof Route !== "undefined" && typeof map !== "undefined" && !!map.getSource("src-trails")'

def open_app(b, url, viewport):
    ctx = b.new_context(viewport=viewport, geolocation=HOME, permissions=['geolocation'])
    pg = ctx.new_page()
    errs = []
    pg.on('pageerror', lambda e: errs.append('pageerror: ' + str(e)))
    pg.on('console', lambda m: errs.append('console: ' + m.text) if m.type == 'error' else None)
    pg.goto(url)
    pg.wait_for_function(READY, timeout=40000)
    pg.wait_for_timeout(2000)
    pg.evaluate('() => document.querySelectorAll(".sheet").forEach(s => s.hidden = true)')
    return ctx, pg, errs

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
  with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', args=['--use-gl=angle', '--use-angle=gl'])
    ctx, pg, errs = open_app(b, URL, {'width': 1280, 'height': 860})

    # ---- the engine, on a pair where the shortcut has to win ----
    r = pg.evaluate('''async (to) => {
      await Route.load();
      Route.open(to);
      await new Promise(r => setTimeout(r, 1500));
      const x = Route.result();
      return x && {len: x.length, without: x.without && x.without.length,
                   used: x.trailsUsed, legs: x.legs.map(l => [l.kind, l.name, Math.round(l.length)])};
    }''', OTHER)
    print('route p11:', json.dumps(r, ensure_ascii=False))
    check('a route comes back', bool(r), r)
    check('it goes through הדקלים - גפן', r and 'p11' in r['used'], r and r['used'])
    check('and is much shorter than the streets', r and r['without'] and r['without'] > 1.5 * r['len'],
          r and (r['len'], r['without']))

    bar = pg.evaluate('''() => ({shown: !document.getElementById('route-bar').hidden,
      main: document.getElementById('route-main').textContent,
      sub: document.getElementById('route-sub').textContent,
      go: !document.getElementById('route-go').hidden,
      layers: ['route-street','route-trail','route-without'].filter(id => map.getLayer(id)),
      pane: !document.getElementById('detail-view').hidden && !!document.querySelector('#detail .route-title'),
      legs: document.querySelectorAll('#detail .route-leg').length,
      url: location.search})''')
    print('bar:', json.dumps(bar, ensure_ascii=False))
    check('the bar says minutes and metres', bar['shown'] and 'דק׳' in bar['main'], bar['main'])
    check('it says what the shortcut saved', 'חוסך' in bar['sub'], bar['sub'])
    check('the route is drawn, with the street-only one under it', len(bar['layers']) == 3, bar['layers'])
    check('the pane lists the legs', bar['pane'] and bar['legs'] >= 2, bar)
    q = parse_qs(urlparse(bar['url']).query)
    check('the link carries the route', 'route' in q, bar['url'])
    pg.wait_for_timeout(900)
    pg.screenshot(path=os.path.join(OUT, 'shot_route.png'))

    # ---- walking it ----
    pg.click('#route-go')
    pg.wait_for_timeout(1500)
    nv = pg.evaluate('''() => ({nav: !document.getElementById('nav').hidden,
      route: !!(nav && nav.item && nav.item.route),
      bar: getComputedStyle(document.getElementById('route-bar')).display,
      state: document.getElementById('nav-state').textContent,
      dist: document.getElementById('nav-dist').textContent})''')
    print('walking:', json.dumps(nv, ensure_ascii=False))
    check('נווט starts navigation along the route', nv['nav'] and nv['route'], nv)
    check('the route bar steps aside for it', nv['bar'] == 'none', nv['bar'])
    check('it counts down along the route', 'מסלול' in nv['state'], nv['state'])
    pg.click('#nav-stop')
    pg.wait_for_timeout(300)
    check('stopping brings the route bar back',
          pg.evaluate('() => getComputedStyle(document.getElementById("route-bar")).display') != 'none')

    # ---- the link, opened somewhere else ----
    link = URL + bar['url']
    ctx2, pg2, errs2 = open_app(b, link, {'width': 1280, 'height': 860})
    pg2.wait_for_timeout(2500)
    again = pg2.evaluate('() => { const x = Route.result(); return x && x.trailsUsed; }')
    check('a link with a route shows the same route', again and 'p11' in again, again)
    errs.extend(errs2)
    ctx2.close()

    # ---- close ----
    pg.click('#route-stop')
    pg.wait_for_timeout(400)
    gone = pg.evaluate('''() => ({bar: document.getElementById('route-bar').hidden,
      layers: ['route-street','route-trail','route-without'].filter(id => map.getLayer(id)).length,
      pins: document.querySelectorAll('.route-pin').length, url: location.search})''')
    check('× clears the bar, the line and the pins',
          gone['bar'] and gone['layers'] == 0 and gone['pins'] == 0, gone)
    check('and the link forgets it', 'route=' not in gone['url'], gone['url'])

    # ---- through real clicks: the card, then a tap on the map ----
    pg.evaluate('''() => { if (map.getTerrain()) map.setTerrain(null);
        map.jumpTo({center: [34.9705, 32.4787], zoom: 16.5, pitch: 0, bearing: 0}); }''')
    pg.wait_for_timeout(1500)
    pg.click('#route-ask')
    pg.wait_for_timeout(300)
    check('the card arms a tap for the destination', pg.evaluate('() => Route.isPicking()'))
    before = pg.evaluate('() => selectedId')
    box = pg.evaluate('() => { const r = map.getCanvas().getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }')
    pg.mouse.click(box[0] + box[2] * 0.7, box[1] + box[3] * 0.4)
    pg.wait_for_timeout(2000)
    tapped = pg.evaluate('''() => ({picking: Route.isPicking(), result: !!Route.result(),
      sel: selectedId, main: document.getElementById('route-main').textContent})''')
    print('after the tap:', json.dumps(tapped, ensure_ascii=False))
    check('the tap sets the destination and routes', tapped['result'] and not tapped['picking'], tapped)
    check('the tap did not also open a trail', tapped['sel'] == before, tapped['sel'])
    pg.click('#route-stop')

    # ---- from a place's page ----
    pg.evaluate('() => select("p14")')
    pg.wait_for_timeout(900)
    has = pg.evaluate('() => !!document.getElementById("route-here")')
    check('a place offers מסלול הליכה לכאן', has)
    if has:
        pg.click('#route-here')
        pg.wait_for_timeout(2500)
        res = pg.evaluate('() => { const x = Route.result(); return x && Math.round(x.length); }')
        check('and it routes there from where you are', bool(res), res)
        pg.click('#route-stop')

    # ---- no GPS: the first tap is א, the second ב ----
    ctx4 = b.new_context(viewport={'width': 1280, 'height': 860})   # no geolocation permission
    pg4 = ctx4.new_page()
    pg4.on('pageerror', lambda e: errs.append('pageerror: ' + str(e)))
    pg4.goto(URL)
    pg4.wait_for_function(READY, timeout=40000)
    pg4.wait_for_timeout(1500)
    pg4.evaluate('''() => { document.querySelectorAll(".sheet").forEach(s => s.hidden = true);
        if (map.getTerrain()) map.setTerrain(null);
        map.jumpTo({center: [34.9705, 32.4787], zoom: 16.5, pitch: 0, bearing: 0}); }''')
    pg4.wait_for_timeout(1500)
    pg4.click('#route-ask')
    pg4.wait_for_timeout(1200)
    box = pg4.evaluate('() => { const r = map.getCanvas().getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }')
    first = (box[0] + box[2] * 0.3, box[1] + box[3] * 0.6)
    second = (box[0] + box[2] * 0.7, box[1] + box[3] * 0.4)
    # Where the two taps land on the ground, before the route re-frames the map.
    ground = pg4.evaluate('''(pts) => pts.map(([x, y]) => { const r = map.getCanvas().getBoundingClientRect();
        const ll = map.unproject([x - r.left, y - r.top]); return [ll.lng, ll.lat]; })''', [first, second])
    pg4.mouse.click(*first)
    pg4.wait_for_timeout(500)
    mid = pg4.evaluate('() => ({picking: Route.isPicking(), main: document.getElementById("route-main").textContent})')
    pg4.mouse.click(*second)
    pg4.wait_for_timeout(2000)
    pins = pg4.evaluate('''(ground) => {
      const r0 = map.getCanvas().getBoundingClientRect();
      const pts = ground.map((g) => { const p = map.project(g); return [p.x + r0.left, p.y + r0.top]; });
      const at = (cls) => { const r = document.querySelector(cls).getBoundingClientRect();
                            return [r.left + r.width / 2, r.top + r.height / 2]; };
      const near = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]) < 25;
      return {a: near(at('.route-from'), pts[0]), b: near(at('.route-to'), pts[1]),
              result: !!Route.result(),
              letters: [document.querySelector('.route-from').textContent,
                        document.querySelector('.route-to').textContent]};
    }''', ground)
    print('no gps:', json.dumps(mid, ensure_ascii=False), json.dumps(pins, ensure_ascii=False))
    check('after the first tap it asks for the destination (ב)', 'ב' in mid['main'], mid['main'])
    check('the first tap is א and the second ב', pins['a'] and pins['b'] and pins['letters'] == ['א', 'ב'], pins)
    check('and the route is worked out', pins['result'])
    ctx4.close()

    # ---- phone ----
    ctx3, pg3, errs3 = open_app(b, URL, {'width': 390, 'height': 800})
    pg3.evaluate('async (to) => { await Route.load(); Route.open(to); }', OTHER)
    pg3.wait_for_timeout(2500)
    pg3.screenshot(path=os.path.join(OUT, 'shot_route_phone.png'))
    errs.extend(errs3)

    check('no page errors', not errs, errs[:4])
    b.close()
finally:
    srv.terminate()

print()
print(f'{len(fails)} failed' if fails else 'all passed')
for f in fails: print('  - ' + f)
sys.exit(1 if fails else 0)
