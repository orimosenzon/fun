"""Drawing a trail: a tap shows up at once, in a real Chrome with the terrain on.

On 24/9/2026 Ori reported a long pause between each tap and its point while
drawing a new shortcut, long enough to make him tap again. Measured in a real
Chrome window it was 0.8s before anything moved and 1-1.3s before the point:
every layer had its own mouseenter/mouseleave for the pointer cursor, MapLibre
answers each with a queryRenderedFeatures, and over terrain each of those reads
pixels back from the GPU. 124 reads over four taps, a display frame apiece.

This checks the fix from the outside:
  * the tap's own frame already shows the new point (the SVG overlay);
  * a tap costs a handful of GPU read-backs, not dozens;
  * the map's own layer catches up and the overlay steps aside;
  * while drawing, a tap on a trail adds a point and selects nothing, and a
    tap on a waypoint's label adds a point too;
  * outside the editor the hover cursor still works.

    python3 tests/test_draw_tap.py

Needs Playwright with Google Chrome (channel='chrome'). Under a minute.
"""
import subprocess, sys, time, os
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8771

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

# Count GPU read-backs, and time each tap to the first frame that shows it.
PROBE = r'''() => {
  const gl = map.painter.context.gl;
  const real = gl.readPixels.bind(gl);
  window.__reads = 0;
  gl.readPixels = (...a) => { window.__reads++; return real(...a); };
  window.__taps = [];
  map.getCanvas().addEventListener('pointerup', () => {
    const rec = { t0: performance.now() };
    window.__taps.push(rec);
    requestAnimationFrame(() => {
      rec.frame = performance.now() - rec.t0;
      rec.after = document.querySelectorAll('.editor-ghost circle').length;
    });
  }, true);
}'''

def tap(page, x, y):
    page.mouse.move(x, y); page.mouse.down(); page.mouse.up()

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(channel='chrome', headless=True,
                              args=['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'])
        ctx = b.new_context(viewport={'width': 1200, 'height': 900}, locale='he-IL')
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1');"
                            " localStorage.removeItem('dk.editor.v1'); } catch (e) {}")
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{PORT}/index.html')
        page.wait_for_function(
            '() => typeof map !== "undefined" && map.loaded() && !!map.getSource("src-trails")',
            timeout=60000)
        time.sleep(4)
        page.evaluate('() => document.querySelectorAll(".sheet").forEach((x) => { x.hidden = true; })')
        check('terrain is on (the case that was slow)', page.evaluate('() => !!map.getTerrain()'))

        box = page.evaluate('() => { const r = map.getCanvas().getBoundingClientRect();'
                            ' return [r.left, r.top, r.width, r.height]; }')

        # A point on a visible trail, in screen coordinates.
        def trail_point():
            return page.evaluate('''() => {
              const hits = map.queryRenderedFeatures({ layers: Layers.trailHitLayers() });
              for (const f of hits) {
                const g = f.geometry, c = g.type === 'LineString' ? g.coordinates
                  : g.type === 'MultiLineString' ? g.coordinates[0] : null;
                if (!c || c.length < 2) continue;
                const mid = c[Math.floor(c.length / 2)];
                const p = map.project(mid);
                const r = map.getCanvas().getBoundingClientRect();
                if (p.x > 60 && p.y > 120 && p.x < r.width - 60 && p.y < r.height - 60)
                  return { x: p.x + r.left, y: p.y + r.top, id: f.properties.id };
              }
              return null;
            }''')

        # --- the hover cursor, outside the editor ---
        tp = trail_point()
        check('found a trail on screen to aim at', tp is not None)
        if tp:
            page.mouse.move(tp['x'], tp['y'])
            time.sleep(0.5)
            check('hover over a trail shows the pointer',
                  page.evaluate('() => map.getCanvas().style.cursor') == 'pointer')
            page.mouse.move(box[0] + 5, box[1] + box[3] - 5)
            time.sleep(0.5)

        # --- into the drawing editor ---
        page.evaluate('''() => { const b = document.createElement('button'); b.dataset.act = 'draw';
          document.getElementById('draft-sheet').appendChild(b); b.click(); b.remove(); }''')
        page.wait_for_function('() => Drafts.isDrafting()')
        time.sleep(2)
        page.evaluate(PROBE)

        for i in range(5):
            tap(page, box[0] + box[2] * 0.3 + i * 40, box[1] + box[3] * 0.55 + (i % 2) * 30)
            time.sleep(0.6)
        taps = page.evaluate('() => window.__taps')
        reads = page.evaluate('() => window.__reads')
        check('five taps, five pointerups', len(taps) == 5, len(taps))
        check('each tap shows its point in its own frame',
              # The overlay is taken down between taps once the map catches up,
              # so what it holds after tap n is the whole line: n points.
              [t.get('after') for t in taps] == [1, 2, 3, 4, 5],
              [t.get('after') for t in taps])
        check('the first frame comes quickly', all(t.get('frame', 1e9) < 250 for t in taps),
              [round(t.get('frame', -1)) for t in taps])
        # Before the fix this was ~30 per tap; what is left is one per MapLibre
        # mouse event plus the tap's own unproject.
        check('few GPU read-backs per tap', reads / 5 <= 12, f'{reads} over 5 taps')

        page.wait_for_function('() => !document.querySelector(".editor-ghost")', timeout=15000)
        n_map = page.evaluate('''() => map.getSource('src-editor')._data.features
                                   .filter((f) => f.geometry.type === 'Point').length''')
        check('the map has all five points once the overlay steps aside', n_map == 5, n_map)

        # A tap on a trail while drawing: a point, not a selection.
        tp = trail_point()
        if tp:
            tap(page, tp['x'], tp['y'])
            time.sleep(0.8)
            check('a tap on a trail while drawing selects nothing',
                  page.evaluate('() => selectedId') in (None, ''), page.evaluate('() => selectedId'))
            check('...and adds a point',
                  page.evaluate('() => document.querySelectorAll(".editor-ghost circle").length'
                                ' || map.getSource("src-editor")._data.features.length - 1') == 6)

        # A tap on a waypoint's label while drawing adds a point.
        pin = page.evaluate('''() => {
          const r = map.getCanvas().getBoundingClientRect();
          for (const b of document.querySelectorAll('.pin b')) {
            const q = b.getBoundingClientRect();
            if (q.width && q.left > r.left + 40 && q.right < r.right - 40
                && q.top > r.top + 120 && q.bottom < r.bottom - 60)
              return { x: q.left + q.width / 2, y: q.top + q.height / 2 };
          }
          return null;
        }''')
        if pin:
            before = page.evaluate('() => map.getSource("src-editor")._data.features.length')
            tap(page, pin['x'], pin['y'])
            time.sleep(0.8)
            after = page.evaluate('() => map.getSource("src-editor")._data.features.length')
            check('a tap on a waypoint label while drawing adds a point', after == before + 1,
                  (before, after))
        else:
            print('  --   no waypoint label on screen; label tap not checked')

        check('no page errors', not errors, errors)
        b.close()
finally:
    srv.terminate()

print()
print('all passed' if not fails else 'FAILED: ' + ', '.join(fails))
sys.exit(1 if fails else 0)
