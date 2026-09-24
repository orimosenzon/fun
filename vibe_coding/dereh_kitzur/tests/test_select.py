"""The chosen trail is the lit trail, in a real Chrome.

Feature state lives inside a MapLibre source and is keyed by the feature's
position in the array handed to setData. Two things fall out of that, and both
of them were bugs on 24/9/2026 - the detail pane named one trail while its
neighbour glowed on the map:

  * a state survives setData, so a rebuild that adds or drops one item shifts
    every position after it and leaves the old states on the wrong trails;
  * setStyle throws the sources away, so a basemap switch dropped the
    highlight entirely - nothing lit, nothing dimmed.

This walks both, plus the ordinary paths: picking from the list, clearing the
selection, an area rather than a line, a place rather than a trail.

    python3 tests/test_select.py

Needs Playwright with Google Chrome (channel='chrome'). Takes about a minute.
"""
import subprocess, sys, time, os, json
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8769

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

# Every feature carrying `sel`, across every source this app draws. `stray`
# catches the contradiction that shifted states used to produce: one feature
# marked both chosen and dimmed.
STATE = r'''() => {
  const lit = [], stray = [];
  Object.keys(map.getStyle().sources).forEach((src) => {
    if (!src.startsWith('src-')) return;
    const s = map.getSource(src);
    const d = s._data || (s.serialize && s.serialize().data);
    if (!d || !d.features) return;
    d.features.forEach((f) => {
      const st = map.getFeatureState({ source: src, id: f.id }) || {};
      const it = Layers.item(f.properties.id);
      if (st.sel && st.dim) stray.push({ src, fid: f.id, name: it && it.name });
      else if (st.sel) lit.push({ src, fid: f.id, id: f.properties.id, name: it && it.name });
    });
  });
  return { lit, stray };
}'''

PANE = r'''() => {
  const d = document.getElementById('detail-view');
  if (!d || d.hidden) return null;
  const h = d.querySelector('h2, h3, .detail-name, .name');
  return h ? h.textContent.trim() : null;
}'''

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(channel='chrome', headless=True,
                              args=['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'])
        ctx = b.new_context(viewport={'width': 1200, 'height': 900}, locale='he-IL')
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); } catch (e) {}")
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(f'http://127.0.0.1:{PORT}/index.html?bg=sat')
        page.wait_for_function(
            '() => typeof Layers !== "undefined" && typeof map !== "undefined" && !!map.getSource("src-trails")',
            timeout=30000)
        time.sleep(3)
        page.evaluate('''() => { const w = document.getElementById("welcome-sheet"); if (w) w.hidden = true;
          document.querySelectorAll('.sheet[aria-modal]').forEach((x) => { x.style.display = 'none'; }); }''')
        # Flat and north-up so nothing here depends on the terrain mesh.
        page.evaluate('''() => { if (map.getTerrain && map.getTerrain()) map.setTerrain(null);
          map.jumpTo({ pitch: 0, bearing: 0 }); }''')
        time.sleep(1.5)

        def lit_is(label, wanted=None):
            """Exactly the wanted item is lit, nothing else, and nothing is both."""
            st = page.evaluate(STATE)
            ok = (len(st['lit']) == (1 if wanted else 0) and not st['stray']
                  and (not wanted or st['lit'][0]['id'] == wanted))
            check(label, ok, json.dumps({'pane': page.evaluate(PANE), 'want': wanted, **st},
                                        ensure_ascii=False))

        # A trail of the initiative's own layer, and the one that follows it in
        # the array - the pair the bug swapped.
        pair = page.evaluate('''() => {
          const l = Layers.list.find((x) => x.id === 'trails');
          const segs = l.segments.filter((s) => s.path && s.path.length > 1);
          return [segs[segs.length - 2].id, segs[segs.length - 1].id];
        }''')
        first, second = pair

        # ---- the ordinary paths ----
        page.evaluate('(id) => select(id, false)', first)
        time.sleep(0.7)
        lit_is('choosing a trail lights that trail', first)

        page.evaluate('() => deselect()')
        time.sleep(0.6)
        lit_is('clearing the selection leaves nothing lit')

        row = page.evaluate('''() => { const r = document.querySelector('#list-view .row');
          if (!r) return null; r.click(); return r.dataset.id; }''')
        time.sleep(0.9)
        lit_is('choosing from the list lights that row', row)

        # ---- a rebuild that shifts every position after the chosen one ----
        page.evaluate('(id) => select(id, false)', second)
        time.sleep(0.7)
        dropped = page.evaluate('''() => {
          const l = Layers.list.find((x) => x.id === 'trails');
          const gone = l.segments.splice(0, 1)[0];
          Layers.refresh('trails');
          return gone.name;
        }''')
        time.sleep(0.9)
        print(f'    dropped "{dropped}" from the head of the trails array')
        lit_is('a shifted array still lights the chosen trail', second)

        # ---- the basemap, which throws every source away ----
        page.reload()
        page.wait_for_function(
            '() => typeof Layers !== "undefined" && typeof map !== "undefined" && !!map.getSource("src-trails")',
            timeout=30000)
        time.sleep(3)
        page.evaluate('''() => { const w = document.getElementById("welcome-sheet"); if (w) w.hidden = true;
          document.querySelectorAll('.sheet[aria-modal]').forEach((x) => { x.style.display = 'none'; }); }''')
        page.evaluate('(id) => select(id, false)', first)
        time.sleep(0.8)
        lit_is('lit again after a reload', first)
        for i in (1, 2):
            page.evaluate('() => document.getElementById("basemap").click()')
            time.sleep(4)
            lit_is(f'still lit across basemap switch {i}', first)
        page.screenshot(path=f'{OUT}/shot_select.png')

        # ---- an area and a place, which live in their own sources ----
        area = page.evaluate('''() => {
          const l = Layers.list.find((x) => (x.waypoints || []).some((p) => p.shape && p.shape.length));
          if (!l) return null;
          Layers.turnOn(l.id);
          return (l.waypoints.find((p) => p.shape && p.shape.length) || {}).id || null;
        }''')
        time.sleep(2.5)
        if area:
            page.evaluate('(id) => select(id, false)', area)
            time.sleep(0.9)
            lit_is('choosing an area lights that area', area)

        place = page.evaluate('''() => {
          const l = Layers.list.find((x) => x.id === 'places');
          if (!l) return null;
          Layers.turnOn(l.id);
          const w = (l.waypoints || []).find((p) => !p.unplaced);
          return w ? w.id : null;
        }''')
        time.sleep(2.5)
        if place:
            page.evaluate('(id) => select(id, false)', place)
            time.sleep(0.9)
            lit_is('choosing a place lights that place', place)

        check('no page errors', not errors, errors[:3])
        b.close()
finally:
    srv.terminate()

print()
print(f'{len(fails)} failed' if fails else 'all passed')
for f in fails: print('  - ' + f)
sys.exit(1 if fails else 0)
