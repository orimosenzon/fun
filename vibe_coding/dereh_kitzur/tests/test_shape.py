"""The shape editor (web/shape.js): reshaping a trail that is already there.

Drives a real Chrome through every gesture the editor offers, on a published
trail (with the editor's write stubbed, so nothing reaches the data repo) and
on a draft in IndexedDB:

  * the button is on an editor's trail page, and opens the editor flat;
  * the trail leaves its map layer while it is being edited;
  * dragging a point moves it; dragging a ⊕ adds one; undo and redo;
  * tapping a point offers delete; tapping a stretch offers delete, and
    deleting a middle stretch leaves two pieces;
  * dropping an end on the other piece's end joins them again;
  * carrying the line on from an end adds a point per tap;
  * saving hands one line to Store.reshape and leaves the editor;
  * an interrupted session is offered back after a reload;
  * a draft is reshaped in IndexedDB.

    python3 tests/test_shape.py

Needs Playwright with Google Chrome (channel='chrome'). About a minute.
"""
import subprocess, sys, time, os, json
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8773

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

STUB = r'''() => {
  Store.isEditor = () => true;
  window.__saved = null;
  Store.reshape = async (id, pieces, name) => {
    window.__saved = { id, pieces, name };
    const doc = JSON.parse(JSON.stringify(DATA));
    const seg = doc.segments.find((s) => s.id === id);
    seg.path = pieces[0];
    return doc;
  };
}'''

def handles(page, cls='sh-v'):
    return page.evaluate('''(cls) => [...document.querySelectorAll('.shape-svg .' + cls)].map((g) => {
      const r = g.querySelector('.dot').getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2, l: +g.dataset.l, i: +g.dataset.i,
               end: g.classList.contains('end') };
    })''', cls)

def lines(page):
    return page.evaluate('() => Shape._state() && Shape._state().lines')

def npoints(page):
    return sum(len(l) for l in lines(page))

def drag(page, a, b, steps=12):
    page.mouse.move(a[0], a[1]); page.mouse.down()
    for k in range(1, steps + 1):
        page.mouse.move(a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps)
        time.sleep(0.01)
    page.mouse.up()
    time.sleep(0.15)

def tap(page, x, y):
    page.mouse.move(x, y); page.mouse.down(); page.mouse.up(); time.sleep(0.15)

def pop_click(page, op):
    page.click(f'.shape-pop [data-op="{op}"]'); time.sleep(0.15)

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
        dialogs = []
        def on_dialog(d):
            dialogs.append(d.message); d.accept()
        page.on('dialog', on_dialog)

        def boot():
            page.wait_for_function(
                '() => typeof map !== "undefined" && map.loaded() && !!map.getSource("src-trails")',
                timeout=60000)
            time.sleep(3)
            page.evaluate('() => document.querySelectorAll(".sheet").forEach((x) => { x.hidden = true; })')
            page.evaluate(STUB)

        page.goto(f'http://127.0.0.1:{PORT}/index.html')
        page.evaluate("() => localStorage.removeItem('dk.shape.v1')")
        boot()

        # A trail with a few corners and some length, so it has room to work in.
        tid = page.evaluate('''() => Layers.byId('trails').segments
          .filter((s) => s.path.length >= 4 && s.path.length <= 30 && s.length > 150)
          .sort((a, b) => b.path.length - a.path.length)[0].id''')
        orig = page.evaluate('(id) => Layers.item(id).path', tid)
        print('  trail', tid, len(orig), 'points')

        page.evaluate('(id) => select(id)', tid)
        time.sleep(0.6)
        check('editor trail page offers "עריכת התוואי"',
              page.locator('#detail [data-pub="shape"]').count() == 1)
        page.click('#detail [data-pub="shape"]')
        time.sleep(1.2)
        check('editor is on', page.evaluate('() => document.body.classList.contains("shaping") && Shape.isOn()'))
        check('map is flat', page.evaluate('() => map.getPitch() < 1'))
        check('bar shown, topbar hidden', page.evaluate(
            '() => !document.getElementById("shape-bar").hidden && getComputedStyle(document.getElementById("topbar")).display === "none"'))
        gone = page.evaluate('''(id) => !map.querySourceFeatures('src-trails').some((f) => f.properties.id === id)''', tid)
        check('the trail left its own layer while edited', gone)
        hv = handles(page)
        check('a handle for every point', len(hv) == len(orig), (len(hv), len(orig)))
        check('save disabled before any change', page.is_disabled('#shape-save'))
        page.screenshot(path=os.path.join(OUT, 'shot_shape_open.png'))

        # 1. drag an interior point
        v = [h for h in hv if not h['end']][0]
        before = lines(page)[0][v['i']]
        drag(page, (v['x'], v['y']), (v['x'] + 45, v['y'] + 30))
        after = lines(page)[0][v['i']]
        check('dragging a point moves it', before != after, (before, after))
        check('same number of points after a drag', npoints(page) == len(orig))
        check('save enabled after a change', not page.is_disabled('#shape-save'))
        check('undo enabled', not page.is_disabled('#shape-undo'))

        # 2. drag a ⊕ out of a stretch
        mids = handles(page, 'sh-mid')
        check('⊕ handles drawn', len(mids) > 0, len(mids))
        # The ⊕ on the longest stretch on screen, well clear of its points.
        hv0 = handles(page)
        at = {(h['l'], h['i']): h for h in hv0}
        def span(m):
            a, c = at.get((m['l'], m['i'])), at.get((m['l'], m['i'] + 1))
            return ((a['x'] - c['x']) ** 2 + (a['y'] - c['y']) ** 2) ** .5 if a and c else 0
        m = max(mids, key=span)
        drag(page, (m['x'], m['y']), (m['x'] - 30, m['y'] + 40))
        check('dragging a ⊕ adds a point', npoints(page) == len(orig) + 1, npoints(page))

        # 3. undo / redo
        page.click('#shape-undo'); time.sleep(0.15)
        check('undo takes the new point back', npoints(page) == len(orig))
        page.click('#shape-redo'); time.sleep(0.15)
        check('redo puts it back', npoints(page) == len(orig) + 1)
        page.keyboard.press('Control+z'); time.sleep(0.15)
        check('Ctrl+Z undoes', npoints(page) == len(orig))

        # 4. tap a point, delete it
        hv = handles(page)
        v = [h for h in hv if not h['end']][0]
        tap(page, v['x'], v['y'])
        check('tapping a point opens its menu', page.is_visible('.shape-pop [data-op="del-v"]'))
        check('an interior point offers a split', page.is_visible('.shape-pop [data-op="split"]'))
        page.screenshot(path=os.path.join(OUT, 'shot_shape_point.png'))
        pop_click(page, 'del-v')
        check('delete removes the point', npoints(page) == len(orig) - 1, npoints(page))
        page.click('#shape-undo'); time.sleep(0.15)

        # 5. tap a middle stretch (a quarter of the way along, off its ⊕), delete it
        hv = [h for h in handles(page) if h['l'] == 0]
        hv.sort(key=lambda h: h['i'])
        # The longest middle stretch, so a quarter along is clear of both ends.
        k = max(range(1, len(hv) - 2), key=lambda j: (hv[j]['x'] - hv[j + 1]['x']) ** 2 + (hv[j]['y'] - hv[j + 1]['y']) ** 2)
        a, c = hv[k], hv[k + 1]
        tx, ty = a['x'] + (c['x'] - a['x']) * 0.28, a['y'] + (c['y'] - a['y']) * 0.28
        tap(page, tx, ty)
        sel = page.evaluate('() => Shape._state().sel')
        check('tapping a stretch selects it', sel and sel['t'] == 'e', sel)
        page.screenshot(path=os.path.join(OUT, 'shot_shape_edge.png'))
        pop_click(page, 'del-e')
        ls = lines(page)
        check('deleting a middle stretch leaves two pieces', len(ls) == 2, [len(l) for l in ls])
        check('the bar says so', 'חלקים' in page.inner_text('#shape-state'))

        # 6. join them again: drag one piece's end onto the other's
        hv = handles(page)
        e0 = [h for h in hv if h['l'] == 0 and h['end'] and h['i'] > 0][0]
        e1 = [h for h in hv if h['l'] == 1 and h['end'] and h['i'] == 0][0]
        drag(page, (e0['x'], e0['y']), (e1['x'] + 3, e1['y'] + 2), steps=20)
        ls = lines(page)
        check('dropping an end on an end joins the pieces', len(ls) == 1, [len(l) for l in ls])

        # 7. carry the line on from its last point
        hv = handles(page)
        last = [h for h in hv if h['end'] and h['i'] > 0][0]
        tap(page, last['x'], last['y'])
        check('an end offers "המשך מכאן"', page.is_visible('.shape-pop [data-op="extend"]'))
        pop_click(page, 'extend')
        n0 = npoints(page)
        tap(page, last['x'] + 60, last['y'] - 70)
        tap(page, last['x'] + 120, last['y'] - 90)
        check('each tap carries the line on', npoints(page) == n0 + 2, (n0, npoints(page)))
        page.screenshot(path=os.path.join(OUT, 'shot_shape_extend.png'))
        pop_click(page, 'done-extend')
        check('done leaves the carrying-on', page.evaluate('() => !Shape._state().extend'))

        # 8. the mirror, then a reload offers it back
        want = lines(page)
        mirror = page.evaluate("() => JSON.parse(localStorage.getItem('dk.shape.v1'))")
        check('the session is mirrored', mirror and mirror['lines'] == want)
        page.goto(f'http://127.0.0.1:{PORT}/index.html')
        boot()
        page.evaluate('() => Shape.restore()')
        time.sleep(1)
        check('an interrupted session is offered back',
              any('שלא נשמרה' in d for d in dialogs), dialogs[-2:])
        check('...and comes back as it was', page.evaluate('() => Shape.isOn()') and lines(page) == want)

        # 9. save
        page.click('#shape-save')
        time.sleep(1)
        saved = page.evaluate('() => window.__saved')
        check('save hands the line to Store.reshape', saved and saved['id'] == tid
              and saved['pieces'] == want, saved and saved['id'])
        check('the editor closed', page.evaluate('() => !Shape.isOn() && !document.body.classList.contains("shaping")'))
        check('the trail is back on its layer with its new line', page.evaluate(
            '''(id) => Layers.item(id).path.length''', tid) == len(want[0]))
        check('mirror cleared', page.evaluate("() => localStorage.getItem('dk.shape.v1') === null"))

        # 10. Escape / cancel with changes asks first
        page.evaluate('(id) => select(id)', tid)
        time.sleep(0.4)
        page.click('#detail [data-pub="shape"]'); time.sleep(1)
        hv = handles(page)
        v = hv[len(hv) // 2]
        drag(page, (v['x'], v['y']), (v['x'] + 20, v['y'] + 20))
        nd = len(dialogs)
        page.click('#shape-cancel'); time.sleep(0.3)
        check('leaving with changes asks first', len(dialogs) == nd + 1 and 'בלי לשמור' in dialogs[-1])
        check('...and leaves', not page.evaluate('() => Shape.isOn()'))

        # 11. a draft
        page.evaluate('''(path) => new Promise((done) => {
          const req = indexedDB.open('derech-kitzur', 1);
          req.onsuccess = () => {
            const t = req.result.transaction('drafts', 'readwrite');
            t.objectStore('drafts').put({ id: 'draft-shape-test', name: 'טיוטת בדיקה', note: '',
              path, mode: 'walk', created: Date.now(), updated: Date.now(), photos: [] });
            t.oncomplete = () => done();
          };
        })''', orig)
        page.goto(f'http://127.0.0.1:{PORT}/index.html')
        boot()
        page.evaluate("() => select('draft-shape-test')")
        time.sleep(0.6)
        page.click('#detail [data-draft="edit"]'); time.sleep(1.2)
        check('a draft opens in the shape editor', page.evaluate('() => Shape.kind() === "draft"'))
        hv = handles(page)
        v = [h for h in hv if not h['end']][0]
        tap(page, v['x'], v['y'])
        pop_click(page, 'del-v')
        page.click('#shape-save'); time.sleep(1)
        stored = page.evaluate('''() => new Promise((done) => {
          const req = indexedDB.open('derech-kitzur', 1);
          req.onsuccess = () => {
            const g = req.result.transaction('drafts').objectStore('drafts').get('draft-shape-test');
            g.onsuccess = () => done(g.result.path.length);
          };
        })''')
        check('the draft is saved with one point fewer', stored == len(orig) - 1, stored)

        # 12. phone width: the bar fits and the handles are reachable
        ctx2 = b.new_context(viewport={'width': 390, 'height': 800}, locale='he-IL',
                             is_mobile=True, has_touch=True, device_scale_factor=2)
        ctx2.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); } catch (e) {}")
        ph = ctx2.new_page()
        ph.on('pageerror', lambda e: errors.append('phone: ' + str(e)))
        ph.on('dialog', lambda d: d.accept())
        ph.goto(f'http://127.0.0.1:{PORT}/index.html')
        ph.wait_for_function(
            '() => typeof map !== "undefined" && map.loaded() && !!map.getSource("src-trails")', timeout=60000)
        time.sleep(3)
        ph.evaluate('() => document.querySelectorAll(".sheet").forEach((x) => { x.hidden = true; })')
        ph.evaluate(STUB)
        ph.evaluate('(id) => { select(id); Shape.open(Layers.item(id)); }', tid)
        time.sleep(1.5)
        hv = handles(ph)
        v = [h for h in hv if not h['end']][0]
        ph.touchscreen.tap(v['x'], v['y']); time.sleep(0.3)
        ph.screenshot(path=os.path.join(OUT, 'shot_shape_phone.png'))
        bar = ph.evaluate('() => { const r = document.getElementById("shape-bar").getBoundingClientRect(); return [r.width, r.height]; }')
        check('phone: bar within the screen', bar[0] <= 390, bar)
        check('phone: a tap selects a point', ph.evaluate('() => !!Shape._state().sel'))
        ctx2.close()

        check('no page errors', not errors, errors[:3])
        b.close()
finally:
    srv.terminate()

print('\n' + ('ALL OK' if not fails else f'{len(fails)} FAILED: ' + ', '.join(fails)))
sys.exit(1 if fails else 0)
