"""Splitting a long recording into the trails inside it (web/split.js).

Drives a real Chrome through the split mode on the longest recording in the
data (a walk with Shahar, as of 10/2026), with the editor's write stubbed so
nothing reaches the data repo, and on a draft in IndexedDB for real:

  * the button is on an editor's recording page, and not on a short trail;
  * opening analyses the line into new / street / known pieces, names the new
    ones after streets, and shows them in the list and on the map;
  * the toggle in the list keeps or drops a piece, and undo takes it back;
  * a tap on the line selects a piece and "חתוך כאן" cuts it; a tap on a ✂
    and "הסר את החיתוך" joins it again;
  * dragging a ✂ moves the boundary along the recording and nowhere else;
  * a typed name sticks and survives a boundary move;
  * saving hands the kept pieces, named, to Store.splitOut;
  * an interrupted split is offered back after a reload;
  * a draft is split into drafts in IndexedDB.

    python3 tests/test_split.py

Needs Playwright with Google Chrome (channel='chrome'). About a minute.
"""
import subprocess, sys, time, os
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8774

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

STUB = r'''() => {
  Store.isEditor = () => true;
  window.__split = null;
  Store.splitOut = async (id, pieces, keepOrig, name) => {
    window.__split = { id, pieces, keepOrig, name };
    return { doc: JSON.parse(JSON.stringify(DATA)), ids: ['x1'] };
  };
}'''

def st(page, expr):
    return page.evaluate('() => { const s = Split._state(); return s && (' + expr + '); }')

def pieces(page):
    return st(page, 's.pieces.map((p) => ({a: p.a, b: p.b, kind: p.kind, keep: p.keep, name: p.name, custom: p.custom}))')

def tap(page, x, y):
    page.mouse.move(x, y); page.mouse.down(); page.mouse.up(); time.sleep(0.2)

def drag(page, a, b, steps=14):
    page.mouse.move(a[0], a[1]); page.mouse.down()
    for k in range(1, steps + 1):
        page.mouse.move(a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps)
        time.sleep(0.01)
    page.mouse.up(); time.sleep(0.25)

def screen_of(page, i):
    """Where point i of the recording is on the screen."""
    return page.evaluate('''(i) => { const s = Split._state(); const r = map.getCanvas().getBoundingClientRect();
      const q = map.project([s.path[i][1], s.path[i][0]]); return [q.x + r.left, q.y + r.top]; }''', i)

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(channel='chrome', headless=True,
                              args=['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'])
        ctx = b.new_context(viewport={'width': 1200, 'height': 900}, locale='he-IL')
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1');"
                            " localStorage.removeItem('dk.editor.v1'); localStorage.removeItem('dk.shape.v1'); } catch (e) {}")
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
        page.evaluate("() => localStorage.removeItem('dk.split.v1')")
        boot()

        tid = page.evaluate('''() => Layers.trailLayers().flatMap((l) => l.segments)
          .filter((s) => Split.canSplit(s)).sort((a, b) => b.path.length - a.path.length)[0].id''')
        short = page.evaluate('''() => Layers.byId('trails').segments.find((s) => s.path.length < 10).id''')
        info = page.evaluate('(id) => ({ name: Layers.item(id).name, n: Layers.item(id).path.length })', tid)
        print('  recording', tid, info)

        page.evaluate('(id) => select(id)', short); time.sleep(0.5)
        check('no split button on a short trail', page.locator('#detail [data-pub="split"]').count() == 0)
        page.evaluate('(id) => select(id)', tid); time.sleep(0.6)
        check('recording page offers "פירוק למסלולים"', page.locator('#detail [data-pub="split"]').count() == 1)
        page.click('#detail [data-pub="split"]')
        page.wait_for_function('() => Split.isOn()', timeout=20000)
        time.sleep(1.2)

        check('split mode on, bar shown, list shown',
              page.evaluate('''() => document.body.classList.contains("splitting")
                && !document.getElementById("split-bar").hidden
                && !document.getElementById("split-view").hidden
                && document.getElementById("detail-view").hidden'''))
        check('map is flat', page.evaluate('() => map.getPitch() < 1'))
        P = pieces(page)
        kinds = {k: sum(1 for x in P if x['kind'] == k) for k in 'OST'}
        print('  pieces', len(P), kinds)
        check('found new, street and known pieces', all(kinds[k] > 0 for k in 'OST'), kinds)
        check('pieces cover the recording end to end, sharing boundaries',
              P[0]['a'] == 0 and P[-1]['b'] == info['n'] - 1
              and all(P[k]['b'] == P[k + 1]['a'] for k in range(len(P) - 1)))
        check('only the new pieces are kept', all(x['keep'] == (x['kind'] == 'O') for x in P))
        named = [x['name'] for x in P if x['keep']]
        check('new pieces are named after streets', sum(' - ' in n for n in named) >= len(named) // 2, named[:5])
        check('one row per piece', page.locator('#split-list .sp-row').count() == len(P))
        check('one ✂ per boundary on screen or off',
              page.locator('.split-svg .sp-cut').count() <= len(P) - 1 and page.locator('.split-svg .sp-cut').count() > 0)
        save_lbl = page.text_content('#split-save')
        check('save says how many', save_lbl == f'שמור {kinds["O"]}', save_lbl)

        # toggle a street piece on from the list, then undo
        si = next(i for i, x in enumerate(P) if x['kind'] == 'S')
        page.click(f'#split-list [data-tg="{si}"]'); time.sleep(0.2)
        check('list toggle keeps a street piece', pieces(page)[si]['keep'] is True)
        check('kept piece now has a name field', page.locator(f'#split-list .sp-name[data-k="{si}"]').count() == 1)
        page.click('#split-undo'); time.sleep(0.2)
        check('undo drops it again', pieces(page)[si]['keep'] is False)

        # the longest new piece: focus it from the list
        k = max((i for i, x in enumerate(P) if x['kind'] == 'O'), key=lambda i: P[i]['b'] - P[i]['a'])
        page.click(f'#split-list .sp-row[data-k="{k}"] .sp-len'); time.sleep(1.0)
        check('row click selects the piece', st(page, 's.sel && s.sel.t === "p" && s.sel.k') == k)

        # tap the middle of it on the map and cut there
        mid = (P[k]['a'] + P[k]['b']) // 2
        x, y = screen_of(page, mid)
        tap(page, x, y)
        check('tap on the line selects that piece', st(page, 's.sel && s.sel.k') == k)
        check('menu offers a cut', page.locator('.split-pop [data-op="cut"]').count() == 1)
        page.click('.split-pop [data-op="cut"]'); time.sleep(0.2)
        P2 = pieces(page)
        check('cut makes one more piece, both kept', len(P2) == len(P) + 1 and P2[k]['keep'] and P2[k + 1]['keep'])
        check('the two halves meet', P2[k]['b'] == P2[k + 1]['a'] and P[k]['a'] < P2[k]['b'] < P[k]['b'])

        # tap the new ✂ and remove the cut
        cx, cy = screen_of(page, P2[k + 1]['a'])
        tap(page, cx, cy)
        check('tap on ✂ selects the cut', st(page, 's.sel && s.sel.t') == 'c')
        page.click('.split-pop [data-op="join"]'); time.sleep(0.2)
        check('removing the cut joins them back', pieces(page) == P)

        # rename it
        inp = page.locator(f'#split-list .sp-name[data-k="{k}"]')
        inp.fill('שביל הבדיקה'); page.click('.split-title'); time.sleep(0.2)
        check('typed name sticks', pieces(page)[k]['name'] == 'שביל הבדיקה' and pieces(page)[k]['custom'])

        # drag the ✂ at the start of piece k along the line
        page.evaluate('''(k) => { const s = Split._state(); const p = s.pieces[k];
          const b = new maplibregl.LngLatBounds(); s.path.slice(Math.max(0, p.a - 40), p.b).forEach(([la, ln]) => b.extend([ln, la]));
          map.fitBounds(b, { padding: 120, duration: 0 }); }''', k)
        time.sleep(0.8)
        a0 = P[k]['a']
        target = a0 + max(3, (P[k]['b'] - a0) // 3)
        fx, fy = screen_of(page, a0)
        tx, ty = screen_of(page, target)
        drag(page, (fx, fy), (tx, ty))
        P3 = pieces(page)
        check('dragging ✂ moves the boundary forward along the recording',
              P3[k]['a'] > a0 and P3[k - 1]['b'] == P3[k]['a'], (a0, P3[k]['a'], target))
        check('typed name survives the move', P3[k]['name'] == 'שביל הבדיקה')
        drag(page, (tx, ty), (tx + 4000, ty + 4000), steps=6)
        P4 = pieces(page)
        check('a ✂ cannot leave its two pieces', P4[k - 1]['a'] < P4[k]['a'] < P4[k]['b'])
        page.click('#split-undo'); time.sleep(0.2)

        # reload and get offered the work back
        before_reload = pieces(page)
        page.reload(); boot(); time.sleep(1)
        page.evaluate('() => Split.restore()'); time.sleep(1.5)
        check('interrupted split offered back', any('נשאר פירוק' in d for d in dialogs), dialogs[-1:] if dialogs else None)
        check('restored with the same pieces', Split_on := page.evaluate('() => Split.isOn()') and pieces(page) == before_reload)

        # save
        keep_n = sum(1 for x in pieces(page) if x['keep'])
        check('keeping the recording is the default for a recording in its own layer',
              page.evaluate('() => document.getElementById("split-keep-orig").checked'))
        page.click('#split-save'); time.sleep(1.2)
        saved = page.evaluate('() => window.__split')
        check('save hands the kept pieces to Store.splitOut',
              saved and saved['id'] == tid and len(saved['pieces']) == keep_n and saved['keepOrig'] is True,
              saved and (len(saved['pieces']), keep_n))
        check('saved names are the list names',
              saved and saved['pieces'][0]['name'] and any(pc['name'] == 'שביל הבדיקה' for pc in saved['pieces']))
        check('saved paths are slices of the recording',
              saved and all(len(pc['path']) >= 2 for pc in saved['pieces']))
        check('split mode closed and list restored',
              not page.evaluate('() => Split.isOn()') and not page.evaluate('() => document.body.classList.contains("splitting")'))
        check('mirror cleared after save', page.evaluate("() => localStorage.getItem('dk.split.v1') === null"))

        # a draft, for real, in IndexedDB
        path = page.evaluate('(id) => Layers.item(id).path', tid)
        page.evaluate('''async (path) => {
          const req = indexedDB.open('derech-kitzur', 1);
          await new Promise((r) => { req.onsuccess = r; });
          const db = req.result;
          const store = 'drafts';
          await new Promise((r) => { const t = db.transaction(store, 'readwrite');
            t.objectStore(store).put({ id: 'draft-splittest', name: 'הקלטת בדיקה', note: '', path, mode: 'walk',
              layer: '', color: '', created: Date.now(), updated: Date.now(), photos: [] });
            t.oncomplete = r; });
        }''', path)
        page.reload(); boot(); time.sleep(1)
        n_before = page.evaluate("() => Layers.byId('drafts').segments.length")
        page.evaluate("() => select('draft-splittest')"); time.sleep(0.6)
        check('draft page offers "פירוק למסלולים"', page.locator('#detail [data-draft="split"]').count() == 1)
        page.click('#detail [data-draft="split"]')
        page.wait_for_function('() => Split.isOn()', timeout=20000); time.sleep(0.8)
        check('a draft is replaced by its pieces by default',
              not page.evaluate('() => document.getElementById("split-keep-orig").checked'))
        # The draft is a copy of a published recording, so all of it is
        # "already on the map": keep two pieces by hand.
        check('a copy of a published recording is all known', all(x['kind'] == 'T' for x in pieces(page)))
        for i in range(min(2, len(pieces(page)))):
            if not pieces(page)[i]['keep']:
                page.click(f'#split-list [data-tg="{i}"]'); time.sleep(0.2)
        kd = sum(1 for x in pieces(page) if x['keep'])
        page.click('#split-save'); time.sleep(1.5)
        n_after = page.evaluate("() => Layers.byId('drafts').segments.length")
        check('draft split into drafts', n_after == n_before - 1 + kd, (n_before, kd, n_after))
        names = page.evaluate("() => Layers.byId('drafts').segments.map((s) => s.name)")
        check('original draft gone', 'הקלטת בדיקה' not in names)

        check('no page errors', not errors, errors[:3])
        b.close()
finally:
    srv.terminate()

print('\n' + ('ALL PASSED' if not fails else f'{len(fails)} FAILED: ' + ', '.join(fails)))
sys.exit(1 if fails else 0)
