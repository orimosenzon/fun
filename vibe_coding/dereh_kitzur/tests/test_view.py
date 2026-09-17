"""The view button: what hangs over the ground, in a real Chrome.

Takes off as the jet over the moshava, waits for photos to hang over the
trails, and steps the view through its three positions with the key and
with the button: normal (photos, names, lit trails), clean (trails alone),
trails (no photos, names, trails lit hard). Then checks that the choice is
kept between flights, and that with no trail on the map the button has two
positions and a stored 'trails' comes back as 'clean'. Same rig as
test_f16.py:

    python3 tests/test_view.py

Needs Playwright with Google Chrome (channel='chrome') and --use-angle=gl.
Takes about a minute. Screenshots land beside this file.
"""
import subprocess, sys, time, os
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8767

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

def dbg(page):
    return page.evaluate('() => Explore.debug()')

def look(page):
    """The button as the eye sees it: its position, its word, its icon."""
    return page.evaluate('''() => {
      const b = document.getElementById('fly-look');
      const shown = (cls) => getComputedStyle(b.querySelector('.' + cls)).display !== 'none';
      return { view: b.dataset.view, name: b.querySelector('b').textContent, title: b.title,
               icons: { normal: shown('look-normal'), clean: shown('look-clean'), trails: shown('look-trails') },
               body: document.body.dataset.flyView,
               core: map.getPaintProperty('fly-trail-core', 'line-width'),
               glow: map.getPaintProperty('fly-trail-glow', 'line-color') };
    }''')

def take_off(page):
    page.click('#explore')
    page.wait_for_function('() => Explore.debug().flying', timeout=15000)
    page.evaluate('() => document.getElementById("fly-intro").click()')   # dismiss intro
    time.sleep(0.3)

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(channel='chrome', headless=True, args=[
            '--autoplay-policy=no-user-gesture-required',
            '--use-gl=angle', '--use-angle=gl',
            '--ignore-gpu-blocklist'])
        ctx = b.new_context(viewport={'width': 1440, 'height': 900}, locale='he-IL')
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); localStorage.setItem('dk.fly.craft', 'jet'); localStorage.removeItem('dk.fly.view'); } catch (e) {}")
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
        page.goto(f'http://127.0.0.1:{PORT}/index.html')
        page.wait_for_function('() => typeof Explore !== "undefined" && typeof map !== "undefined" && map && map.loaded()', timeout=30000)
        time.sleep(1)
        page.evaluate('() => { const w = document.getElementById("welcome-sheet"); if (w) w.hidden = true; }')

        # ---- normal: the flight as it was ----
        take_off(page)
        d = dbg(page)
        check('starts in the normal view', d['view'] == 'normal', d['view'])
        check('three positions with the shortcuts on', d['views'] == ['normal', 'clean', 'trails'] and d['hasTrails'], d['views'])
        L = look(page)
        check('button says רגיל with the photo icon', L['name'] == 'רגיל' and L['icons'] == {'normal': True, 'clean': False, 'trails': False}, L)
        check('title names the next position', 'נקי' in L['title'] and 'V' in L['title'], L['title'])
        check('body carries the view', L['body'] == 'normal', L['body'])
        core_normal, glow_normal = L['core'], L['glow']
        # photos hang over the trails once the reveal has run a few times
        page.wait_for_function('() => Explore.debug().cards > 0', timeout=15000)
        time.sleep(1.5)
        d = dbg(page)
        print(f'    normal: {d["cards"]} cards, {d["chips"]} chips, {len(d["lit"])} trails lit, max glow {max(d["lit"] or [0]):.2f}')
        check('photos hang in the normal view', d['cards'] > 0, d['cards'])
        check('names shown in the normal view', d['chips'] > 0, d['chips'])
        lit_normal = sorted(d['lit'], reverse=True)
        page.screenshot(path=f'{OUT}/shot_view_normal.png')

        # ---- V: clean ----
        page.keyboard.press('KeyV')
        time.sleep(0.5)
        d = dbg(page)
        L = look(page)
        print(f'    clean: {d["cards"]} cards, {d["chips"]} chips, {len(d["lit"])} trails lit')
        check('V steps to clean', d['view'] == 'clean', d['view'])
        check('no photos in the clean view', d['cards'] == 0, d['cards'])
        check('no names in the clean view', d['chips'] == 0, d['chips'])
        check('trails still lit in the clean view', len(d['lit']) > 0 and max(d['lit']) > 0.3, d['lit'][:5])
        check('button says נקי with the struck-out icon', L['name'] == 'נקי' and L['icons'] == {'normal': False, 'clean': True, 'trails': False}, L)
        check('trail paint unchanged in clean', L['core'] == core_normal and L['glow'] == glow_normal, (L['core'], L['glow']))
        check('stays clean for two seconds', (time.sleep(2), dbg(page))[1]['cards'] == 0)
        page.screenshot(path=f'{OUT}/shot_view_clean.png')

        # ---- V: trails ----
        page.keyboard.press('KeyV')
        time.sleep(0.6)
        d = dbg(page)
        L = look(page)
        lit_trails = sorted(d['lit'], reverse=True)
        print(f'    trails: {d["cards"]} cards, {d["chips"]} chips, {len(d["lit"])} trails lit, max glow {max(d["lit"] or [0]):.2f}; core width {L["core"]}, glow {L["glow"]}')
        check('V steps to trails', d['view'] == 'trails', d['view'])
        check('no photos in the trails view', d['cards'] == 0, d['cards'])
        check('names shown in the trails view', d['chips'] > 0, d['chips'])
        check('button says שבילים with the trail icon', L['name'] == 'שבילים' and L['icons'] == {'normal': False, 'clean': False, 'trails': True}, L)
        check('core line wider', L['core'] != core_normal and L['core'][-1] > core_normal[-1], (core_normal, L['core']))
        check('glow a deeper amber', L['glow'] != glow_normal, L['glow'])
        # the same trails at the same place, lit harder: compare the brightest few
        n = min(3, len(lit_normal), len(lit_trails))
        check('trails lit harder', n > 0 and all(t >= s for t, s in zip(lit_trails[:n], lit_normal[:n])), (lit_normal[:n], lit_trails[:n]))
        check('body says trails', L['body'] == 'trails', L['body'])
        page.screenshot(path=f'{OUT}/shot_view_trails.png')

        # ---- the button itself, round to normal ----
        page.click('#fly-look')
        time.sleep(0.5)
        d = dbg(page)
        L = look(page)
        check('the button steps round to normal', d['view'] == 'normal' and L['name'] == 'רגיל', (d['view'], L['name']))
        check('photos back', d['cards'] > 0, d['cards'])
        check('trail paint back', L['core'] == core_normal and L['glow'] == glow_normal, (L['core'], L['glow']))
        check('a press on the button is not the throttle', d['throttle'] < 0.05, d['throttle'])

        # ---- kept between flights ----
        page.click('#fly-look')   # clean
        time.sleep(0.3)
        check('stored', page.evaluate('() => localStorage.getItem("dk.fly.view")') == 'clean')
        page.keyboard.press('Escape')
        page.wait_for_function('() => !Explore.isOn()', timeout=5000)
        time.sleep(1.5)
        check('body view cleared on landing', page.evaluate('() => document.body.dataset.flyView') is None)
        take_off(page)
        d = dbg(page)
        check('next flight takes off clean', d['view'] == 'clean' and look(page)['name'] == 'נקי', d['view'])
        page.keyboard.press('KeyV')   # trails, so the next case has it stored
        time.sleep(0.2)
        check('trails stored', page.evaluate('() => localStorage.getItem("dk.fly.view")') == 'trails')
        page.keyboard.press('Escape')
        page.wait_for_function('() => !Explore.isOn()', timeout=5000)
        time.sleep(1.5)

        # ---- no trail on the map: two positions ----
        page.evaluate('() => { Layers.clearAll(); Layers.turnOn("kitzur-spots"); }')
        time.sleep(0.5)
        take_off(page)
        d = dbg(page)
        L = look(page)
        print(f'    without trails: views {d["views"]}, view {d["view"]}')
        check('no trail in the world', not d['hasTrails'])
        check('two positions', d['views'] == ['normal', 'clean'], d['views'])
        check('a stored trails view comes back as clean', d['view'] == 'clean' and L['name'] == 'נקי', (d['view'], L['name']))
        check('and is stored as clean', page.evaluate('() => localStorage.getItem("dk.fly.view")') == 'clean')
        check('title offers normal, not trails', 'רגיל' in L['title'] and 'שבילים' not in L['title'].split(':')[-1], L['title'])
        page.keyboard.press('KeyV')
        time.sleep(0.2)
        check('V goes to normal', dbg(page)['view'] == 'normal')
        page.keyboard.press('KeyV')
        time.sleep(0.2)
        check('V goes to clean, never trails', dbg(page)['view'] == 'clean')
        check('asking for trails outright gives clean', page.evaluate('() => Explore.setView("trails")') == 'clean')
        page.keyboard.press('Escape')
        page.wait_for_function('() => !Explore.isOn()', timeout=5000)

        # ---- a phone: the button on a line of its own, under the three ----
        page.set_viewport_size({'width': 390, 'height': 844})
        time.sleep(0.5)
        page.evaluate('() => { Layers.turnOn("trails"); }')
        take_off(page)
        r = page.evaluate('''() => {
          const at = (id) => { const r = document.getElementById(id).getBoundingClientRect(); return { l: r.left, r: r.right, t: r.top, b: r.bottom }; };
          return { look: at('fly-look'), x: at('fly-x'), snd: at('fly-snd'), compass: at('fly-compass') };
        }''')
        print(f'    phone: look {r["look"]}, compass {r["compass"]}')
        below = r['look']['t'] >= r['x']['b'] - 2
        clear_of_compass = r['look']['l'] >= r['compass']['r'] or r['look']['r'] <= r['compass']['l'] or r['look']['t'] >= r['compass']['b']
        check('phone: button under the top line', below, r)
        check('phone: button clear of the compass', clear_of_compass, r)
        page.screenshot(path=f'{OUT}/shot_view_phone.png')
        page.keyboard.press('Escape')
        page.wait_for_function('() => !Explore.isOn()', timeout=5000)

        check('no page errors', not errors, errors[:3])
        b.close()
finally:
    srv.terminate()

print()
print(f'{len(fails)} failed' if fails else 'all passed')
for f in fails: print('  - ' + f)
sys.exit(1 if fails else 0)
