"""View presets (modes.js) in a real Chrome: הולך רגל, מתכנן, שקיפות.

A first visit is the walker's view. Each chip switches to exactly its layers,
flattens or tilts the map, and gives the panel its own cards; ticking one more
layer by hand lights no chip; and a link opens in the preset it was sent from.

    python3 tests/test_modes.py
"""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8772
URL = f'http://127.0.0.1:{PORT}/index.html'

fails = []
def check(name, ok, detail=''):
    print(('  ok   ' if ok else '  FAIL ') + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

READY = '() => typeof Modes !== "undefined" && typeof map !== "undefined" && !!map.getSource("src-trails")'
STATE = '''() => ({
  lit: [...document.querySelectorAll('#menu .mode.on')].map(b => b.dataset.mode),
  current: (Modes.current() || {}).id || null,
  on: Layers.onIds().sort(),
  pitch: Math.round(map.getPitch()),
  hint: document.getElementById('mode-hint').textContent,
  planCard: getComputedStyle(document.getElementById('plan-ask')).display,
  routeCard: getComputedStyle(document.getElementById('route-ask')).display,
  legend: Layers.legendIsOpen(),
  url: location.search})'''

def page(b, url, viewport={'width': 1280, 'height': 860}):
    ctx = b.new_context(viewport=viewport)
    pg = ctx.new_page()
    errs = []
    pg.on('pageerror', lambda e: errs.append('pageerror: ' + str(e)))
    pg.on('console', lambda m: errs.append('console: ' + m.text) if m.type == 'error' else None)
    pg.goto(url)
    pg.wait_for_function(READY, timeout=40000)
    pg.wait_for_timeout(1500)
    pg.evaluate('() => document.querySelectorAll(".sheet").forEach(s => s.hidden = true)')
    return ctx, pg, errs

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
  with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', args=['--use-gl=angle', '--use-angle=gl'])
    ctx, pg, errs = page(b, URL)

    s = pg.evaluate(STATE)
    print('first visit:', json.dumps(s, ensure_ascii=False))
    check('a first visit is the walker preset', s['lit'] == ['walk'], s['lit'])
    # Since 6/10/2026 the modes live in the ☰ menu and hide nothing: the
    # directions are on the map's top bar, the planning question in the menu.
    check('the directions button is on the map in every mode', s['routeCard'] != 'none', s['routeCard'])

    pg.click('#menu-btn')
    pg.wait_for_timeout(300)
    pg.click('#menu .mode[data-mode="plan"]')
    pg.wait_for_timeout(1200)
    s = pg.evaluate(STATE)
    print('planner:', json.dumps(s, ensure_ascii=False))
    check('מתכנן lights its chip', s['lit'] == ['plan'], s['lit'])
    check('and turns on land use, plans and the cycling network',
          all(x in s['on'] for x in ['landuse', 'plans', 'bike-existing', 'bike-proposed']), s['on'])
    check('and nothing of the walker it does not need', 'kitzur-spots' not in s['on'], s['on'])
    check('the map goes flat', s['pitch'] < 3, s['pitch'])
    check('the key opens, for the land-use colours', s['legend'])
    check('choosing a view leaves the menu open, to compare', pg.evaluate('() => Menu.isOpen()'))
    check('and hides no action', s['planCard'] != 'none' and s['routeCard'] != 'none')
    check('the link carries the layers', 'landuse' in s['url'], s['url'])
    pg.wait_for_timeout(1500)
    pg.screenshot(path=os.path.join(OUT, 'shot_mode_plan.png'))
    plan_url = URL + s['url']

    pg.click('#menu .mode[data-mode="open"]')
    pg.wait_for_timeout(1500)
    s = pg.evaluate(STATE)
    print('transparency:', json.dumps(s, ensure_ascii=False))
    check('שקיפות lights its chip', s['lit'] == ['open'], s['lit'])
    check('and shows the parcels and the plans', 'parcels' in s['on'] and 'plans' in s['on'], s['on'])
    check('and leaves the land use off', 'landuse' not in s['on'], s['on'])
    pg.screenshot(path=os.path.join(OUT, 'shot_mode_open.png'))

    # One more layer by hand: no preset any more, and every card back.
    pg.evaluate('() => Layers.turnOn("canopy")')
    pg.wait_for_timeout(400)
    s = pg.evaluate(STATE)
    check('a layer ticked by hand lights no chip', s['lit'] == [] and s['current'] is None, s['lit'])
    check('the hint says it is a view of your own', 'משלך' in s['hint'], s['hint'])
    check('and every action is still there', s['planCard'] != 'none' and s['routeCard'] != 'none')

    pg.click('#menu .mode[data-mode="walk"]')
    pg.wait_for_timeout(1200)
    s = pg.evaluate(STATE)
    check('הולך רגל goes back to the shortcuts', s['lit'] == ['walk'] and 'canopy' not in s['on'], s)
    check('and tilts the map again', s['pitch'] > 40, s['pitch'])

    # The chips in the layer sheet are the same chips.
    pg.click('#layers')
    pg.wait_for_timeout(400)
    pg.click('#layer-sheet .mode[data-mode="plan"]')
    pg.wait_for_timeout(600)
    s = pg.evaluate(STATE)
    sheet_ticks = pg.evaluate('''() => [...document.querySelectorAll('#layer-list .lay input:checked')]
        .map(i => i.closest('.lay').dataset.id).sort()''')
    check('the sheet has the presets too', s['lit'] == ['plan'], s['lit'])
    check('and its ticks follow', 'landuse' in sheet_ticks, sheet_ticks)
    pg.click('#layer-sheet .sheet-x')

    # A link opens in the preset it was sent from.
    ctx2, pg2, errs2 = page(b, plan_url)
    s = pg2.evaluate(STATE)
    check('a link sent from מתכנן opens in מתכנן', s['lit'] == ['plan'], s['lit'])
    errs.extend(errs2)
    ctx2.close()

    # Phone
    ctx3, pg3, errs3 = page(b, URL, {'width': 390, 'height': 800})
    pg3.screenshot(path=os.path.join(OUT, 'shot_modes_phone.png'))
    errs.extend(errs3)

    check('no page errors', not errs, errs[:4])
    b.close()
finally:
    srv.terminate()

print()
print(f'{len(fails)} failed' if fails else 'all passed')
for f in fails: print('  - ' + f)
sys.exit(1 if fails else 0)
