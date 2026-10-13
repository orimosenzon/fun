"""The tree-survey layer and its explorer, in a real Chrome.

    python3 tests/test_trees.py

Walks what somebody does with it: opens a link that carries a cut (`tr=`),
checks the map, the counts and the key agree with the cut, opens one tree's
card and comes back to the explorer, presses a preset, and makes sure the
address bar now carries the new cut. Needs Playwright with Google Chrome
(channel='chrome'). About half a minute.
"""
import json, os, subprocess, sys, time
from urllib.parse import quote
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8773

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

DOC = json.load(open(os.path.join(WEB, 'data', 'trees.json')))
TREES = DOC['places']
hollow_by_road = [p for p in TREES if p['tree']['health'] == 'חלול'
                  and set(p['tree'].get('near', [])) & {'מדרכה', 'כביש'}]

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
try:
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel='chrome', args=['--use-gl=angle', '--ignore-gpu-blocklist'])
        pg = b.new_page(viewport={'width': 1300, 'height': 900})
        errors = []
        pg.on('pageerror', lambda e: errors.append(str(e)))
        cut = 'c.near_health.' + quote('חלול') + '_near.' + quote('מדרכה') + '*' + quote('כביש') + '_o'
        pg.goto(f'http://127.0.0.1:{PORT}/?tr={cut}')
        pg.wait_for_function('typeof map !== "undefined" && map && map.loaded() && Layers.byId("trees")', timeout=30000)
        pg.wait_for_timeout(1500)
        pg.evaluate("document.getElementById('welcome-sheet') && (document.getElementById('welcome-sheet').hidden = true)")

        check('the link opens the explorer', pg.is_visible('#trees-view'))
        check('the layer is switched on', pg.evaluate('Layers.byId("trees").on'))
        shown = pg.evaluate('Layers.byId("trees").waypoints.filter(p => !p.hidden).length')
        check('the cut matches the survey', shown == len(hollow_by_road), (shown, len(hollow_by_road)))
        drawn = pg.evaluate('''map.getSource("src-trees")._data.features.length''')
        check('the map draws exactly the cut', drawn == shown, (drawn, shown))
        check('the count says so', f'{shown}' in pg.inner_text('.tx-count'))
        key = pg.evaluate('Layers.byId("trees").groups.map(g => g.name)')
        check('the key is the colouring cut', key[:2] == ['מדרכה', 'כביש'], key)

        # One tree's card, and back.
        tid = pg.evaluate('Layers.byId("trees").waypoints.find(p => !p.hidden).id')
        pg.evaluate(f'select("{tid}", false)')
        pg.wait_for_timeout(500)
        detail = pg.inner_text('#detail')
        check('the card shows the survey values', 'מצב בריאות' in detail and 'חלול' in detail, detail[:200])
        check('the card says it is the 2018-2019 state', 'ייתכן שמצבו השתנה' in detail)
        check('the explorer steps aside for the card', not pg.is_visible('#trees-view'))
        pg.click('#back')
        pg.wait_for_timeout(300)
        check('back from the card returns to the explorer', pg.is_visible('#trees-view'))

        # A preset replaces the whole cut and lands in the address bar.
        pg.click('[data-tx="preset"]:has-text("העצים העתיקים")')
        pg.wait_for_timeout(800)
        ancient = sum(1 for p in TREES if p['tree'].get('age') == 'עתיק')
        shown = pg.evaluate('Layers.byId("trees").waypoints.filter(p => !p.hidden).length')
        check('the preset shows the ancient trees', shown == ancient, (shown, ancient))
        url = pg.evaluate('decodeURIComponent(location.search)')
        check('the address bar carries the preset', 'tr=c.species_age.עתיק_o' in url, url)
        check('the other layers stay in the address bar', 'layers=' in url and 'trees' in url, url)

        # The list counts only what the filter lets through.
        pg.click('[data-tx="close"]')
        pg.wait_for_timeout(300)
        listed = pg.evaluate('Layers.visibleWaypoints().filter(p => p.tree).length')
        check('the list obeys the filter', listed == ancient, (listed, ancient))

        # Reset puts every tree back.
        pg.evaluate('Trees.open()')
        pg.click('[data-tx="reset"]')
        pg.wait_for_timeout(500)
        shown = pg.evaluate('Layers.byId("trees").waypoints.filter(p => !p.hidden).length')
        check('reset shows the whole survey', shown == len(TREES), (shown, len(TREES)))

        check('no page errors', not errors, errors[:3])
        b.close()
finally:
    srv.terminate()

print(f'\n{"נכשלו " + str(len(fails)) if fails else "הכל עבר"}')
sys.exit(1 if fails else 0)
