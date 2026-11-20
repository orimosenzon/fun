"""The two jet bikes in a real Chrome: the first one (the FLYING HOG, the
default) and the F-5 (FÖHN's, for whoever swaps to it), and the swap.

The F-5 is imported from ../../jetbikes/foehn, so the server here runs on the
folder above the project, the way GitHub Pages serves it. The flight itself
is checked in Node (tests/test_f5.mjs) and the first bike in test_bike.py;
this is the seam: which bike a visitor gets, that it is drawn and flies over
the map, that N and the model button swap the two where the bike stands,
that the choice is remembered, that `?bike=f5` asks for the new one, and
that a visitor who never chose gets the first bike.

    python3 tests/test_f5.py

Needs Playwright with Google Chrome (channel='chrome') and the network (map
tiles, and Three.js from the CDN for the first bike). About a minute.
Screenshots land beside this file.
"""
import math, subprocess, sys, time, os
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))      # vibe_coding/
PORT = 8772
URL = f'http://127.0.0.1:{PORT}/dereh_kitzur/web/index.html'

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

bike = lambda page: page.evaluate('() => Explore.debug().bike')
kmh = lambda d: math.hypot(d['vx'], d['vz']) * 3.6

def open_app(ctx, query):
    page = ctx.new_page()
    page.errors = []
    page.on('pageerror', lambda e: page.errors.append(str(e)))
    page.on('console', lambda m: page.errors.append(m.text) if m.type == 'error' else None)
    page.goto(URL + query)
    page.wait_for_function('() => typeof Explore !== "undefined" && typeof map !== "undefined" && map && map.loaded()', timeout=30000)
    time.sleep(1)
    page.evaluate('() => { const w = document.getElementById("welcome-sheet"); if (w) w.hidden = true; }')
    return page

def take_off(page):
    page.click('#explore')
    page.wait_for_function('() => Explore.debug().flying', timeout=30000)
    time.sleep(1.2)

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(channel='chrome', headless=True, args=[
            '--autoplay-policy=no-user-gesture-required', '--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'])
        ctx = b.new_context(viewport={'width': 1440, 'height': 900}, locale='he-IL')
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); } catch (e) {}")

        # ---- the F-5, asked for in the address ----
        page = open_app(ctx, '?craft=bike&bike=f5')
        take_off(page)
        d = bike(page)
        check('?bike=f5 opens on the F-5', d['model'] == 'f5', d['model'])
        check('parked on the ground, drawn, on its own canvas', d['y'] < 0.1 and d['visible'] and d['canvas'], d)
        check('its card is the one shown', page.evaluate(
            '() => getComputedStyle(document.querySelector(".fly-intro-card.f5")).display !== "none" && getComputedStyle(document.querySelector(".fly-intro-card.hog")).display === "none"'))
        check('the model button says F-5', page.inner_text('#fly-model b') == 'F-5')
        page.screenshot(path=os.path.join(OUT, 'shot_f5_intro.png'))
        page.keyboard.press('Slash'); time.sleep(0.3)
        page.screenshot(path=os.path.join(OUT, 'shot_f5_parked.png'))

        page.keyboard.down('KeyW'); time.sleep(0.5); page.keyboard.up('KeyW')
        time.sleep(3.5)
        d = bike(page)
        check('W takes it off and it flies the way the map faced', d['y'] > 1 and kmh(d) > 20, (d['y'], kmh(d)))
        h0 = d['heading']
        page.keyboard.down('ArrowRight'); time.sleep(0.8); page.keyboard.up('ArrowRight')
        d = bike(page)
        turned = (d['heading'] - h0 + 540) % 360 - 180
        check('right arrow turns it right, banked right', turned > 5 and d['bank'] > 5, (turned, d['bank']))
        page.screenshot(path=os.path.join(OUT, 'shot_f5_turn.png'))
        m = page.evaluate('() => Explore.debug().bike.mapCam')
        check('the map follows the bike', m is not None)

        # ---- N: the first bike, standing where this one was ----
        before = bike(page)
        page.keyboard.press('KeyN')
        page.wait_for_function('() => Explore.debug().bike.model === "hog"', timeout=30000)
        time.sleep(1.5)
        d = bike(page)
        check('N swaps to the first bike, parked under where the F-5 was',
              d['model'] == 'hog' and math.hypot(d['x'] - before['x'], d['z'] - before['z']) < 60 and d['y'] < 1.5, d)
        check('its card and its button follow', page.inner_text('#fly-model b') == 'HOG' and page.evaluate(
            '() => !document.body.classList.contains("fly-f5")'))
        check('only one canvas is showing', page.evaluate(
            '() => [...document.querySelectorAll("canvas.fly-bike")].filter((c) => !c.hidden).length') == 1)
        page.screenshot(path=os.path.join(OUT, 'shot_f5_swapped_hog.png'))
        check('the choice is remembered', page.evaluate('() => localStorage.getItem("dk.bikeModel.v2")') == 'hog')

        # ---- the button: back to the F-5 ----
        page.click('#fly-model')
        page.wait_for_function('() => Explore.debug().bike.model === "f5"', timeout=30000)
        time.sleep(1)
        d = bike(page)
        check('the model button swaps back to the F-5', d['model'] == 'f5' and d['visible'] and d['y'] < 0.1, d)
        check('no errors on the way', not page.errors, page.errors[:4])
        page.close()

        # ---- the F-5 was the last one chosen, so it is the one that comes back ----
        page = open_app(ctx, '?craft=bike')
        take_off(page)
        check('the last bike chosen is the one a return visit gets', bike(page)['model'] == 'f5')
        # hovering, for a look at the flames over the photograph
        page.keyboard.press('Slash')
        page.keyboard.down('KeyE'); time.sleep(2.2); page.keyboard.up('KeyE'); time.sleep(1.5)
        page.screenshot(path=os.path.join(OUT, 'shot_f5_hover.png'))
        page.evaluate('() => localStorage.removeItem("dk.bikeModel.v2")')
        page.close()

        # ---- a visitor who never chose gets the first bike ----
        page = open_app(ctx, '?craft=bike')
        take_off(page)
        check('with nothing chosen, the bike is the first one', bike(page)['model'] == 'hog')
        page.close()
        b.close()
finally:
    srv.terminate()

print('\n' + (f'{len(fails)} FAILED: ' + '; '.join(fails) if fails else 'all passed'))
sys.exit(1 if fails else 0)
