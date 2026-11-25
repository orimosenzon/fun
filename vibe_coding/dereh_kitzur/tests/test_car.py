"""The flying car in the flight mode, and the picker it is chosen from.

The flight itself is tested in Node (tests/test_car.mjs). This is the rest:
that the small button on the flight button opens all five aircraft by name
and that choosing one is all it takes, at desktop and at phone width; that
the car loads on its first take-off, stands on the ground, flies and turns;
that it has its two views, from behind with the whole car in the picture and
from the driver's seat with the cockpit around you; that it is not one of
the two bikes the swap button swaps; and that a link remembers it.

    python3 tests/test_car.py

Needs Playwright with Google Chrome. Serves vibe_coding/ itself, because the
car's model is the roadster project's and its flight is FÖHN's, imported
from their own folders. Takes about a minute and a half.
"""
import math, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))      # vibe_coding/
PORT = 8773
URL = f'http://127.0.0.1:{PORT}/dereh_kitzur/web/index.html'

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

car = lambda page: page.evaluate('() => Explore.debug().bike')
kmh = lambda d: math.hypot(d['vx'], d['vz']) * 3.6
alpha = lambda page, x, y: page.evaluate(f'() => Explore.carRig().alphaAt({x}, {y})')

def open_app(ctx, query=''):
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
    page.wait_for_function('() => Explore.debug().flying', timeout=40000)
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

        # ---- the picker ----
        page = open_app(ctx)
        pop = page.locator('#craft-pop')
        img = lambda: page.evaluate('() => document.querySelector("#explore img").getAttribute("src")')
        check('a first visit is the balloon, and the picker is closed',
              page.evaluate('() => Explore.getCraft()') == 'balloon' and not pop.is_visible())
        page.click('#craft')
        check('the small button opens it', pop.is_visible() and page.get_attribute('#craft', 'aria-expanded') == 'true')
        tiles = page.evaluate('() => [...document.querySelectorAll(".craft-tile")].map((t) => [t.dataset.craft, t.textContent.trim(), t.classList.contains("on")])')
        check('five aircraft, each with its name', [t[0] for t in tiles] == ['balloon', 'jet', 'bike', 'f5', 'car'] and all(t[1] for t in tiles), tiles)
        check('the one in use is marked, and only it', [t[0] for t in tiles if t[2]] == ['balloon'], tiles)
        check('every picture loaded', page.evaluate('() => [...document.querySelectorAll(".craft-tile img")].every((i) => i.complete && i.naturalWidth > 0)'))
        box = pop.bounding_box()
        check('all of it on the screen', box['x'] >= 0 and box['y'] >= 0 and box['x'] + box['width'] <= 1440 and box['y'] + box['height'] <= 900, box)
        page.screenshot(path=os.path.join(OUT, 'shot_craft_pick.png'))
        page.keyboard.press('Escape')
        check('Escape closes it and changes nothing', not pop.is_visible() and page.evaluate('() => Explore.getCraft()') == 'balloon')
        page.click('#craft')
        page.mouse.click(300, 300)
        check('so does a press anywhere else', not pop.is_visible())
        check('and the flight has not started', not page.evaluate('() => Explore.isOn()'))

        page.click('#craft')
        page.click('.craft-tile[data-craft="car"]')
        check('choosing the car closes it, and the big button wears the car',
              not pop.is_visible() and page.evaluate('() => Explore.getCraft()') == 'car' and img() == 'img/car.svg')
        check('the title says so', 'מכונית מעופפת' in page.get_attribute('#explore', 'title'))
        check('the choice is kept', page.evaluate('() => [localStorage.getItem("dk.fly.craft"), localStorage.getItem("dk.bikeModel.v2")]') == ['bike', 'car'])
        time.sleep(2)
        q = page.evaluate('() => location.search')
        check('and the address says craft=car', 'craft=car' in q, q)
        page.click('#craft')
        check('reopened, the car is the one marked',
              page.evaluate('() => [...document.querySelectorAll(".craft-tile.on")].map((t) => t.dataset.craft)') == ['car'])
        page.keyboard.press('Escape')
        check('nothing of the car is fetched before a flight', not page.evaluate('() => CarFly.ready()'))

        # ---- the flight ----
        take_off(page)
        d = car(page)
        check('in the air mode as the car', d['model'] == 'car' and page.evaluate('() => Explore.debug().pick') == 'car', d['model'])
        check('parked on the ground, drawn, on its own canvas', d['y'] < 0.1 and d['visible'] and d['canvas'], d)
        check('only one canvas is showing', page.evaluate(
            '() => [...document.querySelectorAll("canvas.fly-bike:not(.fly-rig)")].filter((c) => !c.hidden).length') == 1)
        check('its card is the one shown', page.evaluate(
            '() => ["car", "hog", "f5"].map((c) => getComputedStyle(document.querySelector(".fly-intro-card." + c)).display !== "none")') == [True, False, False])
        check('the bikes\' swap button is not offered', not page.locator('#fly-model').is_visible())
        page.screenshot(path=os.path.join(OUT, 'shot_car_intro.png'))
        page.keyboard.press('Slash'); time.sleep(0.3)
        check('from behind: the car is in the middle of the picture, and the corners are the map',
              alpha(page, 0.5, 0.52) > 0.9 and alpha(page, 0.05, 0.08) < 0.1 and alpha(page, 0.95, 0.08) < 0.1,
              [alpha(page, 0.5, 0.52), alpha(page, 0.05, 0.08)])
        page.screenshot(path=os.path.join(OUT, 'shot_car_parked.png'))

        page.keyboard.down('KeyW'); time.sleep(0.5); page.keyboard.up('KeyW')
        time.sleep(3.5)
        d = car(page)
        check('W takes it off and it flies the way the map faced', d['y'] > 1 and kmh(d) > 20, (d['y'], kmh(d)))
        h0 = d['heading']
        page.keyboard.down('ArrowRight'); time.sleep(1.0)
        d = car(page)
        page.screenshot(path=os.path.join(OUT, 'shot_car_turn.png'))
        page.keyboard.up('ArrowRight')
        turned = (d['heading'] - h0 + 540) % 360 - 180
        check('right arrow turns it right, banked right, and no further than a car banks', turned > 3 and 5 < d['bank'] < 58, (turned, d['bank']))
        check('from behind the picture leans only a little', abs(d['camRoll']) < 17, d['camRoll'])

        # ---- the driver's seat ----
        time.sleep(1.5)
        page.keyboard.press('KeyC'); time.sleep(0.3)
        d = car(page)
        check('C: the seat', d['cam'] == 1 and 'מתוך המכונית' in d['msg'], (d['cam'], d['msg']))
        time.sleep(0.8)
        check('the dashboard fills the bottom of the picture', alpha(page, 0.5, 0.85) > 0.9 and alpha(page, 0.3, 0.9) > 0.9,
              [alpha(page, 0.5, 0.85), alpha(page, 0.3, 0.9)])
        # (lower down the ground's grain is drawn too, faintly, as it is for the bikes)
        check('and the windscreen is clear: the map shows through', alpha(page, 0.5, 0.12) < 0.15 and alpha(page, 0.4, 0.4) < 0.4,
              [alpha(page, 0.5, 0.12), alpha(page, 0.4, 0.4)])
        check('the speed streaks stay off the dashboard', float(page.evaluate('() => document.getElementById("fly-rush").style.opacity || 0')) == 0)
        page.screenshot(path=os.path.join(OUT, 'shot_car_inside.png'))
        page.keyboard.down('ArrowLeft'); time.sleep(1.2)
        d = car(page)
        tf = page.evaluate('() => map.getContainer().style.transform')
        page.screenshot(path=os.path.join(OUT, 'shot_car_inside_turn.png'))
        page.keyboard.up('ArrowLeft')
        check('in a turn the world rolls and the cockpit does not', d['bank'] < -5 and abs(d['camRoll'] - d['bank']) < 12 and 'rotate' in tf
              and alpha(page, 0.5, 0.85) > 0.9, (d['bank'], d['camRoll'], tf))
        time.sleep(1.5)
        page.keyboard.press('KeyN'); time.sleep(0.8)
        check('N does not swap the car for a bike', car(page)['model'] == 'car')
        page.keyboard.press('KeyC'); time.sleep(0.3)
        d = car(page)
        check('C again: from behind, the whole car', d['cam'] == 0 and 'מבט מאחור' in d['msg'] and alpha(page, 0.12, 0.9) < 0.4, (d['cam'], d['msg'], alpha(page, 0.12, 0.9)))

        # ---- another aircraft, chosen in the air, is for the next flight ----
        page.keyboard.press('Escape')
        page.wait_for_function('() => !Explore.isOn()', timeout=8000)
        time.sleep(1.2)
        check('out: the car\'s canvas is put away, the picker\'s button is back',
              page.evaluate('() => [...document.querySelectorAll("canvas.fly-bike")].every((c) => c.hidden)') and page.locator('#craft').is_visible())
        check('no errors on the way', not page.errors, page.errors[:4])
        page.close()

        # ---- links ----
        page = open_app(ctx, '?craft=car')
        check('craft=car in the address opens on the car', page.evaluate('() => Explore.getCraft()') == 'car'
              and page.evaluate('() => document.querySelector("#explore img").getAttribute("src")') == 'img/car.svg')
        page.click('#craft'); page.click('.craft-tile[data-craft="f5"]')
        check('the F-5 has a tile of its own now', page.evaluate('() => Explore.getCraft()') == 'f5'
              and page.evaluate('() => localStorage.getItem("dk.bikeModel.v2")') == 'f5')
        page.click('#craft'); page.click('.craft-tile[data-craft="bike"]')
        check('and so has the first bike', page.evaluate('() => Explore.getCraft()') == 'bike'
              and page.evaluate('() => localStorage.getItem("dk.bikeModel.v2")') == 'hog')
        page.close()
        page = open_app(ctx, '?craft=f5')
        check('craft=f5 opens on the F-5', page.evaluate('() => Explore.getCraft()') == 'f5')
        page.close()
        page = open_app(ctx, '?craft=bike&bike=f5')
        check('an old link, craft=bike&bike=f5, still means the F-5', page.evaluate('() => Explore.getCraft()') == 'f5')
        page.close()
        ctx.close()

        # ---- a phone ----
        ctx = b.new_context(viewport={'width': 390, 'height': 844}, locale='he-IL', has_touch=True, is_mobile=True)
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); } catch (e) {}")
        page = open_app(ctx)
        time.sleep(3)   # the welcome sheet comes late on a phone
        page.evaluate('() => { const w = document.getElementById("welcome-sheet"); if (w) w.hidden = true; }')
        page.tap('#craft')
        box = page.locator('#craft-pop').bounding_box()
        check('phone: the picker is all on the screen', box and box['x'] >= 0 and box['y'] >= 0 and box['x'] + box['width'] <= 390 and box['y'] + box['height'] <= 844, box)
        over = page.evaluate('''() => { const r = document.querySelector('.craft-tile[data-craft="car"]').getBoundingClientRect();
            const t = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2); return !!(t && t.closest('.craft-tile')); }''')
        check('phone: nothing lies over its tiles', over)
        page.screenshot(path=os.path.join(OUT, 'shot_craft_pick_phone.png'))
        page.tap('.craft-tile[data-craft="car"]')
        check('phone: a tap chooses', page.evaluate('() => Explore.getCraft()') == 'car')
        page.tap('#explore')
        page.wait_for_function('() => Explore.debug().flying', timeout=40000)
        time.sleep(1.5)
        page.evaluate('() => document.getElementById("fly-intro").click()')
        time.sleep(0.3)
        page.screenshot(path=os.path.join(OUT, 'shot_car_phone.png'))
        page.evaluate('''() => { const b = [...document.querySelectorAll('.fly-touch .tmenu button')][0];
            b.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerId: 7 })); }''')
        time.sleep(1.0)
        d = car(page)
        check('phone: the camera button puts you in the seat', d['cam'] == 1, d['cam'])
        check('phone: the dashboard below, the windscreen above', alpha(page, 0.5, 0.9) > 0.9 and alpha(page, 0.5, 0.12) < 0.15,
              [alpha(page, 0.5, 0.9), alpha(page, 0.5, 0.12)])
        page.screenshot(path=os.path.join(OUT, 'shot_car_phone_inside.png'))
        check('phone: no errors', not page.errors, page.errors[:4])
        b.close()
finally:
    srv.terminate()

print()
print('all passed' if not fails else f'{len(fails)} FAILED: ' + ', '.join(fails))
sys.exit(1 if fails else 0)
