"""The F-16, flown in a real Chrome (rewritten 26/11/2026, when the jet became
an aeroplane: f16_core.js, f16_pit.js).

The flight model itself is checked in Node, against the clock and without a
browser (tests/test_f16core.mjs). This file checks what only the page can
show: that the map's camera is the pilot's eye (MapLibre 5's roll, and its
pitch above the horizon), that the cockpit is drawn, and that the mouse, the
keys and the touch controls reach the aircraft.

    python3 tests/test_f16.py

Needs Playwright with Google Chrome (channel='chrome') and --use-angle=gl.
About two minutes. Screenshots land beside this file.
"""
import subprocess, sys, time, os
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(OUT))       # vibe_coding: the jet imports three.js from FÖHN's folder
PORT = 8766
KT = 1.943844

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

def dbg(page): return page.evaluate('() => Explore.debug()')
def jet(page): return page.evaluate('() => Explore.debug().jet')
def wait(page, cond, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if page.evaluate(cond): return True
        time.sleep(0.1)
    return False
def adiff(a, b): return ((a - b + 540) % 360) - 180

def open_app(p, viewport, touch=False):
    b = p.chromium.launch(channel='chrome', headless=True, args=[
        '--autoplay-policy=no-user-gesture-required', '--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'])
    ctx = b.new_context(viewport=viewport, locale='he-IL', has_touch=touch, is_mobile=touch)
    ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); localStorage.setItem('dk.fly.craft', 'jet'); } catch (e) {}")
    page = ctx.new_page()
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
    page.goto(f'http://127.0.0.1:{PORT}/dereh_kitzur/web/index.html')
    page.wait_for_function('() => typeof Explore !== "undefined" && typeof map !== "undefined" && map && map.loaded()', timeout=30000)
    time.sleep(1)
    page.evaluate('() => { const w = document.getElementById("welcome-sheet"); if (w) w.hidden = true; }')
    return b, page, errors

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
    with sync_playwright() as p:
        b, page, errors = open_app(p, {'width': 1440, 'height': 900})
        check('MapLibre 5', page.evaluate('() => maplibregl.getVersion()').startswith('5.'))
        check('starts as jet', page.evaluate('() => Explore.getCraft()') == 'jet')
        before = page.evaluate('() => ({ maxPitch: map.getMaxPitch(), fov: map.getVerticalFieldOfView() })')

        # ---- take off ----
        page.click('#explore')
        page.wait_for_function('() => Explore.debug().flying', timeout=20000)
        d = dbg(page); j = d['jet']
        check('in the air as the jet', d['on'] and d['craft'] == 'jet' and j is not None)
        txt = page.locator('.fly-intro-card.jet').inner_text()
        check('intro names the F-16', 'F-16' in txt and 'נץ' in txt)
        check('intro explains the stick and the HUD', 'קצב גלגול' in txt and 'סמן נתיב הטיסה' in txt and 'קשר' in txt and 'רגל' in txt)
        check('intro lists gear, brakes, autopilot, look', all(k in txt for k in ['גלגלים', 'בלמי אוויר', 'טייס אוטומטי', 'מבט חופשי']))
        check('no long dash in the card', '—' not in txt)
        check('2,500 ft and 360 kt', abs(j['y'] - 760) < 25 and abs(j['cas'] * KT - 350) < 25, (j['y'], j['cas'] * KT))
        check('the cockpit is drawn', j['canvas'] and j['pit']['frame'] > 5 and j['pit']['mode'] == 'pit', j['pit'])
        check('the other aircrafts\' gauges and sky are away',
              not page.locator('#fly-compass').is_visible() and not page.locator('#fly-sky').is_visible()
              and not page.locator('.fly-thr').first.is_visible())
        check('the name of what was found is shown', page.locator('.fly-found').is_visible())
        page.screenshot(path=f'{OUT}/shot_f16_intro.png')

        # ---- the stick is the mouse, from its first movement ----
        page.mouse.move(720, 450); page.mouse.move(800, 450); page.mouse.move(720, 450)
        check('the card goes when the mouse moves', wait(page, '() => !Explore.debug().intro', 3))
        time.sleep(3.5)
        j = jet(page)
        check('hands off: wings level, path level', abs(j['roll']) < 2 and abs(j['vs']) < 4, (j['roll'], j['vs']))
        m = j['map']
        check('the map\'s camera is the pilot\'s eye: pitch', abs(m['pitch'] - (90 + j['pitch'] - 12)) < 1.0, (m['pitch'], j['pitch']))
        check('  roll, and the lens', abs(m['roll'] - j['roll']) < 1 and abs(m['fov'] - 60) < 0.5, m)
        page.screenshot(path=f'{OUT}/shot_f16_hud.png')
        f0 = jet(page)['pit']['frame']; time.sleep(2); fps = (jet(page)['pit']['frame'] - f0) / 2
        print(f'    cockpit frames a second: {fps:.0f}')
        check('one cockpit frame for each of the map\'s, and enough of them', 24 < fps < 70, fps)

        page.mouse.move(1150, 450); time.sleep(1.5); page.mouse.move(720, 450); time.sleep(0.4)
        j = jet(page)
        check('mouse to the right rolls right', j['roll'] > 25, j['roll'])
        check('the map rolls with the aircraft', abs(adiff(j['map']['roll'], j['roll'])) < 3, (j['map']['roll'], j['roll']))
        h0 = j['heading']
        page.mouse.move(720, 700); time.sleep(2.0)
        j = jet(page)
        check('pulling in the bank: g, and a turn to the right', j['maxG'] > 2 and adiff(j['heading'], h0) > 8, (j['maxG'], j['heading'], h0))
        page.screenshot(path=f'{OUT}/shot_f16_turn.png')
        page.mouse.move(720, 450)
        page.keyboard.press('r'); time.sleep(0.6)
        j = jet(page)
        check('R: level again, at a safe height', abs(j['roll']) < 3 and j['y'] > 700 and 'ישרה' in j['msg'], (j['roll'], j['y'], j['msg']))

        # ---- above the horizon ----
        page.mouse.move(720, 830); time.sleep(3.0)
        j = jet(page); d = dbg(page)
        check('a hard pull points the nose well above the horizon', j['pitch'] > 30, j['pitch'])
        check('and the map looks up with it, past ninety degrees', j['map']['pitch'] > 105, j['map']['pitch'])
        check('nothing on the ground is named in the sky', d['chips'] == 0 and d['cards'] == 0, (d['chips'], d['cards']))
        page.screenshot(path=f'{OUT}/shot_f16_up.png')
        page.mouse.move(720, 450); page.keyboard.press('r'); time.sleep(0.5)

        # ---- the throttle, the burner, the barrier ----
        page.keyboard.down('w'); time.sleep(1.4); page.keyboard.up('w')
        j = jet(page)
        check('W opens the throttle and it stays', j['lever'] > 0.99, j['lever'])
        page.keyboard.press('t'); time.sleep(0.3)
        check('T: the autopilot holds height', jet(page)['ap'] and 'טייס אוטומטי' in jet(page)['msg'])
        check('the burner lights', wait(page, '() => Explore.debug().jet.abf > 0.9', 6))
        check('through Mach 1 inside half a minute', wait(page, '() => Explore.debug().sonic', 30))
        d = dbg(page)
        check('one boom, and the supersonic look', d['booms'] == 1 and page.evaluate('() => document.body.classList.contains("fly-super")'))
        check('the height was held all the while', abs(d['jet']['y'] - 760) < 40 and abs(d['jet']['roll']) < 2, d['jet']['y'])
        for _ in range(24): page.mouse.wheel(0, 120); time.sleep(0.02)
        time.sleep(0.3)
        check('the wheel closes the throttle', jet(page)['lever'] < 0.05, jet(page)['lever'])
        page.keyboard.press('b'); time.sleep(0.3)
        check('B opens the speed brakes, and says so', 'בלמי אוויר' in jet(page)['msg'], jet(page)['msg'])
        check('  and they come out', wait(page, '() => Explore.debug().jet.sb > 0.9', 3))
        check('and it comes back below Mach 1', wait(page, '() => !Explore.debug().sonic', 25))
        page.keyboard.press('b')
        page.mouse.move(720, 450); page.mouse.move(1350, 450); time.sleep(0.5); page.mouse.move(720, 450)
        check('a hand on the stick takes it from the autopilot', wait(page, '() => !Explore.debug().jet.ap', 3))
        page.keyboard.press('r'); time.sleep(0.4)

        # ---- the head ----
        page.mouse.move(720, 450); page.mouse.down(button='right'); page.mouse.move(380, 540, steps=8); time.sleep(0.7)
        j = jet(page)
        check('right button held: the mouse turns the head, not the aircraft', j['look']['yaw'] < -0.8 and abs(j['roll']) < 4, (j['look'], j['roll']))
        check('and the map looks where the head does', adiff(j['map']['bearing'], j['heading']) < -40, (j['map']['bearing'], j['heading']))
        page.screenshot(path=f'{OUT}/shot_f16_look.png')
        page.mouse.up(button='right'); time.sleep(1.0)
        check('released, the eyes come back to the front', abs(jet(page)['look']['yaw']) < 0.05, jet(page)['look'])
        page.keyboard.down('z'); time.sleep(0.9)
        check('Z zooms', jet(page)['fov'] < 32, jet(page)['fov'])
        page.keyboard.up('z'); time.sleep(0.9)
        page.keyboard.press('c'); time.sleep(1.0)
        j = jet(page)
        check('C puts the cockpit away: the HUD alone, through a longer lens', j['mode'] == 'hud' and j['pit']['mode'] == 'hud' and abs(j['fov'] - 42) < 1.5, (j['mode'], j['fov']))
        page.screenshot(path=f'{OUT}/shot_f16_hudonly.png')
        page.keyboard.press('c'); time.sleep(0.8)
        check('and back', jet(page)['mode'] == 'pit')

        # ---- the hold ----
        page.keyboard.press('p'); time.sleep(0.3)
        a = jet(page); time.sleep(1.0); c = jet(page)
        check('P holds the flight where it is', c['hold'] and abs(a['x'] - c['x']) < 0.01 and abs(a['z'] - c['z']) < 0.01)
        check('the cockpit goes on being drawn', c['pit']['frame'] >= a['pit']['frame'])
        page.keyboard.press('p'); time.sleep(0.2)

        # ---- the gear, and a landing ----
        page.evaluate('() => { const a = Explore.jetAc(); a.reset({ x: a.pos[0], z: a.pos[2], alt: 45, bearing: 30, speed: 76 }); a.lever = 0.32; }')
        page.keyboard.press('g')
        check('G lowers the gear', wait(page, '() => Explore.debug().jet.gear > 0.99', 5))
        # A crude pilot on short finals: hold three metres a second down, then flare.
        t0 = time.time()
        while time.time() - t0 < 30:
            j = jet(page)
            if j['grounded'] or j['crashed']: break
            want = -3.0 if j['y'] > 5 else -0.8
            sy = max(-0.3, min(0.5, 0.08 * (want - j['vs'])))
            page.evaluate(f'() => {{ const a = Explore.jetAc(); a.lever = Math.max(0, Math.min(1, 0.4 + 0.03 * (74 - a.tas))); }}')
            page.mouse.move(720, 450 + (45 + abs(sy) ** (1 / 1.6) * 410) * (1 if sy > 0 else -1))
            time.sleep(0.06)
        page.mouse.move(720, 450)
        j = jet(page)
        check('a landing is a landing', j['grounded'] and not j['crashed'], (j['crashWhy'], j['vs']))
        check('and says so', 'נחיתה' in j['msg'], j['msg'])
        page.screenshot(path=f'{OUT}/shot_f16_ground.png')
        page.evaluate('() => { Explore.jetAc().lever = 0; }')
        page.keyboard.down('Space')
        check('the wheel brakes stop it', wait(page, '() => Explore.debug().jet.tas < 0.5', 25))
        page.keyboard.up('Space')
        j = jet(page)
        check('the eye is a pilot\'s height above the ground', 1.5 < dbg(page)['alt'] < 3.5, dbg(page)['alt'])

        # ---- and a take-off from where it stopped ----
        page.keyboard.down('w'); time.sleep(1.2); page.keyboard.up('w')
        check('full burner on the ground: it rolls', wait(page, '() => Explore.debug().jet.cas * 1.944 > 150', 25))
        page.mouse.move(720, 885)
        check('pull at 150 knots: off the ground', wait(page, '() => !Explore.debug().jet.grounded', 8))
        page.mouse.move(720, 560); time.sleep(2.5); page.mouse.move(720, 450)
        j = jet(page)
        check('climbing away', j['y'] > 15 and not j['crashed'], j['y'])

        # ---- the ground, with the wheels up ----
        page.keyboard.press('g'); time.sleep(0.2)
        page.evaluate('() => { const a = Explore.jetAc(); a.reset({ x: a.pos[0], z: a.pos[2], alt: 160, bearing: 30, speed: 170 }); }')
        page.mouse.move(720, 250)
        check('flown into the ground: a crash', wait(page, '() => Explore.debug().jet.crashed', 12))
        check('named for what it was', 'התרסקות' in jet(page)['msg'], jet(page)['msg'])
        page.mouse.move(720, 450)
        check('and back in the air a few seconds later', wait(page, '() => { const j = Explore.debug().jet; return !j.crashed && j.y > 600; }', 6))

        # ---- out ----
        page.keyboard.press('Escape'); time.sleep(2.2)
        after = page.evaluate('() => ({ maxPitch: map.getMaxPitch(), fov: map.getVerticalFieldOfView(), roll: map.getRoll(), clamped: map.getCenterClampedToGround(), pitch: map.getPitch() })')
        check('the map has its own camera back', abs(after['maxPitch'] - before['maxPitch']) < 0.01 and abs(after['fov'] - before['fov']) < 0.01
              and abs(after['roll']) < 0.01 and after['clamped'] and after['pitch'] <= before['maxPitch'], after)
        check('no errors on the console', not errors, errors[:4])
        b.close()

        # ---- a phone ----
        b, page, errors = open_app(p, {'width': 915, 'height': 412}, touch=True)
        page.evaluate('() => document.getElementById("explore").click()')
        page.wait_for_function('() => Explore.debug().flying', timeout=20000)
        page.evaluate('() => { const i = document.getElementById("fly-intro"); if (i) i.hidden = true; }')
        time.sleep(1.5)
        j = jet(page)
        check('phone: the cockpit starts put away', j['mode'] == 'hud', j['mode'])
        check('phone: the joystick, the throttle and the jet\'s own buttons', page.locator('.fly-touch .tjoy').is_visible()
              and page.locator('.fly-touch .tjet').is_visible() and not page.locator('.fly-touch .tboost').is_visible())
        box = page.locator('.fly-touch .tjoy').bounding_box()
        cx, cy = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
        page.mouse.move(cx, cy); page.mouse.down(); page.mouse.move(cx + 22, cy, steps=3); time.sleep(0.5)
        r = jet(page)['roll']
        page.mouse.up()
        check('phone: the joystick rolls the aircraft', r > 15, r)
        page.locator('.fly-touch .tjet button').nth(0).dispatch_event('pointerdown')
        check('phone: G lowers the gear', wait(page, '() => Explore.debug().jet.gearDown', 2))
        page.screenshot(path=f'{OUT}/shot_f16_phone.png')
        check('phone: no errors', not errors, errors[:4])
        b.close()
finally:
    srv.kill()

print()
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
