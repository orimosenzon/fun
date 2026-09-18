"""The jet bike, checked against the clock in a real Chrome.

Runs a local server on web/, opens the app, steps the aircraft button round
to the bike, takes off, and reads the flight model through Explore.debug()
while pressing the keys and moving the mouse the way a person would. The
audio is tapped through an AnalyserNode hung off the compressor so the
engine can be measured, not assumed. The scene's camera is checked against
the map's: the bike has to sit on the ground the map draws.

    python3 tests/test_bike.py

Needs Playwright with Google Chrome (channel='chrome') and the angle/gl
backend, for the same reason as the balloon's test: under swiftshader the
flight model's clock runs at a twelfth of real time. Fetches Three.js from
the CDN on the first take-off, so it needs the network. About three
minutes. Screenshots land beside this file.
"""
import json, math, statistics, subprocess, sys, time, os
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8769

TAP = """
(() => {
  const AC = window.AudioContext;
  const orig = AC.prototype.createDynamicsCompressor;
  AC.prototype.createDynamicsCompressor = function () {
    const comp = orig.call(this);
    const an = this.createAnalyser();
    an.fftSize = 2048;
    comp.connect(an);
    window.__tap = { ctx: this, an };
    return comp;
  };
})();
"""

RMS = """
() => {
  if (!window.__tap) return { rms: -1, state: 'none' };
  const { an, ctx } = window.__tap;
  const buf = new Float32Array(an.fftSize);
  an.getFloatTimeDomainData(buf);
  let s = 0; for (const v of buf) s += v * v;
  return { rms: Math.sqrt(s / buf.length), state: ctx.state };
}
"""

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

def dbg(page):
    return page.evaluate('() => Explore.debug()')

def bike(page):
    return page.evaluate('() => Explore.debug().bike')

def hspd(d):
    return math.hypot(d['vx'], d['vz'])

def rms(page):
    return page.evaluate(RMS)['rms']

def centre(page):
    page.mouse.move(720, 450); page.mouse.move(721, 451)

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
        ctx.add_init_script(TAP)
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); } catch (e) {}")
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
        page.goto(f'http://127.0.0.1:{PORT}/index.html')
        page.wait_for_function('() => typeof Explore !== "undefined" && typeof Bike !== "undefined" && typeof map !== "undefined" && map && map.loaded()', timeout=30000)
        time.sleep(1)
        page.evaluate('() => { const w = document.getElementById("welcome-sheet"); if (w) w.hidden = true; }')

        # ---- the small button goes round: balloon, jet, bike, balloon ----
        craft = page.locator('#craft')
        img = lambda: page.evaluate('() => document.querySelector("#explore img").getAttribute("src")')
        shown = lambda cls: page.evaluate(f'() => getComputedStyle(document.querySelector("#craft .{cls}")).display')
        check('starts as balloon, small button shows the jet',
              page.evaluate('() => Explore.getCraft()') == 'balloon' and shown('craft-jet') == 'block' and shown('craft-bike') == 'none')
        craft.click()
        check('one press: jet; small button shows the bike',
              page.evaluate('() => Explore.getCraft()') == 'jet' and img() == 'img/f16.svg' and shown('craft-bike') == 'block' and shown('craft-bal') == 'none')
        check('body says jet', page.evaluate('() => document.body.classList.contains("craft-jet") && !document.body.classList.contains("craft-balloon")'))
        craft.click()
        check('two presses: bike; small button shows the balloon',
              page.evaluate('() => Explore.getCraft()') == 'bike' and img() == 'img/bike.svg' and shown('craft-bal') == 'block' and shown('craft-jet') == 'none')
        check('titles say bike, and balloon next',
              'אופנוע' in page.locator('#explore').get_attribute('title') and 'כדור' in craft.get_attribute('title'))
        check('bike choice is kept', page.evaluate('() => localStorage.getItem("dk.fly.craft")') == 'bike')
        ex = page.locator('#explore').bounding_box()
        page.screenshot(path=f'{OUT}/shot_fab_bike.png', clip={'x': ex['x'] - 40, 'y': ex['y'] - 20, 'width': 120, 'height': 100})
        craft.click()
        check('three presses: round to the balloon', page.evaluate('() => Explore.getCraft()') == 'balloon' and img() == 'img/balloon.svg')
        craft.click(); craft.click()
        check('back to the bike', page.evaluate('() => Explore.getCraft()') == 'bike')
        check('three.js not fetched before a flight', not page.evaluate('() => Bike.ready()'))

        # ---- take off ----
        t0 = time.time()
        page.click('#explore')
        page.wait_for_function('() => Explore.debug().flying', timeout=40000)
        print(f'    airborne after {time.time() - t0:.1f} s (three.js fetched on the way)')
        d = dbg(page)
        check('in the mode as the bike', d['on'] and d['craft'] == 'bike')
        check('three.js loaded, rig built', page.evaluate('() => Bike.ready()') and d['bike'] is not None)
        check('body carries fly-bike', page.evaluate('() => document.body.classList.contains("fly-bike") && !document.body.classList.contains("fly-balloon")'))
        check('bike canvas shown, the window\'s size',
              d['bike']['canvas'] and page.evaluate('() => { const c = document.getElementById("fly-bike"); const r = c.getBoundingClientRect(); return r.width === innerWidth && r.height === innerHeight; }'))
        check('map ceilings raised for the bike', page.evaluate('() => map.getMaxPitch()') == 85 and page.evaluate('() => map.getMaxZoom()') == 24)
        check('bike intro shown, the others hidden',
              page.locator('.fly-intro-card.bike').is_visible() and not page.locator('.fly-intro-card.jet').is_visible() and not page.locator('.fly-intro-card.balloon').is_visible())
        check('lift bar and flight computer gauge shown', page.locator('#fly-lift').is_visible() and page.locator('#fly-fc-g').is_visible())
        check('mach and g hidden', not page.locator('#fly-mach-g').is_visible() and not page.locator('#fly-g-g').is_visible())
        check('lift bar sits under the throttle bar',
              page.locator('#fly-lift').bounding_box()['y'] > page.locator('.fly-thr').first.bounding_box()['y'] + 6)
        check('compass shows the wind pointer', page.locator('#fly-drift').is_visible())
        page.screenshot(path=f'{OUT}/shot_bike_intro.png')

        # ---- parked ----
        k = d['bike']
        check('on the ground, engine off', k['grounded'] and k['throttle'] == 0 and k['y'] < 0.7, k)
        check('facing the way the map faced', abs(((k['heading'] - d['bearing'] + 540) % 360) - 180) < 2, (k['heading'], d['bearing']))
        check('chase camera: behind and above, looking down a little',
              k['pose']['pitch'] < 82.5 and 74 < k['pose']['pitch'] and 2 < k['mapCam']['y'] < 5, k['pose'])
        # the scene's camera is rebuilt from the map's; the two must agree
        w, m = k['want'], k['mapCam']
        err = math.hypot(w['x'] - m['x'], w['y'] - m['y'], w['z'] - m['z'])
        check('scene camera sits where the chase camera wants, to a few cm', err < 0.15, (err, w, m))
        check('and looks the same way', abs(w['dx'] - m['dx']) + abs(w['dy'] - m['dy']) + abs(w['dz'] - m['dz']) < 0.02, (w, m))
        check('window field of view is the bike\'s', abs(m['vfov'] - 42) < 0.5, m['vfov'])
        idle = max(rms(page) for _ in range(20))
        print(f'    idle rms {idle:.4f}, ctx {page.evaluate(RMS)["state"]}')
        check('engine audible at idle', idle > 0.004, idle)

        # dismiss the intro with the mouse, take the stick
        centre(page)
        page.evaluate('() => document.getElementById("fly-intro").click()')
        time.sleep(0.2); centre(page); time.sleep(0.2)
        check('intro gone, stick armed', not dbg(page)['intro'] and dbg(page)['armed'])
        page.screenshot(path=f'{OUT}/shot_bike_parked.png')

        # ---- lift off: E climbs, W accelerates, the lever law ----
        page.keyboard.down('KeyE'); time.sleep(2.5); page.keyboard.up('KeyE')
        k = bike(page)
        check('E lifts off and climbs', not k['grounded'] and k['y'] > 6 and k['vert'] > 0.5, k)
        check('altitude gauge reads it', abs(int(page.evaluate('() => parseInt(document.getElementById("fly-alt").textContent)')) - k['y']) < 3)
        page.keyboard.down('KeyW'); time.sleep(0.45); page.keyboard.up('KeyW')
        time.sleep(0.3)
        k = bike(page)
        check('half a second of W is half a lever, a quarter of the engine', 0.15 < k['throttle'] < 0.36, k['throttle'])
        time.sleep(2.5)
        loud = max(rms(page) for _ in range(20))
        print(f'    under throttle rms {loud:.4f}')
        k2 = bike(page)
        check('the lever stays where W left it', abs(k2['throttle'] - k['throttle']) < 0.02, (k['throttle'], k2['throttle']))
        check('and the bike moves off', hspd(k2) > 5 and hspd(k2) > hspd(k) + 3, (hspd(k), hspd(k2)))
        check('engine louder under throttle than at idle, and in the jet\'s league', loud > idle * 1.15 and loud > 0.04, (idle, loud))
        check('speed gauge in km/h', abs(int(page.evaluate('() => parseInt(document.getElementById("fly-speed").textContent)')) - hspd(k2) * 3.6) < 12)
        check('altitude held with the nose level', abs(k2['vy']) < 2.5, k2['vy'])
        page.screenshot(path=f'{OUT}/shot_bike_cruise.png')

        # ---- the mouse is the lean: left of centre banks left and turns left ----
        h0 = bike(page)['heading']
        page.mouse.move(330, 450); time.sleep(2.2)
        k = bike(page)
        turned = ((k['heading'] - h0 + 540) % 360) - 180
        print(f'    mouse left: bank {k["bank"]:.1f}, camRoll {k["camRoll"]:.1f}, turned {turned:.1f}')
        check('cursor left banks left', k['bank'] < -8, k['bank'])
        check('and turns left', turned < -8, turned)
        check('picture leans a share of the bank, the same way', -k['bank'] * 0.45 > -k['camRoll'] > -k['bank'] * 0.15, (k['bank'], k['camRoll']))
        check('map bearing follows the camera', abs(((dbg(page)['bearing'] - page.evaluate('() => (map.getBearing() + 360) % 360') + 540) % 360) - 180) < 25)
        check('map container rolled by the lean',
              abs(float(page.evaluate('() => document.getElementById("map").style.transform.match(/rotate\\((-?[\\d.]+)deg\\)/)[1]')) + k['camRoll']) < 1.5)
        page.screenshot(path=f'{OUT}/shot_bike_turn.png')
        centre(page); time.sleep(1.5)
        k = bike(page)
        check('centred, wings level again', abs(k['bank']) < 6, k['bank'])

        # ---- and below centre raises the nose and climbs ----
        y0 = bike(page)['y']
        page.mouse.move(720, 720); time.sleep(1.6)
        k = bike(page)
        check('cursor below centre raises the nose', k['nose'] > 8, k['nose'])
        check('and the path follows it: climbing', k['vy'] > 2 and k['y'] > y0 + 1, (k['vy'], y0, k['y']))
        centre(page); time.sleep(1.5)
        k = bike(page)
        check('nose back on the horizon', abs(k['nose']) < 5, k['nose'])

        # ---- a hard turn at speed holds its height (the spiral-dive fix) ----
        page.keyboard.down('KeyW'); time.sleep(0.5); page.keyboard.up('KeyW')
        page.keyboard.down('KeyE'); time.sleep(1.5); page.keyboard.up('KeyE')
        time.sleep(2)
        y0 = bike(page)['y']
        page.keyboard.down('ArrowRight')
        ys, banks, noses = [], [], []
        for _ in range(10):
            time.sleep(0.4); k = bike(page); ys.append(k['y']); banks.append(k['bank']); noses.append(k['nose'])
        page.keyboard.up('ArrowRight')
        print(f'    hard turn at {hspd(k) * 3.6:.0f} km/h: bank max {max(banks):.0f}, nose {min(noses):.1f}..{max(noses):.1f}, height {min(ys):.0f}..{max(ys):.0f} from {y0:.0f}')
        check('banks to the cap, not past it', 42 < max(banks) < 56, max(banks))
        check('nose stays near the horizon through the turn', min(noses) > -8 and max(noses) < 8, (min(noses), max(noses)))
        check('height held within a few metres', max(ys) - min(ys) < 6 and abs(ys[-1] - y0) < 6, (y0, ys))
        check('not crashed', bike(page)['crashes'] == 0)
        time.sleep(1.5)
        check('levels itself when the stick is released', abs(bike(page)['bank']) < 8, bike(page)['bank'])

        # ---- the boost, the pulses ----
        page.keyboard.down('Space'); time.sleep(0.8)
        k = bike(page)
        check('Space lights the afterburner', k['ab'] > 0.8, k['ab'])
        check('burn gauge shown', page.locator('#fly-burn').is_visible())
        page.keyboard.up('Space'); time.sleep(0.6)
        check('and it goes out', bike(page)['ab'] < 0.2)
        quiet = rms(page)
        page.keyboard.press('ShiftLeft')
        chuff = max(rms(page) for _ in range(30))
        time.sleep(0.4)
        check('a pulse is heard', chuff > quiet * 1.2 or chuff > 0.02, (quiet, chuff))
        b0 = bike(page)['bank']
        page.keyboard.press('KeyZ'); time.sleep(0.25)
        check('Z pulses a roll', abs(bike(page)['bank'] - b0) > 1.5, (b0, bike(page)['bank']))
        time.sleep(1.5)

        # ---- the computer, the level, the camera ----
        page.keyboard.press('KeyT'); time.sleep(0.2)
        check('T switches the flight computer off, gauge says so', not bike(page)['assist'] and page.evaluate('() => document.getElementById("fly-fc").textContent') == 'ידני')
        check('a message says it', 'כבוי' in bike(page)['msg'])
        page.keyboard.press('KeyT'); time.sleep(0.2)
        check('and on again', bike(page)['assist'])
        page.keyboard.press('KeyX'); time.sleep(0.3)
        page.keyboard.press('KeyR'); time.sleep(0.2)
        check('R levels the bike', abs(bike(page)['bank']) < 2 and abs(bike(page)['nose']) < 2, (bike(page)['bank'], bike(page)['nose']))
        ox0 = page.evaluate('() => getComputedStyle(document.documentElement).getPropertyValue("--fly-ox")')
        page.keyboard.press('KeyC'); time.sleep(0.6)
        k = bike(page)
        ox1 = page.evaluate('() => getComputedStyle(document.documentElement).getPropertyValue("--fly-ox")')
        check('C goes onboard: the frame becomes the jet\'s square', k['cam'] == 1 and ox1 != ox0, (ox0, ox1))
        check('onboard camera is at the rider\'s eyes', abs(k['mapCam']['x'] - k['x']) < 1.2 and abs(k['mapCam']['z'] - k['z']) < 1.2 and 0 < k['mapCam']['y'] - k['y'] < 1.6, (k['mapCam'], k['x'], k['y'], k['z']))
        page.screenshot(path=f'{OUT}/shot_bike_onboard.png')
        page.keyboard.press('KeyC'); time.sleep(0.6)
        check('C again: chase, frame back', bike(page)['cam'] == 0 and page.evaluate('() => getComputedStyle(document.documentElement).getPropertyValue("--fly-ox")') == ox0)

        # ---- the view button and the gallery still work from the bike ----
        page.keyboard.press('KeyV'); time.sleep(0.2)
        check('V cycles the view', dbg(page)['view'] == 'clean')
        page.keyboard.press('KeyV'); page.keyboard.press('KeyV'); time.sleep(0.2)
        check('and round to normal', dbg(page)['view'] == 'normal')
        page.wait_for_function('() => !document.getElementById("fly-name").hidden', timeout=20000)
        page.keyboard.press('Enter'); time.sleep(0.5)
        check('Enter opens the nearest trail\'s pictures and holds the flight',
              not page.evaluate('() => document.getElementById("fly-view").hidden') and page.evaluate('() => document.body.classList.contains("fly-paused")'))
        x0 = bike(page)['x']
        time.sleep(0.8)
        check('held: the bike does not move', abs(bike(page)['x'] - x0) < 0.5)
        page.keyboard.press('Escape'); time.sleep(0.3)
        check('Escape closes it, flight resumes', page.evaluate('() => document.getElementById("fly-view").hidden') and dbg(page)['on'])
        centre(page)

        # ---- the brake, and a landing ----
        page.keyboard.down('KeyS'); time.sleep(0.8)   # the lever to idle
        v0 = hspd(bike(page))
        time.sleep(3)
        v1 = hspd(bike(page))
        page.keyboard.up('KeyS')
        print(f'    brake: {v0 * 3.6:.0f} -> {v1 * 3.6:.0f} km/h in 3 s')
        check('S at idle brakes hard', v1 < v0 * 0.7 and bike(page)['throttle'] == 0, (v0, v1))
        page.keyboard.down('KeyD')
        page.wait_for_function('() => Explore.debug().bike.grounded', timeout=40000)
        time.sleep(1.2); page.keyboard.up('KeyD')
        k = bike(page)
        check('D brings it down onto the ground, in one piece', k['grounded'] and k['crashes'] == 0 and k['y'] < 0.7, k)
        page.screenshot(path=f'{OUT}/shot_bike_landed.png')

        # ---- the crash, and what comes after ----
        page.keyboard.down('KeyE'); time.sleep(2.5); page.keyboard.up('KeyE')
        page.keyboard.down('KeyW'); time.sleep(1.2); page.keyboard.up('KeyW')
        time.sleep(1)
        page.mouse.move(720, 110)   # nose hard down
        page.wait_for_function('() => Explore.debug().bike.crashes > 0', timeout=20000)
        centre(page)
        time.sleep(0.3)
        k = bike(page)
        check('a dive into the ground is a crash: bike gone, message up', not k['visible'] and 'ריסוק' in k['msg'] and page.locator('#fly-msg').is_visible())
        check('the wreck\'s throttle is dropped', k['throttle'] == 0)
        page.screenshot(path=f'{OUT}/shot_bike_crash.png')
        time.sleep(3.6)
        k = bike(page)
        check('respawns where it fell, level, on the ground, engine off', k['visible'] and not k['crashed'] and k['grounded'] and abs(k['nose']) < 2 and k['throttle'] == 0, k)
        check('the crash is counted once', k['crashes'] == 1)

        # ---- out ----
        page.keyboard.press('Escape')
        time.sleep(1.6)
        d = dbg(page)
        check('Escape leaves the mode', not d['on'])
        check('map ceilings back', page.evaluate('() => map.getMaxPitch()') == 80 and page.evaluate('() => map.getMaxZoom()') == 22)
        check('canvas hidden, classes gone, transform cleared',
              page.evaluate('() => document.getElementById("fly-bike").hidden && !document.body.classList.contains("fly-bike") && !document.body.classList.contains("flying") && document.getElementById("map").style.transform === ""'))
        check('the buttons are back', page.locator('#explore').is_visible() and craft.is_visible())
        check('engine quiet after leaving', max(rms(page) for _ in range(10)) < 0.01)

        # ---- a second flight is instant, and the rig is reused ----
        t0 = time.time()
        page.click('#explore')
        page.wait_for_function('() => Explore.debug().flying', timeout=20000)
        print(f'    second take-off after {time.time() - t0:.1f} s')
        k = bike(page)
        check('second flight: parked again at the new place, no crash carried over', k['grounded'] and k['crashes'] == 0 and k['throttle'] == 0 and k['visible'])
        page.keyboard.press('Escape'); time.sleep(1.2)

        # ---- a phone: the joystick and the buttons ----
        page.close()
        ctx2 = b.new_context(viewport={'width': 412, 'height': 915}, locale='he-IL', has_touch=True, is_mobile=True, device_scale_factor=2)
        ctx2.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); localStorage.setItem('dk.fly.craft', 'bike'); localStorage.setItem('dk.fly.sound', '0'); } catch (e) {}")
        pg = ctx2.new_page()
        pg.on('pageerror', lambda e: errors.append(str(e)))
        pg.goto(f'http://127.0.0.1:{PORT}/index.html')
        pg.wait_for_function('() => typeof Explore !== "undefined" && typeof map !== "undefined" && map && map.loaded()', timeout=30000)
        time.sleep(0.8)
        pg.evaluate('() => { const w = document.getElementById("welcome-sheet"); if (w) w.hidden = true; }')
        pg.tap('#explore')
        pg.wait_for_function('() => Explore.debug().flying', timeout=40000)
        pg.evaluate('() => document.getElementById("fly-intro").click()')
        time.sleep(0.3)
        check('phone: touch controls shown', pg.locator('.fly-touch .tjoy').is_visible() and pg.locator('.fly-touch .tboost').is_visible())
        check('phone: reticle hidden, no map overhang beyond the lean', pg.evaluate('() => getComputedStyle(document.querySelector(".fly-reticle")).display') == 'none')
        pg.screenshot(path=f'{OUT}/shot_bike_phone.png')
        # hold the climb button, then the throttle
        boxes = pg.evaluate('() => [...document.querySelectorAll(".fly-touch .tvert")].map((v) => { const r = v.querySelector("button").getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; })')
        thr, col = boxes[0], boxes[1]
        pg.touchscreen.tap(col['x'], col['y'])   # a tap is a short hold; then a real hold
        pg.evaluate('''([x, y]) => { const el = document.elementFromPoint(x, y); el.dispatchEvent(new PointerEvent("pointerdown", { pointerId: 7, pointerType: "touch", clientX: x, clientY: y, bubbles: true })); }''', [col['x'], col['y']])
        time.sleep(2)
        k = bike(pg)
        check('phone: climb button held lifts off', not k['grounded'] and k['y'] > 3, k)
        pg.evaluate('''([x, y]) => { const el = document.elementFromPoint(x, y); el.dispatchEvent(new PointerEvent("pointerup", { pointerId: 7, pointerType: "touch", clientX: x, clientY: y, bubbles: true })); }''', [col['x'], col['y']])
        # a little throttle, or a hover has nothing to vector; then the joystick, dragged right
        pg.evaluate('''([x, y]) => { const el = document.elementFromPoint(x, y); el.dispatchEvent(new PointerEvent("pointerdown", { pointerId: 9, pointerType: "touch", clientX: x, clientY: y, bubbles: true })); }''', [thr['x'], thr['y']])
        time.sleep(0.45)
        pg.evaluate('''([x, y]) => { const el = document.elementFromPoint(x, y); el.dispatchEvent(new PointerEvent("pointerup", { pointerId: 9, pointerType: "touch", clientX: x, clientY: y, bubbles: true })); }''', [thr['x'], thr['y']])
        check('phone: throttle button opens the lever', 0.1 < bike(pg)['throttle'] < 0.5, bike(pg)['throttle'])
        joy = pg.evaluate('() => { const r = document.querySelector(".fly-touch .tjoy").getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; }')
        pg.evaluate('''([x, y]) => { const el = document.querySelector(".fly-touch .tjoy"); el.dispatchEvent(new PointerEvent("pointerdown", { pointerId: 8, pointerType: "touch", clientX: x, clientY: y, bubbles: true })); el.dispatchEvent(new PointerEvent("pointermove", { pointerId: 8, pointerType: "touch", clientX: x + 40, clientY: y, bubbles: true })); }''', [joy['x'], joy['y']])
        h0 = bike(pg)['heading']
        time.sleep(1.5)
        k = bike(pg)
        turned = ((k['heading'] - h0 + 540) % 360) - 180
        check('phone: joystick right steers right (a hover turns on the nozzle, not the bank)', k['steer'] > 0.3 and turned > 1.5, (k['steer'], turned))
        pg.evaluate('''([x, y]) => { const el = document.querySelector(".fly-touch .tjoy"); el.dispatchEvent(new PointerEvent("pointerup", { pointerId: 8, pointerType: "touch", clientX: x, clientY: y, bubbles: true })); }''', [joy['x'], joy['y']])
        time.sleep(0.5)
        check('phone: joystick released centres', abs(bike(pg)['steer']) < 0.05, bike(pg)['steer'])
        pg.close()

        check('no page errors', not errors, errors[:5])
        b.close()
finally:
    srv.terminate()

print()
print(f'{len(fails)} failed' if fails else 'all passed')
for f in fails: print('  -', f)
sys.exit(1 if fails else 0)
