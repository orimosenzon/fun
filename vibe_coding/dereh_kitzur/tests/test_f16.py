"""The F-16, checked against the clock in a real Chrome.

Takes off as the jet, holds the button until the barrier is crossed, and
reads the flight model through Explore.debug() the whole way: the throttle
detent, the acceleration, the Mach number against the local speed of sound,
the boom (measured off the compressor, not assumed), the cone, the g-limited
turn, the coast back down, and the climb to the ceiling at the F-16's own
rate. Same rig as test_balloon.py:

    python3 tests/test_f16.py

Needs Playwright with Google Chrome (channel='chrome') and --use-angle=gl,
or the page draws at one frame a second and every timing here is wrong.
Takes about three minutes. Screenshots land beside this file.
"""
import math, subprocess, sys, time, os
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8766
RAD = math.pi / 180

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

SPEC = """
() => {
  const { an, ctx } = window.__tap;
  const b = new Float32Array(an.frequencyBinCount);
  an.getFloatFrequencyData(b);
  const hz = ctx.sampleRate / an.fftSize;
  let num = 0, den = 0, low = 0;
  for (let i = 1; i < b.length; i++) {
    const p = Math.pow(10, b[i] / 10);
    num += p * i * hz; den += p;
    if (i * hz < 120) low += p;
  }
  return { centroid: num / den, low: low / den };
}
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

# One round trip for everything the crossing is judged by.
PROBE = '() => { const r = (' + RMS + ')(); const s = (' + SPEC + ')(); const d = Explore.debug(); return { ' \
        'rms: r.rms, c: s.centroid, low: s.low, speed: d.speed, mach: d.mach, sonic: d.sonic, booms: d.booms, buffet: d.buffet, ' \
        'cone: document.getElementById("fly-cone").classList.contains("go"), super: document.body.classList.contains("fly-super") }; }'

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

def dbg(page):
    return page.evaluate('() => Explore.debug()')

def sample(page, seconds, every=0.25, until=None):
    """Read the model every `every` seconds for `seconds`, or until `until(d)`."""
    out = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        d = dbg(page)
        d['t'] = round(time.time() - t0, 2)
        d['rms'] = page.evaluate(RMS)['rms']
        d['spec'] = page.evaluate(SPEC)
        d['cone'] = page.evaluate('() => document.getElementById("fly-cone").classList.contains("go")')
        d['super'] = page.evaluate('() => document.body.classList.contains("fly-super")')
        out.append(d)
        if until and until(d): break
        time.sleep(every)
    return out

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
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); localStorage.setItem('dk.fly.craft', 'jet'); } catch (e) {}")
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
        page.goto(f'http://127.0.0.1:{PORT}/index.html')
        page.wait_for_function('() => typeof Explore !== "undefined" && typeof map !== "undefined" && map && map.loaded()', timeout=30000)
        time.sleep(1)
        page.evaluate('() => { const w = document.getElementById("welcome-sheet"); if (w) w.hidden = true; }')

        check('starts as jet', page.evaluate('() => Explore.getCraft()') == 'jet')
        check('button says F-16', 'F-16' in page.locator('#explore').get_attribute('title'))

        # ---- take off ----
        page.click('#explore')
        page.wait_for_function('() => Explore.debug().flying', timeout=15000)
        d = dbg(page)
        check('in the air as the jet', d['on'] and d['craft'] == 'jet')
        card = page.locator('.fly-intro-card.jet')
        check('intro shown', card.is_visible())
        txt = card.inner_text()
        check('intro names the F-16', 'F-16' in txt and 'נץ' in txt)
        check('intro gives the ceiling', '15,240' in txt and '50,000' in txt)
        check('intro gives the Mach limits', 'מאך 1.2' in txt and 'מאך 2' in txt)
        check('intro mentions the boom', 'בום' in txt)
        check('intro says a real one does not stop', 'נעצר באוויר' in txt)
        check('intro explains the HUD units', 'קשר' in txt and 'רגל' in txt)
        # The jet's instruments are the F-16's own HUD and panel (cockpit.js);
        # the generic gauges and the compass belong to the other two aircraft.
        check('cockpit shown, generic gauges and compass hidden',
              page.locator('.ckpt').is_visible() and page.locator('#hud-mach').is_visible()
              and page.locator('.ckpt-panel').is_visible()
              and not page.locator('.fly-readout').is_visible() and not page.locator('#fly-compass').is_visible())
        check('vario hidden', not page.locator('#fly-vsi-g').is_visible())
        page.screenshot(path=f'{OUT}/shot_f16_intro.png')
        env = d['envelope']
        print(f'    at {d["alt"]:.0f} m: a = {env["a"]:.1f} m/s, top = {env["top"]:.0f} m/s ({env["top"]*3.6:.0f} km/h), dry = {env["mil"]:.0f} m/s')
        check('speed of sound at 260 m is ISA', abs(env['a'] - 339.2) < 0.5, env['a'])
        # the limits slide from the sea-level figure to the high one over 12 km
        t = d['alt'] / 12000
        check('top speed at 260 m is Mach 1.2 (plus the slide)', abs(env['top'] / env['a'] - (1.2 + 0.85 * t)) < 0.005, env)
        check('dry ceiling at 260 m is Mach 0.95', abs(env['mil'] / env['a'] - (0.95 + 0.15 * t)) < 0.005, env)

        # ---- the mouse, from the first movement ----
        # The card goes when the mouse travels, the same movement takes the
        # stick, and the stick's authority comes in over a second and a half
        # rather than all at once: a cursor left at the edge is a turn that
        # grows, not a lurch.
        page.mouse.move(700, 450)
        for x in range(700, 1300, 40): page.mouse.move(x, 450); time.sleep(0.02)
        time.sleep(0.15)
        d = dbg(page)
        check('take-off card put away by moving the mouse', not d['intro'] and not card.is_visible())
        check('stick taken by the same movement', d['armed'])
        y0 = abs(d['yaw']); time.sleep(0.4); y1 = abs(dbg(page)['yaw']); time.sleep(1.6); y2 = abs(dbg(page)['yaw'])
        print(f'    yaw after the stick was taken at the edge: {y0:.1f} -> {y1:.1f} -> {y2:.1f} deg/s')
        check('the turn comes in softly, not as a lurch', y0 < 8 and y1 < 0.5 * y2 and y2 > 20, (y0, y1, y2))
        page.mouse.move(720, 450)
        time.sleep(1.2)
        # The help reopened with H is a different thing: it was asked for, so
        # moving the mouse does not put it away.
        page.keyboard.press('KeyH')
        time.sleep(0.1)
        for x in range(720, 1100, 40): page.mouse.move(x, 450); time.sleep(0.02)
        check('help reopened with H stays while the mouse moves', dbg(page)['intro'])
        page.keyboard.press('KeyH')
        time.sleep(0.1)
        check('and H again puts it away', not dbg(page)['intro'])

        # ---- the stick, and the intro out of the way ----
        page.mouse.move(720, 450); page.mouse.move(721, 451)
        page.evaluate('() => document.getElementById("fly-intro").click()')
        time.sleep(0.2)
        page.mouse.move(720, 450); page.mouse.move(722, 452)
        check('stick armed', not page.evaluate('() => document.body.classList.contains("fly-unarmed")'))

        # ---- the HUD ----
        HUD = '''() => { const d = Explore.debug(); const t = (id) => document.getElementById(id).textContent;
          const tf = (id) => document.getElementById(id).getAttribute('transform');
          const hz = document.querySelector('#hud-ladder .rung.horizon');
          return { bearing: d.bearing, alt: d.alt, speed: d.speed, bank: d.bank, mach: d.mach, gee: d.gee,
                   hdg: t('hud-hdg'), spd: t('hud-spd'), alt_: t('hud-alt'), g: t('hud-g'), machS: t('hud-mach'),
                   ded: t('hud-ded'), stpt: t('hud-stpt'), lad: tf('hud-ladder'), fpm: tf('hud-fpm'),
                   hz: hz && hz.style.display !== 'none' ? hz.getAttribute('transform') : null,
                   fpmLim: document.getElementById('hud-fpm').classList.contains('lim') }; }'''
        hud = page.evaluate(HUD)
        print(f'    bearing {hud["bearing"]:.1f}: HUD heading "{hud["hdg"]}", {hud["spd"]} kt, {hud["alt_"]} ft, g {hud["g"]}, ded "{hud["ded"]}"')
        want = round(hud['bearing']) % 360 or 360
        check('heading box is the bearing, three digits', hud['hdg'] == f'{want:03d}', (hud['bearing'], hud['hdg']))
        check('airspeed box in knots', abs(int(hud['spd']) - abs(hud['speed']) * 1.943844) < 1.5, (hud['speed'], hud['spd']))
        check('altitude box in feet, to ten', abs(int(hud['alt_'].replace(',', '')) - hud['alt'] * 3.28084) <= 5, (hud['alt'], hud['alt_']))
        check('radar altitude below 1,500 ft', page.evaluate('() => [...document.querySelectorAll(".ckpt-hud text")].some(t => t.textContent === "R" && t.style.display !== "none")'))
        check('DED carries heading, knots and feet', 'HDG' in hud['ded'] and 'KT' in hud['ded'] and 'FT' in hud['ded'], hud['ded'])
        check('DED names the steerpoint, the nearest trail', bool(hud['stpt'].strip()), hud['stpt'])
        # Conformal: level, the flight path marker rides the horizon line.
        def ty(tr): return float(tr.split()[-1].rstrip(')'))
        check('horizon rung drawn and the FPM on it, level', hud['hz'] is not None and abs(ty(hud['hz']) - ty(hud['fpm'])) < 2 and not hud['fpmLim'], (hud['hz'], hud['fpm']))
        page.keyboard.down('ArrowRight'); time.sleep(1.5); page.keyboard.up('ArrowRight')
        time.sleep(1.2)   # the turn dies down after the key
        hud2 = page.evaluate(HUD)
        want2 = round(hud2['bearing']) % 360 or 360
        check('heading box follows a turn', hud2['bearing'] > hud['bearing'] + 20 and hud2['hdg'] == f'{want2:03d}', (hud['bearing'], hud2['bearing'], hud2['hdg']))
        # The ladder rolls with the picture: the stick to the side banks it.
        page.mouse.move(1300, 450); time.sleep(0.9)
        hud3 = page.evaluate(HUD)
        rot = float(hud3['lad'].split('(')[1].split()[0])
        check('ladder rolled against the bank', abs(hud3['bank']) > 5 and abs(rot + hud3['bank']) < 0.5, (hud3['bank'], hud3['lad']))
        check('g on the HUD while turning', float(hud3['g']) >= 1.0 and hud3['g'] == f'{hud3["gee"]:.1f}', (hud3['g'], hud3['gee']))
        page.mouse.move(720, 450); time.sleep(1.5)
        page.screenshot(path=f'{OUT}/shot_f16_hud.png', clip={'x': 480, 'y': 40, 'width': 480, 'height': 400})

        # ---- a short press: a walking pace, not a launch ----
        page.mouse.down(); time.sleep(0.5); page.mouse.up()
        s = sample(page, 4)
        vmax = max(x['speed'] for x in s)
        print(f'    half-second press: peak {vmax*3.6:.0f} km/h, throttle peaked {max(x["throttle"] for x in s):.2f}')
        check('a tap is a nudge, under 120 km/h', 0 < vmax * 3.6 < 120, vmax * 3.6)
        check('no burner from a tap', all(not x['super'] and x['burn'] < 0.5 for x in s))
        check('and it stops again', s[-1]['speed'] * 3.6 < 30, s[-1]['speed'] * 3.6)

        # ---- the run to the barrier ----
        page.mouse.down()
        t_run = time.time()
        run = sample(page, 60, 0.2, until=lambda x: x['mach'] > 0.975)
        v = [x['speed'] for x in run]
        ups = sum(1 for a, c in zip(v, v[1:]) if c >= a - 0.5)
        check('speed climbs steadily under the button', ups >= len(v) - 3, (ups, len(v)))
        burn_at = next((x['t'] for x in run if x['burn'] > 0.5), None)
        thr_at = next((x['throttle'] for x in run if x['burn'] > 0.5), None)
        print(f'    burner lit at t={burn_at}s, throttle {thr_at}')
        check('burner lights past the detent, about 1.3 s in', burn_at is not None and 1.0 < burn_at < 2.2 and thr_at > 0.64, (burn_at, thr_at))
        check('near the barrier within 45 s', run[-1]['mach'] > 0.975, run[-1]['mach'])
        check('no boom yet', run[-1]['booms'] == 0 and not run[-1]['sonic'])

        # The crossing, read as fast as the round trip allows: the boom is
        # two thumps a tenth of a second apart and the engine ducked under
        # them for a third of a second, none of which a sampler on a
        # quarter-second beat would be sure to see.
        probe = []
        t0 = time.time(); t_boom = None
        while time.time() - t0 < 15:
            r = page.evaluate(PROBE); r['t'] = time.time() - t0; probe.append(r)
            if r['sonic'] and t_boom is None: t_boom = r['t']
            if t_boom is not None and r['t'] - t_boom > 1.4: break
        check('went supersonic', t_boom is not None)
        t_cross = time.time() - t_run - (probe[-1]['t'] - t_boom)
        at = next(x for x in probe if x['sonic'])
        print(f'    Mach 1 at t={t_cross:.1f}s, {at["speed"]*3.6:.0f} km/h, Mach {at["mach"]:.3f}; {len(probe)} reads through the crossing')
        check('the barrier is half a minute of held button', 18 < t_cross < 45, t_cross)
        check('boom counted once', probe[-1]['booms'] == 1, probe[-1]['booms'])
        check('buffet felt through the band', max(x['buffet'] for x in run + probe) > 0.5)
        check('fly-super on the body', any(x['super'] for x in probe if x['sonic']))
        check('cone played', any(x['cone'] for x in probe if x['sonic']))
        pre = [x for x in probe if x['t'] < t_boom][-25:]
        post = [x for x in probe if 0 <= x['t'] - t_boom < 0.8]
        c_pre = min(x['c'] for x in pre); c_post = min(x['c'] for x in post)
        l_pre = max(x['low'] for x in pre); l_post = max(x['low'] for x in post)
        r_pre = max(x['rms'] for x in pre); r_peak = max(x['rms'] for x in post); r_dip = min(x['rms'] for x in post)
        print(f'    engine before: rms {r_pre:.3f}, centroid {c_pre:.0f} Hz; boom: peak {r_peak:.3f}, centroid {c_post:.0f} Hz, under 120 Hz {l_pre:.2f} -> {l_post:.2f}, then ducked to {r_dip:.3f}')
        check('boom louder than the burner', r_peak > 1.3 * r_pre and r_peak > 0.35, (r_pre, r_peak))
        check('boom is a thump: energy drops under 120 Hz', c_post < 110 and l_post > 0.9, (c_post, l_post))
        check('engine ducked under the boom', r_dip < 0.5 * min(x['rms'] for x in pre), (r_dip, min(x['rms'] for x in pre)))
        page.screenshot(path=f'{OUT}/shot_f16_sonic.png')
        check('Mach on the HUD reads over one', float(page.evaluate('() => document.getElementById("hud-mach").textContent')) >= 1.0)

        # ---- to the top, and the turn there ----
        top = sample(page, 40, 0.5, until=lambda x: x['speed'] > 0.97 * x['envelope']['top'])
        d = top[-1]
        print(f'    top: {d["speed"]*3.6:.0f} km/h, Mach {d["mach"]:.2f} at {d["alt"]:.0f} m after {top[-1]["t"]:.0f}s more')
        check('reaches the F-16\'s top speed', d['speed'] > 0.97 * d['envelope']['top'], (d['speed'], d['envelope']['top']))
        check('which is about 1,470 km/h', 1400 < d['speed'] * 3.6 < 1500, d['speed'] * 3.6)
        check('boom still counted once', d['booms'] == 1, d['booms'])
        # Past the barrier the engine is left behind: only the air is heard.
        q = [page.evaluate(PROBE) for _ in range(12)]
        q_rms = max(x['rms'] for x in q); q_c = min(x['c'] for x in q)
        print(f'    supersonic, burner lit: rms {q_rms:.3f} (was {r_pre:.3f} subsonic), centroid {q_c:.0f} Hz (was {c_pre:.0f})')
        check('engine hushed past Mach 1', q_rms < 0.4 * r_pre, (q_rms, r_pre))
        check('what is left is the air: the sound moved up', q_c > 3 * c_pre, (q_c, c_pre))
        # stick hard over: the turn is g-limited, not stick-limited
        page.mouse.move(1430, 450)
        time.sleep(1.5)
        turn = sample(page, 2, 0.2)
        cap = max(9 * 9.81 / x['speed'] / RAD for x in turn)
        yaw = max(abs(x['yaw']) for x in turn)
        gmax = max(x['gee'] for x in turn)
        print(f'    at Mach {turn[-1]["mach"]:.2f}: yaw {yaw:.1f} deg/s (9 g cap {cap:.1f}), {gmax:.1f} g')
        check('turn rate held to 9 g', yaw <= cap + 0.3, (yaw, cap))
        check('turn rate is a fighter\'s, not a mouse\'s', 8 < yaw < 15, yaw)
        check('g gauge near nine', 8.3 < gmax <= 9.3, gmax)
        page.mouse.move(720, 450)

        # ---- let go: a coast, not a brake, and the barrier crossed back ----
        page.mouse.up()
        t_rel = time.time()
        coast = sample(page, 40, 0.5, until=lambda x: not x['sonic'])
        t_sub = time.time() - t_rel
        print(f'    subsonic again {t_sub:.1f}s after release')
        check('drops back under Mach 1 within 12 s', 2 < t_sub < 12, t_sub)
        check('crossing back is no boom', coast[-1]['booms'] == 1)
        check('fly-super gone', not sample(page, 0.3, 0.1)[-1]['super'])
        time.sleep(1.5)
        back = [page.evaluate(PROBE) for _ in range(12)]
        b_rms = max(x['rms'] for x in back); b_c = min(x['c'] for x in back)
        print(f'    subsonic again: rms {b_rms:.3f}, centroid {b_c:.0f} Hz')
        # The throttle is at idle by now, so what comes back is a rumble and
        # not the burner: the spectrum's centre falling is the surer sign.
        check('engine heard again under Mach 1', b_rms > 1.3 * q_rms and b_c < 0.5 * q_c, (b_rms, q_rms, b_c, q_c))
        slow = sample(page, 40, 0.5, until=lambda x: x['speed'] * 3.6 < 100)
        t_slow = time.time() - t_rel
        print(f'    under 100 km/h {t_slow:.1f}s after release')
        check('down to a car\'s speed within 35 s', t_slow < 35, t_slow)

        # ---- the ceiling ----
        page.keyboard.down('ArrowUp')
        climb = sample(page, 75, 0.5, until=lambda x: x['alt'] >= 15239)
        page.keyboard.up('ArrowUp')
        # One sample every half second with a round trip inside it is too
        # coarse for a rate per sample; the rate over the whole climb is not.
        hi = [x for x in climb if 400 < x['alt'] < 15239]
        rate = (hi[-1]['alt'] - hi[0]['alt']) / (hi[-1]['t'] - hi[0]['t'])
        print(f'    ceiling {climb[-1]["alt"]:.0f} m after {climb[-1]["t"]:.0f}s; climb {rate:.0f} m/s averaged above 400 m')
        check('reaches the F-16\'s ceiling', climb[-1]['alt'] >= 15239, climb[-1]['alt'])
        check('climbs at 50,000 ft/min', abs(rate - 254) < 254 * 0.06, rate)
        check('a minute\'s climb', 50 < climb[-1]['t'] < 72, climb[-1]['t'])
        d = dbg(page)
        env = d['envelope']
        print(f'    at the ceiling: a = {env["a"]:.1f} m/s, top = Mach {env["top"]/env["a"]:.2f} ({env["top"]*3.6:.0f} km/h)')
        check('sound is slower up here', abs(env['a'] - 295.1) < 0.5, env['a'])
        check('top speed is Mach 2.05', abs(env['top'] / env['a'] - 2.05) < 0.01, env)
        check('the view still draws at the ceiling', 1 < d['pitch'] < 90 and page.evaluate('() => map.getZoom()') > 5)
        page.screenshot(path=f'{OUT}/shot_f16_ceiling.png')
        check('altitude box reads the ceiling, 50,000 ft', page.evaluate('() => document.getElementById("hud-alt").textContent') == '50,000')
        check('no radar altitude up here', not page.evaluate('() => [...document.querySelectorAll(".ckpt-hud text")].some(t => t.textContent === "R" && t.style.display !== "none")'))
        check('FPM clamped and dashed, looking down from the ceiling', page.evaluate('() => document.getElementById("hud-fpm").classList.contains("lim")'))

        # a dive is faster than a climb
        page.keyboard.down('ArrowDown'); dive = sample(page, 3, 0.5); page.keyboard.up('ArrowDown')
        drate = (dive[0]['alt'] - dive[-1]['alt']) / (dive[-1]['t'] - dive[0]['t'])
        print(f'    dive {drate:.0f} m/s')
        check('dives at about 400 m/s', 330 < drate < 420, drate)

        # ---- out ----
        page.keyboard.press('Escape')
        page.wait_for_function('() => !Explore.isOn()', timeout=5000)
        time.sleep(1)
        check('fly-super cleared on exit', not page.evaluate('() => document.body.classList.contains("fly-super")'))
        check('cone reset on exit', not page.evaluate('() => document.getElementById("fly-cone").classList.contains("go")'))

        bad = [e for e in errors if 'AbortError' not in e and 'favicon' not in e and '404' not in e]
        check('no page errors', not bad, bad)
        b.close()
finally:
    srv.terminate()

print()
print('FAILED: ' + ', '.join(fails) if fails else 'ALL PASSED')
sys.exit(1 if fails else 0)
