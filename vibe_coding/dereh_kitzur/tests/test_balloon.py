"""Balloon mode, checked against the clock in a real Chrome.

Runs a local server on web/, opens the app, swaps the aircraft, takes off,
and reads the flight model through Explore.debug() while pressing the keys
the way a person would. The audio is tapped through an AnalyserNode hung off
the compressor so the burner's sound can be measured, not assumed.

    python3 tests/test_balloon.py

Needs Playwright with Google Chrome (channel='chrome') and the angle/gl
backend: under swiftshader the page draws at one frame a second, the flight
model's clock (dt is capped per frame) runs at a twelfth of real time, and
every timing check below is meaningless. Takes about four minutes, most of
it waiting for a balloon to do what balloons do. Screenshots land beside
this file.
"""
import json, math, statistics, subprocess, sys, time, os
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8765

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

def sample(page, seconds, every=0.25):
    out = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        d = dbg(page)
        d['t'] = round(time.time() - t0, 2)
        d['rms'] = page.evaluate(RMS)['rms']
        out.append(d)
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
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); } catch (e) {}")
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
        page.goto(f'http://127.0.0.1:{PORT}/index.html')
        page.wait_for_function('() => typeof Explore !== "undefined" && typeof map !== "undefined" && map && map.loaded()', timeout=30000)
        time.sleep(1)
        page.evaluate('() => { const w = document.getElementById("welcome-sheet"); if (w) w.hidden = true; }')

        # ---- the small button ----
        craft = page.locator('#craft')
        check('craft button visible', craft.is_visible())
        ex = page.locator('#explore').bounding_box(); cb = craft.bounding_box()
        check('craft button sits on the jet\'s lower-left shoulder',
              cb['x'] < ex['x'] + 10 and cb['y'] + cb['height'] > ex['y'] + ex['height'] - 10, (ex, cb))
        # A first visit gets the balloon; the jet is one press away, and the
        # choice is kept.
        check('starts as balloon', page.evaluate('() => Explore.getCraft()') == 'balloon')
        check('big button starts with the balloon, before and after app.js',
              page.evaluate('() => document.querySelector("#explore img").getAttribute("src")') == 'img/balloon.svg'
              and page.evaluate('() => document.body.classList.contains("craft-balloon")'))
        craft.click()
        check('click swaps to jet', page.evaluate('() => Explore.getCraft()') == 'jet')
        check('big button shows the jet, title says F-16',
              page.evaluate('() => document.querySelector("#explore img").getAttribute("src")') == 'img/f16.svg'
              and 'F-16' in page.locator('#explore').get_attribute('title'))
        check('jet choice is kept', page.evaluate('() => localStorage.getItem("dk.fly.craft")') == 'jet')
        page.screenshot(path=f'{OUT}/shot_fab_jet.png', clip={'x': ex['x'] - 40, 'y': ex['y'] - 20, 'width': 120, 'height': 100})
        craft.click()
        check('click swaps back to balloon', page.evaluate('() => Explore.getCraft()') == 'balloon')
        check('big button shows the balloon', page.evaluate('() => document.querySelector("#explore img").getAttribute("src")') == 'img/balloon.svg')
        check('choice is kept', page.evaluate('() => localStorage.getItem("dk.fly.craft")') == 'balloon')
        check('title says balloon', 'כדור' in page.locator('#explore').get_attribute('title'))
        page.screenshot(path=f'{OUT}/shot_fab_balloon.png', clip={'x': ex['x'] - 40, 'y': ex['y'] - 20, 'width': 120, 'height': 100})

        # ---- take off ----
        page.click('#explore')
        page.wait_for_function('() => Explore.debug().flying', timeout=15000)
        d = dbg(page)
        check('in the air as a balloon', d['on'] and d['craft'] == 'balloon')
        check('body carries fly-balloon', page.evaluate('() => document.body.classList.contains("fly-balloon")'))
        check('craft button hidden in flight', not craft.is_visible())
        check('vario gauge shown', page.locator('#fly-vsi-g').is_visible())
        check('basket rim shown', page.locator('.fly-basket').is_visible())
        check('compass shown, with the wind pointer', page.locator('#fly-compass').is_visible() and page.locator('#fly-drift').is_visible())
        check('heading in words', any(w in page.evaluate('() => document.getElementById("fly-heading").textContent') for w in ('צפון', 'דרום', 'מזרח', 'מערב')))
        check('balloon intro shown, jet intro hidden',
              page.locator('.fly-intro-card.balloon').is_visible() and not page.locator('.fly-intro-card.jet').is_visible())
        page.screenshot(path=f'{OUT}/shot_intro.png')
        check('launch burn lit', d['balloon']['launch'] > 0 and d['burn'] > 0.3, d)
        check('burner gauge reads דולק', page.evaluate('() => document.querySelector("#fly-burn i").textContent') == 'דולק')

        # the launch burn: the roar is on, and the sound is measurable
        s = sample(page, 1.5)
        loud = max(x['rms'] for x in s)
        print(f'    launch rms max {loud:.4f}, ctx {page.evaluate(RMS)["state"]}')
        check('burner audible during launch', loud > 0.01, loud)
        # the roar's texture: level over 40 quick reads, and where its energy sits
        fl = [page.evaluate(RMS)['rms'] for _ in range(40)]
        spec = page.evaluate('''() => { const { an, ctx } = window.__tap; const b = new Float32Array(an.frequencyBinCount);
            an.getFloatFrequencyData(b); const hz = ctx.sampleRate / an.fftSize; let num = 0, den = 0, peak = 0, pf = 0;
            for (let i = 1; i < b.length; i++) { const p = Math.pow(10, b[i] / 10); num += p * i * hz; den += p; if (p > peak) { peak = p; pf = i * hz; } }
            return { centroid: num / den, peak: pf }; }''')
        cv = statistics.pstdev(fl) / statistics.mean(fl)
        print(f'    burner flutter: mean {statistics.mean(fl):.3f}, cv {cv:.2f}; spectral centroid {spec["centroid"]:.0f} Hz, peak {spec["peak"]:.0f} Hz')
        check('burner flutters', cv > 0.1, cv)
        check('burner is a roar, not a hiss', 150 < spec['centroid'] < 1500, spec)

        # arm the stick by passing the cursor through the middle, then park it
        page.mouse.move(720, 450); page.mouse.move(721, 451)
        page.evaluate('() => document.getElementById("fly-intro").click()')   # dismiss intro
        time.sleep(0.2)
        page.mouse.move(720, 450); page.mouse.move(722, 452)

        # ---- after the launch burn: climbing, then it should ease ----
        page.wait_for_function('() => Explore.debug().balloon.launch <= 0', timeout=8000)
        s = sample(page, 3)
        quiet = max(x['rms'] for x in s[4:])
        print(f'    after launch rms max {quiet:.5f}')
        check('silent once the burner is off', quiet < 0.002, quiet)
        check('climbing after the launch burn', s[-1]['balloon']['vz'] > 0.5 and s[-1]['alt'] > s[0]['alt'], (s[0]['alt'], s[-1]['alt'], s[-1]['balloon']['vz']))
        check('burner glow off', s[-1]['burn'] < 0.1)

        # ---- the wind, at altitude: hold the button, speed builds; let go, it fades slowly, never to nothing ----
        v0 = dbg(page)['speed']
        page.mouse.down(); s = sample(page, 5); page.mouse.up()
        v1 = s[-1]['speed']; r1 = s[-1]['balloon']['ride']
        print(f'    alt {s[-1]["alt"]:.0f}, speed {v0*3.6:.0f} -> {v1*3.6:.0f} km/h, ride {r1:.2f}')
        check('holding the button catches the wind', v1 > v0 + 3 and r1 > 0.85, (v0, v1, r1))
        check('the wind is a balloon\'s, not a jet\'s', v1 * 3.6 < 90, v1 * 3.6)
        s = sample(page, 8)
        r2 = s[-1]['balloon']['ride']
        check('released, the wind lets go slowly', r1 - 0.4 < r2 < r1 - 0.12, (r1, r2))
        check('but never stops', s[-1]['speed'] > 1 and r2 > 0.3, (s[-1]['speed'], r2))
        check('silent throughout', max(x['rms'] for x in s) < 0.002)

        # ---- turning: the drift follows the facing with a lag ----
        b0 = dbg(page)['bearing']
        page.keyboard.down('ArrowRight'); time.sleep(2.5); page.keyboard.up('ArrowRight')
        d = dbg(page)
        turned = (d['bearing'] - b0 + 540) % 360 - 180
        lag = (d['bearing'] - d['balloon']['drift'] + 540) % 360 - 180
        print(f'    turned {turned:.1f} deg in 2.5 s, drift lags by {lag:.1f}')
        check('turn is slow and heavy', 15 < turned < 70, turned)
        check('drift lags the basket', 2 < lag < 40, lag)
        time.sleep(4)
        d = dbg(page)
        lag2 = (d['bearing'] - d['balloon']['drift'] + 540) % 360 - 180
        check('drift comes round behind the basket', abs(lag2) < abs(lag) * 0.5, (lag, lag2))

        # ---- burner: from an eased climb, a 3 s burn, then hands off. The climb should arrive late and outlast the burn ----
        page.keyboard.down('ArrowDown'); time.sleep(2); page.keyboard.up('ArrowDown')   # take the launch heat off
        page.wait_for_function('() => Explore.debug().balloon.vz < 1.2', timeout=60000)
        before = dbg(page)
        print(f'    settled: alt {before["alt"]:.0f}, vz {before["balloon"]["vz"]:.2f}, dT {before["balloon"]["dT"]:.0f}')
        page.keyboard.down('ArrowUp'); burnS = sample(page, 3, 0.2); page.keyboard.up('ArrowUp')
        after = sample(page, 30, 0.3)
        check('burner audible while held', max(x['rms'] for x in burnS) > 0.01, max(x['rms'] for x in burnS))
        check('burner glow + gauge while held', burnS[-1]['burn'] > 0.8)
        check('heat rises with the burn', burnS[-1]['balloon']['dT'] > before['balloon']['dT'] + 4, (before['balloon']['dT'], burnS[-1]['balloon']['dT']))
        vz = [x['balloon']['vz'] for x in after]
        dT = [x['balloon']['dT'] for x in after]
        peak_i = max(range(len(vz)), key=lambda i: vz[i])
        print(f'    vz at release {vz[0]:.2f}, peak {vz[peak_i]:.2f} at +{after[peak_i]["t"]}s, end {vz[-1]:.2f}; dT {before["balloon"]["dT"]:.0f} -> {dT[0]:.0f} -> max {max(dT):.0f} -> {dT[-1]:.0f}')
        check('heat keeps arriving after release (the plume)', max(dT) > dT[0] + 3, (dT[0], max(dT)))
        check('and then cools', dT[-1] < max(dT) - 1, (max(dT), dT[-1]))
        check('climb keeps building after the burner is released', after[peak_i]['t'] > 1.0 and vz[peak_i] > vz[0] + 0.3, (vz[0], vz[peak_i], after[peak_i]['t']))
        check('and then eases', vz[-1] < vz[peak_i] - 0.2, (vz[-1], vz[peak_i]))
        check('silent after release', max(x['rms'] for x in after[3:]) < 0.002)

        # ---- the basket sways: the roll wanders, and never past the basket's limit ----
        bank = [x['bank'] for x in after]
        print(f'    bank range {min(bank):.2f}..{max(bank):.2f}, stdev {statistics.pstdev(bank):.2f}')
        check('roll moves', statistics.pstdev(bank) > 0.15, bank)
        check('roll stays gentle', max(abs(v) for v in bank) < 9, bank)
        tf = page.evaluate('() => map.getContainer().style.transform')
        check('picture is rotated by the sway', 'rotate(' in tf, tf)

        # ---- vent: 2.5 s open, the heat drops and the climb turns to a sink ----
        dT0 = dbg(page)['balloon']['dT']
        page.keyboard.down('ArrowDown'); s = sample(page, 2.5); page.keyboard.up('ArrowDown')
        d = dbg(page)
        check('vent dumps heat', d['balloon']['dT'] < dT0 - 12, (dT0, d['balloon']['dT']))
        s2 = sample(page, 7)
        check('vent turns the climb into a sink', s2[-1]['balloon']['vz'] < -0.5, s2[-1]['balloon'])
        check('silent while venting', max(x['rms'] for x in s) < 0.002)

        # ---- looking down ----
        p0 = dbg(page)['pitch']
        page.keyboard.down('KeyW'); time.sleep(2); page.keyboard.up('KeyW')
        p1 = dbg(page)['pitch']
        check('W looks down over the edge', p1 < p0 - 8, (p0, p1))
        page.screenshot(path=f'{OUT}/shot_balloon_down.png')
        time.sleep(2.5)
        page.screenshot(path=f'{OUT}/shot_balloon.png')

        # ---- ground: vent all the way down, land, and a burn lifts off again ----
        page.keyboard.down('ArrowDown')
        page.wait_for_function('() => Explore.debug().alt <= 40.5', timeout=90000)
        page.keyboard.up('ArrowDown')
        d = dbg(page)
        check('the ground is at 40 m', abs(d['alt'] - 40) < 0.6, d['alt'])
        check('grounded envelope keeps some heat', d['balloon']['dT'] >= 61.5, d['balloon']['dT'])
        page.keyboard.down('ArrowUp'); time.sleep(6); page.keyboard.up('ArrowUp')
        time.sleep(5)
        d = dbg(page)
        print(f'    after a 6 s burn on the ground and 5 s wait: alt {d["alt"]:.1f}, vz {d["balloon"]["vz"]:.2f}, dT {d["balloon"]["dT"]:.0f}')
        check('a long burn lifts off again', d['alt'] > 41 and d['balloon']['vz'] > 0.5, (d['alt'], d['balloon']['vz']))

        # ---- out, and the jet still flies ----
        page.keyboard.press('Escape')
        page.wait_for_function('() => !Explore.isOn()', timeout=5000)
        time.sleep(1.2)
        check('fly-balloon class gone', not page.evaluate('() => document.body.classList.contains("fly-balloon")'))
        check('craft button back', craft.is_visible())
        craft.click()
        check('swap back to jet', page.evaluate('() => Explore.getCraft()') == 'jet')
        page.click('#explore')
        page.wait_for_function('() => Explore.debug().flying', timeout=15000)
        page.evaluate('() => document.getElementById("fly-intro").click()')
        d = dbg(page)
        check('jet has no balloon dressing', not page.evaluate('() => document.body.classList.contains("fly-balloon")') and not page.locator('#fly-vsi-g').is_visible() and not page.locator('.fly-basket').is_visible())
        check('jet burner gauge reads אחורי', page.evaluate('() => document.querySelector("#fly-burn i").textContent') == 'אחורי')
        a0 = d['alt']
        page.keyboard.down('ArrowUp'); time.sleep(1); page.keyboard.up('ArrowUp')
        check('jet climbs at once', dbg(page)['alt'] > a0 * 1.3, (a0, dbg(page)['alt']))
        rms = page.evaluate(RMS)['rms']
        check('jet engine idles audibly', rms > 0.003, rms)
        page.screenshot(path=f'{OUT}/shot_jet.png')
        page.keyboard.press('Escape')
        page.wait_for_function('() => !Explore.isOn()', timeout=5000)

        bad = [e for e in errors if 'AbortError' not in e and 'favicon' not in e and '404' not in e]
        check('no page errors', not bad, bad)
        b.close()
finally:
    srv.terminate()

print()
print('FAILED: ' + ', '.join(fails) if fails else 'ALL PASSED')
sys.exit(1 if fails else 0)
