"""A long walk in the app, and the phone waking up to a pile of fixes.

On 21/11/2026 Ori walked a long way with the app recording and the screen off.
When he woke the phone the app did not answer for about five minutes, then
came back. The native watcher sends a fix every second (distanceFilter 0); if
the page is asleep they wait in line and arrive together when it wakes, and if
each one costs work that grows with the walk, the line takes minutes to drain
and the page cannot answer a tap until it does.

This fakes the app (window.Capacitor with the plugins native.js uses), builds a
walk of WALK points, then hands the page BURST fixes in one go, as on waking,
with the CPU slowed to about a phone's speed. It checks that the burst is
drained quickly, every point is kept, and the bar, the map line and the live
mirror catch up.

    python3 tests/test_walk_backlog.py

Needs Playwright with Google Chrome (channel='chrome'). About a minute.
"""
import subprocess, sys, time, os
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8781
WALK = 5000       # points already in the walk: a couple of hours
BURST = 1200      # fixes waiting on wake: twenty minutes of screen-off
SLOWDOWN = 4      # CPU throttle, roughly a mid-range phone against a desktop
LIMIT_MS = 15000  # the burst has to be drained within this

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

# Just enough of the app for native.js: a watcher whose callback the test
# keeps, a log that goes nowhere, and an App plugin that listens to nothing.
SHIM = r"""
try { localStorage.setItem('dk.welcome.v1', '1'); localStorage.removeItem('dk.editor.v1');
      localStorage.removeItem('dk.gpslog.v1'); } catch (e) {}
window.__fix = null;
window.__nativeLines = 0;
window.Capacitor = {
  isNativePlatform: () => true,
  Plugins: {
    BackgroundGeolocation: {
      checkPermissions: async () => ({ location: 'granted' }),
      requestPermissions: async () => ({ location: 'granted' }),
      addWatcher: async (opts, cb) => { window.__fix = cb; return 'w1'; },
      removeWatcher: async () => {},
      openSettings: () => {}
    },
    LocalNotifications: { requestPermissions: async () => ({}) },
    DkLog: { log: async () => { window.__nativeLines++; }, read: async () => ({ text: '' }),
             clear: async () => {}, share: async () => {} },
    App: { addListener: () => ({ remove() {} }), minimizeApp: () => {} }
  }
};
"""

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(channel='chrome', headless=True,
                              args=['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'])
        ctx = b.new_context(viewport={'width': 400, 'height': 800}, locale='he-IL', has_touch=True)
        ctx.add_init_script(SHIM)
        page = ctx.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('dialog', lambda d: d.accept())
        page.goto(f'http://127.0.0.1:{PORT}/index.html')
        page.wait_for_function('() => typeof map !== "undefined" && map.loaded()', timeout=60000)
        time.sleep(2)
        check('the shim makes this the app', page.evaluate('() => Native.isApp === true'))

        page.evaluate('() => document.querySelectorAll(".sheet").forEach((x) => { x.hidden = true; })')
        page.evaluate('''() => { const s = document.getElementById('draft-sheet'); s.hidden = false;
           const b = document.createElement('button'); b.dataset.act = 'walk';
           s.appendChild(b); b.click(); }''')
        page.wait_for_function('() => typeof window.__fix === "function"', timeout=10000)

        # A walk north-east, a point every ~5 m, so each is past MIN_STEP_M.
        # The first WALK go in as they would on a walk with the screen on: one
        # by one, each given its turn, so the log and the mirror fill up.
        page.evaluate('''async (n) => {
          const t0 = Date.now() - n * 1000;
          for (let i = 0; i < n; i++) {
            window.__fix({ latitude: 32.47 + i * 0.00004, longitude: 34.96 + i * 0.00002,
                           accuracy: 6, time: t0 + i * 1000 });
            if (i % 50 === 49) await new Promise((r) => setTimeout(r, 0));
          }
        }''', WALK)
        time.sleep(1.5)
        have = page.evaluate('() => Drafts.isDrafting() && document.getElementById("draft-state").textContent')
        check(f'the walk has {WALK} points', f'{WALK} נקודות' in (have or ''), have)

        cdp = ctx.new_cdp_session(page)
        cdp.send('Emulation.setCPUThrottlingRate', {'rate': SLOWDOWN})
        # The wake: BURST fixes delivered back to back, each as its own task
        # (that is how the bridge hands them over), then one more task behind
        # them standing for the walker's tap. How long until it runs is how
        # long the app looked dead.
        ms = page.evaluate('''([n, start]) => new Promise((done) => {
          const t0 = performance.now();
          const tBase = Date.now() - n * 1000;
          for (let i = 0; i < n; i++) {
            const k = start + i;
            setTimeout(() => window.__fix({ latitude: 32.47 + k * 0.00004,
              longitude: 34.96 + k * 0.00002, accuracy: 6, time: tBase + i * 1000 }), 0);
          }
          setTimeout(() => done(performance.now() - t0), 0);
        })''', [BURST, WALK])
        # and whatever the page put off until the next frame
        page.evaluate('() => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)))')
        cdp.send('Emulation.setCPUThrottlingRate', {'rate': 1})
        print(f'       burst of {BURST} fixes on a {WALK}-point walk, CPU /{SLOWDOWN}: {ms/1000:.1f} s')
        check(f'the burst is drained in under {LIMIT_MS/1000:.0f} s', ms < LIMIT_MS, f'{ms/1000:.1f} s')

        time.sleep(1.5)
        total = WALK + BURST
        st = page.evaluate('() => document.getElementById("draft-state").textContent')
        check(f'every point is kept ({total})', f'{total} נקודות' in st, st)
        drawn = page.evaluate('''() => { const s = map.getSource('src-editor') || null;
           const d = s && (s._data || (s.serialize && s.serialize().data));
           const f = d && d.features && d.features.find((x) => x.geometry.type === 'LineString');
           return f ? f.geometry.coordinates.length : -1; }''')
        if drawn != -1:
            check('the line on the map has caught up', drawn == total, drawn)
        mirror = page.evaluate("() => (JSON.parse(localStorage.getItem('dk.editor.v1') || '{}').path || []).length")
        check('the live mirror has caught up', mirror == total, mirror)
        check('the native log got a line per fix', page.evaluate('() => window.__nativeLines') >= total)
        log = page.evaluate("() => JSON.parse(localStorage.getItem('dk.gpslog.v1') || '[]')")
        check('the page log has the latest fix', any(f'push #{total}' in l for l in log), log[-3:])

        page.evaluate('() => Drafts.stop()')
        time.sleep(0.5)
        check('stopping drops the mirror', page.evaluate("() => localStorage.getItem('dk.editor.v1') === null"))
        check('no page errors', not errors, errors[:3])
        b.close()
finally:
    srv.terminate()

print()
print('FAILED: ' + ', '.join(fails) if fails else 'all ok')
sys.exit(1 if fails else 0)
