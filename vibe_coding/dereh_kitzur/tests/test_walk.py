"""Recording a trail by walking it, with a faked GPS.

On 11/10/2026 Ori walked with the recorder on and nothing was recorded. The
walk log (?gpslog) showed why: Chrome had been given only approximate
location, one fix at exactly 2000 m and then silence. This checks:
  * precise fixes become points, a vague one is skipped;
  * a 2000 m fix brings up the "precise location" note, once, with the
    settings button on Android, and the recording carries on afterwards;
  * every fix lands in the walk log, and ?gpslog shows it;
  * in a browser, Native.isApp is false and nothing native is touched.

    python3 tests/test_walk.py

Needs Playwright with Google Chrome (channel='chrome'). About half a minute.
"""
import subprocess, sys, time, os
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8779
ANDROID_UA = ('Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/155.0.0.0 Mobile Safari/537.36')

fails = []
def check(name, ok, detail=''):
    print(('  ok  ' if ok else '  FAIL') + ' ' + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(channel='chrome', headless=True,
                              args=['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist'])
        ctx = b.new_context(viewport={'width': 400, 'height': 800}, locale='he-IL',
                            user_agent=ANDROID_UA, has_touch=True,
                            permissions=['geolocation'],
                            geolocation={'latitude': 32.475, 'longitude': 34.97, 'accuracy': 2000})
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1');"
                            " localStorage.removeItem('dk.editor.v1');"
                            # once per tab, or the ?gpslog load would wipe it
                            " if (!sessionStorage.getItem('t')) { sessionStorage.setItem('t', 1);"
                            " localStorage.removeItem('dk.gpslog.v1'); } } catch (e) {}")
        page = ctx.new_page()
        errors, dialogs = [], []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('dialog', lambda d: (dialogs.append(d.message), d.accept()))
        page.goto(f'http://127.0.0.1:{PORT}/index.html')
        page.wait_for_function('() => typeof map !== "undefined" && map.loaded()', timeout=60000)
        time.sleep(2)
        check('a browser is not the app', page.evaluate('() => Native.isApp === false'))

        page.evaluate('() => document.querySelectorAll(".sheet").forEach((x) => { x.hidden = true; })')
        # The walk button lives in a sheet whose contents are drawn on demand;
        # a button with the same data-act goes through the same handler.
        page.evaluate('''() => { const s = document.getElementById('draft-sheet'); s.hidden = false;
           const b = document.createElement('button'); b.dataset.act = 'walk';
           s.appendChild(b); b.click(); }''')
        time.sleep(1.5)
        check('a 2000 m fix brings up the precise-location note',
              page.locator('.dk-modal').count() == 1)
        check('...with the settings button on Android', page.locator('[data-a=open]').count() == 1)
        check('the bar says the location is approximate',
              'משוער' in page.inner_text('#draft-state'), page.inner_text('#draft-state'))
        page.click('[data-a=close]')

        for i in range(1, 6):
            ctx.set_geolocation({'latitude': 32.475 + i * 0.0001, 'longitude': 34.97,
                                 'accuracy': 120 if i == 3 else 8})
            time.sleep(1.2)
        time.sleep(1)
        n = page.evaluate('() => document.getElementById("draft-state").textContent')
        check('four precise fixes became four points, the vague one did not', '4 נקודות' in n, n)
        check('the note came once only', page.locator('.dk-modal').count() == 0)
        page.evaluate('() => Drafts.stop()')

        log = page.evaluate("() => JSON.parse(localStorage.getItem('dk.gpslog.v1') || '[]')")
        check('the log has the start and the stop',
              any('walk start' in l for l in log) and any('walk stop' in l for l in log))
        check('the log has the coarse fix', any('acc=2000' in l and 'weak' in l for l in log))
        check('the log has the points', sum('push #' in l for l in log) == 4,
              [l for l in log if 'fix' in l])

        page.goto(f'http://127.0.0.1:{PORT}/index.html?gpslog')
        time.sleep(2)
        check('?gpslog shows the log', 'walk start' in page.inner_text('.dk-modal pre'))
        check('no page errors', not errors, errors)
        check('no alerts along the way', not dialogs, dialogs)
        b.close()
finally:
    srv.terminate()

print('\nall passed' if not fails else f'\n{len(fails)} failed')
sys.exit(1 if fails else 0)
