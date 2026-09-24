"""The floating column of buttons, at phone and desktop width.

Written when "מה מתוכנן כאן?" joined the column on 24/9/2026 and everything
above it had to move up: the flight button, and the little aircraft chip that
rides on its shoulder. The chip is positioned against the same anchors as the
button but is not its child, so the two can part company silently - which is
exactly the kind of thing a screenshot catches and a person does not.

    python3 tests/test_fabs.py

Needs Playwright with Google Chrome. Takes about half a minute.
"""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright
OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT=8794
fails=[]
def check(n, ok, d=''):
    print(('  ok   ' if ok else '  FAIL ')+n+('' if ok else '   '+str(d)))
    if not ok: fails.append(n)

srv = subprocess.Popen([sys.executable,'-m','http.server',str(PORT),'--bind','127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
  with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', args=['--use-gl=angle','--use-angle=gl'])
    for label, vp in (('phone', {'width':390,'height':844}), ('desktop', {'width':1280,'height':900})):
        ctx = b.new_context(viewport=vp, locale='he-IL')
        ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1','1'); } catch(e){}")
        pg = ctx.new_page()
        errs=[]
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(f'http://127.0.0.1:{PORT}/index.html')
        pg.wait_for_function('() => typeof PlanHere !== "undefined" && typeof map !== "undefined"', timeout=40000)
        pg.wait_for_timeout(4000)
        pg.evaluate('() => { const w=document.getElementById("welcome-sheet"); if (w) w.hidden=true; }')
        pg.wait_for_timeout(600)
        r = pg.evaluate('''() => {
          const at = (id) => { const n = document.getElementById(id); if (!n) return null;
            const r = n.getBoundingClientRect();
            return {id, t: Math.round(r.top), b: Math.round(r.bottom),
                    l: Math.round(r.left), r: Math.round(r.right), h: Math.round(r.height)}; };
          return {list: ['locate','basemap','tilt','layers','plan-fab','explore','craft'].map(at),
                  vh: window.innerHeight};
        }''')
        rows = [x for x in r['list'] if x and x['id'] != 'craft']
        craft = next(x for x in r['list'] if x and x['id'] == 'craft')
        print(f'\n{label}:')
        for x in rows: print(f"   {x['id']:9s} top {x['t']:4d}  bottom {x['b']:4d}")
        print(f"   {'craft':9s} top {craft['t']:4d}  bottom {craft['b']:4d}")
        # going up the column, each must sit strictly above the previous
        order = [x['t'] for x in rows]
        check(f'{label}: the column reads bottom to top in order',
              order == sorted(order, reverse=True), order)
        gaps = [rows[i]['t'] - rows[i+1]['b'] for i in range(len(rows)-1)]
        check(f'{label}: nothing overlaps', all(g >= 0 for g in gaps), gaps)
        check(f'{label}: all of it on screen', all(x['t'] >= 0 for x in rows), rows)
        ex = next(x for x in rows if x['id'] == 'explore')
        check(f'{label}: the craft chip rides the flight button',
              craft['t'] >= ex['t'] - 12 and craft['b'] <= ex['b'] + 12, {'craft': craft, 'explore': ex})
        pf = next(x for x in rows if x['id'] == 'plan-fab')
        check(f'{label}: the planning button sits below the balloon',
              pf['t'] > ex['b'] - 1, {'plan': pf, 'explore': ex})

        # and it works
        pg.click('#plan-fab')
        st = pg.evaluate('''() => ({armed: PlanHere.isArmed(),
            fab: document.getElementById('plan-fab').classList.contains('on'),
            card: document.getElementById('plan-ask').classList.contains('on')})''')
        check(f'{label}: it arms the question and both buttons light up',
              st['armed'] and st['fab'] and st['card'], st)
        pg.keyboard.press('Escape')
        pg.wait_for_timeout(200)
        check(f'{label}: Escape disarms both',
              pg.evaluate('''() => !PlanHere.isArmed()
                && !document.getElementById('plan-fab').classList.contains('on')
                && !document.getElementById('plan-ask').classList.contains('on')'''))
        check(f'{label}: no page errors', not errs, errs[:2])
        pg.screenshot(path=os.path.join(OUT, f'shot_fabs_{label}.png'))
        ctx.close()
    b.close()
finally:
    srv.terminate()
print()
print(f'{len(fails)} failed' if fails else 'all passed')
for f in fails: print('  - '+f)
sys.exit(1 if fails else 0)
