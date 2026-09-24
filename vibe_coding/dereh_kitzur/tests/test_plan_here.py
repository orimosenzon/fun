"""מה מתוכנן כאן: the live planning lookup, in a real Chrome.

Runs against recorded answers by default, because the planning service is a
government host behind a WAF that stops answering an address after a burst.
`--live` asks the real one.
"""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8770
URL = f'http://127.0.0.1:{PORT}/index.html'
SPOT = {'lng': 34.97507, 'lat': 32.47411}     # the open space: gush 10102, parcel 110
LIVE = '--live' in sys.argv
S = os.path.join(OUT, 'fixtures')

fails = []
def check(name, ok, detail=''):
    print(('  ok   ' if ok else '  FAIL ') + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

# Answer the two services from the recorded captures, by URL.
STUB = '''(fx) => {
  const real = window.fetch;
  window.fetch = (url, init) => {
    const u = String(url);
    const body = init && init.body ? String(init.body) : '';
    let doc = null;
    if (u.includes('/MapServer/1/query')) {
      doc = body.includes('plan_area_name') ? fx.deposit : fx.plans;
    } else if (u.includes('/MapServer/4/query')) {
      doc = fx.uses;
    } else if (u.includes('geoserver')) {
      doc = fx.parcel;
    }
    if (doc) return Promise.resolve(new Response(JSON.stringify(doc),
      {status: 200, headers: {'Content-Type': 'application/json'}}));
    return real(url, init);
  };
}'''

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
  with sync_playwright() as p:
     b = p.chromium.launch(channel='chrome', args=['--use-gl=angle', '--use-angle=gl'])
     pg = b.new_page(viewport={'width': 1200, 'height': 900})
     errs = []
     pg.on('pageerror', lambda e: errs.append('pageerror: ' + str(e)))
     pg.on('console', lambda m: errs.append('console: ' + m.text) if m.type == 'error' else None)
     pg.goto(URL)
     pg.wait_for_function('() => typeof PlanHere !== "undefined" && typeof map !== "undefined" && !!map.getSource("src-trails")', timeout=40000)
     pg.wait_for_timeout(2500)
     pg.evaluate('''() => { document.querySelectorAll('.sheet[aria-modal]').forEach(x => x.style.display='none');
       try { sessionStorage.clear(); } catch (e) {} }''')
     if not LIVE:
         pg.evaluate(STUB, json.load(open(os.path.join(S, 'planning.json'))))
     print('mode:', 'LIVE' if LIVE else 'recorded answers')

     # ---- the lookup ----
     found = pg.evaluate('''async (spot) => {
       const r = await PlanHere.lookUp(spot);
       return {plot: r.plot,
         live: r.live && {use: r.live.mavat_name, plan: r.live.pl_number, cell: r.live.num},
         plans: r.plans.map(a => ({num: a.pl_number, name: a.pl_name,
           status: a.internet_short_status, shuts: PlanHere.date(PlanHere.shutsOn(a)),
           units: a.quantity_delta_120}))};
     }''', SPOT)
     print('parcel:', json.dumps(found['plot'], ensure_ascii=False))
     print('designation in force:', json.dumps(found['live'], ensure_ascii=False))
     print(f'plans here: {len(found["plans"])}')
     for a in found['plans']:
         print(f"   {a['num']:16s} {str(a['status']):14s} יח\"ד {str(a['units']):>6}  {a['name'][:46]}")

     check('the parcel is found', found['plot'] and found['plot']['gush'] == 10102)
     check('a designation is in force', bool(found['live']), found['live'])
     check('the plans over the point come back', len(found['plans']) >= 5, len(found['plans']))
     # The whole point: the latest plan in force wins, not the first answer.
     check('the designation is the latest plan that is law',
           found['live'] and found['live']['plan'] == '308-0701094', found['live'])

     # ---- the sheet ----
     pg.evaluate('async (spot) => { await PlanHere.at(spot); }', SPOT)
     pg.wait_for_timeout(900)
     shown = pg.evaluate('''() => {
       const c = document.getElementById('plan-card');
       const txt = (s) => { const n = c.querySelector(s); return n ? n.textContent.replace(/\\s+/g,' ').trim() : null; };
       return {open: !document.getElementById('plan-sheet').hidden,
               lead: txt('.sheet-lead'), use: txt('.ph-use b'),
               plans: c.querySelectorAll('.ph-plan').length,
               openOnes: c.querySelectorAll('.ph-open').length,
               adds: [...c.querySelectorAll('.ph-adds')].map(n => n.textContent.trim()),
               limits: c.querySelectorAll('.ph-limits li').length,
               links: c.querySelectorAll('.ph-links a').length};
     }''')
     print('\nsheet:', json.dumps(shown, ensure_ascii=False, indent=1))
     check('the sheet opens', shown['open'])
     check('it names the parcel', shown['lead'] and 'גוש 10102' in shown['lead'], shown['lead'])
     check('it names the designation', shown['use'] == 'שטח ציבורי פתוח', shown['use'])
     check('every plan is listed', shown['plans'] == len(found['plans']), shown['plans'])
     check('it says what the plans add', any('יחידות דיור' in a or 'יחידת דיור' in a for a in shown['adds']), shown['adds'])
     check('it says what it cannot know', shown['limits'] >= 3, shown['limits'])
     pg.screenshot(path=os.path.join(OUT, 'shot_plan_here.png'))

     # ---- the banner ----
     rows = pg.evaluate('async () => await PlanHere.openForObjection()')
     print(f'\nopen for objection town-wide: {len(rows)}')
     for r in rows[:6]:
         print(f"   {r['num']:16s} נסגרת בעוד {r['left']:4d} ימים  יח\"ד {r['units']:>5}  {r['name'][:44]}")
     check('only plans whose window is still open', all(r['left'] >= 0 for r in rows), rows[:2])
     check('soonest first', rows == sorted(rows, key=lambda r: r['left']))

     banner = pg.evaluate('''async () => { await PlanHere.paintBanner();
       const b = document.getElementById('plan-banner');
       return {hidden: b.hidden, text: b.textContent.replace(/\\s+/g,' ').trim()}; }''')
     print('banner:', json.dumps(banner, ensure_ascii=False))
     check('the banner matches the count', banner['hidden'] == (len(rows) == 0))
     if rows:
         check('the banner names the nearest deadline', 'נסגרת' in banner['text'], banner['text'])

     # ---- the banner's own list, which is live and not the built file ----
     pg.evaluate('() => document.getElementById("plan-banner").click()')
     pg.wait_for_timeout(900)
     lst = pg.evaluate('''() => {
      const c = document.getElementById('plan-card');
      return {open: !document.getElementById('plan-sheet').hidden,
              rows: c.querySelectorAll('.ph-plan').length,
              allOpen: c.querySelectorAll('.ph-plan').length === c.querySelectorAll('.ph-open').length,
              deadlines: c.querySelectorAll('.ph-deadline').length};
     }''')
     print('banner list:', json.dumps(lst, ensure_ascii=False))
     check('the banner opens a list of them', lst['open'] and lst['rows'] == len(rows), lst)
     check('every row is an open one, with its date', lst['allOpen'] and lst['deadlines'] == lst['rows'], lst)
     pg.evaluate('() => PlanHere.close()')

  # ---- the arming flow, through real clicks ----
     pg.evaluate('() => { document.querySelectorAll(".sheet").forEach(s => { s.hidden = true; s.style.display=""; }); }')
     pg.evaluate('''() => { if (map.getTerrain()) map.setTerrain(null);
         map.jumpTo({center: [34.97507, 32.47411], zoom: 17, pitch: 0, bearing: 0}); }''')
     pg.wait_for_timeout(1200)
     pg.click('#plan-ask')
     armed = pg.evaluate('() => ({armed: PlanHere.isArmed(), body: document.body.classList.contains("asking-plan")})')
     check('the button arms the next tap', armed['armed'] and armed['body'], armed)

     before = pg.evaluate('() => selectedId')
     pg.mouse.click(400, 430)
     pg.wait_for_timeout(1600)
     after = pg.evaluate('''() => ({open: !document.getElementById('plan-sheet').hidden,
       armed: PlanHere.isArmed(), plans: document.querySelectorAll('#plan-card .ph-plan').length,
       selected: typeof selectedId !== 'undefined' ? selectedId : null})''')
     print('after tapping the map:', json.dumps(after, ensure_ascii=False))
     check('the tap asks the question', after['open'] and after['plans'] > 0, after)
     check('it disarms itself afterwards', not after['armed'])
     check('the tap did not also pick a trail', after['selected'] == before, after['selected'])

     # Escape backs out
     pg.keyboard.press('Escape')
     pg.wait_for_timeout(300)
     check('Escape closes the answer', not pg.evaluate('() => PlanHere.isOpen()'))
     pg.screenshot(path=os.path.join(OUT, 'shot_plan_flow.png'))

     check('no page errors', not errs, errs[:3])
     b.close()

finally:
    srv.terminate()

print()
print(f'{len(fails)} failed' if fails else 'all passed')
for f in fails: print('  - ' + f)
sys.exit(1 if fails else 0)
