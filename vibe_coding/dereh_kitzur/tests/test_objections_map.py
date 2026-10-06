"""Parcels open to objection, glowing red on the parcels layer, in a real Chrome.

The morning's file (web/data/objections.json, from watch_objections.py) is
served in place of the data repo's copy, and Xplan's live answer is a fixture:
one plan the file also has, with its date, and one plan the file does not
know yet, which has no parcels and must glow by its blue line. Checks:
  * the banner counts both sources, once each;
  * with the parcels on, far out, every open plan has a red disc and the
    parcels of the file's plans are painted;
  * the plan the file does not know glows by its outline;
  * the list says which windows are missing from the national service, and
    "show on the map" turns the parcels on and lands on the plan;
  * a tap on a red parcel says it is open, until when and where to object;
  * a tap on a disc far out opens the list on that plan.
"""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(OUT), 'web')
PORT = 8773
URL = f'http://127.0.0.1:{PORT}/index.html'
FILE = json.load(open(os.path.join(WEB, 'data', 'objections.json'), encoding='utf-8'))
# The file is refreshed each morning from the real plans, so it can still hold
# one whose objection period closed since; the app drops those, and so does this.
_today = time.strftime('%Y-%m-%d')
FILE['plans'] = [p for p in FILE['plans'] if not p.get('shuts') or p['shuts'] >= _today]
fails = []
def check(name, ok, detail=''):
    print(('  ok   ' if ok else '  FAIL ') + name + ('' if ok else '   ' + str(detail)))
    if not ok: fails.append(name)

known = FILE['plans'][0]
LIVE = {'features': [
    {'attributes': {'pl_number': known['num'], 'pl_name': known['name'], 'pl_url': known['url'],
                    'pl_last_deposit_date': None, 'pl_rejection_date': None, 'pl_by_auth_of': 2,
                    'quantity_delta_120': known['units']},
     'geometry': {'rings': known['rings']}},
    {'attributes': {'pl_number': 'TEST-1', 'pl_name': 'תכנית בדיקה חדשה', 'pl_url': 'https://mavat.iplan.gov.il/',
                    'pl_last_deposit_date': int((time.time() + 30 * 86400) * 1000),
                    'pl_rejection_date': None, 'pl_by_auth_of': 3, 'quantity_delta_120': 4},
     'geometry': {'rings': [[[34.9700, 32.4600], [34.9712, 32.4600], [34.9712, 32.4610],
                             [34.9700, 32.4610], [34.9700, 32.4600]]]}},
]}
# The live row for `known` has no date, so it must not drop the file's.
LIVE['features'][0]['attributes']['pl_last_deposit_date'] = None
n_open = len(FILE['plans']) + 1
hidden = sum(1 for p in FILE['plans'] if p['hidden'])
target = next(p for p in FILE['plans'] if p['hidden'] and p['parcels'])

srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'],
                       cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(0.8)
try:
  with sync_playwright() as p:
     b = p.chromium.launch(channel='chrome', args=['--use-gl=angle', '--use-angle=gl'])
     ctx = b.new_context(viewport={'width': 1200, 'height': 900}, locale='he-IL')
     ctx.add_init_script("try { localStorage.setItem('dk.welcome.v1', '1'); sessionStorage.clear(); } catch (e) {}")
     ctx.route('**/data/objections.json', lambda r: r.fulfill(
         status=200, content_type='application/json', body=json.dumps(FILE)))
     def xplan(route):
         body = route.request.post_data or ''
         if 'internet_short_status' in body:
             route.fulfill(status=200, content_type='application/json', body=json.dumps(LIVE))
         else:
             route.fulfill(status=200, content_type='application/json', body='{"features": []}')
     ctx.route('https://ags.iplan.gov.il/**', xplan)
     pg = ctx.new_page()
     errs = []
     pg.on('pageerror', lambda e: errs.append('pageerror: ' + str(e)))
     pg.on('console', lambda m: errs.append('console: ' + m.text) if m.type == 'error' else None)
     pg.goto(URL)
     pg.wait_for_function('() => typeof Layers !== "undefined" && typeof map !== "undefined"'
                          ' && !!map.getSource("src-trails")', timeout=40000)
     pg.wait_for_function('() => !document.getElementById("plan-banner").hidden', timeout=20000)
     banner = pg.inner_text('#plan-banner')
     check('the banner counts the file and the live answer, once each', f'{n_open} תכניות' in banner, banner)

     pg.evaluate('''() => { document.querySelectorAll('.sheet').forEach((x) => { x.hidden = true; });
       if (map.getTerrain()) map.setTerrain(null);
       map.jumpTo({center: [34.972, 32.472], zoom: 13.2, pitch: 0, bearing: 0});
       Layers.turnOn("parcels"); }''')
     pg.wait_for_function('() => map.getSource("grd-parcels") && map.isSourceLoaded("grd-parcels")'
                          ' && map.getLayer("pgo-f-parcels")', timeout=30000)
     pg.wait_for_timeout(2500)
     far = pg.evaluate('''() => ({
       discs: new Set(map.queryRenderedFeatures({ layers: ['pgo-p-parcels'] }).map((f) => f.properties.num)).size,
       parcels: new Set(map.queryRenderedFeatures({ layers: ['pgo-f-parcels'] }).map((f) => f.id)).size,
       shapes: map.queryRenderedFeatures({ layers: ['pgo-sf-parcels'] }).length,
       vis: map.getLayoutProperty('pgo-f-parcels', 'visibility') })''')
     print('  far out:', far)
     want_ids = len({i for q in FILE['plans'] for i in q['parcels']})
     check('a red disc for every open plan', far['discs'] == n_open, far)
     check('the parcels of the file are painted', far['parcels'] >= want_ids - 1, (far, want_ids))
     check('the plan the file does not know glows by its outline', far['shapes'] >= 1, far)
     pg.screenshot(path=os.path.join(OUT, 'shot_objections_far.png'))

     # The list.
     check('the ☰ button counts the open plans', pg.inner_text('#menu-badge') == str(n_open), pg.inner_text('#menu-badge'))
     pg.click('#menu-btn')
     pg.wait_for_timeout(300)
     pg.click('#plan-banner')
     pg.wait_for_selector('#plan-card .ph-plan', timeout=10000)
     rows = pg.locator('#plan-card .ph-plan').count()
     alert = pg.inner_text('#plan-card .ph-alert') if pg.locator('#plan-card .ph-alert').count() else ''
     check('the list has every open plan', rows == n_open, rows)
     check('it says how many the national service is missing', str(hidden) in alert or (hidden == 1 and 'אחת' in alert), alert)
     card = pg.inner_text('#plan-card')
     check('estimates are said as estimates', 'מועד משוער' in card, card[:300])
     check('where to object is said', 'בסמכות הוועדה המקומית' in card and 'מקוון במבא"ת' in card, card[:300])

     pg.click(f'#plan-card [data-act="obj-fly"][data-num="{target["num"]}"]')
     pg.wait_for_timeout(2500)
     near = pg.evaluate(f'''() => ({{ zoom: map.getZoom(), sheet: document.getElementById('plan-sheet').hidden,
       red: map.queryRenderedFeatures({{ layers: ['pgo-f-parcels'] }}).map((f) => f.id) }})''')
     print('  on the plan:', {k: v for k, v in near.items() if k != 'red'}, len(near['red']))
     check('"show on the map" closes the list and lands close in', near['sheet'] and near['zoom'] > 16, near)
     check('and its parcel is red there', target['parcels'][0] in near['red'], near['red'][:5])
     pg.screenshot(path=os.path.join(OUT, 'shot_objections_near.png'))

     # A tap on the red parcel.
     pt = pg.evaluate(f'''() => {{ const c = {json.dumps(target['centre'])}; const p = map.project(c); return [p.x, p.y]; }}''')
     box = pg.locator('#map canvas').bounding_box()
     pg.mouse.click(box['x'] + pt[0], box['y'] + pt[1])
     pg.wait_for_timeout(1500)
     detail = pg.inner_text('body')
     check('the parcel says it is open to objection', 'פתוחה להתנגדות' in detail and target['num'] in detail, '')

     # A disc far out.
     pg.evaluate(f'''() => {{ deselect && deselect(); map.jumpTo({{center: {json.dumps(target['centre'])}, zoom: 13.5}}); }}''')
     pg.wait_for_timeout(1500)
     pt = pg.evaluate(f'''() => {{ const p = map.project({json.dumps(target['centre'])}); return [p.x, p.y]; }}''')
     pg.mouse.click(box['x'] + pt[0], box['y'] + pt[1])
     pg.wait_for_selector('#plan-card .ph-focus', timeout=8000)
     first = pg.locator('#plan-card .ph-plan').first.inner_text()
     check('a disc opens the list on its plan', target['num'] in first, first[:120])

     real = [e for e in errs if 'img/' not in e and 'favicon' not in e]
     check('no errors', not real, real[:5])
     b.close()
finally:
  srv.terminate()

print()
print('FAIL: ' + ', '.join(fails) if fails else 'all ok')
sys.exit(1 if fails else 0)
