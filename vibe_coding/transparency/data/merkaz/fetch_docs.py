"""Download a plan's public documents from מבא"ת, from inside the plan page itself."""
import base64, json, os, re, sys, time
from playwright.sync_api import sync_playwright
KEY = "6LeUKkMoAAAAAH4UacB4zewg4ult8Rcriv-ce0Db"
JS = """async ([key, url]) => {
  const tok = await new Promise(r => grecaptcha.ready(() => grecaptcha.execute(key, {action: 'importantAction'}).then(r)));
  const res = await fetch(url, {headers: {Authorization: tok}});
  const buf = new Uint8Array(await res.arrayBuffer());
  let s = ''; for (let i = 0; i < buf.length; i += 8192) s += String.fromCharCode(...buf.subarray(i, i + 8192));
  return [res.status, res.headers.get('content-type'), btoa(s)];
}"""
def wanted(mid, d, include):
    p = d['planDetails']['NUMB']
    out = []
    for r in (d.get('rsPlanDocs') or []) + (d.get('rsPlanDocsAdd') or []):
        if r.get('FILE_DATA') and any(w in r['DOC_NAME'] for w in include):
            out.append((r['ID'], r.get('PLAN_ENTITY_DOC_NUM') or r['FILE_DATA']['edNum'], r['DOC_NAME'], r['FILE_TYPE'].strip()))
    if 'החלטות' in include:
        for r in d.get('rsDes') or []:
            for suf in ['', '1', '_10']:
                eid = r.get('ENTITY_DOC_ID' + suf)
                if eid: out.append((int(eid), 'temp-default', f"החלטה {r['MEETING_DATE'].replace('/','-')} {suf}", (r.get('FILE_TYPE' + suf) or 'pdf').strip()))
    return p, out
jobs = json.loads(sys.argv[1])  # {mid: [doc-name substrings]}
with sync_playwright() as pw:
    b = pw.chromium.launch(channel="chrome", headless=True)
    ctx = b.new_context(); ctx.route(re.compile(r"govmap\.gov\.il"), lambda r: r.abort())
    pg = ctx.new_page()
    for mid, include in jobs.items():
        d = json.load(open(f"mavat/{mid}.json"))
        pnum, docs = wanted(mid, d, include)
        pg.goto(f"https://mavat.iplan.gov.il/SV4/1/{mid}/310", timeout=60000); pg.wait_for_timeout(5000)
        for eid, edn, name, ft in docs:
            fn = f"docs/{pnum.replace('/','_').replace(' ','')}__{name.replace('/','_')}.{ft}"
            if os.path.exists(fn): continue
            url = f"https://mavat.iplan.gov.il/rest/api/Attacments/?eid={int(eid)}&fn={name}.{ft}&edn={edn}&pn={pnum}"
            try:
                st, ct, data = pg.evaluate(JS, [KEY, url])
                raw = base64.b64decode(data)
                if st == 200 and len(raw) > 500:
                    open(fn, "wb").write(raw); print("ok", fn, len(raw), ct)
                else: print("bad", st, ct, len(raw), fn)
            except Exception as e: print("fail", fn, str(e)[:120])
            time.sleep(1.5)
    b.close()
