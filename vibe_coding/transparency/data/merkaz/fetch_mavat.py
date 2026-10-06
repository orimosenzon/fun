"""Full מבא"ת JSON for each plan page (opened in Chrome like a person would)."""
import json, os, re, sys, time
from playwright.sync_api import sync_playwright
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mavat")
os.makedirs(OUT, exist_ok=True)
mids = sys.argv[1:]
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    ctx = b.new_context()
    ctx.route(re.compile(r"govmap\.gov\.il|fonts\.(googleapis|gstatic)"), lambda r: r.abort())
    page = ctx.new_page()
    for mid in mids:
        if os.path.exists(f"{OUT}/{mid}.json"):
            continue
        try:
            with page.expect_response(lambda r: "rest/api/SV4/1?mid=" in r.url, timeout=60000) as info:
                page.goto(f"https://mavat.iplan.gov.il/SV4/1/{mid}/310", wait_until="commit", timeout=60000)
            json.dump(info.value.json(), open(f"{OUT}/{mid}.json", "w"), ensure_ascii=False, indent=1)
            print("ok", mid)
        except Exception as e:
            print("fail", mid, str(e)[:100])
        time.sleep(2)
    b.close()
