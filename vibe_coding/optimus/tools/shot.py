#!/usr/bin/env python3
"""צילום מסך של הסימולטור. שימוש:
python3 tools/shot.py "index.html?auto=1" out.png [wait_s] [js_before] [wait2] [w] [h]
דורש שרת: python3 -m http.server 8791 מתוך תיקיית הפרויקט.
"""
import sys, asyncio
from playwright.async_api import async_playwright

async def main():
    url = sys.argv[1]; out = sys.argv[2]
    wait = float(sys.argv[3]) if len(sys.argv) > 3 else 4
    js = sys.argv[4] if len(sys.argv) > 4 else ''
    wait2 = float(sys.argv[5]) if len(sys.argv) > 5 else 2
    w = int(sys.argv[6]) if len(sys.argv) > 6 else 1400
    h = int(sys.argv[7]) if len(sys.argv) > 7 else 800
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=False, args=['--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=gl', '--autoplay-policy=no-user-gesture-required'])
        pg = await b.new_page(viewport={'width': w, 'height': h})
        logs = []
        pg.on('console', lambda m: logs.append(f'[{m.type}] {m.text}'))
        pg.on('pageerror', lambda e: logs.append(f'[pageerror] {e}'))
        await pg.goto('http://localhost:8791/' + url)
        await pg.wait_for_timeout(int(wait * 1000))
        if js:
            r = await pg.evaluate(js)
            if r is not None: print('js:', r)
            await pg.wait_for_timeout(int(wait2 * 1000))
        await pg.screenshot(path=out)
        fps = await pg.evaluate('window.__fps')
        print('fps', fps)
        for l in logs[:30]: print(l)
        await b.close()

asyncio.run(main())
