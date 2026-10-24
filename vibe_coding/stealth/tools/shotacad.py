#!/usr/bin/env python3
"""צילום כל המעבדות באקדמיה. שימוש: python3 tools/shotacad.py outdir [lab-id ...]
דורש: python3 -m http.server 8791 מתוך תיקיית הפרויקט."""
import sys, asyncio
from playwright.async_api import async_playwright
out = sys.argv[1]; only = sys.argv[2:]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=False, args=['--ignore-gpu-blocklist', '--use-angle=gl'])
        pg = await b.new_page(viewport={'width': 1450, 'height': 950})
        logs = []
        pg.on('console', lambda m: logs.append(f'[{m.type}] {m.text}'))
        pg.on('pageerror', lambda e: logs.append(f'[pageerror] {e}'))
        await pg.goto('http://localhost:8791/academy.html'); await pg.wait_for_timeout(2500)
        labs = await pg.eval_on_selector_all('.lab', 'els => els.map(e => e.id)')
        for lid in labs:
            if only and lid not in only: continue
            el = await pg.query_selector('#' + lid)
            await el.scroll_into_view_if_needed(); await pg.wait_for_timeout(2500)
            await el.screenshot(path=f'{out}/{lid}.png')
        await pg.evaluate('window.scrollTo(0,0)'); await pg.wait_for_timeout(500)
        await pg.screenshot(path=f'{out}/acad_top.png')
        for l in logs[:30]: print(l)
        await b.close()
asyncio.run(main())
