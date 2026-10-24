#!/usr/bin/env python3
"""צילום כל הלשוניות של שולחן התכנון. שימוש: python3 tools/tabs.py outdir [preset]
דורש: python3 -m http.server 8791 מתוך תיקיית הפרויקט."""
import sys, asyncio
from playwright.async_api import async_playwright
out = sys.argv[1]; preset = sys.argv[2] if len(sys.argv) > 2 else ''
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=False, args=['--ignore-gpu-blocklist', '--use-angle=gl'])
        pg = await b.new_page(viewport={'width': 1500, 'height': 950})
        logs = []
        pg.on('console', lambda m: logs.append(f'[{m.type}] {m.text}'))
        pg.on('pageerror', lambda e: logs.append(f'[pageerror] {e}'))
        await pg.goto('http://localhost:8791/index.html')
        await pg.wait_for_timeout(3000)
        if preset:
            await pg.click(f'#presets .btn:nth-child({preset})'); await pg.wait_for_timeout(2500)
        for t in ['aero', 'ir', 'more', 'mission']:
            await pg.click(f'#tabs button[data-t={t}]')
            await pg.wait_for_timeout(3500 if t == 'mission' else 1500)
            if t == 'mission':
                await pg.check('#mRef'); await pg.wait_for_timeout(4000)
                await pg.click('#mPlay'); await pg.wait_for_timeout(9000)
            el = await pg.query_selector('#t-' + t)
            await el.screenshot(path=f'{out}/tab_{t}.png')
        for l in logs[:30]: print(l)
        await b.close()
asyncio.run(main())
