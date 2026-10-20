#!/usr/bin/env python3
"""צילומי מסך של המרכז המקצועי: python3 tools/shotpage.py out_prefix id1 id2 ..."""
import sys, asyncio
from playwright.async_api import async_playwright
async def main():
    pre = sys.argv[1]; ids = sys.argv[2:]
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=False, args=['--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=gl'])
        pg = await b.new_page(viewport={'width': 1400, 'height': 900})
        logs = []
        pg.on('console', lambda m: logs.append(f'[{m.type}] {m.text}') if m.type in ('error','warning') else None)
        pg.on('pageerror', lambda e: logs.append(f'[pageerror] {e}'))
        await pg.goto('http://localhost:8791/academy.html')
        await pg.wait_for_timeout(1500)
        for i in ids:
            await pg.evaluate(f"document.getElementById('{i}').scrollIntoView({{block:'start'}})")
            await pg.wait_for_timeout(int(__import__('os').environ.get('W', '2200')))
            await pg.screenshot(path=f'{pre}_{i}.png')
        for l in logs[:30]: print(l)
        await b.close()
asyncio.run(main())
