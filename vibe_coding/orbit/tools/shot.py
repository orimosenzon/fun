# צילום מסך עם Playwright: python3 tools/shot.py <hash> <out.png> [wait_ms] [w] [h] [js]
import sys, asyncio
from playwright.async_api import async_playwright
async def main():
    hsh, out = sys.argv[1], sys.argv[2]
    wait = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    w = int(sys.argv[4]) if len(sys.argv) > 4 else 1440
    hgt = int(sys.argv[5]) if len(sys.argv) > 5 else 900
    js = sys.argv[6] if len(sys.argv) > 6 else None
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
        pg = await b.new_page(viewport={'width': w, 'height': hgt})
        logs = []
        pg.on('console', lambda m: logs.append(f'{m.type}: {m.text}'))
        pg.on('pageerror', lambda e: logs.append(f'PAGEERROR: {e}'))
        await pg.goto(f'http://localhost:8765/index.html#{hsh}')
        await pg.wait_for_timeout(wait)
        if js:
            await pg.evaluate(js)
            await pg.wait_for_timeout(int(sys.argv[7]) if len(sys.argv) > 7 else 2000)
        await pg.screenshot(path=out)
        for l in logs: print(l)
        await b.close()
asyncio.run(main())
