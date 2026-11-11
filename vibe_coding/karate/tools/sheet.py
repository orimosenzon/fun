#!/usr/bin/env python3
"""גיליון תנוחות לבדיקה חזותית. שימוש:
python3 tools/sheet.py out.png "gyaku_c:0.2:0.5" "oi_c:0.4:-0.4" ...   (טכניקה:זמן:זווית מצלמה[:mir])
דורש שרת: python3 -m http.server 8793 מתוך תיקיית הפרויקט.
"""
import sys, asyncio, io
from playwright.async_api import async_playwright
from PIL import Image

async def main():
    out = sys.argv[1]; specs = sys.argv[2:]
    W, H = 520, 520
    imgs = []
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=False, args=['--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=gl'])
        pg = await b.new_page(viewport={'width': W, 'height': H})
        pg.on('pageerror', lambda e: print('[pageerror]', e))
        for s in specs:
            a = s.split(':')
            url = f'http://localhost:8793/index.html?pose={a[0]}&t={a[1]}&az={a[2]}&zoom={a[4] if len(a) > 4 else 0.5}' + ('&mir=1' if len(a) > 3 and a[3] == '1' else '') + ('&opp=1&d=' + a[5] if len(a) > 5 else '')
            await pg.goto(url); await pg.wait_for_timeout(1300)
            imgs.append(Image.open(io.BytesIO(await pg.screenshot())))
        await b.close()
    cols = min(4, len(imgs)); rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * W, rows * H))
    for i, im in enumerate(imgs): sheet.paste(im, ((i % cols) * W, (i // cols) * H))
    sheet.save(out)
asyncio.run(main())
