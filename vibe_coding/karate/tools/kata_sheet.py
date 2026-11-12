#!/usr/bin/env python3
"""גיליון של כל תנועות קאטה (שיא כל תנועה). שימוש:
python3 tools/kata_sheet.py heian1 out.png [view=0] [from] [to] [cols=7] [ph]
דורש שרת 8793.
"""
import sys, asyncio, io
from playwright.async_api import async_playwright
from PIL import Image, ImageDraw, ImageFont

async def main():
    kid, out = sys.argv[1], sys.argv[2]
    view = sys.argv[3] if len(sys.argv) > 3 else '0'
    a = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    b_ = int(sys.argv[5]) if len(sys.argv) > 5 else 999
    cols = int(sys.argv[6]) if len(sys.argv) > 6 else 7
    ph = sys.argv[7] if len(sys.argv) > 7 else ''
    W, H = 300, 340
    imgs = []
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=False, args=['--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=gl'])
        pg = await b.new_page(viewport={'width': W, 'height': H})
        errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto(f'http://localhost:8793/index.html?kata={kid}&ui=0&view={view}&zoom=0.62'); await pg.wait_for_timeout(1500)
        n = await pg.evaluate('window.__kata.K.steps.length')
        for i in range(a, min(b_, n) + 1):
            lab = await pg.evaluate(f'''(() => {{ const k = window.__kata, m = k.TL.marks[{i}-1]; k.t = {"m.tp" if ph == "prep" else "m.tk + 0.001"}; k.playing = false;
              const s = k.K.steps[{i}-1]; return s.count + (s.m.sub ? "b" : ""); }})()''')
            await pg.wait_for_timeout(450)
            im = Image.open(io.BytesIO(await pg.screenshot())).convert('RGB')
            d = ImageDraw.Draw(im); d.rectangle([0, 0, 46, 22], fill='black')
            d.text((3, 1), str(lab), fill='yellow', font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17))
            imgs.append(im)
        for e in errs[:5]: print('[pageerror]', e)
        await b.close()
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * W, rows * H))
    for i, im in enumerate(imgs): sheet.paste(im, ((i % cols) * W, (i // cols) * H))
    sheet.save(out); print(len(imgs), sheet.size)
asyncio.run(main())
