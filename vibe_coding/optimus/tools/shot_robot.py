#!/usr/bin/env python3
"""צילומי הרובוט מכמה זוויות, בלי HUD. שימוש: python3 tools/shot_robot.py out_prefix [wait] [extra_query]
זוויות: front, side, back, closeup (ראש וכתפיים), hands"""
import sys, asyncio
from playwright.async_api import async_playwright
VIEWS = {
  'front': ([2.6, 1.15, 0.0], [0, 0.95, 0]),
  'threeq': ([1.9, 1.35, -1.6], [0, 0.95, 0]),
  'side': ([0.0, 1.1, -2.7], [0, 0.95, 0]),
  'back': ([-2.5, 1.3, 0.6], [0, 0.95, 0]),
  'head': ([0.75, 1.62, -0.35], [0, 1.45, 0]),
  'hands': ([0.55, 0.95, -0.6], [0.05, 0.75, -0.15]),
  'face': ([0.85, 1.6, -0.12], [0, 1.58, 0]),
  'hips': ([1.2, 0.95, -0.5], [0, 0.85, 0]),
}
async def main():
    pre = sys.argv[1]; wait = float(sys.argv[2]) if len(sys.argv) > 2 else 5
    extra = sys.argv[3] if len(sys.argv) > 3 else ''
    views = sys.argv[4].split(',') if len(sys.argv) > 4 else list(VIEWS)
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=False, args=['--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=gl'])
        pg = await b.new_page(viewport={'width': 1000, 'height': 1000})
        logs = []
        pg.on('console', lambda m: logs.append(f'[{m.type}] {m.text}') if m.type in ('error', 'warning') else None)
        pg.on('pageerror', lambda e: logs.append(f'[pageerror] {e}'))
        await pg.goto(f'http://localhost:8791/index.html?auto=1&clean=1&scene=empty{extra}')
        await pg.wait_for_timeout(int(wait * 1000))
        for v in views:
            (cp, tg) = VIEWS[v]
            await pg.evaluate(f"""(() => {{ const a = window.__app; a.paused = true; const r = a.rig; r.setMode('free');
              r.camera.position.set({cp[0]},{cp[1]},{cp[2]}); r.orbit.target.set({tg[0]},{tg[1]},{tg[2]}); r.orbit.update(); }})()""")
            await pg.wait_for_timeout(400)
            await pg.screenshot(path=f'{pre}_{v}.png')
        for l in logs[:20]: print(l)
        await b.close()
asyncio.run(main())
