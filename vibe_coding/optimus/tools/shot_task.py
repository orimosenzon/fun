#!/usr/bin/env python3
"""צילומי משימה ברגעים מסוימים. שימוש:
python3 tools/shot_task.py <task> out_prefix t1,t2,... [cam: px,py,pz,tx,ty,tz בקואורדינטות סימולציה] [clean]
הזמן הוא זמן סימולציה. המצלמה בקואורדינטות MuJoCo (z למעלה)."""
import sys, asyncio
from playwright.async_api import async_playwright
async def main():
    task, pre, times = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(',')]
    cam = [float(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 and sys.argv[4] else None
    clean = len(sys.argv) > 5 and sys.argv[5] == 'clean'
    extra = sys.argv[6] if len(sys.argv) > 6 else ''
    async with async_playwright() as p:
        b = await p.chromium.launch(headless=False, args=['--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=gl'])
        pg = await b.new_page(viewport={'width': 1200, 'height': 800})
        logs = []
        pg.on('console', lambda m: logs.append(f'[{m.type}] {m.text}') if m.type in ('error', 'warning') else None)
        pg.on('pageerror', lambda e: logs.append(f'[pageerror] {e}'))
        await pg.goto(f'http://localhost:8791/index.html?task={task}' + ('&clean=1' if clean else '') + extra)
        await pg.wait_for_timeout(2500)
        # עוצרים, ומריצים את הסימולציה ידנית עד כל זמן מבוקש (מהר ודטרמיניסטי)
        await pg.evaluate("window.__app.paused = true; document.getElementById('intro').classList.add('hidden')")
        for t in times:
            st = await pg.evaluate(f"""(() => {{ const a = window.__app; while (a.world.t < {t}) window.__simStep(); a.paused = true; return a.tasks.status; }})()""")
            if cam:
                await pg.evaluate(f"""(() => {{ const r = window.__app.rig; r.setMode('free');
                  r.camera.position.set({cam[0]},{cam[2]},{-cam[1]}); r.orbit.target.set({cam[3]},{cam[5]},{-cam[4]}); r.orbit.update(); }})()""")
            await pg.evaluate("window.__render()")
            await pg.wait_for_timeout(300)
            await pg.screenshot(path=f'{pre}_{t:05.1f}.png')
            print(t, st)
        for l in logs[:20]: print(l)
        await b.close()
asyncio.run(main())
