"""Drive the game with real keyboard events (headless) to check the input path."""
import asyncio, pathlib, subprocess, sys, time
from playwright.async_api import async_playwright
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
GPU = ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"]
SCRIPT = [("Enter", 0.2), (None, 4.5), ("KeyE", 2.5), (None, 1.0), ("KeyW", 1.5), (None, 4.0), ("ArrowRight", 1.5), (None, 2.0),
          ("ArrowLeft", 1.5), ("ArrowUp", 1.0), ("ArrowDown", 1.5), ("KeyS", 2.0), ("ArrowDown", 2.5), (None, 2.0), ("KeyC", 0.1), (None, 1.5), ("KeyD", 2.0)]
async def main():
    srv = subprocess.Popen([sys.executable, "-m", "http.server", "8792", "--bind", "127.0.0.1"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with async_playwright() as p:
            b = await p.chromium.launch(channel="chrome", headless=True, args=GPU)
            pg = await b.new_page(viewport={"width": 1280, "height": 720})
            logs = []; pg.on("console", lambda m: logs.append(m.text)); pg.on("pageerror", lambda e: logs.append(f"ERR {e}"))
            await pg.goto("http://127.0.0.1:8792/index.html")
            await pg.wait_for_function("window.game !== undefined", timeout=120000)
            i = 0
            for key, dur in SCRIPT:
                if key: await pg.keyboard.down(key)
                await asyncio.sleep(dur)
                if key: await pg.keyboard.up(key)
                st = await pg.evaluate("({s: game.state, fps: game.fps, p: game.bike.p.map(v=>v.toFixed(1)).join(','), v: (Math.hypot(...game.bike.v)*3.6).toFixed(0), main: game.bike.main.T.toFixed(0), lift: game.bike.lift.map(e=>e.T.toFixed(0)).join('/')})")
                print(f"after {key or 'wait'} {dur}s: {st}", flush=True)
                await pg.screenshot(path=str(OUT / f"k_{i:02d}.jpg"), type="jpeg", quality=85); i += 1
            print("\n".join(l for l in logs if 'ERR' in l or 'crash' in l))
            await b.close()
    finally: srv.terminate()
asyncio.run(main())
