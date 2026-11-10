"""Fly with real keyboard events and log telemetry: python3 tools/play.py OUT event "KeyW:dur:at,ArrowDown:dur:at,..." secs"""
import asyncio, pathlib, subprocess, sys, time
from playwright.async_api import async_playwright
ROOT = pathlib.Path(__file__).resolve().parent.parent
GPU = ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"]
async def main():
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    ev, keys, secs = sys.argv[2], sys.argv[3], float(sys.argv[4])
    shots = [float(x) for x in sys.argv[5:]]
    srv = subprocess.Popen([sys.executable, "-m", "http.server", "8794", "--bind", "127.0.0.1"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with async_playwright() as p:
            b = await p.chromium.launch(channel="chrome", headless=True, args=GPU)
            pg = await b.new_page(viewport={"width": 1280, "height": 720})
            logs = []
            pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
            await pg.goto(f"http://127.0.0.1:8794/index.html?event={ev}&fixed")
            await pg.wait_for_function("window.game !== undefined && game.state === 'fly'", timeout=90000)
            start = time.time()
            async def keyrun(k, dur, at):
                await asyncio.sleep(at); await pg.keyboard.down(k); await asyncio.sleep(dur); await pg.keyboard.up(k)
            tasks = [asyncio.create_task(keyrun(k.split(':')[0], float(k.split(':')[1]), float(k.split(':')[2]))) for k in keys.split(',') if k]
            nxt = 0
            while time.time() - start < secs:
                el = time.time() - start
                st = await pg.evaluate("(()=>{const b=game.bike; const e=game.sim.fcs.tele; return `${game.state} cp=${game.cp} V=${(Math.hypot(...b.v)*3.6).toFixed(0)} agl=${b.agl.toFixed(1)} vz=${b.v[1].toFixed(1)} vt=${(game.vt*3.6).toFixed(0)} tilt=${(b.tilt*57.3).toFixed(0)} heat=${b.heat.toFixed(2)} p=${b.p.map(v=>v.toFixed(0))} fps=${Math.round(game.fps||0)} msg=${document.getElementById('msg').classList.contains('on')?document.getElementById('msg').textContent:''}`})()")
                print(f"{el:5.1f} {st}", flush=True)
                if shots and el >= shots[0]:
                    await pg.screenshot(path=str(out / f"p_{int(shots.pop(0)*10):04d}.jpg"), type="jpeg", quality=85)
                await asyncio.sleep(1.0)
            print("\n".join(logs)); err = await pg.evaluate("document.getElementById('err').textContent"); print("ERR:" + err[:800] if err else "")
            await b.close()
    finally:
        srv.terminate()
asyncio.run(main())
