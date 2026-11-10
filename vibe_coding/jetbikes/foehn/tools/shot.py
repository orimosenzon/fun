"""Headless run on the GPU with screenshots.
   python3 tools/shot.py OUT "query" t1 t2 ... [--w 1600 --h 900] [--js "expr run before each shot"] [--keys "ArrowDown:2,KeyW:3"]"""
import argparse, asyncio, pathlib, subprocess, sys, time
from playwright.async_api import async_playwright
ROOT = pathlib.Path(__file__).resolve().parent.parent
GPU = ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu", "--autoplay-policy=no-user-gesture-required"]

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("query"); ap.add_argument("times", nargs="*", type=float)
    ap.add_argument("--w", type=int, default=1600); ap.add_argument("--h", type=int, default=900)
    ap.add_argument("--js", default=""); ap.add_argument("--port", type=int, default=8793); ap.add_argument("--keys", default="")
    ap.add_argument("--click", default="")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(a.port), "--bind", "127.0.0.1"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with async_playwright() as p:
            b = await p.chromium.launch(channel="chrome", headless=True, args=GPU)
            pg = await b.new_page(viewport={"width": a.w, "height": a.h})
            logs = []
            pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
            pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
            t0 = time.time()
            await pg.goto(f"http://127.0.0.1:{a.port}/index.html?{a.query}")
            try:
                await pg.wait_for_function("window.game !== undefined", timeout=90000)
            except Exception:
                print("\n".join(logs[-30:])); err = await pg.evaluate("document.getElementById('err').textContent"); print("ERR:", err); await pg.screenshot(path=str(out / "fail.jpg")); raise
            print(f"loaded in {time.time()-t0:.1f}s", flush=True)
            if a.click: await pg.click(a.click)
            start = time.time()
            holds = [(k.split(':')[0], float(k.split(':')[1]), float(k.split(':')[2]) if k.count(':') > 1 else 0) for k in a.keys.split(',') if k]
            async def keyrun(k, dur, at):
                await asyncio.sleep(at); await pg.keyboard.down(k); await asyncio.sleep(dur); await pg.keyboard.up(k)
            tasks = [asyncio.create_task(keyrun(*h)) for h in holds]
            for t in a.times:
                await asyncio.sleep(max(0, t - (time.time() - start)))
                if a.js:
                    await pg.evaluate(a.js); await asyncio.sleep(0.5)
                st = await pg.evaluate("({s: game.state, cp: game.cp, t: +game.total.toFixed(1), fps: Math.round(game.fps||0), agl: +game.bike.agl.toFixed(1), v: Math.round(Math.hypot(...game.bike.v)*3.6), calls: dbg.renderer.info.render.calls, tris: dbg.renderer.info.render.triangles})")
                print(f"{t:5.1f}s {st}", flush=True)
                await pg.screenshot(path=str(out / f"s_{int(t*10):04d}.jpg"), type="jpeg", quality=88)
            for l in logs:
                if "rror" in l or "arn" in l: print(l[:400])
            err = await pg.evaluate("document.getElementById('err').textContent")
            if err: print("ERR:", err[:1500])
            await b.close()
    finally:
        srv.terminate()
asyncio.run(main())
