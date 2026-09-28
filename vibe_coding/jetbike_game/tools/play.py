"""Headless play-test: run the game (demo autopilot by default) on the GPU, sample state,
take screenshots.   python3 tools/play.py OUT_DIR [--secs 90] [--every 10] [--q ...]"""
import argparse, asyncio, pathlib, subprocess, sys, time
from playwright.async_api import async_playwright
ROOT = pathlib.Path(__file__).resolve().parent.parent
GPU = ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu", "--autoplay-policy=no-user-gesture-required"]

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--secs", type=float, default=90); ap.add_argument("--every", type=float, default=10)
    ap.add_argument("--w", type=int, default=1600); ap.add_argument("--h", type=int, default=900); ap.add_argument("--q", default="demo")
    ap.add_argument("--port", type=int, default=8791)
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
            await pg.goto(f"http://127.0.0.1:{a.port}/index.html?{a.q}")
            try:
                await pg.wait_for_function("window.game !== undefined", timeout=120000)
            except Exception:
                print("\n".join(logs)); raise
            print(f"loaded in {time.time()-t0:.1f}s", flush=True)
            start = time.time(); nxt = 0
            while time.time() - start < a.secs:
                el = time.time() - start
                if el >= nxt:
                    st = await pg.evaluate("({s: game.state, r: game.ringIndex, t: game.raceT.toFixed(1), fps: game.fps, p: game.bike.p.map(v=>v.toFixed(0)).join(','), v: (Math.hypot(...game.bike.v)*3.6).toFixed(0)})")
                    print(f"{el:5.1f}s {st}", flush=True)
                    await pg.screenshot(path=str(out / f"g_{int(el):03d}.jpg"), type="jpeg", quality=85)
                    nxt += a.every
                await asyncio.sleep(0.5)
            for l in logs:
                if "rror" in l or "arn" in l or "crash" in l or "probe" in l: print(l)
            await b.close()
    finally:
        srv.terminate()
asyncio.run(main())
