"""Headless Chrome on the GPU, with screenshots.
   python3 tools/shot.py OUT 'query#hash' STEP [STEP ...] [--w 1600 --h 900 --port 8941]
   A STEP is  name:seconds  or  name:seconds:javascript  (the script runs first, then the wait, then the picture).
   Keys can be held with --keys "ArrowDown:2:1,KeyW:3" (code:duration:start)."""
import argparse, asyncio, pathlib, subprocess, sys, time
from playwright.async_api import async_playwright
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent      # vibe_coding: the page imports ../jetbikes/foehn
GPU = ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu", "--autoplay-policy=no-user-gesture-required"]

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("query"); ap.add_argument("steps", nargs="*")
    ap.add_argument("--w", type=int, default=1600); ap.add_argument("--h", type=int, default=900)
    ap.add_argument("--port", type=int, default=8941); ap.add_argument("--keys", default=""); ap.add_argument("--page", default="index.html")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(a.port), "--bind", "127.0.0.1"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        await asyncio.sleep(0.6)
        async with async_playwright() as p:
            b = await p.chromium.launch(channel="chrome", headless=True, args=GPU)
            pg = await b.new_page(viewport={"width": a.w, "height": a.h})
            logs = []
            pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
            pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
            q = a.query if a.query.startswith(("?", "#")) or not a.query else "?" + a.query
            await pg.goto(f"http://127.0.0.1:{a.port}/mandelscape/{a.page}{q}")
            try:
                await pg.wait_for_function("window.dbg !== undefined", timeout=30000)
            except Exception:
                print("\n".join(logs[-30:])); print("ERR:", await pg.evaluate("document.getElementById('err').textContent")); await pg.screenshot(path=str(out / "fail.jpg")); raise
            holds = [(k.split(':')[0], float(k.split(':')[1]), float(k.split(':')[2]) if k.count(':') > 1 else 0) for k in a.keys.split(',') if k]
            async def keyrun(k, dur, at):
                await asyncio.sleep(at); await pg.keyboard.down(k); await asyncio.sleep(dur); await pg.keyboard.up(k)
            tasks = [asyncio.create_task(keyrun(*h)) for h in holds]
            for st in a.steps:
                name, wait, *js = st.split(":", 2)
                if js and js[0].strip():
                    r = await pg.evaluate(js[0])
                    if r is not None: print(f"  {name} js -> {r}", flush=True)
                await asyncio.sleep(float(wait or 0))
                if not float(wait or 0): continue
                await pg.screenshot(path=str(out / f"{name}.jpg"), type="jpeg", quality=90)
                print(f"shot {name}", flush=True)
            for l in logs:
                if "rror" in l or "arn" in l or "[log]" in l: print(l[:600])
            err = await pg.evaluate("document.getElementById('err') ? document.getElementById('err').textContent : ''")
            if err: print("ERR:", err[:2500])
            await b.close()
    finally:
        srv.terminate()
asyncio.run(main())
