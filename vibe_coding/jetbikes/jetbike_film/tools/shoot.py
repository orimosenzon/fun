"""Grab still frames from the film at given times (for checking looks).

usage: python3 tools/shoot.py OUT_DIR t1 t2 ... [--w 1280 --h 720 --sub 1]
"""
import argparse, asyncio, pathlib, subprocess, sys, time
from playwright.async_api import async_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
GPU_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"]


def serve(port):
    return subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("times", nargs="+", type=float)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    srv = serve(a.port)
    try:
        async with async_playwright() as p:
            b = await p.chromium.launch(channel="chrome", headless=True, args=GPU_ARGS)
            pg = await b.new_page(viewport={"width": a.w, "height": a.h})
            logs = []
            pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
            pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
            t0 = time.time()
            await pg.goto(f"http://127.0.0.1:{a.port}/index.html?render&w={a.w}&h={a.h}&sub={a.sub}")
            try:
                await pg.wait_for_function("window.filmReady === true", timeout=90000)
            except Exception:
                print("\n".join(logs)); raise
            print(f"ready in {time.time() - t0:.1f}s")
            fps = await pg.evaluate("film.fps")
            for t in a.times:
                i = int(round(t * fps))
                t1 = time.time()
                await pg.evaluate(f"film.renderFrame({i})")
                await pg.screenshot(path=str(out / f"f_{t:07.2f}.png"))
                print(f"t={t:.2f} frame {i} {time.time() - t1:.2f}s")
            for l in logs:
                if "error" in l.lower() or "warn" in l.lower():
                    print(l)
            await b.close()
    finally:
        srv.terminate()


asyncio.run(main())
