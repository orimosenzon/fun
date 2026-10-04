"""Render the film frame by frame (resumable), then encode.

usage:
  python3 tools/render.py frames [--w 1920 --h 1080 --sub 8 --start 0 --end N]   # write out/frames/*.jpg
  python3 tools/render.py audio-data                                           # write out/audio_data.json
  python3 tools/render.py encode [--crf 17]                                    # frames + out/audio.wav -> out/jetbike.mp4
"""
import argparse, asyncio, base64, json, pathlib, subprocess, sys, time
from playwright.async_api import async_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
GPU_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"]


def serve(port):
    return subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def open_page(p, a):
    b = await p.chromium.launch(channel="chrome", headless=True, args=GPU_ARGS)
    pg = await b.new_page(viewport={"width": a.w, "height": a.h})
    logs = []
    pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
    pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
    await pg.goto(f"http://127.0.0.1:{a.port}/index.html?render&w={a.w}&h={a.h}&sub={a.sub}&fps={a.fps}")
    try:
        await pg.wait_for_function("window.filmReady === true", timeout=180000)
    except Exception:
        print("\n".join(logs))
        raise
    return b, pg


async def frames(a):
    fdir = OUT / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        b, pg = await open_page(p, a)
        cdp = await pg.context.new_cdp_session(pg)
        n = await pg.evaluate("film.frames")
        end = min(n, a.end) if a.end else n
        print(f"{n} frames total, rendering {a.start}..{end - 1}", flush=True)
        t0 = time.time()
        # particles need continuity: always warm up from a few seconds before the start
        done = 0
        for i in range(a.start, end):
            f = fdir / f"{i:05d}.jpg"
            if f.exists() and not a.force:
                continue
            await pg.evaluate(f"film.renderFrame({i})")
            shot = await cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 95, "optimizeForSpeed": True})
            f.write_bytes(base64.b64decode(shot["data"]))
            done += 1
            if done % 25 == 0:
                el = time.time() - t0
                print(f"frame {i}  {el / done:.2f}s/frame  eta {(end - i) * el / done / 60:.1f} min", flush=True)
        await b.close()


async def audio_data(a):
    async with async_playwright() as p:
        a.sub = 1
        b, pg = await open_page(p, a)
        data = await pg.evaluate("film.audioData(120)")
        (OUT / "audio_data.json").write_text(json.dumps(data))
        print("wrote audio_data.json", len(data["eng"]), "samples")
        await b.close()


def encode(a):
    wav = OUT / "audio.wav"
    cmd = ["ffmpeg", "-y", "-framerate", str(a.fps), "-i", str(OUT / "frames" / "%05d.jpg")]
    if wav.exists():
        cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "256k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(OUT / a.name)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frames", "audio-data", "encode"])
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--sub", type=int, default=8)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--name", default="jetbike.mp4")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.cmd == "encode":
        return encode(a)
    srv = serve(a.port)
    try:
        asyncio.run(frames(a) if a.cmd == "frames" else audio_data(a))
    finally:
        srv.terminate()


main()
