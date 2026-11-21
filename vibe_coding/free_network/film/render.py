"""Turns the film page into a video file: photographs every frame, lays the narration under
them, and writes public/video/intro.mp4 with its poster.

  python3 film/render.py [--workers 4] [--fps 30]
Needs a static server on the project root (python3 -m http.server 8812) and the recordings and
stills that film/narrate.py and film/shoot.py make.
"""
import argparse, array, json, subprocess, sys, wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
PAGE = "http://127.0.0.1:8812/film/index.html?still"
RATE = 24000  # the narration is recorded at this rate, in one channel of 16 bits


def open_film(p):
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(PAGE)
    page.wait_for_function("window.film", timeout=120000)
    return browser, page


def shoot(job):
    """One worker photographs a run of frames straight into a piece of video."""
    from playwright.sync_api import sync_playwright
    index, first, last, fps = job
    piece = OUT / f"piece_{index}.mp4"
    encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg", "-i", "-",
                                "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-g", str(fps * 2), str(piece)], stdin=subprocess.PIPE)
    with sync_playwright() as p:
        browser, page = open_film(p)
        for frame in range(first, last):
            page.evaluate("t => film.seek(t)", frame / fps)
            encoder.stdin.write(page.screenshot(type="jpeg", quality=93))
        browser.close()
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError(f"the encoder failed on piece {index}")
    return piece


def sound_track(sound, duration):
    samples = array.array("h", bytes(2 * int(duration * RATE)))
    for part in sound:
        with wave.open(str(HERE / "audio" / f"{part['id']}.wav")) as recording:
            assert (recording.getframerate(), recording.getnchannels(), recording.getsampwidth()) == (RATE, 1, 2), part
            voice = array.array("h", recording.readframes(recording.getnframes()))
        start = int(part["at"] * RATE)
        samples[start:start + len(voice)] = voice
    path = OUT / "sound.wav"
    with wave.open(str(path), "wb") as track:
        track.setnchannels(1), track.setsampwidth(2), track.setframerate(RATE)
        track.writeframes(samples.tobytes())
    return path


def main():
    from playwright.sync_api import sync_playwright
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser, page = open_film(p)
        duration = page.evaluate("film.duration")
        sound = page.evaluate("film.sound")
        page.evaluate("t => film.seek(t)", sound[2]["at"] + 9)
        target = HERE.parent / "public" / "video"
        target.mkdir(exist_ok=True)
        page.screenshot(path=target / "intro.jpg", type="jpeg", quality=88)
        browser.close()
    frames = int(duration * args.fps)
    edges = [round(frames * i / args.workers) for i in range(args.workers + 1)]
    jobs = [(i, edges[i], edges[i + 1], args.fps) for i in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        pieces = list(pool.map(shoot, jobs))
    listing = OUT / "pieces.txt"
    listing.write_text("".join(f"file '{piece.name}'\n" for piece in pieces))
    track = sound_track(sound, duration)
    film = target / "intro.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(track),
                    "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,aformat=channel_layouts=mono", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(film)], check=True)
    print(f"{film}: {duration:.1f} seconds, {film.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
