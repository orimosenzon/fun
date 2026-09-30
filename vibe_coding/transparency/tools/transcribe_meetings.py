#!/usr/bin/env python3
"""תמלול של ישיבות מליאת המועצה מיוטיוב, עם המודל העברי של ivrit.ai.

עובר על data/meetings.json מהחדשה לישנה. לכל ישיבה: מוריד אודיו ל-private/audio,
מתמלל ל-private/transcripts/<תאריך>_<מזהה>.json (קטעים + חותמות זמן למילים, כדי
להצמיד דוברים בשלב הבא). ישיבה שכבר תומללה מדולגת, כך שאפשר לעצור ולהמשיך.
הכל מחוץ ל-git, כי בתמלולים יש שמות של דוברים.

    nohup python3 tools/transcribe_meetings.py &
    tail -f private/transcribe.log
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AUDIO = os.path.join(ROOT, "private", "audio")
OUT = os.path.join(ROOT, "private", "transcripts")
LOG = os.path.join(ROOT, "private", "transcribe.log")
MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def download(vid):
    path = os.path.join(AUDIO, f"{vid}.m4a")
    cmd = ["yt-dlp", "-q", "--no-progress", "-f", "bestaudio", "-x", "--audio-format", "m4a",
           "-o", os.path.join(AUDIO, "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={vid}"]
    node = shutil.which("node")
    if node:
        cmd[1:1] = ["--js-runtimes", f"node:{node}"]
    for attempt in range(3):  # בלילה של 27/9 היו כישלונות חולפים של יוטיוב
        if os.path.exists(path):
            break
        if subprocess.run(cmd).returncode != 0 and attempt < 2:
            time.sleep(60)
    if not os.path.exists(path):
        raise RuntimeError(f"yt-dlp failed 3 times for {vid}")
    return path


SR = 16000
CHUNK = 20 * 60  # ישיבה של 3.5 שעות בבת אחת הפילה את התהליך על חוסר זיכרון (27/9)


def cut_points(audio):
    """נקודות חיתוך בערך כל 20 דקות, כל אחת ברגע השקט ביותר בטווח של חצי דקה סביבה."""
    import numpy as np
    points, total = [0], len(audio)
    target = CHUNK * SR
    while total - points[-1] > target * 1.25:
        mid = points[-1] + target
        lo, hi = mid - 30 * SR, mid + 30 * SR
        frames = audio[lo:hi][: (hi - lo) // 800 * 800].reshape(-1, 800)  # חלונות של 50ms
        quietest = int(np.argmin((frames ** 2).mean(axis=1)))
        points.append(lo + quietest * 800 + 400)
    return points + [total]


def transcribe_chunked(model, path):
    from faster_whisper import decode_audio
    audio = decode_audio(path, sampling_rate=SR)
    pts = cut_points(audio)
    segments = []
    for a, b in zip(pts, pts[1:]):
        off = a / SR
        segs, _ = model.transcribe(audio[a:b], language="he", beam_size=5,
                                   vad_filter=True, word_timestamps=True)
        for s in segs:
            segments.append({
                "start": round(s.start + off, 2), "end": round(s.end + off, 2),
                "text": s.text.strip(),
                "words": [[round(w.start + off, 2), round(w.end + off, 2), w.word]
                          for w in (s.words or [])],
            })
    return segments, len(audio) / SR


def main():
    os.makedirs(AUDIO, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    meetings = json.load(open(os.path.join(ROOT, "data", "meetings.json"), encoding="utf-8"))
    only = set(sys.argv[1:])  # אפשר להעביר מזהים כדי לתמלל רק אותם

    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL, device="cuda", compute_type="int8")
    log(f"model loaded: {MODEL}; {len(meetings)} meetings")

    for m in meetings:
        if only and m["id"] not in only:
            continue
        out = os.path.join(OUT, f"{m['date']}_{m['id']}.json")
        if os.path.exists(out):
            continue
        try:
            t0 = time.time()
            audio = download(m["id"])
            segments, duration = transcribe_chunked(model, audio)
            el = time.time() - t0
            json.dump({"meeting": m, "model": MODEL, "duration": duration,
                       "segments": segments}, open(out + ".tmp", "w", encoding="utf-8"),
                      ensure_ascii=False)
            os.replace(out + ".tmp", out)
            log(f"done {m['date']} {m['id']}: {duration/60:.0f} min audio in "
                f"{el/60:.1f} min (x{duration/el:.1f}), {len(segments)} segments")
        except Exception as e:
            log(f"FAIL {m['date']} {m['id']}: {e}\n{traceback.format_exc()}")
    log("all done")


if __name__ == "__main__":
    main()
