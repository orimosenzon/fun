#!/usr/bin/env python3
"""זיהוי דוברים בישיבות המליאה (pyannote), והצמדה שלהם לתמלול.

רץ בסביבה של הפרויקט, כי ה-torch הגלובלי (cu128) לא תומך ב-GTX 1060:

    .venv/bin/python tools/diarize_meetings.py            # כל הישיבות שיש להן תמלול
    .venv/bin/python tools/diarize_meetings.py 8ow4RDfJRoM

לכל תמלול ב-private/transcripts נוצר private/diarization/<אותו שם>.json עם תורות
הדיבור, ו-private/transcripts_spk/<אותו שם>.json: התמלול מחולק מחדש לפי דובר, כך
שכל קטע הוא רצף של דובר אחד ("SPEAKER_03"). שמות אמיתיים מוצמדים בשלב נפרד.
"""
import glob
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.path.join(ROOT, "private")
LOG = os.path.join(PRIV, "diarize.log")
SR = 16000


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())[None, :]


def speaker_at(turns, t0, t1):
    """הדובר עם החפיפה הגדולה ביותר לטווח [t0, t1]."""
    best, best_ov = None, 0.0
    for s, e, spk in turns:
        if s > t1:
            break
        ov = min(e, t1) - max(s, t0)
        if ov > best_ov:
            best, best_ov = spk, ov
    return best


def merge(transcript, turns):
    """מחלק את התמלול מחדש לקטעים לפי דובר, על בסיס חותמות הזמן של המילים."""
    turns = sorted(turns)
    out, cur = [], None
    for seg in transcript["segments"]:
        for ws, we, word in seg["words"] or [[seg["start"], seg["end"], seg["text"]]]:
            spk = speaker_at(turns, ws, we) or (cur["speaker"] if cur else None)
            if cur and spk == cur["speaker"] and ws - cur["end"] < 3.0:
                cur["text"] += word
                cur["end"] = we
            else:
                cur = {"speaker": spk, "start": ws, "end": we, "text": word}
                out.append(cur)
    for c in out:
        c["text"] = c["text"].strip()
    return out


MAX_PART = 4 * 3600 * SR  # ישיבה של 5 שעות בבת אחת נהרגה על זיכרון ב-28/9


def diarize_once(pipe, wav, offset=0.0):
    res = pipe({"waveform": wav, "sample_rate": SR})
    ann = getattr(res, "exclusive_speaker_diarization", res)
    turns = [[round(s.start + offset, 2), round(s.end + offset, 2), spk]
             for s, _, spk in ann.itertracks(yield_label=True)]
    labels = res.speaker_diarization.labels() if hasattr(res, "speaker_diarization") else []
    emb = getattr(res, "speaker_embeddings", None)
    cents = {lab: emb[i] for i, lab in enumerate(labels)} if emb is not None else {}
    return turns, cents


def diarize(pipe, wav):
    """ישיבה ארוכה מתחלקת לחלקים ברגע שקט, והדוברים מאוחדים לפי דמיון של טביעות הקול."""
    n = wav.shape[1]
    if n <= MAX_PART:
        return diarize_once(pipe, wav)[0]
    parts = -(-n // MAX_PART)
    cuts = [0]
    for k in range(1, parts):
        mid = k * n // parts
        win = wav[0, mid - 30 * SR: mid + 30 * SR]
        energy = win[: len(win) // 800 * 800].reshape(-1, 800).pow(2).mean(1)
        cuts.append(mid - 30 * SR + int(energy.argmin()) * 800)
    cuts.append(n)
    all_turns, known = [], {}  # known: תווית אחידה -> טביעת קול
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        turns, cents = diarize_once(pipe, wav[:, a:b], offset=a / SR)
        torch.cuda.empty_cache()
        mapping = {}
        for lab, c in cents.items():
            best, best_sim = None, 0.6  # סף דמיון קוסינוס לזיהוי אותו דובר
            for g, gc in known.items():
                if g in mapping.values():
                    continue
                sim = float(np.dot(c, gc) / (np.linalg.norm(c) * np.linalg.norm(gc) + 1e-9))
                if sim > best_sim:
                    best, best_sim = g, sim
            if best is None:
                best = f"SPEAKER_{len(known):02d}"
                known[best] = c
            mapping[lab] = best
        all_turns += [[s, e, mapping.get(spk, f"P{i}_{spk}")] for s, e, spk in turns]
    return all_turns


def main():
    from pyannote.audio import Pipeline
    only = set(sys.argv[1:])
    os.makedirs(os.path.join(PRIV, "diarization"), exist_ok=True)
    os.makedirs(os.path.join(PRIV, "transcripts_spk"), exist_ok=True)
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=True)
    pipe.to(torch.device("cuda"))
    log("pipeline loaded")

    for tpath in sorted(glob.glob(os.path.join(PRIV, "transcripts", "*.json")), reverse=True):
        name = os.path.basename(tpath)
        vid = name.split("_", 1)[1][:-5]
        if only and vid not in only:
            continue
        dpath = os.path.join(PRIV, "diarization", name)
        spath = os.path.join(PRIV, "transcripts_spk", name)
        if os.path.exists(spath):
            continue
        try:
            t0 = time.time()
            if os.path.exists(dpath):
                turns = json.load(open(dpath))["turns"]
            else:
                wav = load_audio(os.path.join(PRIV, "audio", f"{vid}.m4a"))
                turns = diarize(pipe, wav)
                del wav
                json.dump({"turns": turns}, open(dpath, "w"))
            tr = json.load(open(tpath, encoding="utf-8"))
            segs = merge(tr, turns)
            json.dump({"meeting": tr["meeting"], "duration": tr["duration"], "segments": segs},
                      open(spath, "w", encoding="utf-8"), ensure_ascii=False)
            n = len({t[2] for t in turns})
            log(f"done {name}: {n} speakers, {len(segs)} turns, {(time.time()-t0)/60:.1f} min "
                f"for {tr['duration']/60:.0f} min audio")
        except Exception as e:
            log(f"FAIL {name}: {e}\n{traceback.format_exc()}")
            torch.cuda.empty_cache()
    log("all done")


if __name__ == "__main__":
    main()
