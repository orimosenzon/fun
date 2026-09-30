#!/usr/bin/env python3
"""בניית אינדקס החיפוש מתמלולי ישיבות המליאה.

חותך כל ישיבה לקטעים של עד ~90 שניות (עם חפיפה קטנה), כל קטע עם שמות הדוברים
שבו, ומחשב לכל קטע וקטור משמעות עם BAAI/bge-m3 (רב-לשוני, חזק בעברית, רץ מקומית).

    .venv/bin/python rag/build_index.py   ->   private/index/{chunks.json, emb.npy}
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from export_notebooklm import label  # noqa: E402  אותו ניסוח של שם דובר בכל מקום

PRIV = os.path.join(HERE, "..", "private")
OUT = os.path.join(PRIV, "index")
EMBED_MODEL = "BAAI/bge-m3"
MAX_SECONDS = 90
OVERLAP = 1  # כמה פסקאות מסוף קטע חוזרות בתחילת הבא, כדי שלא ייחתך רעיון באמצע


def meeting_chunks(path):
    d = json.load(open(path, encoding="utf-8"))
    m = d["meeting"]
    sp = os.path.join(PRIV, "speakers", os.path.basename(path))
    names = json.load(open(sp, encoding="utf-8"))["speakers"] if os.path.exists(sp) else {}
    paras = []
    for s in d["segments"]:
        if paras and paras[-1]["speaker"] == s["speaker"] and s["start"] - paras[-1]["start"] < 45:
            paras[-1]["text"] += " " + s["text"]
            paras[-1]["end"] = s["end"]
        else:
            paras.append(dict(s))
    chunks, cur = [], []
    for p in paras:
        if cur and p["end"] - cur[0]["start"] > MAX_SECONDS:
            chunks.append(cur)
            cur = cur[-OVERLAP:] if OVERLAP else []
        cur.append(p)
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:
        text = "\n".join(f"{label(p['speaker'], names)}: {p['text']}" for p in c)
        speakers = sorted({label(p["speaker"], names) for p in c})
        out.append({"date": m["date"], "video": m["id"], "start": int(c[0]["start"]),
                    "end": int(c[-1]["end"]), "speakers": speakers, "text": text})
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    chunks = []
    for path in sorted(glob.glob(os.path.join(PRIV, "transcripts_spk", "*.json"))):
        chunks += meeting_chunks(path)
    print(f"{len(chunks)} chunks from {len({c['video'] for c in chunks})} meetings", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device="cuda")
    model.max_seq_length = 1024
    emb = model.encode([c["text"] for c in chunks], batch_size=8, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    np.save(os.path.join(OUT, "emb.npy"), emb.astype(np.float16))
    json.dump({"model": EMBED_MODEL, "chunks": chunks},
              open(os.path.join(OUT, "chunks.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", emb.shape)


if __name__ == "__main__":
    main()
