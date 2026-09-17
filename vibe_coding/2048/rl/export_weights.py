"""
ייצוא משקולות של סוכן מאומן לקובץ JS, כדי שהסוכן יוכל לשחק בתוך ממשק הווב (game/js/agent.js).

הפורמט: קובץ JS שמגדיר window.AGENT_WEIGHTS[<name>] = {kind, filters, hidden, tensors: {name: {shape, dtype, data}}}
כאשר data הוא base64 של float16 (חצי מהגודל של float32; דיוק מספיק לבחירת פעולה).

הרצה:
    python rl/export_weights.py --agent dqn --checkpoint checkpoints/dqn_best.pt
    python rl/export_weights.py --agent ac  --checkpoint checkpoints/a2c_best.pt
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["dqn", "ac"], required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    ckpt = torch.load(os.path.join(ROOT, args.checkpoint), map_location="cpu")
    a = ckpt["args"]
    tensors = {}
    total = 0
    for name, t in ckpt["model"].items():
        arr = t.detach().cpu().numpy().astype(np.float16)
        total += arr.size
        tensors[name] = {
            "shape": list(arr.shape),
            "dtype": "float16",
            "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        }
    payload = {
        "kind": args.agent,
        "filters": a["filters"],
        "hidden": a["hidden"],
        "checkpoint": args.checkpoint,
        "eval": ckpt.get("eval"),
        "transitions": ckpt.get("transitions"),
        "tensors": tensors,
    }
    out_dir = os.path.join(ROOT, "game", "weights")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.agent}.js")
    with open(out, "w", encoding="utf-8") as f:
        f.write("window.AGENT_WEIGHTS = window.AGENT_WEIGHTS || {};\n")
        f.write(f"window.AGENT_WEIGHTS[{json.dumps(args.agent)}] = ")
        json.dump(payload, f)
        f.write(";\n")
    print(f"wrote {os.path.relpath(out, ROOT)}: {total:,} parameters, {os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
