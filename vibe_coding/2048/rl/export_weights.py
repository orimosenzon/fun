"""
ייצוא משקולות של סוכן מאומן לקובץ JS, כדי שהסוכן יוכל לשחק בתוך ממשק הווב (game/js/agent.js).

הפורמט: קובץ JS שמגדיר window.AGENT_WEIGHTS[<name>] = {kind, filters, hidden, tensors: {name: {shape, dtype, data}}}
כאשר data הוא base64 של float16 (חצי מהגודל של float32; דיוק מספיק לבחירת פעולה).

רשת n-tuple (הגרסה הקטנה, 5 רביעיות): {kind: "ntuple", tuples, table: {dtype: "float32", data}}, עם
הערכה של 1,000 משחקים שמחושבת כאן (numba, שניות ספורות). הטבלה של 4x6 (268MB) לא מיוצאת: גדולה מדי לדפדפן.

הרצה:
    python rl/export_weights.py --agent dqn --checkpoint checkpoints/dqn_best.pt
    python rl/export_weights.py --agent ac  --checkpoint checkpoints/a2c_best.pt
    python rl/export_weights.py --agent ppo --checkpoint checkpoints/ppo_best.pt
    python rl/export_weights.py --agent ntuple --checkpoint checkpoints/ntuple_small_best.npz
    python rl/export_weights.py --agent asnet --checkpoint checkpoints/asnet_best.pt
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_BROWSER_MB = 8  # מעבר לזה הקובץ כבד מדי להורדה בדף המשחק


def export_torch(agent: str, checkpoint: str) -> tuple[dict, int]:
    import torch

    ckpt = torch.load(os.path.join(ROOT, checkpoint), map_location="cpu")
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
        "kind": agent,
        "filters": a["filters"],
        "hidden": a["hidden"],
        "checkpoint": checkpoint,
        "eval": ckpt.get("eval"),
        "transitions": ckpt.get("transitions"),
        "reward_scale": a.get("reward_scale", 1e-3),
        "tensors": tensors,
    }
    return payload, total


def export_ntuple(checkpoint: str, eval_games: int) -> tuple[dict, int]:
    from common import summarize
    from ntuple_td import greedy_eval, load_checkpoint, net_from_meta

    table, meta = load_checkpoint(os.path.join(ROOT, checkpoint))
    net = net_from_meta(meta)
    if table.nbytes / 1e6 > MAX_BROWSER_MB:
        raise SystemExit(f"the table is {table.nbytes / 1e6:.0f}MB; only the small network fits in the browser "
                         f"(train with --net small)")
    # הערכה עצמאית של 1,000 משחקים (הערכת האימון היא על 100 בלבד), כדי שהפאנל בדפדפן יציג מספר אמין
    summ = summarize(greedy_eval(table.astype(np.float32), eval_games, seed=2048, net=net))
    print(f"eval on {eval_games} games: mean {summ['score_mean']:,.0f}, median {summ['score_median']:,.0f}, "
          f"2048 in {100 * summ['reach_2048']:.1f}%, 4096 in {100 * summ['reach_4096']:.1f}%")
    payload = {
        "kind": "ntuple",
        "net": meta.get("net", "small"),
        "tuples": net.tuples.tolist(),
        "n_values": 16,
        "checkpoint": checkpoint,
        "eval": summ,
        "transitions": meta.get("transitions"),
        "episodes": meta.get("episodes"),
        "train_minutes": (meta.get("args") or {}).get("time_limit_min"),
        "table": {"dtype": "float32", "size": int(table.size),
                  "data": base64.b64encode(table.astype(np.float32).tobytes()).decode("ascii")},
    }
    return payload, int(table.size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["dqn", "ac", "ppo", "ntuple", "asnet"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-games", type=int, default=1000, help="ל-ntuple: כמה משחקים להערכה שנשמרת בקובץ")
    args = p.parse_args()

    if args.agent == "ntuple":
        payload, total = export_ntuple(args.checkpoint, args.eval_games)
    else:
        payload, total = export_torch(args.agent, args.checkpoint)
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
