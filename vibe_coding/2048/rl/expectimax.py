"""
ניסוי 1: expectimax מעל טבלת ה-n-tuple המאומנת, בלי אימון נוסף.

הסוכן של הדו"ח מסתכל מהלך אחד קדימה: argmax על r + V(afterstate). כאן מוסיפים חיפוש:
  עומק 1  = הסוכן החמדן (כמו בדו"ח)
  עומק 2  = לכל afterstate מחשבים תוחלת על האריח האקראי (עד 15 משבצות × {2, 4}), ובכל
            לוח שנוצר בוחרים שוב את המהלך הטוב ביותר לפי r + V(afterstate)
  עומק 3  = עוד שכבת אריח ומהלך

זה שינוי בהערכה בלבד: הטבלה לא משתנה, רק הדרך שבה משתמשים בה בזמן המשחק. בספרות (יה ואחרים
2016, יאשקובסקי 2016) חיפוש כזה בעומק 3–5 מכפיל בערך את הניקוד הממוצע.

עומק k עולה פי ~100 מעומק k-1 (30 המשכים × 4 מהלכים), ולכן ההערכה רצה במקביל על כל הליבות
(numba prange), ועומק 3 על פחות משחקים.

הרצה:
    python rl/expectimax.py --depths 1 2 3 --games 1000 1000 100
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from numba import njit, prange

from common import record_game, summarize
from ntuple_td import _features, _max_tile, _reset, _slide, _spawn, _value, load_checkpoint, net_from_meta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "reports", "data")


@njit(cache=True)
def _after_value(table, feats, offsets, after, depth, scratch, idx):
    """ערך של afterstate עם עוד depth שכבות של (אריח אקראי, מהלך). depth=0: קריאה מהטבלה.

    scratch (levels, 2, 16): לכל רמת רקורסיה לוח לאריח ולוח להחלקה; idx (levels, n_feats): אינדקסים לעלים.
    """
    if depth == 0:
        _features(after, feats, offsets, idx[0])
        return _value(table, idx[0])
    board = scratch[depth, 0]
    out = scratch[depth, 1]
    total = 0.0
    n_empty = 0
    for i in range(16):
        if after[i] != 0:
            continue
        n_empty += 1
        for k in range(2):
            val = 1 if k == 0 else 2          # אריח 2 בהסתברות 0.9, אריח 4 בהסתברות 0.1
            p = 0.9 if k == 0 else 0.1
            for j in range(16):
                board[j] = after[j]
            board[i] = val
            best = -1e30
            for a in range(4):
                sc, changed = _slide(board, a, out)
                if not changed:
                    continue
                v = sc + _after_value(table, feats, offsets, out, depth - 1, scratch, idx)
                if v > best:
                    best = v
            if best < -1e29:              # אין מהלך חוקי: הלוח סופי, ערך 0
                best = 0.0
            total += p * best
    return total / n_empty


@njit(cache=True)
def _best_move(table, feats, offsets, board, depth, scratch, idx, after):
    """המהלך הטוב ביותר בלוח לפי חיפוש לעומק depth מהלכים (1 = חמדן). מחזיר (פעולה, ניקוד מיידי)."""
    best_a = -1
    best_v = -1e30
    best_r = 0
    for a in range(4):
        sc, changed = _slide(board, a, after[a])
        if not changed:
            continue
        v = sc + _after_value(table, feats, offsets, after[a], depth - 1, scratch, idx)
        if v > best_v:
            best_v = v
            best_a = a
            best_r = sc
    return best_a, best_r


@njit(parallel=True, cache=True)
def play_expectimax(table, feats, offsets, depth, seeds, ep_scores, ep_tiles, ep_moves):
    """משחקים מלאים במקביל (משחק לכל איטרציה של prange), בלי למידה."""
    n_games = seeds.shape[0]
    n_feats = feats.shape[0]
    for g in prange(n_games):
        np.random.seed(seeds[g])
        board = np.zeros(16, dtype=np.uint8)
        after = np.zeros((4, 16), dtype=np.uint8)
        scratch = np.zeros((depth + 1, 2, 16), dtype=np.uint8)
        idx = np.zeros((depth + 1, n_feats), dtype=np.int64)
        _reset(board)
        score = 0
        moves = 0
        while True:
            a, r = _best_move(table, feats, offsets, board, depth, scratch, idx, after)
            if a < 0:
                break
            for i in range(16):
                board[i] = after[a, i]
            score += r
            moves += 1
            _spawn(board)
        ep_scores[g] = score
        ep_tiles[g] = _max_tile(board)
        ep_moves[g] = moves


def make_expectimax_policy(table, net, depth: int):
    """מדיניות בממשק של evaluate.py (לוחות (N,4,4) -> פעולות), לוח אחרי לוח; נועדה להקלטת משחק בודד."""
    scratch = np.zeros((depth + 1, 2, 16), dtype=np.uint8)
    idx = np.zeros((depth + 1, net.n_feats), dtype=np.int64)
    after = np.zeros((4, 16), dtype=np.uint8)

    def policy(boards, valid):
        acts = np.zeros(boards.shape[0], dtype=np.int64)
        for i in range(boards.shape[0]):
            a, _ = _best_move(table, net.feats, net.offsets, np.ascontiguousarray(boards[i].reshape(16)), depth, scratch, idx, after)
            acts[i] = max(a, 0)
        return acts

    return policy


def evaluate_depth(table, net, depth: int, games: int, seed: int, record: bool) -> dict:
    seeds = (seed + np.arange(games)).astype(np.int64)
    sc = np.zeros(games, dtype=np.int64)
    ti = np.zeros(games, dtype=np.int64)
    mv = np.zeros(games, dtype=np.int64)
    t0 = time.time()
    play_expectimax(table, net.feats, net.offsets, depth, seeds, sc, ti, mv)
    secs = time.time() - t0
    results = [{"score": int(s), "max_tile": int(t), "moves": int(m)} for s, t, m in zip(sc, ti, mv)]
    summ = summarize(results)
    out = {
        "depth": depth, "games": games, "seconds": secs, "moves_per_sec": float(mv.sum() / max(secs, 1e-9)),
        "summary": summ, "scores": sc.tolist(), "max_tiles": ti.tolist(), "moves": mv.tolist(),
    }
    print(f"depth {depth}: {games} games in {secs / 60:.1f} min ({out['moves_per_sec']:,.0f} moves/s) | mean {summ['score_mean']:,.0f} "
          f"median {summ['score_median']:,.0f} max {summ['score_max']:,.0f} | 2048 {100 * summ['reach_2048']:.1f}% 4096 {100 * summ['reach_4096']:.1f}% "
          f"8192 {100 * (ti >= 8192).mean():.1f}% 16384 {100 * (ti >= 16384).mean():.1f}%", flush=True)
    if record:
        # משחק מוקלט לצפייה בדו"ח (זרע קבוע, לא בהכרח הטוב ביותר)
        g = record_game(make_expectimax_policy(table, net, depth), seed=10_000)
        out["recorded_game"] = g
        print(f"    recorded game: {g['score']:,} points, {g['moves']:,} moves, tile {g['max_tile']:,}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/ntuple_best.npz")
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--games", type=int, nargs="+", default=[1000, 1000, 100], help="מספר משחקים לכל עומק")
    p.add_argument("--seed", type=int, default=2048)
    p.add_argument("--record", type=int, default=2, help="להקליט משחק לצפייה בעומק הזה")
    p.add_argument("--out", default="eval_ntuple_expectimax.json")
    args = p.parse_args()

    table, meta = load_checkpoint(os.path.join(ROOT, args.checkpoint))
    net = net_from_meta(meta)
    table = np.ascontiguousarray(table, dtype=np.float32)
    print(f"table: {table.size:,} weights, {net.n_feats} features; depths {args.depths}", flush=True)

    # קומפילציה על משחק אחד קצר, כדי שהזמן הנמדד יהיה של המשחקים בלבד
    play_expectimax(table, net.feats, net.offsets, 1, np.array([0], dtype=np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64))

    path = os.path.join(DATA_DIR, args.out)
    data = {"checkpoint": args.checkpoint, "seed": args.seed, "depths": {}}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    for depth, games in zip(args.depths, args.games):
        data["depths"][str(depth)] = evaluate_depth(table, net, depth, games, args.seed, record=(depth == args.record))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print("  ->", os.path.relpath(path, ROOT), flush=True)


if __name__ == "__main__":
    main()
