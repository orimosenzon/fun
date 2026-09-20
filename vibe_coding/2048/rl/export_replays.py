"""
ייצוא משחקים מוקלטים של רשת ה-n-tuple המלאה (4 שישיות, 268MB) לדפדפן, כדי שאפשר יהיה לראות אותה
משחקת בממשק האמיתי גם בלי הטבלה: הטבלה נשארת בפייתון, והדפדפן מקבל רק את המהלכים והאריחים.

כל מהלך הוא שני תווים: הפעולה (0..3) והאריח שנפל אחריה כספרה בבסיס 32 (אינדקס המשבצת, ועוד 16 אם
האריח היה 4). משחק של 10,000 מהלכים הוא 20KB. הדפדפן מריץ את חוקי המשחק בעצמו (game_manager.js
עם forcedTile), ולכן הניקוד והאנימציות הם של המשחק, וכל סטייה הייתה מתגלה מיד.

מקורות: שני המשחקים שכבר מוקלטים ב-reports/data/eval_ntuple.json (הטוב ביותר והטיפוסי), ובנוסף
הקלטות חדשות מנקודות ביקורת, כל אחת עם תווית שמוצגת בפאנל (--source נתיב:מספר_משחקים:תווית, אפשר כמה).

הרצה:
    python rl/export_replays.py --source "checkpoints/ntuple_best.npz:4:73 דקות" --source "checkpoints/ntuple_long_best.npz:4:4 שעות"
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from game2048 import move_boards

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIGITS = "0123456789abcdefghijklmnopqrstuv"


def encode_game(game: dict, label: str) -> dict:
    """מהקלטת לוחות (frames) למחרוזת של פעולות ואריחים."""
    frames = game["frames"]
    steps = []
    for prev, cur in zip(frames, frames[1:]):
        a = cur["action"]
        after, _, changed = move_boards(np.array(prev["board"], dtype=np.uint8)[None], np.array([a]))
        assert changed[0], "recorded move did not change the board"
        after = after[0].flatten()
        board = np.array(cur["board"], dtype=np.uint8).flatten()
        spawned = np.nonzero((after == 0) & (board != 0))[0]
        assert len(spawned) == 1, f"expected exactly one new tile, found {len(spawned)}"
        cell = int(spawned[0])
        exp = int(board[cell])
        assert exp in (1, 2)
        steps.append(f"{a}{DIGITS[cell + (16 if exp == 2 else 0)]}")
    return {
        "label": label,
        "score": game["score"],
        "max_tile": game["max_tile"],
        "moves": len(steps),
        "start": [int(v) for v in np.array(frames[0]["board"]).flatten()],
        "steps": "".join(steps),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="append", default=[], help="נתיב:מספר_משחקים:תווית (אפשר כמה פעמים)")
    p.add_argument("--seed", type=int, default=777)
    args = p.parse_args()

    games = []
    ev_path = os.path.join(ROOT, "reports", "data", "eval_ntuple.json")
    if os.path.exists(ev_path):
        with open(ev_path, encoding="utf-8") as f:
            ev = json.load(f)
        for key, label in (("best_game", "73 minutes, best of 40 recorded"), ("typical_game", "73 minutes, typical game")):
            if ev.get(key):
                games.append(encode_game(ev[key], label))
                print(f"{key}: {games[-1]['score']:,} points, {games[-1]['moves']:,} moves")

    for src in args.source:
        path, n, label = src.split(":", 2)
        from common import record_game
        from ntuple_td import load_checkpoint, make_policy, net_from_meta
        table, meta = load_checkpoint(os.path.join(ROOT, path))
        policy = make_policy(table, net_from_meta(meta))
        for i in range(int(n)):
            g = record_game(policy, seed=args.seed + i)
            games.append(encode_game(g, label))
            print(f"{label} seed {args.seed + i}: {g['score']:,} points, {g['moves']:,} moves, tile {g['max_tile']:,}")
        del table, policy

    games.sort(key=lambda g: -g["score"])
    out = os.path.join(ROOT, "game", "weights", "ntuple_replays.js")
    with open(out, "w", encoding="utf-8") as f:
        f.write("window.AGENT_REPLAYS = ")
        json.dump({"kind": "replay", "source": "rl/ntuple_td.py, 4x6 n-tuple network (checkpoints/ntuple_best.npz)", "games": games}, f)
        f.write(";\n")
    print(f"wrote {os.path.relpath(out, ROOT)}: {len(games)} games, {os.path.getsize(out) / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
