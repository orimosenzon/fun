"""
הערכת סוכנים מאומנים וקווי בסיס, וייצוא הנתונים לדו"חות.

לכל סוכן: 1,000 משחקים (זרע קבוע, אותם משחקים לכולם), סטטיסטיקות מסכמות,
התפלגויות מלאות, והקלטה של המשחק הטוב ביותר לצפייה חוזרת.

הרצה:
    python rl/evaluate.py --all
    python rl/evaluate.py --agent dqn --checkpoint checkpoints/dqn_best.pt --games 1000
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from common import (
    ActorCriticNetwork,
    QNetwork,
    get_device,
    play_games,
    random_policy,
    record_game,
    summarize,
)
from game2048 import move_boards

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "reports", "data")


def greedy_score_policy(boards: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """קו בסיס: המהלך עם הניקוד המיידי הגבוה ביותר (שוברים שוויון לפי סדר קבוע)."""
    n = boards.shape[0]
    gains = np.full((n, 4), -1, dtype=np.int64)
    for a in range(4):
        _, sc, changed = move_boards(boards, np.full(n, a))
        gains[:, a] = np.where(changed, sc, -1)
    gains[~valid] = -1
    # שבירת שוויון קבועה: שמאלה > למטה > ימינה > למעלה (העדפה קבועה עוזרת לשמור פינה)
    tie_break = np.array([0.0, 0.1, 0.2, 0.3])[None, :]
    return (gains + tie_break).argmax(axis=1)


def load_agent(kind: str, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    if kind == "dqn":
        net = QNetwork(a["filters"], a["hidden"]).to(device)
    else:
        net = ActorCriticNetwork(a["filters"], a["hidden"]).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt


def agent_policy(kind: str, net, device: torch.device):
    from common import encode_boards, masked_argmax

    @torch.no_grad()
    def policy(boards, valid):
        out = net(encode_boards(boards, device))
        logits = out if kind == "dqn" else out[0]
        return masked_argmax(logits, torch.from_numpy(valid).to(device)).cpu().numpy()

    return policy


def evaluate_policy(name: str, label: str, policy, games: int, seed: int = 2048) -> dict:
    results = play_games(policy, games, seed=seed)
    summ = summarize(results)
    # הקלטת משחקים: מחפשים משחק טוב (עד 40 ניסיונות) ושומרים את הטוב ביותר, וגם משחק "טיפוסי"
    best = None
    recorded = []
    for s in range(40):
        g = record_game(policy, seed=10_000 + s)
        recorded.append(g)
        if best is None or g["score"] > best["score"]:
            best = g
    recorded.sort(key=lambda g: g["score"])
    typical = recorded[len(recorded) // 2]
    # מאפייני אסטרטגיה מתוך 40 המשחקים המוקלטים: העדפת כיוונים, והאם האריח הגדול נשמר בפינה
    action_counts = [0, 0, 0, 0]
    corner_hits, frames_n = 0, 0
    for g in recorded:
        for fr in g["frames"]:
            if fr["action"] is not None:
                action_counts[fr["action"]] += 1
            b = np.array(fr["board"])
            m = b.max()
            if m >= 5:  # מ-32 ומעלה, לפני זה "פינה" חסרת משמעות
                frames_n += 1
                corners = (b[0, 0], b[0, 3], b[3, 0], b[3, 3])
                corner_hits += int(m in corners)
    out = {
        "name": name,
        "label": label,
        "summary": summ,
        "action_counts": action_counts,
        "corner_rate": corner_hits / max(1, frames_n),
        "scores": [r["score"] for r in results],
        "max_tiles": [r["max_tile"] for r in results],
        "moves": [r["moves"] for r in results],
        "best_game": best,
        "typical_game": typical,
    }
    print(f"{label:22s} games={games} score_mean={summ['score_mean']:.0f} median={summ['score_median']:.0f} "
          f"max={summ['score_max']:.0f} p1024={summ['reach_1024']:.2f} p2048={summ['reach_2048']:.2f} "
          f"best_recorded={best['score']}")
    return out


def save(data: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"eval_{data['name']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("  ->", os.path.relpath(path, ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="הערכת שני הסוכנים ושני קווי הבסיס")
    p.add_argument("--agent", choices=["dqn", "ac"])
    p.add_argument("--checkpoint")
    p.add_argument("--games", type=int, default=1000)
    args = p.parse_args()
    device = get_device()

    if args.all or args.agent is None:
        rng = np.random.default_rng(0)
        save(evaluate_policy("random", "מדיניות אקראית", random_policy(rng), args.games))
        save(evaluate_policy("greedy", "חמדן צעד אחד", greedy_score_policy, args.games))
        for kind, name, label in (("dqn", "dqn", "DQN"), ("ac", "ac", "Actor-Critic")):
            path = os.path.join(ROOT, "checkpoints", f"{name}_best.pt")
            if not os.path.exists(path):
                print(f"skip {name}: no checkpoint at {path}")
                continue
            net, ckpt = load_agent(kind, path, device)
            data = evaluate_policy(name, label, agent_policy(kind, net, device), args.games)
            data["checkpoint"] = os.path.relpath(path, ROOT)
            data["train_transitions"] = ckpt.get("transitions")
            data["train_args"] = ckpt.get("args")
            save(data)
    else:
        net, ckpt = load_agent(args.agent, args.checkpoint, device)
        label = "DQN" if args.agent == "dqn" else "Actor-Critic"
        data = evaluate_policy(args.agent, label, agent_policy(args.agent, net, device), args.games)
        data["checkpoint"] = os.path.relpath(args.checkpoint, ROOT)
        data["train_transitions"] = ckpt.get("transitions")
        data["train_args"] = ckpt.get("args")
        save(data)


if __name__ == "__main__":
    main()
