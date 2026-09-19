"""
הערכת סוכנים מאומנים וקווי בסיס, וייצוא הנתונים לדו"חות.

לכל סוכן: 1,000 משחקים (זרע קבוע, אותם משחקים לכולם), סטטיסטיקות מסכמות,
התפלגויות מלאות, והקלטה של המשחק הטוב ביותר לצפייה חוזרת.

הרצה:
    python rl/evaluate.py --all
    python rl/evaluate.py --agent dqn --checkpoint checkpoints/dqn_best.pt --games 1000
    python rl/evaluate.py --agent ppo --checkpoint checkpoints/ppo_best.pt
    python rl/evaluate.py --agent ntuple --checkpoint checkpoints/ntuple_best.npz
    python rl/evaluate.py --agent ntuple --checkpoint checkpoints/ntuple_long_best.npz --name ntuple_long
    python rl/evaluate.py --agent asnet --checkpoint checkpoints/asnet_best.pt
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from common import (
    ActorCriticNetwork,
    AfterstateValueNetwork,
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

# (סוג הסוכן, שם הריצה של נקודת הביקורת, תווית). ac ו-ppo חולקים את אותה רשת; ntuple הוא טבלאות (npz), בלי torch.
AGENT_KINDS = (("dqn", "dqn", "DQN"), ("ac", "a2c", "Actor-Critic"), ("ppo", "ppo", "PPO"), ("ntuple", "ntuple", "N-Tuple TD"),
               ("asnet", "asnet", "Afterstate TD (רשת)"))
LABELS = {k: label for k, _, label in AGENT_KINDS}
CKPT_EXT = {"ntuple": ".npz"}


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
    if kind == "ntuple":
        from ntuple_td import load_checkpoint, net_from_meta
        table, meta = load_checkpoint(path)
        return (table, net_from_meta(meta)), meta
    ckpt = torch.load(path, map_location=device)
    a = ckpt["args"]
    if kind == "dqn":
        net = QNetwork(a["filters"], a["hidden"]).to(device)
    elif kind == "asnet":
        net = AfterstateValueNetwork(a["filters"], a["hidden"]).to(device)
    else:
        net = ActorCriticNetwork(a["filters"], a["hidden"]).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    net._reward_scale = a.get("reward_scale", 1e-3)
    return net, ckpt


def ckpt_reward_scale(net) -> float:
    return getattr(net, "_reward_scale", 1e-3)


def agent_policy(kind: str, net, device: torch.device):
    from common import encode_boards, masked_argmax

    if kind == "ntuple":
        from ntuple_td import make_policy
        table, ntnet = net
        return make_policy(table, ntnet)
    if kind == "asnet":
        from afterstate_net import make_policy as make_asnet_policy
        return make_asnet_policy(net, device, ckpt_reward_scale(net))

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
    corner_hits, edge_hits, frames_n = 0, 0, 0
    pos = np.zeros((4, 4))  # איפה יושב האריח הגדול ביותר
    for g in recorded:
        for fr in g["frames"]:
            if fr["action"] is not None:
                action_counts[fr["action"]] += 1
            b = np.array(fr["board"])
            m = b.max()
            if m >= 5:  # מ-32 ומעלה, לפני זה "פינה" חסרת משמעות
                frames_n += 1
                r, c = np.unravel_index(int(b.argmax()), b.shape)
                pos[r, c] += 1
                is_corner = r in (0, 3) and c in (0, 3)
                is_edge = r in (0, 3) or c in (0, 3)
                corner_hits += int(is_corner)
                edge_hits += int(is_edge)
    out = {
        "name": name,
        "label": label,
        "summary": summ,
        "action_counts": action_counts,
        "corner_rate": corner_hits / max(1, frames_n),
        "edge_rate": edge_hits / max(1, frames_n),
        "max_tile_pos": (pos / max(1, frames_n)).round(4).tolist(),
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
    p.add_argument("--all", action="store_true", help="הערכת כל הסוכנים ושני קווי הבסיס")
    p.add_argument("--agent", choices=[k for k, _, _ in AGENT_KINDS])
    p.add_argument("--checkpoint")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--name", help="שם קובץ הפלט (eval_<name>.json); ברירת מחדל: סוג הסוכן. למשל ntuple_long לריצה הארוכה")
    args = p.parse_args()
    device = get_device()

    if args.all or args.agent is None:
        rng = np.random.default_rng(0)
        save(evaluate_policy("random", "מדיניות אקראית", random_policy(rng), args.games))
        save(evaluate_policy("greedy", "חמדן צעד אחד", greedy_score_policy, args.games))
        for kind, run, label in AGENT_KINDS:
            path = os.path.join(ROOT, "checkpoints", f"{run}_best{CKPT_EXT.get(kind, '.pt')}")
            if not os.path.exists(path):
                print(f"skip {kind}: no checkpoint at {path}")
                continue
            net, ckpt = load_agent(kind, path, device)
            data = evaluate_policy(kind, label, agent_policy(kind, net, device), args.games)
            data["checkpoint"] = os.path.relpath(path, ROOT)
            data["train_transitions"] = ckpt.get("transitions")
            data["train_args"] = ckpt.get("args")
            save(data)
    else:
        net, ckpt = load_agent(args.agent, args.checkpoint, device)
        label = LABELS[args.agent]
        data = evaluate_policy(args.name or args.agent, label, agent_policy(args.agent, net, device), args.games)
        data["checkpoint"] = os.path.relpath(args.checkpoint, ROOT)
        data["train_transitions"] = ckpt.get("transitions")
        data["train_args"] = ckpt.get("args")
        save(data)


if __name__ == "__main__":
    main()
