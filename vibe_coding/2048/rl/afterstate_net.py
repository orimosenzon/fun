"""
ניסוי 3: TD(0) על afterstates עם רשת הקונבולוציה של הפרויקט במקום טבלאות n-tuple.

רשת ה-n-tuple ניצחה את שלוש הרשתות העצביות בפער גדול, ויש לזה שני הסברים אפשריים שקשה
להפריד ביניהם: האלגוריתם (ערך של afterstate, TD(0), מדיניות חמדנית בלי חקירה) והייצוג
(טבלאות על חלונות עם סימטריה). הניסוי הזה מפריד: אותו אלגוריתם בדיוק כמו ב-ntuple_td.py,
אבל V מיוצג ברשת הקונבולוציה שמשמשת את DQN, A2C ו-PPO (גוף משותף + ראש ערך אחד).

  * מדיניות: ארבע החלקות דטרמיניסטיות, argmax על r(s,a) + V(afterstate). חמדנית גם באימון.
  * למידה: אחרי כל מהלך, V(afterstate הקודם) נדחף לעבר r + V(afterstate הנוכחי) (ולעבר 0 בסיום
    משחק). חצי-גרדיאנט: היעד מנותק מהגרף, כמו ב-DQN. gamma = 1 כמו ברשת ה-n-tuple.
  * N סביבות במקביל, וצעד גרדיאנט אחד (Adam) על N העדכונים של כל צעד סביבה. בלי זיכרון חוויות,
    בלי רשת מטרה, בלי אנטרופיה, בלי GAE.
  * התגמול הגולמי כפול 0.001, כמו בשלוש הרשתות (רשת עצבית רגישה לסקאלה; טבלה לא).

עלות: כל צעד דורש V של 4N afterstates (בלי גרדיאנט) ועוד מעבר עם גרדיאנט על N, כלומר פי ~5 חישוב
רשת לצעד לעומת A2C.

הריצה הראשונה (asnet_plain), האלגוריתם כלשונו: 34,606 אחרי 5 דקות (רמת A2C, עם פי 34 פחות מעברים), ואז
תנודות, קפיצה של V ל-146 אלף, וקריסה בדקה 18 ל-V קבוע (ניקוד 3,130, כמו החמדן) בלי התאוששות. זה "השילוש
הקטלני" (קירוב פונקציה + bootstrapping מלא) בלי שום מנגנון ייצוב. הטבלה חסינה לזה כי העדכון שלה מקומי
ולינארי. לכן יש כאן שלושה מייצבים סטנדרטיים, כולם כבויים כברירת מחדל, שהריצה השנייה מפעילה:
  --gamma 0.99      היוון כמו בשלוש הרשתות (כיווץ)
  --target-tau      רשת מטרה עם עדכון רך (Polyak), כמו ב-DQN: היעד r + V_target(s') נע לאט
  --huber 1         הפסד Huber במקום ריבועי, כמו ב-DQN: שגיאה גדולה לא מייצרת גרדיאנט ענק

הרצה לדוגמה:
    python rl/afterstate_net.py --time-limit-min 73 --run-name asnet_plain            # כלשונו (קרס)
    python rl/afterstate_net.py --time-limit-min 73 --run-name asnet --gamma 0.99 --target-tau 0.005 --huber 1
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    AfterstateValueNetwork,
    JsonlLogger,
    RecentStats,
    count_parameters,
    encode_boards,
    get_device,
    play_games,
    summarize,
)
from game2048 import VecGame2048, move_boards


def all_afterstates(boards: np.ndarray):
    """(N,4,4) -> afterstates (4,N,4,4), ניקוד (4,N), האם המהלך חוקי (4,N), לארבעת הכיוונים."""
    n = boards.shape[0]
    after = np.empty((4, n, 4, 4), dtype=np.uint8)
    reward = np.empty((4, n), dtype=np.int64)
    changed = np.empty((4, n), dtype=bool)
    for a in range(4):
        after[a], reward[a], changed[a] = move_boards(boards, np.full(n, a))
    return after, reward, changed


@torch.no_grad()
def choose(net, boards: np.ndarray, device: torch.device, reward_scale: float):
    """המדיניות החמדנית: מחזירה (פעולות (N,), afterstate שנבחר (N,4,4), ניקוד מיידי (N,), V שלו (N,) על המכשיר)."""
    n = boards.shape[0]
    after, reward, changed = all_afterstates(boards)
    v = net(encode_boards(after.reshape(4 * n, 4, 4), device)).view(4, n)
    score = torch.from_numpy(reward * reward_scale).to(device).float() + v
    score = score.masked_fill(~torch.from_numpy(changed).to(device), float("-inf"))
    actions = score.argmax(dim=0).cpu().numpy()
    ar = np.arange(n)
    return actions, after[actions, ar], reward[actions, ar], v[torch.from_numpy(actions).to(device), torch.arange(n, device=device)]


def make_policy(net, device: torch.device, reward_scale: float):
    def policy(boards: np.ndarray, valid: np.ndarray) -> np.ndarray:
        return choose(net, boards, device, reward_scale)[0]
    return policy


def train(args: argparse.Namespace):
    device = get_device() if args.device == "auto" else torch.device(args.device)
    torch.manual_seed(args.seed)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(root, "logs", f"{args.run_name}_train.jsonl"))

    net = AfterstateValueNetwork(args.filters, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    # רשת מטרה (אופציונלית): עותק שמתקדם לאט אחרי הרשת, ומשמש רק ליעד ה-bootstrapping
    target = None
    if args.target_tau > 0:
        target = copy.deepcopy(net)
        for p_ in target.parameters():
            p_.requires_grad_(False)
    print(f"device={device}  parameters={count_parameters(net):,}  envs={args.n_envs}  gamma={args.gamma}  "
          f"target_tau={args.target_tau}  huber={args.huber}", flush=True)

    env = VecGame2048(args.n_envs, seed=args.seed, reward_scale=1.0)
    stats = RecentStats(window=200)
    eval_policy = make_policy(net, device, args.reward_scale)

    N = args.n_envs
    prev_after = np.zeros((N, 4, 4), dtype=np.uint8)   # ה-afterstate הקודם של כל סביבה
    has_prev = np.zeros(N, dtype=bool)                  # האם יש כזה (לא במהלך הראשון של משחק)
    transitions = 0
    updates = 0
    t_start = time.time()
    next_log = args.log_every
    next_eval = args.eval_every
    best_eval = -1.0
    acc = {"td_loss": 0.0, "td_abs": 0.0, "v_mean": 0.0, "n": 0}

    while transitions < args.total_transitions:
        if args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min:
            print("time limit reached", flush=True)
            break

        # ------------------ בחירה: argmax על r + V(afterstate), ארבע החלקות לכל סביבה ------------------
        boards = env.boards
        actions, chosen_after, r_t, v_t = choose(net, boards, device, args.reward_scale)
        if target is not None:
            with torch.no_grad():
                v_t = target(encode_boards(chosen_after, device))   # היעד נמדד ברשת המטרה, ההחלטה ברשת הלומדת
        _, dones, finished = env.step(actions)
        stats.add(finished)
        transitions += N

        # ------------------ עדכון TD(0): היעד ל-afterstate הקודם הוא r_t + gamma * V(afterstate נוכחי) ------------------
        # ובסיום משחק, היעד ל-afterstate הנוכחי הוא 0. שני הסוגים באצווה אחת.
        y_prev = torch.from_numpy(r_t * args.reward_scale).to(device).float() + args.gamma * v_t
        hp = torch.from_numpy(has_prev).to(device)
        dn = torch.from_numpy(dones).to(device)
        x_boards = np.concatenate([prev_after[has_prev], chosen_after[dones]])
        y = torch.cat([y_prev[hp], torch.zeros(int(dones.sum()), device=device)])
        if x_boards.shape[0] > 0:
            v_pred = net(encode_boards(x_boards, device))
            td = y - v_pred
            loss = F.smooth_l1_loss(v_pred, y) if args.huber else 0.5 * (td ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            updates += 1
            if target is not None:
                with torch.no_grad():
                    for p_t, p_o in zip(target.parameters(), net.parameters()):
                        p_t.lerp_(p_o, args.target_tau)
            acc["td_loss"] += loss.item()
            acc["td_abs"] += td.abs().mean().item()
            acc["v_mean"] += v_pred.mean().item()
            acc["n"] += 1

        prev_after = chosen_after
        has_prev = ~dones

        # ------------------ לוג ------------------
        if transitions >= next_log:
            next_log += args.log_every
            snap = stats.snapshot()
            elapsed = time.time() - t_start
            n = max(1, acc["n"])
            rec = dict(
                transitions=transitions,
                episodes=stats.total_episodes,
                updates=updates,
                td_loss=acc["td_loss"] / n,
                td_abs=acc["td_abs"] / n / args.reward_scale,   # בנקודות משחק, כמו ברשת ה-n-tuple
                v_mean=acc["v_mean"] / n / args.reward_scale,
                sps=int(transitions / elapsed),
                **{f"recent_{k}": v for k, v in snap.items()},
            )
            logger.log(**rec)
            print(
                f"[{elapsed/60:6.1f} min] trans={transitions:>11,} |td|={rec['td_abs']:7.1f} V={rec['v_mean']:8.0f} "
                f"score={snap['score_mean']:7.0f} tile={snap['tile_mean']:6.0f} "
                f"p1024={snap['reach_1024']:.2f} p2048={snap['reach_2048']:.2f} sps={rec['sps']}",
                flush=True,
            )
            acc = {"td_loss": 0.0, "td_abs": 0.0, "v_mean": 0.0, "n": 0}

        # ------------------ הערכה חמדנית ושמירה ------------------
        if transitions >= next_eval:
            next_eval += args.eval_every
            net.eval()
            results = play_games(eval_policy, args.eval_games, seed=999)
            net.train()
            summ = summarize(results)
            logger.log(transitions=transitions, updates=updates, eval=summ)
            print(
                f"    EVAL  score_mean={summ['score_mean']:.0f} median={summ['score_median']:.0f} "
                f"max={summ['score_max']:.0f} p1024={summ['reach_1024']:.2f} p2048={summ['reach_2048']:.2f}",
                flush=True,
            )
            ckpt = {"model": net.state_dict(), "args": vars(args), "transitions": transitions, "eval": summ}
            torch.save(ckpt, os.path.join(ckpt_dir, f"{args.run_name}_last.pt"))
            if summ["score_mean"] > best_eval:
                best_eval = summ["score_mean"]
                torch.save(ckpt, os.path.join(ckpt_dir, f"{args.run_name}_best.pt"))

    torch.save({"model": net.state_dict(), "args": vars(args), "transitions": transitions},
               os.path.join(ckpt_dir, f"{args.run_name}_final.pt"))
    logger.close()
    print(f"done. {transitions:,} transitions, {updates:,} updates, {(time.time()-t_start)/60:.1f} min", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Afterstate TD(0) with the project's convolutional network, for 2048")
    p.add_argument("--run-name", default="asnet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--total-transitions", type=int, default=200_000_000)
    p.add_argument("--time-limit-min", type=float, default=0, help="0 = ללא הגבלת זמן")
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--gamma", type=float, default=1.0, help="1 כמו ברשת ה-n-tuple (משחק סופי); 0.99 כמו בשלוש הרשתות")
    p.add_argument("--target-tau", type=float, default=0.0, help="0 = בלי רשת מטרה; 0.005 = עדכון רך כמו ב-DQN")
    p.add_argument("--huber", type=int, default=0, help="1 = הפסד Huber במקום ריבועי")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=1e-3)
    p.add_argument("--filters", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--log-every", type=int, default=500_000)
    p.add_argument("--eval-every", type=int, default=2_000_000)
    p.add_argument("--eval-games", type=int, default=100)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
