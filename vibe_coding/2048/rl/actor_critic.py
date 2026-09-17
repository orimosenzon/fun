"""
Actor-Critic (בגרסת A2C, Advantage Actor-Critic) ל-2048.

בניגוד ל-DQN, שלומד ערכי Q ומסיק מהם מדיניות, כאן הרשת מוציאה ישירות
מדיניות: התפלגות הסתברות על ארבעת הכיוונים (ה"שחקן", actor). לצידה ראש נוסף
שמעריך את ערך המצב V(s) (ה"מבקר", critic). המבקר לא בוחר פעולות; הוא משמש
כקנה מידה: האם הפעולה שבחרנו הובילה לתוצאה טובה יותר או פחות מהצפוי?
ההפרש הזה הוא ה-advantage (יתרון), והוא מה שמכוון את עדכון השחקן:
פעולות עם יתרון חיובי מקבלות הסתברות גבוהה יותר, ולהפך.

מרכיבים:
  * N סביבות במקביל, רולאאוט (rollout) של T צעדים בכל סיבוב אימון.
  * GAE (Generalized Advantage Estimation): הערכת היתרון שמאזנת בין הטיה
    לשונות בעזרת הפרמטר lambda.
  * בונוס אנטרופיה: מונע מהמדיניות לקרוס מוקדם מדי לפעולה אחת (חקירה).
  * מסכת פעולות: לוגיטים של מהלכים לא חוקיים מקבלים -inf, כך שההסתברות
    שלהם היא בדיוק אפס, גם בדגימה וגם בחישוב האנטרופיה.
  * נרמול היתרונות בכל אצווה, כדי שסקאלת התגמול לא תשפיע על יחס הכוחות
    בין הגרדיאנט של המדיניות לבין בונוס האנטרופיה.

הרצה לדוגמה:
    python rl/actor_critic.py --time-limit-min 70 --run-name a2c
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    ActorCriticNetwork,
    JsonlLogger,
    RecentStats,
    count_parameters,
    encode_boards,
    get_device,
    masked_argmax,
    play_games,
    summarize,
)
from game2048 import VecGame2048


def masked_log_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=1)


def make_policy(net: ActorCriticNetwork, device: torch.device, greedy: bool = True):
    """מדיניות להערכה: הפעולה הסבירה ביותר (greedy) או דגימה מההתפלגות."""

    @torch.no_grad()
    def policy(boards: np.ndarray, valid: np.ndarray) -> np.ndarray:
        logits, _ = net(encode_boards(boards, device))
        v = torch.from_numpy(valid).to(device)
        if greedy:
            return masked_argmax(logits, v).cpu().numpy()
        probs = masked_log_softmax(logits, v).exp()
        return torch.multinomial(probs, 1).squeeze(1).cpu().numpy()

    return policy


def train(args: argparse.Namespace):
    device = get_device() if args.device == "auto" else torch.device(args.device)
    torch.manual_seed(args.seed)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(root, "logs", f"{args.run_name}_train.jsonl"))

    net = ActorCriticNetwork(args.filters, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    print(f"device={device}  parameters={count_parameters(net):,}", flush=True)

    env = VecGame2048(args.n_envs, seed=args.seed, reward_scale=args.reward_scale)
    stats = RecentStats(window=200)
    eval_policy = make_policy(net, device, greedy=True)

    N, T = args.n_envs, args.n_steps
    # מאגרי הרולאאוט
    rb_boards = np.zeros((T, N, 4, 4), dtype=np.uint8)
    rb_valid = np.zeros((T, N, 4), dtype=np.bool_)
    rb_actions = torch.zeros((T, N), dtype=torch.long, device=device)
    rb_rewards = torch.zeros((T, N), device=device)
    rb_dones = torch.zeros((T, N), device=device)
    rb_values = torch.zeros((T, N), device=device)

    boards = env.boards.copy()
    transitions = 0
    updates = 0
    t_start = time.time()
    next_log = args.log_every
    next_eval = args.eval_every
    best_eval = -1.0
    acc = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "n": 0}

    while transitions < args.total_transitions:
        if args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min:
            print("time limit reached", flush=True)
            break

        # ------------------ איסוף רולאאוט של T צעדים ב-N סביבות ------------------
        with torch.no_grad():
            for t in range(T):
                valid = env.valid_moves()
                logits, value = net(encode_boards(boards, device))
                v_t = torch.from_numpy(valid).to(device)
                probs = masked_log_softmax(logits, v_t).exp()
                actions = torch.multinomial(probs, 1).squeeze(1)
                rewards, dones, finished = env.step(actions.cpu().numpy())
                stats.add(finished)

                rb_boards[t] = boards
                rb_valid[t] = valid
                rb_actions[t] = actions
                rb_values[t] = value
                rb_rewards[t] = torch.from_numpy(rewards).to(device)
                rb_dones[t] = torch.from_numpy(dones.astype(np.float32)).to(device)
                boards = env.boards.copy()
            transitions += T * N

            # ------------------ GAE: חישוב היתרונות אחורה בזמן ------------------
            _, last_value = net(encode_boards(boards, device))
            advantages = torch.zeros((T, N), device=device)
            last_gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                next_nonterminal = 1.0 - rb_dones[t]
                next_value = last_value if t == T - 1 else rb_values[t + 1]
                delta = rb_rewards[t] + args.gamma * next_value * next_nonterminal - rb_values[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + rb_values

        # ------------------ עדכון (A2C: מעבר אחד על כל האצווה) ------------------
        b_boards = rb_boards.reshape(T * N, 4, 4)
        b_valid = torch.from_numpy(rb_valid.reshape(T * N, 4)).to(device)
        b_actions = rb_actions.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        if args.norm_adv:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        logits, values = net(encode_boards(b_boards, device))
        logp_all = masked_log_softmax(logits, b_valid)
        logp = logp_all.gather(1, b_actions[:, None]).squeeze(1)
        # אנטרופיה על ההתפלגות הממוסכת (איברים לא חוקיים תורמים 0)
        p_all = logp_all.exp()
        entropy = -(p_all * logp_all.masked_fill(~b_valid, 0.0)).sum(dim=1).mean()

        pg_loss = -(b_adv * logp).mean()
        v_loss = 0.5 * F.mse_loss(values, b_ret)
        loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()
        updates += 1
        acc["pg_loss"] += pg_loss.item()
        acc["v_loss"] += v_loss.item()
        acc["entropy"] += entropy.item()
        acc["n"] += 1

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
                pg_loss=acc["pg_loss"] / n,
                v_loss=acc["v_loss"] / n,
                entropy=acc["entropy"] / n,
                sps=int(transitions / elapsed),
                **{f"recent_{k}": v for k, v in snap.items()},
            )
            logger.log(**rec)
            print(
                f"[{elapsed/60:6.1f} min] trans={transitions:>11,} ent={rec['entropy']:.3f} "
                f"vloss={rec['v_loss']:.4f} score={snap['score_mean']:7.0f} tile={snap['tile_mean']:6.0f} "
                f"p1024={snap['reach_1024']:.2f} p2048={snap['reach_2048']:.2f} sps={rec['sps']}",
                flush=True,
            )
            acc = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "n": 0}

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
    p = argparse.ArgumentParser(description="Actor-Critic (A2C) for 2048")
    p.add_argument("--run-name", default="a2c")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--total-transitions", type=int, default=200_000_000)
    p.add_argument("--time-limit-min", type=float, default=0, help="0 = ללא הגבלת זמן")
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=16, help="אורך הרולאאוט T")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--norm-adv", type=int, default=1)
    p.add_argument("--reward-scale", type=float, default=1e-3)
    p.add_argument("--filters", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--log-every", type=int, default=500_000)
    p.add_argument("--eval-every", type=int, default=5_000_000)
    p.add_argument("--eval-games", type=int, default=100)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
