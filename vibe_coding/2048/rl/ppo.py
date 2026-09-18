"""
PPO (Proximal Policy Optimization, שולמן ואחרים 2017) ל-2048.

PPO הוא Actor-Critic שמרשה לעצמו ללמוד מכל אצווה יותר מפעם אחת. ב-A2C כל אצווה
של רולאאוט משמשת לצעד גרדיאנט אחד ונזרקת, כי אחרי הצעד המדיניות כבר השתנתה
והנתונים כבר לא "שלה" (on-policy). PPO עוקף את זה בשני מנגנונים:

  * יחס הסתברויות (importance ratio): לכל פעולה שנדגמה זוכרים את ההסתברות שהמדיניות
    הישנה נתנה לה, ומשקללים את היתרון ביחס  r = pi_new(a|s) / pi_old(a|s).
    זה מתקן את ההטיה שנובעת מכך שהנתונים נאספו על ידי מדיניות קצת אחרת.
  * קטימה (clipping): פונקציית המטרה היא  min(r·A, clip(r, 1-eps, 1+eps)·A).
    ברגע שהיחס יוצא מהחלון [1-eps, 1+eps] בכיוון "המשתלם", הגרדיאנט מתאפס.
    כך המדיניות לא יכולה להתרחק יותר מדי מהמדיניות שאספה את הנתונים, ואפשר לעשות
    כמה מעברים (epochs) של מיני-אצוות על אותה אצווה בבטחה.

כל השאר זהה ל-A2C שבקובץ actor_critic.py: אותה רשת, אותן 64 סביבות, רולאאוט של 16
צעדים, GAE, בונוס אנטרופיה, מסכת מהלכים ונרמול יתרונות. ההבדל היחיד הוא שלב העדכון.
ככה ההשוואה בדו"ח מבודדת בדיוק את מה ש-PPO מוסיף.

הרצה לדוגמה:
    python rl/ppo.py --time-limit-min 73 --run-name ppo
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from actor_critic import make_policy, masked_log_softmax
from common import (
    ActorCriticNetwork,
    JsonlLogger,
    RecentStats,
    count_parameters,
    encode_boards,
    get_device,
    play_games,
    summarize,
)
from game2048 import VecGame2048


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
    B = N * T
    assert B % args.n_minibatches == 0, "גודל האצווה חייב להתחלק במספר המיני-אצוות"
    mb_size = B // args.n_minibatches
    # מאגרי הרולאאוט. בניגוד ל-A2C שומרים גם את לוג-ההסתברות של הפעולה שנבחרה,
    # כי בעדכון נצטרך את היחס בין המדיניות החדשה לזו שאספה את הנתונים.
    rb_boards = np.zeros((T, N, 4, 4), dtype=np.uint8)
    rb_valid = np.zeros((T, N, 4), dtype=np.bool_)
    rb_actions = torch.zeros((T, N), dtype=torch.long, device=device)
    rb_logp = torch.zeros((T, N), device=device)
    rb_rewards = torch.zeros((T, N), device=device)
    rb_dones = torch.zeros((T, N), device=device)
    rb_values = torch.zeros((T, N), device=device)

    boards = env.boards.copy()
    transitions = 0
    iterations = 0  # רולאאוטים
    updates = 0     # צעדי גרדיאנט (epochs * minibatches לכל רולאאוט)
    t_start = time.time()
    next_log = args.log_every
    next_eval = args.eval_every
    best_eval = -1.0
    keys = ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac")
    acc = {k: 0.0 for k in keys}
    acc["n"] = 0

    while transitions < args.total_transitions:
        if args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min:
            print("time limit reached", flush=True)
            break

        # ------------------ איסוף רולאאוט של T צעדים ב-N סביבות (זהה ל-A2C) ------------------
        with torch.no_grad():
            for t in range(T):
                valid = env.valid_moves()
                logits, value = net(encode_boards(boards, device))
                v_t = torch.from_numpy(valid).to(device)
                logp_all = masked_log_softmax(logits, v_t)
                actions = torch.multinomial(logp_all.exp(), 1).squeeze(1)
                rewards, dones, finished = env.step(actions.cpu().numpy())
                stats.add(finished)

                rb_boards[t] = boards
                rb_valid[t] = valid
                rb_actions[t] = actions
                rb_logp[t] = logp_all.gather(1, actions[:, None]).squeeze(1)
                rb_values[t] = value
                rb_rewards[t] = torch.from_numpy(rewards).to(device)
                rb_dones[t] = torch.from_numpy(dones.astype(np.float32)).to(device)
                boards = env.boards.copy()
            transitions += B
            iterations += 1

            # ------------------ GAE (זהה ל-A2C) ------------------
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

            # שיטוח האצווה. הקידוד one-hot של כל הלוחות נעשה פעם אחת ונשאר על ה-GPU,
            # כי כל לוח ייכנס לרשת ppo_epochs פעמים.
            x_all = encode_boards(rb_boards.reshape(B, 4, 4), device)
            b_valid = torch.from_numpy(rb_valid.reshape(B, 4)).to(device)
            b_actions = rb_actions.reshape(-1)
            b_logp_old = rb_logp.reshape(-1)
            b_adv = advantages.reshape(-1)
            b_ret = returns.reshape(-1)
            b_val_old = rb_values.reshape(-1)

        # ------------------ עדכון PPO: K מעברים במיני-אצוות על אותה אצווה ------------------
        for epoch in range(args.ppo_epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, mb_size):
                idx = perm[start:start + mb_size]
                mb_adv = b_adv[idx]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                logits, values = net(x_all[idx])
                logp_all = masked_log_softmax(logits, b_valid[idx])
                logp = logp_all.gather(1, b_actions[idx][:, None]).squeeze(1)
                p_all = logp_all.exp()
                entropy = -(p_all * logp_all.masked_fill(~b_valid[idx], 0.0)).sum(dim=1).mean()

                # יחס ההסתברויות בין המדיניות הנוכחית למדיניות שאספה את הנתונים.
                # במיני-אצווה הראשונה של ה-epoch הראשון היחס הוא בדיוק 1 (אותה רשת).
                log_ratio = logp - b_logp_old[idx]
                ratio = log_ratio.exp()
                # פונקציית המטרה הקטומה. הסימן הפוך כי אנחנו ממזערים.
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if args.clip_vloss:
                    # גרסת "PPO2": גם המבקר לא רשאי לזוז יותר מ-eps מהערך הישן בצעד אחד
                    v_clipped = b_val_old[idx] + torch.clamp(values - b_val_old[idx], -args.clip_eps, args.clip_eps)
                    v_loss = 0.5 * torch.max((values - b_ret[idx]) ** 2, (v_clipped - b_ret[idx]) ** 2).mean()
                else:
                    v_loss = 0.5 * F.mse_loss(values, b_ret[idx])

                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                optimizer.step()
                updates += 1

                with torch.no_grad():
                    # אומדן KL בין המדיניות הישנה לחדשה (האומדן k3 של שולמן: לא מוטה ושונות נמוכה),
                    # ואיזה חלק מהדגימות נמצא מחוץ לחלון הקטימה. שניהם מדדי בריאות של PPO.
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > args.clip_eps).float().mean()
                acc["pg_loss"] += pg_loss.item()
                acc["v_loss"] += v_loss.item()
                acc["entropy"] += entropy.item()
                acc["approx_kl"] += approx_kl.item()
                acc["clip_frac"] += clip_frac.item()
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
                iterations=iterations,
                updates=updates,
                sps=int(transitions / elapsed),
                **{k: acc[k] / n for k in keys},
                **{f"recent_{k}": v for k, v in snap.items()},
            )
            logger.log(**rec)
            print(
                f"[{elapsed/60:6.1f} min] trans={transitions:>11,} ent={rec['entropy']:.3f} kl={rec['approx_kl']:.4f} "
                f"clip={rec['clip_frac']:.2f} vloss={rec['v_loss']:.4f} score={snap['score_mean']:7.0f} tile={snap['tile_mean']:6.0f} "
                f"p1024={snap['reach_1024']:.2f} p2048={snap['reach_2048']:.2f} sps={rec['sps']}",
                flush=True,
            )
            acc = {k: 0.0 for k in keys}
            acc["n"] = 0

        # ------------------ הערכה חמדנית ושמירה (זהה ל-A2C) ------------------
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
    print(f"done. {transitions:,} transitions, {iterations:,} rollouts, {updates:,} gradient steps, "
          f"{(time.time()-t_start)/60:.1f} min", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO for 2048")
    p.add_argument("--run-name", default="ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--total-transitions", type=int, default=200_000_000)
    p.add_argument("--time-limit-min", type=float, default=0, help="0 = ללא הגבלת זמן")
    # צינור הנתונים: זהה ל-A2C
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=16, help="אורך הרולאאוט T")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--norm-adv", type=int, default=1, help="נרמול היתרונות בכל מיני-אצווה")
    p.add_argument("--reward-scale", type=float, default=1e-3)
    p.add_argument("--filters", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    # מה ש-PPO מוסיף
    p.add_argument("--ppo-epochs", type=int, default=4, help="כמה מעברים על כל אצווה של רולאאוט")
    p.add_argument("--n-minibatches", type=int, default=4, help="לכמה מיני-אצוות מחלקים את האצווה בכל מעבר")
    p.add_argument("--clip-eps", type=float, default=0.2, help="רוחב חלון הקטימה של יחס ההסתברויות")
    p.add_argument("--clip-vloss", type=int, default=0, help="1 = לקטום גם את הפסד המבקר (גרסת PPO2)")
    p.add_argument("--log-every", type=int, default=500_000)
    p.add_argument("--eval-every", type=int, default=5_000_000)
    p.add_argument("--eval-games", type=int, default=100)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
