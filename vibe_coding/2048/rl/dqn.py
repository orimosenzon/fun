"""
DQN (Deep Q-Network) ל-2048.

הרעיון: לומדים פונקציה Q(s, a) שמעריכה את סכום התגמולים העתידי (המונחת בגורם
היוון gamma) אם במצב s נבצע את הפעולה a ומשם נמשיך לשחק "הכי טוב שאנחנו יודעים".
כשיש לנו Q טובה, המדיניות היא פשוט: בחר את הפעולה עם ה-Q הגבוה ביותר.

מרכיבים (וכולם מתועדים בהרחבה ב-README):
  * זיכרון חוויות (replay buffer): שומרים מעברים (s, a, r, s', done) ומאמנים
    על דגימות אקראיות מהם, כדי לשבור את המתאם בין מעברים עוקבים.
  * רשת מטרה (target network): עותק איטי של הרשת, שממנו מחשבים את היעד
    r + gamma * max Q(s', a'), כדי שהיעד לא ירדוף אחרי עצמו.
  * Double DQN: הפעולה הטובה ב-s' נבחרת ע"י הרשת המקוונת, אבל הערך שלה נלקח
    מרשת המטרה. זה מקטין את ההטיה האופטימית של max.
  * אפסילון-חמדן (epsilon-greedy): בהסתברות epsilon פעולה אקראית (חקירה),
    אחרת הפעולה הטובה לפי Q (ניצול). epsilon יורד בהדרגה במהלך האימון.
  * מסכת פעולות: מהלך שלא משנה את הלוח אינו חוקי, ולכן גם בבחירת הפעולה וגם
    בחישוב היעד מתעלמים ממנו.

הרצה לדוגמה:
    python rl/dqn.py --total-transitions 20000000 --run-name dqn
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    JsonlLogger,
    QNetwork,
    RecentStats,
    count_parameters,
    encode_boards,
    get_device,
    masked_argmax,
    play_games,
    summarize,
)
from game2048 import VecGame2048, valid_moves


class ReplayBuffer:
    """זיכרון חוויות מעגלי במערכי NumPy קומפקטיים (לוח = 16 בתים)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.s = np.zeros((capacity, 4, 4), dtype=np.uint8)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, 4, 4), dtype=np.uint8)
        self.done = np.zeros(capacity, dtype=np.bool_)
        self.valid2 = np.zeros((capacity, 4), dtype=np.bool_)  # מהלכים חוקיים ב-s'
        self.k = np.ones(capacity, dtype=np.int64)  # כמה צעדים מכסה המעבר (n-step)
        self.idx = 0
        self.size = 0

    def add_batch(self, s, a, r, s2, done, valid2, k):
        n = len(a)
        idx = (self.idx + np.arange(n)) % self.capacity
        self.s[idx] = s
        self.a[idx] = a
        self.r[idx] = r
        self.s2[idx] = s2
        self.done[idx] = done
        self.valid2[idx] = valid2
        self.k[idx] = k
        self.idx = (self.idx + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        i = rng.integers(0, self.size, batch_size)
        return self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i], self.valid2[i], self.k[i]


class NStepCollector:
    """
    הופך מעברים של צעד אחד למעברים של n צעדים, בנפרד לכל אחת מ-N הסביבות.

    מעבר של n צעדים: (s_t, a_t, R, s_{t+n}, done) כאשר
        R = r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1}
    והיעד ללמידה הוא R + gamma^n * max Q(s_{t+n}). כשמשחק מסתיים לפני שהצטברו
    n צעדים, נפלטים מעברים קצרים יותר עם done=True (אין bootstrap).
    """

    def __init__(self, n_envs: int, n: int, gamma: float):
        self.n_envs, self.n, self.gamma = n_envs, n, gamma
        self.buf: list[list[tuple]] = [[] for _ in range(n_envs)]

    def push(self, s, a, r, s2, done, valid2):
        """מוסיף צעד אחד לכל סביבה ומחזיר מערכים של מעברי n-צעדים שהושלמו."""
        out_s, out_a, out_r, out_s2, out_d, out_v2, out_k = [], [], [], [], [], [], []
        for i in range(self.n_envs):
            q = self.buf[i]
            q.append((s[i], a[i], float(r[i])))
            if done[i]:
                # המשחק נגמר: כל מה שבתור מסתיים כאן, בלי bootstrap
                while q:
                    R = 0.0
                    for k, (_, _, rk) in enumerate(q):
                        R += (self.gamma ** k) * rk
                    out_s.append(q[0][0]); out_a.append(q[0][1]); out_r.append(R)
                    out_s2.append(s2[i]); out_d.append(True); out_v2.append(valid2[i]); out_k.append(len(q))
                    q.pop(0)
            elif len(q) == self.n:
                R = 0.0
                for k, (_, _, rk) in enumerate(q):
                    R += (self.gamma ** k) * rk
                out_s.append(q[0][0]); out_a.append(q[0][1]); out_r.append(R)
                out_s2.append(s2[i]); out_d.append(False); out_v2.append(valid2[i]); out_k.append(self.n)
                q.pop(0)
        if not out_a:
            return None
        return (
            np.stack(out_s), np.array(out_a, dtype=np.int64), np.array(out_r, dtype=np.float32),
            np.stack(out_s2), np.array(out_d, dtype=np.bool_), np.stack(out_v2), np.array(out_k, dtype=np.int64),
        )


def make_greedy_policy(net: QNetwork, device: torch.device):
    """מדיניות חמדנית (בלי חקירה) להערכה: argmax על Q בין המהלכים החוקיים."""

    @torch.no_grad()
    def policy(boards: np.ndarray, valid: np.ndarray) -> np.ndarray:
        q = net(encode_boards(boards, device))
        v = torch.from_numpy(valid).to(device)
        return masked_argmax(q, v).cpu().numpy()

    return policy


def train(args: argparse.Namespace):
    device = get_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(root, "checkpoints")
    log_dir = os.path.join(root, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(log_dir, f"{args.run_name}_train.jsonl"))

    online = QNetwork(args.filters, args.hidden).to(device)
    target = QNetwork(args.filters, args.hidden).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    print(f"device={device}  parameters={count_parameters(online):,}", flush=True)

    env = VecGame2048(args.n_envs, seed=args.seed, reward_scale=args.reward_scale)
    buffer = ReplayBuffer(args.buffer_size)
    collector = NStepCollector(args.n_envs, args.n_step, args.gamma)
    stats = RecentStats(window=200)
    greedy_policy = make_greedy_policy(online, device)

    boards = env.boards.copy()
    transitions = 0
    grad_steps = 0
    loss_acc, q_acc, loss_n = 0.0, 0.0, 0
    best_eval = -1.0
    t_start = time.time()
    next_log = args.log_every
    next_eval = args.eval_every
    eps_decay_transitions = int(args.total_transitions * args.eps_decay_frac)

    while transitions < args.total_transitions:
        if args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min:
            print("time limit reached", flush=True)
            break

        # --- בחירת פעולות: אפסילון-חמדן עם מסכת מהלכים חוקיים ---
        frac = min(1.0, transitions / max(1, eps_decay_transitions))
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)
        valid = env.valid_moves()
        with torch.no_grad():
            q = online(encode_boards(boards, device))
            greedy = masked_argmax(q, torch.from_numpy(valid).to(device)).cpu().numpy()
        noise = rng.random(valid.shape)
        noise[~valid] = -1.0
        random_actions = noise.argmax(axis=1)
        explore = rng.random(args.n_envs) < eps
        actions = np.where(explore, random_actions, greedy)

        # --- צעד בסביבה ושמירה בזיכרון ---
        rewards, dones, finished = env.step(actions)
        next_boards = env.boards.copy()  # למשחק שנגמר זה כבר לוח חדש, אבל done מבטל את ה-bootstrap
        packed = collector.push(boards, actions, rewards, next_boards, dones, valid_moves(next_boards))
        if packed is not None:
            buffer.add_batch(*packed)
        boards = next_boards
        transitions += args.n_envs
        stats.add(finished)

        # --- צעדי למידה ---
        if buffer.size >= args.learning_starts:
            for _ in range(args.grad_steps):
                s, a, r, s2, d, v2, k = buffer.sample(args.batch_size, rng)
                s_t = encode_boards(np.concatenate([s, s2]), device)
                a_t = torch.from_numpy(a).to(device)
                r_t = torch.from_numpy(r).to(device)
                d_t = torch.from_numpy(d).to(device)
                v2_t = torch.from_numpy(v2).to(device)
                discount = torch.from_numpy(args.gamma ** k).float().to(device)

                q_all = online(s_t)
                q_s, q_s2_online = q_all[: args.batch_size], q_all[args.batch_size:]
                q_sa = q_s.gather(1, a_t[:, None]).squeeze(1)
                with torch.no_grad():
                    # Double DQN: בחירת הפעולה ברשת המקוונת, הערכתה ברשת המטרה
                    a2 = masked_argmax(q_s2_online, v2_t)
                    q_s2_target = target(s_t[args.batch_size:]).gather(1, a2[:, None]).squeeze(1)
                    y = r_t + discount * (~d_t).float() * q_s2_target
                loss = F.smooth_l1_loss(q_sa, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), args.grad_clip)
                optimizer.step()
                grad_steps += 1
                loss_acc += loss.item()
                q_acc += q_sa.mean().item()
                loss_n += 1

                # עדכון רך של רשת המטרה (Polyak)
                with torch.no_grad():
                    for p_t, p_o in zip(target.parameters(), online.parameters()):
                        p_t.mul_(1 - args.tau).add_(p_o, alpha=args.tau)

        # --- לוג ---
        if transitions >= next_log:
            next_log += args.log_every
            snap = stats.snapshot()
            elapsed = time.time() - t_start
            rec = dict(
                transitions=transitions,
                episodes=stats.total_episodes,
                grad_steps=grad_steps,
                epsilon=round(eps, 4),
                loss=(loss_acc / loss_n) if loss_n else None,
                q_mean=(q_acc / loss_n) if loss_n else None,
                sps=int(transitions / elapsed),
                **{f"recent_{k}": v for k, v in snap.items()},
            )
            logger.log(**rec)
            print(
                f"[{elapsed/60:6.1f} min] trans={transitions:>10,} eps={eps:.3f} "
                f"loss={rec['loss'] if rec['loss'] is None else round(rec['loss'], 4)} "
                f"score={snap['score_mean']:7.0f} tile={snap['tile_mean']:6.0f} "
                f"p1024={snap['reach_1024']:.2f} p2048={snap['reach_2048']:.2f} sps={rec['sps']}",
                flush=True,
            )
            loss_acc, q_acc, loss_n = 0.0, 0.0, 0

        # --- הערכה חמדנית ושמירת נקודת ביקורת ---
        if transitions >= next_eval:
            next_eval += args.eval_every
            online.eval()
            results = play_games(greedy_policy, args.eval_games, seed=999)
            online.train()
            summ = summarize(results)
            logger.log(transitions=transitions, grad_steps=grad_steps, eval=summ)
            print(
                f"    EVAL  score_mean={summ['score_mean']:.0f} median={summ['score_median']:.0f} "
                f"max={summ['score_max']:.0f} p1024={summ['reach_1024']:.2f} p2048={summ['reach_2048']:.2f}",
                flush=True,
            )
            ckpt = {"model": online.state_dict(), "args": vars(args), "transitions": transitions, "eval": summ}
            torch.save(ckpt, os.path.join(ckpt_dir, f"{args.run_name}_last.pt"))
            if summ["score_mean"] > best_eval:
                best_eval = summ["score_mean"]
                torch.save(ckpt, os.path.join(ckpt_dir, f"{args.run_name}_best.pt"))

    ckpt = {"model": online.state_dict(), "args": vars(args), "transitions": transitions}
    torch.save(ckpt, os.path.join(ckpt_dir, f"{args.run_name}_final.pt"))
    logger.close()
    print(f"done. {transitions:,} transitions, {grad_steps:,} grad steps, {(time.time()-t_start)/60:.1f} min", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DQN for 2048")
    p.add_argument("--run-name", default="dqn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-transitions", type=int, default=20_000_000)
    p.add_argument("--time-limit-min", type=float, default=0, help="0 = ללא הגבלת זמן")
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-step", type=int, default=1, help="אורך החזר n-step (1 = DQN קלאסי; 3 נוסה ונמצא גרוע יותר, ראו README)")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-steps", type=int, default=2, help="צעדי למידה לכל צעד סביבה")
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay-frac", type=float, default=0.25, help="חלק מהאימון שבו epsilon יורד לינארית")
    p.add_argument("--tau", type=float, default=0.001, help="קצב עדכון רך של רשת המטרה")
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--reward-scale", type=float, default=1e-3)
    p.add_argument("--filters", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--log-every", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=1_000_000)
    p.add_argument("--eval-games", type=int, default=100)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
