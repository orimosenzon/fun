"""
TD(0) על afterstates עם רשת n-tuple ל-2048 (Szubert & Jaśkowski, 2014).

זו השיטה שהספרות של 2048 מזהה כחזקה ביותר, ומבחינה אלגוריתמית היא הפשוטה מכולן:

  * אין רשת עצבית. פונקציית הערך V היא סכום של m טבלאות חיפוש (lookup tables):
    כל טבלה מסתכלת על n משבצות קבועות בלוח (n-tuple), מקודדת את n הערכים
    שבהן למספר אחד, ומחזירה את המשקל ששמור באינדקס הזה. כאן m=4 טבלאות של
    n=6 משבצות (התצורה "4x6" של Yeh ואחרים), כל טבלה 16^6 כניסות, ובנוסף
    דגימה סימטרית: כל טבלה מופעלת על 8 הסיבובים/השיקופים של הלוח, כך
    ש-V(לוח) הוא סכום של 32 קריאות מטבלה.
  * אין ערכי Q ואין מדיניות מפורשת. לומדים V של afterstate, כלומר של הלוח
    *אחרי* ההחלקה ולפני האריח האקראי. במצב s בוחרים את הפעולה שממקסמת
    r(s,a) + V(afterstate(s,a)) — ארבע החלקות דטרמיניסטיות, בלי תוחלת על
    האריח האקראי.
  * הלמידה היא TD(0) קלאסי, צעד אחד, בלי היוון (gamma=1) ובלי חקירה: אחרי
    כל מהלך, ערכו של ה-afterstate הקודם נדחף לעבר r + V(afterstate הנוכחי),
    ובסיום משחק לעבר 0. העדכון מתחלק שווה בין 32 המשקלים (alpha/32 לכל אחד).
  * הכול רץ על ליבת CPU אחת, בלולאה סדרתית משחק-אחרי-משחק (numba), בדיוק
    כמו במאמרים. אין GPU, אין אצוות.

הרצה לדוגמה:
    python rl/ntuple_td.py --time-limit-min 73 --run-name ntuple
    python rl/ntuple_td.py --net small --time-limit-min 20 --run-name ntuple_small   # הגרסה לדפדפן
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from numba import njit

from common import JsonlLogger, RecentStats, play_games, summarize
from game2048 import SCORE_LEFT, TABLE_LEFT, move_boards

# ----------------------------------------------------------------------------
# הרשת: 4 שישיות (Yeh et al. 2016), אינדקס משבצת = שורה*4 + עמודה
# ----------------------------------------------------------------------------

TUPLES_4x6 = np.array([
    [0, 1, 2, 3, 4, 5],     # השורה הראשונה + שתי משבצות מהשנייה
    [4, 5, 6, 7, 8, 9],     # השורה השנייה + שתיים מהשלישית
    [0, 1, 2, 4, 5, 6],     # מלבן 2x3 בפינה
    [4, 5, 6, 8, 9, 10],    # מלבן 2x3 באמצע
], dtype=np.int64)

# הרשת הקטנה לדפדפן: הרשת המקורית של Szubert & Jaśkowski (2014), רביעיות בלבד. במקור 17 רביעיות
# (4 שורות, 4 עמודות, 9 ריבועים 2x2) בלי דגימה סימטרית; עם 8 הסימטריות מספיקות 5 רביעיות מייצגות.
# 5 x 16^4 = 327,680 משקלים (1.3MB), פי 200 פחות מהרשת של 4x6, ולכן אפשר לטעון אותה בדפדפן.
TUPLES_SMALL = np.array([
    [0, 1, 2, 3],           # שורה חיצונית (ובסימטריה: השורה התחתונה ושתי העמודות החיצוניות)
    [4, 5, 6, 7],           # שורה פנימית
    [0, 1, 4, 5],           # ריבוע 2x2 בפינה
    [1, 2, 5, 6],           # ריבוע 2x2 באמצע הצלע
    [5, 6, 9, 10],          # ריבוע 2x2 במרכז
], dtype=np.int64)
NETWORKS = {"4x6": TUPLES_4x6, "small": TUPLES_SMALL}
N_VALUES = 16                       # מעריכים 0..15 (ריק עד 32768)


def board_symmetries() -> np.ndarray:
    """(8,16): לכל אחת משמונה הסימטריות של הריבוע, איזו משבצת מקורית נוחתת בכל מיקום."""
    idx = np.arange(16).reshape(4, 4)
    perms = []
    for k in range(4):
        r = np.rot90(idx, k)
        perms.append(r.flatten())
        perms.append(np.fliplr(r).flatten())
    return np.array(perms, dtype=np.int64)


def build_features(tuples: np.ndarray, table_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    דגימה סימטרית: כל חלון מופעל על 8 הסימטריות. מחזיר
      feats   (8*m, n): אינדקסי המשבצות של כל מאפיין
      offsets (8*m,):   התחלת הטבלה של החלון שאליו המאפיין שייך
    """
    syms = board_symmetries()
    feats, offsets = [], []
    for s in range(8):
        for j, cells in enumerate(tuples):
            feats.append(syms[s][cells])
            offsets.append(j * table_size)
    return np.array(feats, dtype=np.int64), np.array(offsets, dtype=np.int64)


class Net:
    """תיאור רשת n-tuple: החלונות, המאפיינים (חלונות x סימטריות) ומיקומי הטבלאות במערך השטוח."""

    def __init__(self, tuples):
        self.tuples = np.asarray(tuples, dtype=np.int64)
        self.n_tables, self.tuple_size = self.tuples.shape
        self.table_size = N_VALUES ** self.tuple_size
        self.feats, self.offsets = build_features(self.tuples, self.table_size)
        self.n_feats = self.feats.shape[0]
        self.n_weights = self.n_tables * self.table_size
        self.pow = (N_VALUES ** np.arange(self.tuple_size - 1, -1, -1)).astype(np.int64)

    def new_table(self) -> np.ndarray:
        return np.zeros(self.n_weights, dtype=np.float32)

    def describe(self) -> str:
        return (f"{self.n_tables} x {self.tuple_size}-tuples, {self.n_feats} features with symmetry, "
                f"{self.n_weights:,} weights ({4 * self.n_weights / 1e6:.1f} MB)")


NET_4x6 = Net(TUPLES_4x6)
# שמות ברמת המודול לרשת הראשית (הבדיקות והקוד הישן משתמשים בהם)
FEATS, OFFSETS, N_FEATS, N_TABLES, TABLE_SIZE, TUPLE_SIZE = (
    NET_4x6.feats, NET_4x6.offsets, NET_4x6.n_feats, NET_4x6.n_tables, NET_4x6.table_size, NET_4x6.tuple_size)


def net_from_meta(meta: dict) -> Net:
    """הרשת ששמורה בנקודת ביקורת (החלונות נשמרים ב-meta, ולכן הקובץ מספיק לעצמו)."""
    return Net(meta["tuples"])


# ----------------------------------------------------------------------------
# גרסת NumPy (וקטורית) של V ושל המדיניות: להערכה ולבדיקת המימוש ב-numba
# ----------------------------------------------------------------------------

def feature_indices_np(boards: np.ndarray, net: Net = NET_4x6) -> np.ndarray:
    """(N,4,4) -> (N,n_feats) אינדקסים לטבלה השטוחה."""
    flat = boards.reshape(boards.shape[0], 16).astype(np.int64)
    vals = flat[:, net.feats]                 # (N,n_feats,tuple_size)
    return (vals * net.pow).sum(axis=2) + net.offsets


def values_np(table: np.ndarray, boards: np.ndarray, net: Net = NET_4x6) -> np.ndarray:
    return table[feature_indices_np(boards, net)].sum(axis=1)


def make_policy(table: np.ndarray, net: Net = NET_4x6):
    """מדיניות חמדנית של צעד אחד: argmax על r + V(afterstate), רק בין מהלכים חוקיים."""

    def policy(boards: np.ndarray, valid: np.ndarray) -> np.ndarray:
        n = boards.shape[0]
        best = np.full((n, 4), -np.inf)
        for a in range(4):
            after, sc, changed = move_boards(boards, np.full(n, a))
            v = sc + values_np(table, after, net)
            best[:, a] = np.where(changed & valid[:, a], v, -np.inf)
        return best.argmax(axis=1)

    return policy


# ----------------------------------------------------------------------------
# הליבה הסדרתית ב-numba: משחק, מאפיינים, ערך, ולולאת TD
# ----------------------------------------------------------------------------

@njit(cache=True)
def _slide(board, a, out):
    """מחליק לוח שטוח (16,) לכיוון a לתוך out. מחזיר (ניקוד, האם השתנה).

    כל קו (שורה או עמודה) נקרא בסדר שבו "שמאלה" בטבלה פירושו "לכיוון a",
    ולכן מספיקה טבלת ההחלקה שמאלה של game2048.
    """
    score = 0
    changed = False
    for line in range(4):
        if a == 3:            # שמאלה: השורה משמאל לימין
            c0, step = line * 4, 1
        elif a == 1:          # ימינה: השורה מימין לשמאל
            c0, step = line * 4 + 3, -1
        elif a == 0:          # למעלה: העמודה מלמעלה למטה
            c0, step = line, 4
        else:                 # למטה: העמודה מלמטה למעלה
            c0, step = 12 + line, -4
        code = (np.int64(board[c0]) << 12) | (np.int64(board[c0 + step]) << 8) | \
               (np.int64(board[c0 + 2 * step]) << 4) | np.int64(board[c0 + 3 * step])
        new = np.int64(TABLE_LEFT[code])
        score += SCORE_LEFT[code]
        if new != code:
            changed = True
        out[c0] = (new >> 12) & 15
        out[c0 + step] = (new >> 8) & 15
        out[c0 + 2 * step] = (new >> 4) & 15
        out[c0 + 3 * step] = new & 15
    return score, changed


@njit(cache=True)
def _features(board, feats, offsets, out):
    """32 אינדקסים לטבלה השטוחה עבור לוח שטוח."""
    for f in range(feats.shape[0]):
        idx = 0
        for j in range(feats.shape[1]):
            idx = idx * 16 + np.int64(board[feats[f, j]])
        out[f] = offsets[f] + idx


@njit(cache=True)
def _value(table, idx):
    s = 0.0
    for f in range(idx.shape[0]):
        s += table[idx[f]]
    return s


@njit(cache=True)
def _spawn(board):
    """אריח חדש במשבצת ריקה אקראית: 2 בהסתברות 0.9, 4 בהסתברות 0.1 (כמו במשחק)."""
    n_empty = 0
    for i in range(16):
        if board[i] == 0:
            n_empty += 1
    if n_empty == 0:
        return
    k = int(np.random.random() * n_empty)
    for i in range(16):
        if board[i] == 0:
            if k == 0:
                board[i] = 2 if np.random.random() < 0.1 else 1
                return
            k -= 1


@njit(cache=True)
def _reset(board):
    for i in range(16):
        board[i] = 0
    _spawn(board)
    _spawn(board)


@njit(cache=True)
def _max_tile(board):
    m = 0
    for i in range(16):
        if board[i] > m:
            m = board[i]
    return 2 ** np.int64(m)


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True)
def train_chunk(table, feats, offsets, alpha_w, n_moves, board, prev_idx, state,
                ep_scores, ep_tiles, ep_moves):
    """
    מריץ n_moves מהלכים של TD(0) על afterstates, ממשיך ממצב שמור, וממלא את מערכי
    האפיזודות שהסתיימו. מחזיר (מהלכים שבוצעו, אפיזודות שהסתיימו, סכום |delta|, מספר עדכונים).

    state: [ניקוד נוכחי, מהלכים במשחק הנוכחי, האם יש afterstate קודם]
    """
    after = np.zeros((4, 16), dtype=np.uint8)
    idx = np.zeros((4, feats.shape[0]), dtype=np.int64)
    score = state[0]
    moves = state[1]
    has_prev = state[2] > 0
    n_done = 0
    sum_abs_delta = 0.0
    n_updates = 0
    done_moves = 0
    for _ in range(n_moves):
        # ---- בחירת פעולה: argmax על r + V(afterstate) בין ארבע ההחלקות החוקיות ----
        best_a = -1
        best_v = -1e30
        best_r = 0
        for a in range(4):
            sc, changed = _slide(board, a, after[a])
            if not changed:
                continue
            _features(after[a], feats, offsets, idx[a])
            v = sc + _value(table, idx[a])
            if v > best_v:
                best_v = v
                best_a = a
                best_r = sc
        if best_a < 0:
            # ---- סיום משחק: ה-afterstate האחרון נדחף לעבר 0 ----
            if has_prev:
                delta = 0.0 - _value(table, prev_idx)
                for f in range(prev_idx.shape[0]):
                    table[prev_idx[f]] += alpha_w * delta
                sum_abs_delta += abs(delta)
                n_updates += 1
            ep_scores[n_done] = score
            ep_tiles[n_done] = _max_tile(board)
            ep_moves[n_done] = moves
            n_done += 1
            _reset(board)
            score = 0
            moves = 0
            has_prev = False
            if n_done == ep_scores.shape[0]:
                break
            continue
        # ---- עדכון TD(0) של ה-afterstate הקודם לעבר r_t + V(afterstate_t) ----
        if has_prev:
            delta = best_v - _value(table, prev_idx)      # best_v = r_t + V(s'_t)
            for f in range(prev_idx.shape[0]):
                table[prev_idx[f]] += alpha_w * delta
            sum_abs_delta += abs(delta)
            n_updates += 1
        # ---- ביצוע המהלך ----
        for f in range(prev_idx.shape[0]):
            prev_idx[f] = idx[best_a, f]
        has_prev = True
        for i in range(16):
            board[i] = after[best_a, i]
        score += best_r
        moves += 1
        _spawn(board)
        done_moves += 1
    state[0] = score
    state[1] = moves
    state[2] = 1 if has_prev else 0
    return done_moves, n_done, sum_abs_delta, n_updates


@njit(cache=True)
def play_greedy(table, feats, offsets, n_games, ep_scores, ep_tiles, ep_moves):
    """n_games משחקים חמדניים מלאים (להערכה בזמן האימון), בלי למידה."""
    board = np.zeros(16, dtype=np.uint8)
    after = np.zeros((4, 16), dtype=np.uint8)
    idx = np.zeros(feats.shape[0], dtype=np.int64)
    for g in range(n_games):
        _reset(board)
        score = 0
        moves = 0
        while True:
            best_a = -1
            best_v = -1e30
            best_r = 0
            for a in range(4):
                sc, changed = _slide(board, a, after[a])
                if not changed:
                    continue
                _features(after[a], feats, offsets, idx)
                v = sc + _value(table, idx)
                if v > best_v:
                    best_v = v
                    best_a = a
                    best_r = sc
            if best_a < 0:
                break
            for i in range(16):
                board[i] = after[best_a, i]
            score += best_r
            moves += 1
            _spawn(board)
        ep_scores[g] = score
        ep_tiles[g] = _max_tile(board)
        ep_moves[g] = moves


def greedy_eval(table: np.ndarray, n_games: int, seed: int, net: Net = NET_4x6) -> list[dict]:
    seed_numba(seed)
    sc = np.zeros(n_games, dtype=np.int64)
    ti = np.zeros(n_games, dtype=np.int64)
    mv = np.zeros(n_games, dtype=np.int64)
    play_greedy(table, net.feats, net.offsets, n_games, sc, ti, mv)
    return [{"score": int(s), "max_tile": int(t), "moves": int(m)} for s, t, m in zip(sc, ti, mv)]


# ----------------------------------------------------------------------------
# שמירה וטעינה
# ----------------------------------------------------------------------------

def save_checkpoint(path: str, table: np.ndarray, meta: dict) -> None:
    np.savez(path, table=table, meta=json.dumps(meta, ensure_ascii=False))


def load_checkpoint(path: str) -> tuple[np.ndarray, dict]:
    z = np.load(path)
    return z["table"], json.loads(str(z["meta"]))


# ----------------------------------------------------------------------------
# אימון
# ----------------------------------------------------------------------------

def train(args: argparse.Namespace):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(root, "logs", f"{args.run_name}_train.jsonl"))

    net = Net(NETWORKS[args.net])
    FEATS, OFFSETS, N_FEATS = net.feats, net.offsets, net.n_feats
    table = net.new_table()
    print(f"n-tuple network '{args.net}': {net.describe()}, alpha={args.alpha} ({args.alpha / N_FEATS:.5f} per weight)"
          f"{', decay x0.1 at 50% and x0.01 at 75% of the budget' if args.lr_decay == 1 else ', late decay x0.3 at 75% and x0.1 at 90%' if args.lr_decay == 2 else ', constant'}", flush=True)

    def current_alpha(frac_done: float) -> float:
        """קצב הלמידה של V.
        --lr-decay 1: לוח הזמנים המקובל (גואי 2022): 0.1 → 0.01 בחצי → 0.001 בשלושה רבעים (הפסיד בניסוי ההסרה).
        --lr-decay 2: דעיכה מתונה ומאוחרת: 0.1 → 0.03 בשלושה רבעים → 0.01 בעשירית האחרונה (הריצה הארוכה)."""
        if not args.lr_decay:
            return args.alpha
        if args.lr_decay == 2:
            return args.alpha * (1.0 if frac_done < 0.75 else 0.3 if frac_done < 0.9 else 0.1)
        return args.alpha * (1.0 if frac_done < 0.5 else 0.1 if frac_done < 0.75 else 0.01)

    seed_numba(args.seed)
    board = np.zeros(16, dtype=np.uint8)
    _reset(board)
    prev_idx = np.zeros(N_FEATS, dtype=np.int64)
    state = np.zeros(3, dtype=np.int64)
    chunk = args.chunk
    ep_scores = np.zeros(chunk // 8 + 16, dtype=np.int64)
    ep_tiles = np.zeros_like(ep_scores)
    ep_moves = np.zeros_like(ep_scores)

    stats = RecentStats(window=200)
    transitions = 0
    episodes = 0
    t_start = time.time()
    next_log = args.log_every
    next_eval = args.eval_every
    best_eval = -1.0
    best_table, best_meta, best_dirty = None, None, False
    last_save = time.time()
    best_path = os.path.join(ckpt_dir, f"{args.run_name}_best.npz")
    acc_delta, acc_n = 0.0, 0

    def flush_best():
        # הטבלה של 4x6 שוקלת 268MB, ולכן הגרסה הטובה ביותר נשמרת בזיכרון ונכתבת לדיסק רק מדי כמה דקות
        nonlocal best_dirty, last_save
        if best_dirty:
            save_checkpoint(best_path, best_table, best_meta)
            best_dirty = False
        last_save = time.time()

    while transitions < args.total_transitions:
        if args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min:
            print("time limit reached", flush=True)
            break
        frac = (time.time() - t_start) / (60 * args.time_limit_min) if args.time_limit_min else transitions / args.total_transitions
        alpha = current_alpha(frac)
        n_mv, n_done, s_delta, n_upd = train_chunk(table, FEATS, OFFSETS, np.float32(alpha / N_FEATS), chunk,
                                                   board, prev_idx, state, ep_scores, ep_tiles, ep_moves)
        transitions += n_mv
        episodes += n_done
        acc_delta += s_delta
        acc_n += n_upd
        stats.add([{"score": int(ep_scores[i]), "max_tile": int(ep_tiles[i]), "moves": int(ep_moves[i])} for i in range(n_done)])

        # ------------------ לוג ------------------
        if transitions >= next_log:
            next_log += args.log_every
            snap = stats.snapshot()
            elapsed = time.time() - t_start
            rec = dict(
                transitions=transitions,
                episodes=episodes,
                updates=transitions,
                td_abs=acc_delta / max(1, acc_n),
                visited=float(np.count_nonzero(table)) / table.size,
                alpha=alpha,
                sps=int(transitions / elapsed),
                **{f"recent_{k}": v for k, v in snap.items()},
            )
            logger.log(**rec)
            print(
                f"[{elapsed/60:6.1f} min] trans={transitions:>13,} eps={episodes:>8,} |td|={rec['td_abs']:8.1f} "
                f"visited={100 * rec['visited']:.2f}% score={snap['score_mean']:7.0f} tile={snap['tile_mean']:6.0f} "
                f"p1024={snap['reach_1024']:.2f} p2048={snap['reach_2048']:.2f} sps={rec['sps']:,}",
                flush=True,
            )
            acc_delta, acc_n = 0.0, 0

        # ------------------ הערכה חמדנית ושמירה ------------------
        if transitions >= next_eval:
            next_eval += args.eval_every
            results = greedy_eval(table, args.eval_games, seed=999, net=net)
            summ = summarize(results)
            logger.log(transitions=transitions, updates=transitions, eval=summ)
            print(
                f"    EVAL  score_mean={summ['score_mean']:.0f} median={summ['score_median']:.0f} "
                f"max={summ['score_max']:.0f} p1024={summ['reach_1024']:.2f} p2048={summ['reach_2048']:.2f}",
                flush=True,
            )
            if summ["score_mean"] > best_eval:
                best_eval = summ["score_mean"]
                if best_table is None:
                    best_table = table.copy()
                else:
                    np.copyto(best_table, table)
                best_meta = {"args": vars(args), "transitions": transitions, "episodes": episodes, "eval": summ,
                             "net": args.net, "tuples": net.tuples.tolist()}
                best_dirty = True
            if time.time() - last_save > 60 * args.save_every_min:
                flush_best()

    flush_best()
    meta = {"args": vars(args), "transitions": transitions, "episodes": episodes, "net": args.net, "tuples": net.tuples.tolist()}
    save_checkpoint(os.path.join(ckpt_dir, f"{args.run_name}_final.npz"), table, meta)
    logger.close()
    print(f"done. {transitions:,} transitions, {episodes:,} episodes, {(time.time()-t_start)/60:.1f} min", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Afterstate TD(0) with an n-tuple network for 2048")
    p.add_argument("--run-name", default="ntuple")
    p.add_argument("--net", choices=list(NETWORKS), default="4x6", help="4x6 = 4 שישיות (268MB); small = 5 רביעיות (1.3MB, לדפדפן)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-transitions", type=int, default=20_000_000_000)
    p.add_argument("--time-limit-min", type=float, default=0, help="0 = ללא הגבלת זמן")
    p.add_argument("--alpha", type=float, default=0.1, help="קצב הלמידה של V; כל אחד מהמשקלים שנקראו מקבל alpha/n_feats")
    p.add_argument("--lr-decay", type=int, default=0, help="1 = פי 10 בחצי התקציב ופי 100 בשלושה רבעים; 2 = מתון ומאוחר: פי 3 בשלושה רבעים ופי 10 ב-90%")
    p.add_argument("--chunk", type=int, default=200_000, help="מהלכים בכל קריאה ללולאה המקומפלת")
    p.add_argument("--log-every", type=int, default=5_000_000)
    p.add_argument("--eval-every", type=int, default=20_000_000)
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--save-every-min", type=float, default=10, help="כל כמה דקות לכתוב לדיסק את הטבלה הטובה ביותר")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
