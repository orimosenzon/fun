"""
קוד משותף לשני האלגוריתמים: קידוד הלוח לרשת, ארכיטקטורת הרשת, הערכת ביצועים ולוגים.

קידוד הלוח (state encoding):
    הלוח הוא 16 משבצות, כל אחת מחזיקה מעריך 0..15 (0 = ריק, k = אריח 2^k).
    לרשת נותנים ייצוג one-hot: טנזור בגודל (16 ערוצים, 4, 4), כאשר הערוץ k
    מכיל 1 במשבצות שבהן יש אריח 2^k. הייצוג הזה הרבה יותר נוח לרשת מאשר
    הערכים הגולמיים (2, 4, ..., 2048), שנפרסים על פני שלושה סדרי גודל.

ארכיטקטורת הרשת (משותפת ל-DQN ול-Actor-Critic):
    שתי שכבות קונבולוציה 3x3 (עם ריפוד, כך שהלוח נשאר 4x4), ואז שכבה
    מלאה. אחרי שתי שכבות 3x3 כל נוירון "רואה" את הלוח כולו, ולכן הרשת יכולה
    ללמוד תבניות גלובליות כמו "האריח הגדול בפינה" או "שורה מונוטונית".
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game2048 import VecGame2048, Game2048

NUM_CHANNELS = 16  # מעריכים 0..15
NUM_ACTIONS = 4


def get_device() -> torch.device:
    # cudnn.benchmark: מאפשר ל-cuDNN לבחור את אלגוריתם הקונבולוציה המהיר ביותר
    # לגדלים הקבועים שלנו (לוח 4x4). על ה-GTX 1060 זה מכפיל את מהירות האימון.
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_boards(boards: np.ndarray, device: torch.device) -> torch.Tensor:
    """(N,4,4) uint8 מעריכים -> (N,16,4,4) float32 one-hot על ה-GPU."""
    t = torch.from_numpy(np.ascontiguousarray(boards)).to(device, non_blocking=True).long()
    return F.one_hot(t, NUM_CHANNELS).permute(0, 3, 1, 2).float()


class Trunk(nn.Module):
    """גוף הרשת: קונבולוציות על הלוח ואז וקטור מאפיינים באורך hidden."""

    def __init__(self, filters: int = 128, hidden: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_CHANNELS, filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.fc = nn.Linear(filters * 16, hidden)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        return F.relu(self.fc(x))


class QNetwork(nn.Module):
    """רשת Q ל-DQN: לוח -> ארבעה ערכי Q, אחד לכל כיוון."""

    def __init__(self, filters: int = 128, hidden: int = 256):
        super().__init__()
        self.trunk = Trunk(filters, hidden)
        self.head = nn.Linear(hidden, NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class ActorCriticNetwork(nn.Module):
    """רשת Actor-Critic: גוף משותף, ראש מדיניות (4 לוגיטים) וראש ערך (מספר אחד)."""

    def __init__(self, filters: int = 128, hidden: int = 256):
        super().__init__()
        self.trunk = Trunk(filters, hidden)
        self.policy_head = nn.Linear(hidden, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def masked_argmax(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """argmax רק על פעולות חוקיות. values (N,4), valid (N,4) bool."""
    return values.masked_fill(~valid, float("-inf")).argmax(dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# הערכת ביצועים
# ----------------------------------------------------------------------------

def play_games(policy, n_games: int, seed: int = 12345, n_envs: int = 128) -> list[dict]:
    """
    מריץ n_games משחקים מלאים עם מדיניות נתונה ומחזיר רשימת תוצאות.

    policy: פונקציה (boards (N,4,4) uint8, valid (N,4) bool) -> actions (N,)
    התוצאה לכל משחק: {"score", "max_tile", "moves"}.

    חשוב: כל סביבה תורמת רק את המשחק *הראשון* שלה. אם היינו לוקחים את המשחקים
    הראשונים שמסתיימים מבין N סביבות מקביליות, היינו מקבלים הטיה לטובת משחקים
    קצרים (כלומר גרועים), כי הם מסתיימים קודם. לכן מחכים שכל הסביבות יסיימו את
    המשחק הראשון, ועוברים לסבב הבא עם זרע חדש.
    """
    results: list[dict] = []
    round_seed = seed
    while len(results) < n_games:
        n = min(n_envs, n_games - len(results))
        env = VecGame2048(n, seed=round_seed)
        first: list[dict | None] = [None] * n
        remaining = n
        while remaining:
            valid = env.valid_moves()
            actions = policy(env.boards, valid)
            _, _, finished = env.step(actions)
            for f in finished:
                if first[f["env"]] is None:
                    first[f["env"]] = f
                    remaining -= 1
        results.extend(first)  # type: ignore[arg-type]
        round_seed += 1
    return results


def record_game(policy, seed: int = 0) -> dict:
    """משחק בודד עם הקלטת כל הלוחות, לצפייה חוזרת בדו"ח."""
    g = Game2048(seed=seed)
    frames = [{"board": g.board.astype(int).tolist(), "score": 0, "action": None}]
    while not g.game_over:
        valid = g.valid_moves()
        a = int(policy(g.board[None], valid[None])[0])
        gained, ok = g.step(a)
        if not ok:  # לא אמור לקרות עם מסכת פעולות, הגנה ליתר ביטחון
            a = int(np.nonzero(valid)[0][0])
            g.step(a)
        frames.append({"board": g.board.astype(int).tolist(), "score": g.score, "action": a})
    return {"frames": frames, "score": g.score, "max_tile": g.max_tile, "moves": g.moves}


def summarize(results: list[dict]) -> dict:
    """סטטיסטיקות מסכמות של קבוצת משחקים (לדו"חות)."""
    scores = np.array([r["score"] for r in results], dtype=np.float64)
    tiles = np.array([r["max_tile"] for r in results], dtype=np.int64)
    moves = np.array([r["moves"] for r in results], dtype=np.float64)
    tile_values = [2 ** k for k in range(1, 15)]
    dist = {str(t): int((tiles == t).sum()) for t in tile_values if (tiles == t).any()}
    return {
        "games": len(results),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "moves_mean": float(moves.mean()),
        "max_tile_dist": dist,
        "reach_512": float((tiles >= 512).mean()),
        "reach_1024": float((tiles >= 1024).mean()),
        "reach_2048": float((tiles >= 2048).mean()),
        "reach_4096": float((tiles >= 4096).mean()),
    }


def random_policy(rng: np.random.Generator):
    def policy(boards, valid):
        noise = rng.random(valid.shape)
        noise[~valid] = -1.0
        return noise.argmax(axis=1)
    return policy


# ----------------------------------------------------------------------------
# לוגים
# ----------------------------------------------------------------------------

class JsonlLogger:
    """כותב שורת JSON לכל רשומה, כדי שהדו"חות יוכלו לקרוא את עקומת הלמידה."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self.t0 = time.time()

    def log(self, **record):
        record["elapsed_sec"] = round(time.time() - self.t0, 1)
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


class RecentStats:
    """שומר את תוצאות המשחקים האחרונים (חלון גולש) לצורך דיווח במהלך האימון."""

    def __init__(self, window: int = 200):
        self.window = window
        self.scores: list[int] = []
        self.tiles: list[int] = []
        self.total_episodes = 0

    def add(self, finished: list[dict]):
        for f in finished:
            self.scores.append(f["score"])
            self.tiles.append(f["max_tile"])
        self.total_episodes += len(finished)
        if len(self.scores) > self.window:
            self.scores = self.scores[-self.window:]
            self.tiles = self.tiles[-self.window:]

    def snapshot(self) -> dict:
        if not self.scores:
            return {"score_mean": 0.0, "score_max": 0, "tile_mean": 0.0, "reach_1024": 0.0, "reach_2048": 0.0}
        s = np.array(self.scores)
        t = np.array(self.tiles)
        return {
            "score_mean": float(s.mean()),
            "score_max": int(s.max()),
            "tile_mean": float(t.mean()),
            "reach_1024": float((t >= 1024).mean()),
            "reach_2048": float((t >= 2048).mean()),
        }
