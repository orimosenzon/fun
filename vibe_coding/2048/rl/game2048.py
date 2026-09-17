"""
מנוע 2048 מהיר ומקבילי (NumPy) לצורכי למידת חיזוק.

הרעיון המרכזי: לוח 4x4 מיוצג כמערך של מעריכים (exponent) בבסיס 2:
    0 = משבצת ריקה,  1 = אריח 2,  2 = אריח 4,  ...  11 = אריח 2048
כל שורה בלוח היא ארבעה מספרים בין 0 ל-15, ולכן ניתן לקודד אותה במספר
שלם של 16 סיביות (4 סיביות לכל משבצת). יש בסך הכל 65,536 שורות אפשריות,
ולכן אפשר לחשב מראש, פעם אחת, מה קורה לכל שורה אפשרית כשמזיזים אותה
שמאלה או ימינה: מה השורה החדשה, כמה נקודות התקבלו, והאם השורה השתנתה.

אחרי החישוב המקדים, מהלך שלם על אלפי לוחות במקביל הוא רק כמה פעולות
אינדוקס במערכי NumPy, בלי לולאות פייתון. זה מה שמאפשר לאסוף מיליוני
מהלכים בשעה, וזה הבסיס לשני האלגוריתמים (DQN ו-Actor-Critic).

מוסכמת הפעולות (זהה לממשק הווב ולמקור):
    0 = למעלה,  1 = ימינה,  2 = למטה,  3 = שמאלה
"""

from __future__ import annotations

import numpy as np

ACTION_NAMES = ["up", "right", "down", "left"]
ACTION_NAMES_HE = ["למעלה", "ימינה", "למטה", "שמאלה"]

# משקלות לקידוד שורה של 4 מעריכים למספר של 16 סיביות (המשבצת השמאלית היא הבכירה)
_ROW_WEIGHTS = np.array([4096, 256, 16, 1], dtype=np.int64)
_ROW_SHIFTS = np.array([12, 8, 4, 0], dtype=np.int64)


def _slide_row_left(row: list[int]) -> tuple[list[int], int]:
    """מזיז שורה אחת שמאלה לפי חוקי המשחק המקורי ומחזיר (שורה חדשה, ניקוד).

    חוקי המיזוג של המקור: הסתכלות מהצד שאליו זזים, אריח שמוזג פעם אחת לא
    ממוזג שוב באותו מהלך, ולכן [2,2,2,2] הופך ל-[4,4,0,0] ולא ל-[8,0,0,0].
    """
    tiles = [v for v in row if v != 0]
    out: list[int] = []
    score = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = min(tiles[i] + 1, 15)  # 15 = 32768, תקרת הקידוד (לא נגיע לשם בפועל)
            out.append(merged)
            score += 2 ** merged
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (4 - len(out))
    return out, score


def _build_tables():
    """בונה את טבלאות החיפוש לכל 65,536 השורות האפשריות, לשמאל ולימין."""
    left = np.zeros(65536, dtype=np.uint16)
    right = np.zeros(65536, dtype=np.uint16)
    score_l = np.zeros(65536, dtype=np.int32)
    score_r = np.zeros(65536, dtype=np.int32)
    for code in range(65536):
        row = [(code >> s) & 15 for s in (12, 8, 4, 0)]
        new, sc = _slide_row_left(row)
        left[code] = (new[0] << 12) | (new[1] << 8) | (new[2] << 4) | new[3]
        score_l[code] = sc
        rev = row[::-1]
        new_r, sc_r = _slide_row_left(rev)
        new_r = new_r[::-1]
        right[code] = (new_r[0] << 12) | (new_r[1] << 8) | (new_r[2] << 4) | new_r[3]
        score_r[code] = sc_r
    codes = np.arange(65536, dtype=np.uint16)
    changed_l = left != codes
    changed_r = right != codes
    return left, right, score_l, score_r, changed_l, changed_r


TABLE_LEFT, TABLE_RIGHT, SCORE_LEFT, SCORE_RIGHT, CHANGED_LEFT, CHANGED_RIGHT = _build_tables()


def encode_rows(boards: np.ndarray) -> np.ndarray:
    """(N,4,4) מעריכים -> (N,4) קודי שורות של 16 סיביות."""
    return (boards.astype(np.int64) @ _ROW_WEIGHTS).astype(np.int64)


def decode_rows(codes: np.ndarray) -> np.ndarray:
    """(N,4) קודי שורות -> (N,4,4) מעריכים (uint8)."""
    return ((codes[..., None].astype(np.int64) >> _ROW_SHIFTS) & 15).astype(np.uint8)


def move_boards(boards: np.ndarray, actions: np.ndarray):
    """מבצע מהלך אחד על קבוצת לוחות, בלי הוספת אריח חדש.

    boards : (N,4,4) uint8 מעריכים
    actions: (N,) מספרים 0..3
    מחזיר (לוחות חדשים, ניקוד לכל לוח, האם הלוח השתנה)
    """
    n = boards.shape[0]
    out = np.empty_like(boards)
    score = np.zeros(n, dtype=np.int32)
    changed = np.zeros(n, dtype=bool)
    for a in range(4):
        idx = np.nonzero(actions == a)[0]
        if idx.size == 0:
            continue
        b = boards[idx]
        if a in (0, 2):  # למעלה / למטה: עובדים על העמודות, כלומר על הלוח המשוחלף
            b = b.transpose(0, 2, 1)
        codes = encode_rows(b)
        if a in (0, 3):  # למעלה = שמאלה על הלוח המשוחלף
            new_codes = TABLE_LEFT[codes]
            sc = SCORE_LEFT[codes].sum(axis=1)
        else:
            new_codes = TABLE_RIGHT[codes]
            sc = SCORE_RIGHT[codes].sum(axis=1)
        nb = decode_rows(new_codes)
        if a in (0, 2):
            nb = nb.transpose(0, 2, 1)
        out[idx] = nb
        score[idx] = sc
        changed[idx] = (new_codes != codes).any(axis=1)
    return out, score, changed


def valid_moves(boards: np.ndarray) -> np.ndarray:
    """(N,4,4) -> (N,4) bool: אילו מהלכים משנים את הלוח (כלומר חוקיים)."""
    codes = encode_rows(boards)
    codes_t = encode_rows(boards.transpose(0, 2, 1))
    up = CHANGED_LEFT[codes_t].any(axis=1)
    right = CHANGED_RIGHT[codes].any(axis=1)
    down = CHANGED_RIGHT[codes_t].any(axis=1)
    left = CHANGED_LEFT[codes].any(axis=1)
    return np.stack([up, right, down, left], axis=1)


def spawn_tiles(boards: np.ndarray, rng: np.random.Generator, mask: np.ndarray | None = None) -> None:
    """מוסיף אריח אקראי (2 בהסתברות 0.9, 4 בהסתברות 0.1) למשבצת ריקה, במקום.

    mask: (N,) bool, אילו לוחות לעדכן (ברירת מחדל: כולם). לוח בלי משבצת ריקה לא משתנה.
    """
    n = boards.shape[0]
    if mask is None:
        mask = np.ones(n, dtype=bool)
    flat = boards.reshape(n, 16)
    empty = flat == 0
    has_empty = empty.any(axis=1) & mask
    if not has_empty.any():
        return
    # בחירה אחידה בין המשבצות הריקות: מגרילים מספר לכל משבצת ולוקחים את הגדול מבין הריקות
    noise = rng.random((n, 16))
    noise[~empty] = -1.0
    cell = noise.argmax(axis=1)
    value = np.where(rng.random(n) < 0.9, 1, 2).astype(np.uint8)
    rows = np.nonzero(has_empty)[0]
    flat[rows, cell[rows]] = value[rows]


def new_boards(n: int, rng: np.random.Generator) -> np.ndarray:
    """יוצר n לוחות התחלתיים, כל אחד עם שני אריחים אקראיים (כמו במקור)."""
    boards = np.zeros((n, 4, 4), dtype=np.uint8)
    spawn_tiles(boards, rng)
    spawn_tiles(boards, rng)
    return boards


def max_tile(boards: np.ndarray) -> np.ndarray:
    """(N,4,4) -> (N,) ערך האריח הגדול ביותר בכל לוח (2^k, לא המעריך)."""
    return (2 ** boards.reshape(boards.shape[0], 16).max(axis=1).astype(np.int64))


def board_to_text(board: np.ndarray) -> str:
    """הדפסה ידידותית של לוח יחיד (4,4)."""
    lines = []
    for row in board:
        lines.append(" ".join(f"{(2 ** int(v)) if v else '.':>5}" for v in row))
    return "\n".join(lines)


class VecGame2048:
    """
    N משחקי 2048 שרצים במקביל, עם איפוס אוטומטי של משחקים שהסתיימו.

    זו הסביבה שבה משתמשים שני האלגוריתמים. בכל צעד מקבלים וקטור פעולות
    (אחת לכל משחק) ומחזירים: תגמול, האם המשחק הסתיים, ומידע על משחקים שנגמרו.

    מהלך לא חוקי (שלא משנה את הלוח) לא מקדם את המשחק ומקבל תגמול 0.
    הסוכנים מסתירים מהלכים כאלה בעזרת valid_moves ולכן זה כמעט לא קורה בפועל.
    """

    def __init__(self, n: int, seed: int | None = None, reward_scale: float = 1.0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.reward_scale = reward_scale
        self.boards = new_boards(n, self.rng)
        self.scores = np.zeros(n, dtype=np.int64)
        self.moves = np.zeros(n, dtype=np.int64)
        self.episodes_done = 0

    def reset_all(self) -> np.ndarray:
        self.boards = new_boards(self.n, self.rng)
        self.scores[:] = 0
        self.moves[:] = 0
        return self.boards

    def valid_moves(self) -> np.ndarray:
        return valid_moves(self.boards)

    def step(self, actions: np.ndarray):
        """
        מבצע צעד בכל המשחקים.
        מחזיר:
            rewards   (N,) float32  ניקוד שהתקבל במהלך (כפול reward_scale)
            dones     (N,) bool     האם המשחק הסתיים אחרי המהלך
            finished  רשימת dict עם score / max_tile / moves של משחקים שהסתיימו בצעד זה
        לוחות של משחקים שהסתיימו מאופסים אוטומטית; self.boards תמיד מכיל מצב חי.
        """
        actions = np.asarray(actions, dtype=np.int64)
        nb, gained, changed = move_boards(self.boards, actions)
        self.boards = np.where(changed[:, None, None], nb, self.boards)
        self.scores += gained
        self.moves += changed
        spawn_tiles(self.boards, self.rng, mask=changed)
        dones = ~valid_moves(self.boards).any(axis=1)
        finished = []
        if dones.any():
            idx = np.nonzero(dones)[0]
            mt = max_tile(self.boards[idx])
            for k, i in enumerate(idx):
                finished.append({"env": int(i), "score": int(self.scores[i]), "max_tile": int(mt[k]), "moves": int(self.moves[i])})
            self.episodes_done += len(idx)
            fresh = new_boards(len(idx), self.rng)
            self.boards[idx] = fresh
            self.scores[idx] = 0
            self.moves[idx] = 0
        rewards = gained.astype(np.float32) * self.reward_scale
        return rewards, dones, finished


class Game2048:
    """משחק בודד, נוח לבדיקות, להערכה ולהקלטת משחקים (לצורך צפייה חוזרת)."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.board = new_boards(1, self.rng)[0]
        self.score = 0
        self.moves = 0
        self.history: list[dict] = []
        return self.board

    def valid_moves(self) -> np.ndarray:
        return valid_moves(self.board[None])[0]

    @property
    def game_over(self) -> bool:
        return not self.valid_moves().any()

    @property
    def max_tile(self) -> int:
        return int(max_tile(self.board[None])[0])

    def step(self, action: int, record: bool = False) -> tuple[int, bool]:
        """מבצע מהלך. מחזיר (ניקוד שהתקבל, האם המהלך היה חוקי)."""
        nb, gained, changed = move_boards(self.board[None], np.array([action]))
        if not changed[0]:
            return 0, False
        self.board = nb[0]
        self.score += int(gained[0])
        self.moves += 1
        spawn_tiles(self.board[None], self.rng)
        if record:
            self.history.append({"action": int(action), "board": self.board.astype(int).tolist(), "score": self.score})
        return int(gained[0]), True


if __name__ == "__main__":
    # בדיקה קטנה ומהירה: משחק אקראי אחד והדפסת הלוח הסופי
    g = Game2048(seed=0)
    while not g.game_over:
        vm = g.valid_moves()
        a = g.rng.choice(np.nonzero(vm)[0])
        g.step(int(a))
    print(board_to_text(g.board))
    print("score", g.score, "max tile", g.max_tile, "moves", g.moves)
