"""
בדיקות למנוע המשחק: השוואה מול מימוש איטי ופשוט של החוקים המקוריים.

הרצה:  python -m pytest rl/test_game2048.py -q   (או פשוט python rl/test_game2048.py)
"""

import numpy as np

from game2048 import (
    Game2048,
    VecGame2048,
    _slide_row_left,
    move_boards,
    new_boards,
    spawn_tiles,
    valid_moves,
)


def slow_move(board: np.ndarray, action: int):
    """מימוש ייחוס איטי: מסובבים כך שהמהלך הופך ל'שמאלה', מזיזים שורה-שורה, מסובבים חזרה."""
    b = board.astype(int).copy()
    # מספר סיבובים נגד כיוון השעון כדי להפוך את הכיוון לשמאלה
    k = {3: 0, 0: 1, 1: 2, 2: 3}[action]
    b = np.rot90(b, k)
    score = 0
    out = np.zeros_like(b)
    for r in range(4):
        row, sc = _slide_row_left(list(b[r]))
        out[r] = row
        score += sc
    out = np.rot90(out, -k)
    return out.astype(np.uint8), score, not np.array_equal(out, board)


def test_row_rules():
    assert _slide_row_left([1, 1, 0, 0]) == ([2, 0, 0, 0], 4)
    assert _slide_row_left([1, 1, 1, 1]) == ([2, 2, 0, 0], 8)  # לא ממזגים פעמיים באותו מהלך
    assert _slide_row_left([1, 0, 1, 2]) == ([2, 2, 0, 0], 4)
    assert _slide_row_left([2, 1, 1, 0]) == ([2, 2, 0, 0], 4)
    assert _slide_row_left([0, 0, 0, 3]) == ([3, 0, 0, 0], 0)
    assert _slide_row_left([1, 2, 1, 2]) == ([1, 2, 1, 2], 0)
    assert _slide_row_left([0, 1, 1, 1]) == ([2, 1, 0, 0], 4)  # ממזגים את שני האריחים הקרובים לכיוון


def test_move_matches_slow_reference():
    rng = np.random.default_rng(1)
    boards = rng.integers(0, 6, size=(2000, 4, 4)).astype(np.uint8)
    boards[rng.random((2000, 4, 4)) < 0.3] = 0
    for a in range(4):
        fast, sc, ch = move_boards(boards, np.full(2000, a))
        for i in range(2000):
            sb, ssc, sch = slow_move(boards[i], a)
            assert np.array_equal(fast[i], sb), (a, boards[i], fast[i], sb)
            assert sc[i] == ssc
            assert ch[i] == sch


def test_valid_moves_matches_changed():
    rng = np.random.default_rng(2)
    boards = rng.integers(0, 5, size=(1000, 4, 4)).astype(np.uint8)
    boards[rng.random((1000, 4, 4)) < 0.2] = 0
    vm = valid_moves(boards)
    for a in range(4):
        _, _, ch = move_boards(boards, np.full(1000, a))
        assert np.array_equal(vm[:, a], ch)


def test_spawn_only_fills_empty_cells_with_2_or_4():
    rng = np.random.default_rng(3)
    boards = new_boards(500, rng)
    assert ((boards > 0).sum(axis=(1, 2)) == 2).all()
    assert set(np.unique(boards)).issubset({0, 1, 2})
    before = boards.copy()
    spawn_tiles(boards, rng)
    diff = boards != before
    assert (diff.sum(axis=(1, 2)) == 1).all()
    assert (before[diff] == 0).all()
    # התפלגות 2/4 בערך 90/10
    frac4 = (boards[diff] == 2).mean()
    assert 0.05 < frac4 < 0.16


def test_vec_env_runs_random_games():
    env = VecGame2048(64, seed=4)
    rng = np.random.default_rng(5)
    total_finished = []
    for _ in range(3000):
        vm = env.valid_moves()
        # מהלך אקראי חוקי
        noise = rng.random(vm.shape)
        noise[~vm] = -1
        actions = noise.argmax(axis=1)
        rewards, dones, finished = env.step(actions)
        assert (rewards >= 0).all()
        total_finished += finished
    assert len(total_finished) > 100
    scores = np.array([f["score"] for f in total_finished])
    tiles = np.array([f["max_tile"] for f in total_finished])
    assert 500 < scores.mean() < 2000  # משחק אקראי: בערך אלף נקודות
    assert tiles.max() >= 128


def test_single_game_score_consistency():
    g = Game2048(seed=6)
    total = 0
    while not g.game_over:
        vm = g.valid_moves()
        a = int(g.rng.choice(np.nonzero(vm)[0]))
        gained, ok = g.step(a, record=True)
        assert ok
        total += gained
    assert total == g.score
    assert len(g.history) == g.moves
    # לוח סופי: אין מהלך חוקי
    assert not g.valid_moves().any()


if __name__ == "__main__":
    import time

    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            fn()
            print(f"{name}: OK ({time.time() - t:.2f}s)")
    # מדידת מהירות
    env = VecGame2048(256, seed=0)
    rng = np.random.default_rng(0)
    t = time.time()
    steps = 2000
    for _ in range(steps):
        vm = env.valid_moves()
        noise = rng.random(vm.shape)
        noise[~vm] = -1
        env.step(noise.argmax(axis=1))
    dt = time.time() - t
    print(f"speed: {steps * 256 / dt:,.0f} transitions/sec (256 envs)")
