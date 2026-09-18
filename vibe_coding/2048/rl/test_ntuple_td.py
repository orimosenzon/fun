"""
בדיקות למימוש ה-numba של TD על afterstates: ההחלקה, המאפיינים והמדיניות מול הגרסאות
הווקטוריות ב-NumPy (game2048.move_boards ו-feature_indices_np), ומדידת מהירות.

הרצה:  python rl/test_ntuple_td.py
"""

from __future__ import annotations

import time

import numpy as np

from game2048 import move_boards, valid_moves
from ntuple_td import (
    FEATS, N_FEATS, N_TABLES, OFFSETS, TABLE_SIZE,
    _features, _slide, feature_indices_np, make_policy, play_greedy, seed_numba, train_chunk, values_np,
)


def random_boards(rng, n, max_exp=12):
    b = rng.integers(0, max_exp + 1, size=(n, 4, 4)).astype(np.uint8)
    b[rng.random((n, 4, 4)) < 0.4] = 0
    return b


def test_slide_matches_move_boards():
    rng = np.random.default_rng(1)
    boards = random_boards(rng, 3000)
    out = np.zeros(16, dtype=np.uint8)
    for a in range(4):
        ref, ref_sc, ref_ch = move_boards(boards, np.full(len(boards), a))
        for i in range(len(boards)):
            sc, ch = _slide(boards[i].reshape(16), a, out)
            assert ch == ref_ch[i], (a, boards[i])
            assert sc == ref_sc[i], (a, boards[i], sc, ref_sc[i])
            assert np.array_equal(out.reshape(4, 4), ref[i]), (a, boards[i])
    print("slide: ok (12,000 moves)")


def test_features_match_numpy():
    rng = np.random.default_rng(2)
    boards = random_boards(rng, 2000, max_exp=15)
    ref = feature_indices_np(boards)
    out = np.zeros(N_FEATS, dtype=np.int64)
    for i in range(len(boards)):
        _features(boards[i].reshape(16), FEATS, OFFSETS, out)
        assert np.array_equal(out, ref[i])
    assert ref.min() >= 0 and ref.max() < N_TABLES * TABLE_SIZE
    # סימטריה: לוח מסובב נותן את אותה קבוצת אינדקסים (בסדר אחר), ולכן אותו V
    table = rng.random(N_TABLES * TABLE_SIZE, dtype=np.float32)
    for b in boards[:50]:
        v0 = values_np(table, b[None])[0]
        for k in range(4):
            r = np.rot90(b, k)
            assert np.isclose(values_np(table, r[None])[0], v0, rtol=1e-5)
            assert np.isclose(values_np(table, np.fliplr(r)[None])[0], v0, rtol=1e-5)
    print("features: ok, symmetric sampling invariant under the 8 symmetries")


def test_policy_and_train_smoke():
    seed_numba(0)
    table = np.zeros(N_TABLES * TABLE_SIZE, dtype=np.float32)
    board = np.zeros(16, dtype=np.uint8)
    board[0] = 1
    board[5] = 1
    prev_idx = np.zeros(N_FEATS, dtype=np.int64)
    state = np.zeros(3, dtype=np.int64)
    n = 300_000
    eps = np.zeros(n // 8 + 16, dtype=np.int64)
    tiles = np.zeros_like(eps)
    mv = np.zeros_like(eps)
    train_chunk(table, FEATS, OFFSETS, np.float32(0.1 / N_FEATS), 20_000, board, prev_idx, state, eps, tiles, mv)  # קומפילציה
    t0 = time.time()
    done, n_done, s_delta, n_upd = train_chunk(table, FEATS, OFFSETS, np.float32(0.1 / N_FEATS), n, board, prev_idx, state, eps, tiles, mv)
    dt = time.time() - t0
    print(f"train: {done:,} moves in {dt:.2f}s = {done / dt:,.0f} moves/s, {n_done} episodes, mean score {eps[:n_done].mean():.0f}, "
          f"visited {np.count_nonzero(table):,} weights")
    assert n_done > 10 and eps[:n_done].mean() > 500

    # המדיניות הווקטורית מסכימה עם החמדן הסדרתי: אותו argmax על לוחות אקראיים
    policy = make_policy(table)
    rng = np.random.default_rng(3)
    boards = random_boards(rng, 500, max_exp=8)
    valid = valid_moves(boards)
    keep = valid.any(axis=1)
    boards, valid = boards[keep], valid[keep]
    acts = policy(boards, valid)
    out = np.zeros(16, dtype=np.uint8)
    idx = np.zeros(N_FEATS, dtype=np.int64)
    mism = 0
    for i, b in enumerate(boards):
        best_a, best_v = -1, -1e30
        for a in range(4):
            sc, ch = _slide(b.reshape(16), a, out)
            if not ch:
                continue
            _features(out, FEATS, OFFSETS, idx)
            v = sc + float(table[idx].sum())
            if v > best_v:
                best_v, best_a = v, a
        mism += int(best_a != acts[i])
    print(f"policy: numpy vs numba argmax mismatches {mism}/{len(boards)}")
    assert mism <= 2  # שוויון בערכי float יכול להישבר אחרת בין שני המימושים

    sc = np.zeros(20, dtype=np.int64)
    ti = np.zeros(20, dtype=np.int64)
    m = np.zeros(20, dtype=np.int64)
    t0 = time.time()
    play_greedy(table, FEATS, OFFSETS, 20, sc, ti, m)
    print(f"greedy eval: 20 games, mean score {sc.mean():.0f}, {m.sum() / (time.time() - t0):,.0f} moves/s")


if __name__ == "__main__":
    test_slide_matches_move_boards()
    test_features_match_numpy()
    test_policy_and_train_smoke()
    print("all ok")
