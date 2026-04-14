"""Shared linear algebra utilities for all views."""
import numpy as np
import sympy as sp


# ── 2D transformation helpers ──────────────────────────────────────────────────

def lerp_matrix(T, t):
    """Interpolate: (1-t)*I + t*T  (t=0 → identity, t=1 → T)."""
    I = np.eye(2)
    return I * (1 - t) + np.array(T, dtype=float) * t


def grid_lines(T, bound=4, n=9):
    """
    Return [(xs, ys), ...] for horizontal and vertical grid lines
    transformed by 2×2 matrix T.
    T is linear → straight lines stay straight → just transform endpoints.
    """
    T = np.array(T, dtype=float)
    lines = []
    vals = np.linspace(-bound, bound, n)
    for k in vals:
        # Horizontal: y = k
        p1, p2 = T @ [-bound, k], T @ [bound, k]
        lines.append(([p1[0], p2[0]], [p1[1], p2[1]]))
        # Vertical: x = k
        p1, p2 = T @ [k, -bound], T @ [k, bound]
        lines.append(([p1[0], p2[0]], [p1[1], p2[1]]))
    return lines


def apply_matrix_2d(T, points):
    """Apply 2×2 matrix T to a list of [x,y] points."""
    T = np.array(T, dtype=float)
    return [(T @ np.array(p)).tolist() for p in points]


def eigeninfo(T):
    """Return dict with det, trace, real eigenpairs."""
    T = np.array(T, dtype=float)
    det = float(np.linalg.det(T))
    tr = float(np.trace(T))
    vals, vecs = np.linalg.eig(T)
    real_pairs = []
    for i in range(len(vals)):
        if abs(vals[i].imag) < 1e-9:
            v = vecs[:, i].real
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v = v / norm
            real_pairs.append({"val": float(vals[i].real), "vec": v.tolist()})
    return {"det": det, "trace": tr, "pairs": real_pairs,
            "rank": int(np.linalg.matrix_rank(T))}


# ── Gaussian elimination ────────────────────────────────────────────────────────

def gaussian_steps(rows):
    """
    Gauss-Jordan elimination using exact sympy arithmetic.
    Returns list of step dicts:
      { matrix: [[str,...]], op: str, pivot_row: int|None, pivot_col: int|None }
    """
    mat = sp.Matrix(rows).applyfunc(
        lambda x: sp.Rational(x).limit_denominator(10_000)
    )
    m, n = mat.shape

    steps = [_step(mat, "מטריצה התחלתית")]
    pivot_row = 0

    for col in range(n):
        if pivot_row >= m:
            break

        # Find non-zero pivot in this column
        found = next((r for r in range(pivot_row, m) if mat[r, col] != 0), None)
        if found is None:
            continue

        if found != pivot_row:
            mat.row_swap(found, pivot_row)
            steps.append(_step(mat, f"R{pivot_row+1} ↔ R{found+1}",
                               pivot_row, col))

        pval = mat[pivot_row, col]
        if pval != 1:
            mat[pivot_row, :] /= pval
            steps.append(_step(mat, f"R{pivot_row+1} ÷ {_fmt(pval)}",
                               pivot_row, col))

        for row in range(m):
            if row == pivot_row:
                continue
            factor = mat[row, col]
            if factor == 0:
                continue
            mat[row, :] -= factor * mat[pivot_row, :]
            sign = "−" if factor > 0 else "+"
            steps.append(_step(mat,
                               f"R{row+1} {sign} {_fmt(abs(factor))}·R{pivot_row+1}",
                               pivot_row, col))

        pivot_row += 1

    return steps


def _step(mat, op, pivot_row=None, pivot_col=None):
    return {
        "matrix": [[str(mat[i, j]) for j in range(mat.cols)]
                   for i in range(mat.rows)],
        "op": op,
        "pivot_row": pivot_row,
        "pivot_col": pivot_col,
    }


def _fmt(val):
    """Format a sympy value as a readable string."""
    s = str(val)
    return s if "/" not in s else f"({s})"


# ── 3-D span helpers ────────────────────────────────────────────────────────────

def span_geometry(vectors, t_range=(-2.5, 2.5), n=25):
    """
    Compute geometry for the span of a list of 3-D vectors.
    Returns (kind, data) where kind ∈ {'none','line','plane','space'}.
    data is a dict of x/y/z lists suitable for Plotly.
    """
    vecs = [np.array(v, dtype=float) for v in vectors]
    nonzero = [v for v in vecs if np.linalg.norm(v) > 1e-10]
    if not nonzero:
        return "none", {}

    # Gram-Schmidt to find independent directions
    basis = []
    for v in nonzero:
        q = v.copy()
        for b in basis:
            q -= np.dot(q, b) * b
        nrm = np.linalg.norm(q)
        if nrm > 1e-10:
            basis.append(q / nrm)
        if len(basis) == 3:
            break

    dim = len(basis)
    lo, hi = t_range

    if dim == 1:
        b = nonzero[0]
        t = np.linspace(lo, hi, n)
        pts = np.outer(t, b)
        return "line", {"x": pts[:, 0].tolist(),
                        "y": pts[:, 1].tolist(),
                        "z": pts[:, 2].tolist()}

    if dim == 2:
        v1, v2 = nonzero[0], nonzero[1]
        s = np.linspace(lo, hi, n)
        S, T = np.meshgrid(s, s)
        X = S * v1[0] + T * v2[0]
        Y = S * v1[1] + T * v2[1]
        Z = S * v1[2] + T * v2[2]
        return "plane", {"x": X.tolist(), "y": Y.tolist(), "z": Z.tolist()}

    return "space", {}


# ── 3-D surface ─────────────────────────────────────────────────────────────────

def eval_surface(expr_str, bound=4, n=60):
    """
    Evaluate a function f(x,y) over a grid.
    expr_str: e.g. 'sin(x)*cos(y)', 'x**2 - y**2'
    Returns (X, Y, Z) numpy arrays or raises ValueError.
    """
    import math
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed.update({"x": 0, "y": 0, "abs": abs, "round": round})

    xs = np.linspace(-bound, bound, n)
    ys = np.linspace(-bound, bound, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)

    for i in range(n):
        for j in range(n):
            env = dict(allowed)
            env["x"] = float(X[i, j])
            env["y"] = float(Y[i, j])
            try:
                Z[i, j] = eval(expr_str, {"__builtins__": {}}, env)  # noqa: S307
            except Exception:
                Z[i, j] = float("nan")

    return X, Y, Z
