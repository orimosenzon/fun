"""report.py — shim. בוני הדוחות עברו ל-`haskala_grading/math_report.py`.

ראה את ההסבר ב-`core.py` שלצידו. **אין לערוך כאן** — לערוך ב-lib/.
"""
from __future__ import annotations

from haskala_grading.math_report import (  # noqa: F401
    DEFAULT_POINTS_MAX,
    GRADE_SCALE,
    VERDICT_COLOR,
    VERDICT_HE,
    _fmt_pts,
    _num,
    build_result_docx,
    build_result_html,
    compute_grade,
    compute_totals,
    grade_basis_note,
)
