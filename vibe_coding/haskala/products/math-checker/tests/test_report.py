"""טסטים לבוני הדוחות — כולל שני הבאגים מ-2026-06-29 (esc על מספר, escaping של LaTeX)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import report  # noqa: E402


def _pages(**overrides):
    problem = {
        "id": "1", "topic": "אלגברה", "statement_latex": "x^2-6x+10",
        "student_steps": [{"latex": "x<5", "ok": True, "comment": ""}],
        "final_answer_latex": "x=3", "verdict": "correct",
        "points_max": 10, "points_earned": 8.5,
        "score_suggestion": "טעות קטנה בסוף", "feedback": "עבודה יפה",
    }
    problem.update(overrides)
    return [{"page": 1, "rotation_applied": 0, "image_b64": "",
             "analysis": {"page_summary": "עמוד ראשון", "has_diagram": False,
                          "diagram_description": "", "problems": [problem]}}]


# ─── _num / _fmt_pts ─────────────────────────────────────────────────────────

def test_num_coercions():
    assert report._num(7) == 7.0
    assert report._num("7.5") == 7.5
    assert report._num(None) is None
    assert report._num("") is None
    assert report._num("abc") is None


def test_fmt_pts_drops_trailing_zero():
    assert report._fmt_pts(7.0) == "7"
    assert report._fmt_pts(7.5) == "7.5"
    assert report._fmt_pts(None) == "—"


# ─── compute_totals ──────────────────────────────────────────────────────────

def test_compute_totals():
    earned, total = report.compute_totals(_pages())
    assert (earned, total) == (8.5, 10.0)


def test_compute_totals_skips_missing_max():
    earned, total = report.compute_totals(_pages(points_max=None))
    assert (earned, total) == (0.0, 0.0)


def test_compute_totals_empty():
    assert report.compute_totals([]) == (0.0, 0.0)
    assert report.compute_totals(None) == (0.0, 0.0)


# ─── build_result_html ───────────────────────────────────────────────────────

def test_html_handles_numeric_points():
    """הבאג מ-29/6: esc(8.5) קרס על float. חייב לעבוד עם מספרים בכל שדה."""
    html = report.build_result_html(_pages(), "תרגיל.jpg")
    assert "8.5 / 10" in html


def test_html_escapes_latex_lt():
    """הבאג השני מ-29/6: x<5 חייב לצאת escaped כדי לא להישבר כתג HTML."""
    html = report.build_result_html(_pages(), "ת.jpg")
    assert "x&lt;5" in html
    assert "<x" not in html.replace("<xmp", "")  # אין תג פתוח שנבלע


def test_html_total_and_approval_badges():
    draft = report.build_result_html(_pages(), "ת.jpg", approved=False)
    assert "טיוטה — טרם אושר" in draft
    ok = report.build_result_html(_pages(), "ת.jpg", approved=True)
    assert 'אושר ע"י המורה' in ok


def test_html_no_totals_box_when_no_points():
    html = report.build_result_html(_pages(points_max=None), "ת.jpg")
    assert "ציון כולל" not in html


def test_html_escapes_filename():
    html = report.build_result_html(_pages(), '<script>alert(1)</script>.jpg')
    assert "<script>alert" not in html


# ─── build_result_docx ───────────────────────────────────────────────────────

def test_docx_builds_valid_zip():
    data = report.build_result_docx(_pages(), "תרגיל.jpg")
    assert data[:2] == b"PK"  # docx הוא zip
    assert len(data) > 1000


def test_docx_handles_numeric_points():
    data = report.build_result_docx(_pages(points_earned=7, points_max=10), "ת.jpg")
    assert data[:2] == b"PK"
