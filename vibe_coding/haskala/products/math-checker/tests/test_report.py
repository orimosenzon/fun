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


# ─── compute_grade — always on a 100 scale ───────────────────────────────────

def _multi(*maxes):
    """One page holding one problem per given points_max, each fully earned."""
    probs = [dict(_pages()[0]["analysis"]["problems"][0],
                  id=str(i + 1), points_max=m, points_earned=m)
             for i, m in enumerate(maxes)]
    page = _pages()[0]
    page["analysis"]["problems"] = probs
    return [page]


def _production_case():
    """24 sub-parts at the 10-point default, 134 earned — the report a teacher
    who set the exam out of 100 actually received."""
    pages = _multi(*([10] * 24))
    probs = pages[0]["analysis"]["problems"]
    for i, pr in enumerate(probs):
        pr["points_earned"] = 10 if i < 13 else (4 if i == 13 else 0)
    return pages


def test_compute_grade_rescales_to_100():
    grade, ours = report.compute_grade(_production_case())
    assert grade == 56.0          # 134/240, not "134 out of 240"
    assert ours is True


def test_compute_grade_flags_a_teacher_supplied_scale():
    grade, ours = report.compute_grade(_multi(20, 30, 50))
    assert (grade, ours) == (100.0, False)


def test_compute_grade_uniform_default_is_flagged():
    _, ours = report.compute_grade(_multi(10, 10, 10))
    assert ours is True


def test_compute_grade_empty_is_none():
    assert report.compute_grade([]) == (None, False)
    assert report.compute_grade(_pages(points_max=None)) == (None, False)


def test_compute_grade_allows_half_points():
    grade, _ = report.compute_grade(_pages(points_max=10, points_earned=8.55))
    assert grade == 85.5


def test_grade_basis_note_says_who_chose_the_scale():
    pages = _multi(10, 10)
    assert "לא סופק מחוון" in report.grade_basis_note(pages, True)
    assert "2 סעיפים" in report.grade_basis_note(pages, True)
    assert "לא סופק מחוון" not in report.grade_basis_note(pages, False)


def test_default_points_max_matches_the_prompt():
    """The flag is only honest while the constant tracks math_core's rule."""
    from haskala_grading import math_core
    assert "**10 בדיוק**" in math_core.ANALYSIS_PROMPT
    assert report.DEFAULT_POINTS_MAX == 10.0


def test_reports_show_the_scaled_grade_not_the_raw_sum():
    """The regression that sent 134/240 to a teacher who set the exam out of 100."""
    html = report.build_result_html(_production_case(), "מבחן.pdf")
    assert "ציון כולל: 56 / 100" in html
    assert "/ 240" not in html


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
