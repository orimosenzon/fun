"""טסטים לפונקציות הטהורות של poller.py (בלי רשת/Classroom)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poller  # noqa: E402


# ─── ext_for ─────────────────────────────────────────────────────────────────

def test_ext_for_mime_wins():
    assert poller.ext_for("application/pdf", "scan.weird") == ".pdf"
    assert poller.ext_for("image/jpeg", None) == ".jpg"
    assert poller.ext_for("image/png", "a.png") == ".png"


def test_ext_for_title_fallback():
    assert poller.ext_for("application/octet-stream", "photo.JPEG") == ".jpg"
    assert poller.ext_for(None, "scan.pdf") == ".pdf"


def test_ext_for_unsupported():
    assert poller.ext_for("application/vnd.google-apps.document", "doc") is None
    assert poller.ext_for(None, "essay.docx") is None


# ─── merge_pages ─────────────────────────────────────────────────────────────

def test_merge_pages_renumbers():
    a = [{"page": 1, "analysis": {}}, {"page": 2, "analysis": {}}]
    b = [{"page": 1, "analysis": {}}]
    merged = poller.merge_pages([a, b])
    assert [p["page"] for p in merged] == [1, 2, 3]
    assert a[0]["page"] == 1  # המקור לא שונה


def test_merge_pages_empty():
    assert poller.merge_pages([]) == []


# ─── scale_grade ─────────────────────────────────────────────────────────────

def test_scale_grade_scales_to_max_points():
    assert poller.scale_grade(8, 10, 100) == 80.0
    assert poller.scale_grade(40, 110, 100) == 36.4


def test_scale_grade_identity_when_same_scale():
    assert poller.scale_grade(8.5, 10, 10) == 8.5


def test_scale_grade_none_cases():
    assert poller.scale_grade(5, 0, 100) is None      # רובריקה בלי ניקוד
    assert poller.scale_grade(5, 10, None) is None    # מטלה בלי maxPoints


# ─── safe_name ───────────────────────────────────────────────────────────────

def test_safe_name_hebrew_kept():
    assert poller.safe_name("דנה כהן") == "דנה_כהן"


def test_safe_name_strips_dangerous():
    assert "/" not in poller.safe_name("a/b\\c:d")
    assert poller.safe_name("") == "unnamed"


# ─── watched works ───────────────────────────────────────────────────────────

def test_watched_works_all():
    works = [{"id": "1", "title": "תרגיל 1"}, {"id": "2", "title": "מבחן"}]
    assert poller._watched_works(works, "all") == works


def test_watched_works_by_title_case_insensitive():
    works = [{"id": "1", "title": "Parabola HW"}, {"id": "2", "title": "מבחן"}]
    out = poller._watched_works(works, ["parabola hw"])
    assert [w["id"] for w in out] == ["1"]


def test_watched_works_by_id():
    works = [{"id": "abc", "title": "x"}, {"id": "def", "title": "y"}]
    out = poller._watched_works(works, ["def"])
    assert [w["id"] for w in out] == ["def"]
