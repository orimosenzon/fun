"""טסטים ל-form_schema — הפונקציות הטהורות שממפות תשובת טופס לפרמטרי בדיקה.

בלי רשת, בלי Sheets, בלי קריאות למודל.

מריצים עם pytest אם הוא מותקן, אחרת:  python tests/test_form_schema.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import form_schema as fs  # noqa: E402
from haskala_grading import core  # noqa: E402


# כותרות בשמות שהטופס באמת מייצר, כולל סימני שאלה וסוגריים
HEADER = [
    "חותמת זמן",
    "כתובת אימייל",
    "שם המורה",
    "באיזה בית ספר את/ה מלמד/ת?",
    "שכבת הגיל של התלמידים",
    "לפי איזו רובריקה לבדוק?",
    "הוראות המשימה — מה התלמידים התבקשו לכתוב?",
    "שפת התרגיל",
    "התרגיל לבדיקה (העלאת קבצים)",
]


# ─── מיפוי עמודות ────────────────────────────────────────────────────────────

def test_map_columns_basic():
    cm = fs.map_columns(HEADER)
    assert cm["timestamp"] == 0
    assert cm["email"] == 1
    assert cm["teacher_name"] == 2
    assert cm["school"] == 3
    assert cm["grade_level"] == 4
    assert cm["rubric"] == 5
    assert cm["instructions"] == 6


def test_map_columns_longest_alias_wins():
    """'שפת התרגיל' ו'התרגיל לבדיקה' שניהם מכילים 'תרגיל'.

    זה בדיוק המקרה שכלל ההתאמה-הארוכה-ביותר נועד לו: התאמה ראשונה-מנצחת
    הייתה ממפה את שתי העמודות לפי הסדר בטופס."""
    cm = fs.map_columns(HEADER)
    assert cm["exercise_lang"] == 7
    assert cm["files"] == 8


def test_map_columns_survives_reordering():
    """אבישי יזיז שאלות. מיפוי לפי מיקום היה נשבר בשקט; לפי כותרת — לא."""
    shuffled = [HEADER[8], HEADER[0], HEADER[5], HEADER[1], HEADER[7]]
    cm = fs.map_columns(shuffled)
    assert cm["files"] == 0
    assert cm["timestamp"] == 1
    assert cm["rubric"] == 2
    assert cm["email"] == 3
    assert cm["exercise_lang"] == 4


def test_map_columns_survives_rewording():
    reworded = ["Timestamp", "Email Address", "מה שם בית הספר שלך?",
                "אנא צרפו את הקבצים", "כיתה"]
    cm = fs.map_columns(reworded)
    assert cm["timestamp"] == 0
    assert cm["email"] == 1
    assert cm["school"] == 2
    assert cm["files"] == 3
    assert cm["grade_level"] == 4


def test_map_columns_missing_required_raises():
    """חסרה עמודת קבצים — כישלון רועש, לא בדיקה של חצי שורה."""
    try:
        fs.map_columns(["חותמת זמן", "כתובת אימייל", "שם המורה"])
    except ValueError as e:
        assert "files" in str(e)
    else:
        raise AssertionError("expected ValueError for a missing required column")


def test_map_columns_ignores_unknown_columns():
    cm = fs.map_columns(HEADER + ["הערות פנימיות", "ציון ידני"])
    assert len(cm) == 9


# ─── חילוץ מזהי קבצים ────────────────────────────────────────────────────────

def test_extract_file_ids_forms_format():
    cell = ("https://drive.google.com/open?id=1AbCdEfGhIjKlMnOpQrStUvWxYz012345, "
            "https://drive.google.com/open?id=9ZyXwVuTsRqPoNmLkJiHgFeDcBa987654")
    assert fs.extract_file_ids(cell) == [
        "1AbCdEfGhIjKlMnOpQrStUvWxYz012345",
        "9ZyXwVuTsRqPoNmLkJiHgFeDcBa987654",
    ]


def test_extract_file_ids_alternate_url_shapes():
    """צורת ה-URL שטפסים כותב השתנתה יותר מפעם אחת."""
    cell = ("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz012345/view "
            "https://docs.google.com/document/d/9ZyXwVuTsRqPoNmLkJiHgFeDcBa987654/edit")
    assert fs.extract_file_ids(cell) == [
        "1AbCdEfGhIjKlMnOpQrStUvWxYz012345",
        "9ZyXwVuTsRqPoNmLkJiHgFeDcBa987654",
    ]


def test_extract_file_ids_preserves_order_and_dedups():
    """סדר העמודים הוא סדר ההעלאה — ערבוב שלו הופך את הדוח ללא קריא."""
    a, b = "1" + "a" * 32, "2" + "b" * 32
    base = "https://drive.google.com/open?id="
    assert fs.extract_file_ids(f"{base}{a}, {base}{b}, {base}{a}") == [a, b]


def test_extract_file_ids_bare_id():
    """מזהה שהודבק ביד, בלי URL — קורה בבדיקות ידניות."""
    a = "1" + "a" * 32
    assert fs.extract_file_ids(a) == [a]


def test_extract_file_ids_empty():
    assert fs.extract_file_ids("") == []
    assert fs.extract_file_ids(None) == []


# ─── רזולוציית רובריקה / שפה / מודל ──────────────────────────────────────────

def test_resolve_rubric_by_name():
    first = core.list_rubrics()[0]
    assert fs.resolve_rubric_choice(first["name"]) == (first["id"], False)


def test_resolve_rubric_from_upload():
    assert fs.resolve_rubric_choice(fs.RUBRIC_FROM_UPLOAD) == (None, True)


def test_resolve_rubric_unknown_falls_back():
    assert fs.resolve_rubric_choice("משהו שלא קיים") == (None, False)


def test_resolve_rubric_empty():
    assert fs.resolve_rubric_choice("") == (None, False)


def test_resolve_lang():
    assert fs.resolve_lang("אנגלית") == "en"
    assert fs.resolve_lang("עברית") == "he"
    assert fs.resolve_lang("English") == "en"
    assert fs.resolve_lang("") == core.DEFAULT_EXERCISE_LANG
    assert fs.resolve_lang("קלינגונית") == core.DEFAULT_EXERCISE_LANG


def test_resolve_model():
    assert fs.resolve_model_choice("") == core.DEFAULT_MODEL
    assert fs.resolve_model_choice("claude") == "claude"
    assert fs.resolve_model_choice("Claude (Sonnet 5)") == "claude"
    assert fs.resolve_model_choice("sonnet") == "claude"
    assert fs.resolve_model_choice("משהו אחר") == core.DEFAULT_MODEL


# ─── פענוח שורה ──────────────────────────────────────────────────────────────

def test_parse_row_full():
    cm = fs.map_columns(HEADER)
    row = ["02/08/2026 09:14:00", "dana@shamir.org.il", "דנה", "שמיר תל אביב",
           "כיתה ט", core.list_rubrics()[0]["name"], "Write a letter to a friend.",
           "אנגלית", "https://drive.google.com/open?id=" + "1" * 33]
    p = fs.parse_row(row, cm)
    assert p["email"] == "dana@shamir.org.il"
    assert p["teacher_name"] == "דנה"
    assert p["school"] == "שמיר תל אביב"
    assert p["grade_level"] == "כיתה ט"
    assert p["exercise_lang"] == "en"
    assert p["rubric_id"] == core.list_rubrics()[0]["id"]
    assert p["file_ids"] == ["1" * 33]


def test_parse_row_short_row():
    """Sheets חותך תאים ריקים בסוף — שורה קצרה מהכותרת היא המצב הרגיל."""
    cm = fs.map_columns(HEADER)
    p = fs.parse_row(["02/08/2026 09:14:00", "dana@shamir.org.il"], cm)
    assert p["email"] == "dana@shamir.org.il"
    assert p["school"] == ""
    assert p["file_ids"] == []
    assert p["model_key"] == core.DEFAULT_MODEL


# ─── בניית ה"שאלה" ───────────────────────────────────────────────────────────

def test_build_question_includes_context():
    q = fs.build_question({"instructions": "Write a letter.",
                           "grade_level": "כיתה ז", "school": "שמיר"})
    assert q.startswith("Write a letter.")   # ניסוח המורה נשאר ראשון
    assert "כיתה ז" in q
    assert "שמיר" in q


def test_build_question_instructions_only():
    assert fs.build_question({"instructions": "Write a letter."}) == "Write a letter."


def test_build_question_empty():
    assert fs.build_question({}) == ""


if __name__ == "__main__":
    # ריצה בלי pytest: מריץ כל test_* בקובץ ומדווח.
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
