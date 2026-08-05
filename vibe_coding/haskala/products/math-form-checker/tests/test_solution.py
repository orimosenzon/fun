"""טסטים לפתרון בית הספר — השאלה שאבישי הוסיף לטופס המתמטיקה ב-2026-08-05.

בלי רשת, בלי Drive, בלי קריאות למודל. כל מה שנבדק כאן הוא לוגיקה טהורה:
איזו עמודה ממופה לפתרון, מה נחשב "מצורף בתיקייה", אילו קבצים בתיקייה הם פתרון
ולא עבודת תלמיד, ומה בדיוק מגיע למודל.

שני הטסטים החשובים כאן הם שני כשלים שקטים — כאלה שלא היו זורקים שגיאה, רק
מחזירים ציונים שגויים בלי שאף אחד ידע:

  1. עמודת ההעלאה של עבודות התלמידים נקראת לפעמים "פתרונות התלמידים", והמילה
     "פתרון" היא תת-מחרוזת של "פתרונות". alias חשוף אחד היה חוטף את העמודה
     הזאת, מגיש את כל המחברות כפתרון בית הספר, וכל התלמידים היו מקבלים 100.
  2. פתרון בית ספר בכתב יד נראה ל-looks_typed בדיוק כמו עבודת תלמיד, ולכן
     היה נבדק כאילו הוא מחברת של מישהו — ציון מלא על שם תלמיד אקראי.

מריצים עם pytest, אחרת:  python tests/test_solution.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import checker  # noqa: E402
import form_schema as fs  # noqa: E402
from haskala_grading import math_core  # noqa: E402

FOLDER_URL = "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
UPLOAD_URL = "https://drive.google.com/open?id=1SoLuTiOnFiLeIdAbCdEfGhIjKlMnOp"
UPLOAD_ID = "1SoLuTiOnFiLeIdAbCdEfGhIjKlMnOp"

# הטופס של אבישי: כמו טופס המתמטיקה הקודם, בתוספת שאלת פתרון בית הספר.
HEADER = [
    "חותמת זמן",
    "כתובת אימייל",
    "שם המורה",
    "קישור לתיקייה משותפת",
    "מחוון הבדיקה — איך לנקד?",
    "צירוף פתרון בית ספר",
    "הערות ובקשות",
]


def _row(**over):
    row = ["2026-08-05 10:00:00", "avishai@bdika.net", "אבישי שלוש",
           FOLDER_URL, "", "", ""]
    idx = {"rubric": 4, "solution": 5, "comments": 6}
    for k, v in over.items():
        row[idx[k]] = v
    return row


def _parse(**over):
    return fs.parse_row(_row(**over), fs.map_columns(HEADER))


# ─── מיפוי העמודה ────────────────────────────────────────────────────────────

def test_solution_column_is_mapped():
    assert fs.map_columns(HEADER)["solution"] == 5


def test_avishais_real_question_title_maps():
    """הכותרת שאבישי בפועל השתמש בה.

    אומתה ב-5/8/2026 מתוך התיקייה ש-Forms יצר לשאלה:
    'התרגיל והפתרון המוצע (File responses)'. לפני שה-alias הזה נוסף, העמודה
    הזאת לא הותאמה לאף שדה — הפתרון פשוט לא היה מגיע למודל אף פעם.
    """
    header = HEADER[:5] + ["התרגיל והפתרון המוצע"] + HEADER[6:]
    assert fs.map_columns(header)["solution"] == 5


def test_assignment_description_upload_does_not_leak_a_url_into_the_rubric():
    """בטופס המתמטיקה 'Assignment description' היא שאלת העלאת קובץ, לא פסקה.

    אם קוראים אותה כטקסט, build_rubric מדביק "על המבחן: https://drive..."
    לתוך המחוון — שורת רעש שמוגשת למודל כהקשר על המבחן.
    """
    header = ["חותמת זמן", "כתובת אימייל", "קישור לתיקייה משותפת",
              "Assignment description"]
    cm = fs.map_columns(header)
    assert cm["instructions"] == 3
    p = fs.parse_row(["2026-08-05", "a@b.net", FOLDER_URL, UPLOAD_URL], cm)
    assert p["instructions"] == ""
    assert p["instructions_file_ids"] == [UPLOAD_ID]
    assert "drive.google.com" not in fs.build_rubric(p)


def test_a_typed_assignment_description_still_reaches_the_rubric():
    header = ["חותמת זמן", "כתובת אימייל", "קישור לתיקייה משותפת",
              "Assignment description"]
    p = fs.parse_row(["2026-08-05", "a@b.net", FOLDER_URL, "פרק 3 — משוואות"],
                     fs.map_columns(header))
    assert p["instructions"] == "פרק 3 — משוואות"
    assert "פרק 3 — משוואות" in fs.build_rubric(p)


def test_solution_question_may_be_reworded():
    """התאמה לפי מילות מפתח — אבישי יכול לנסח מחדש בלי לשבור כלום."""
    for title in ["צירוף פתרון בית ספר",
                  "אפשר לצרף כאן את פתרון בית הספר (לא חובה)",
                  "פתרון המורה",
                  "העלאת הפתרון הרשמי",
                  "School solution (optional)"]:
        header = HEADER[:5] + [title] + HEADER[6:]
        assert fs.map_columns(header).get("solution") == 5, title


def test_student_upload_column_is_not_stolen_by_the_solution_field():
    """הכשל השקט: "פתרון" הוא תת-מחרוזת של "פתרונות".

    עמודת ההעלאה של המחברות חייבת להישאר files. אם היא נחטפת ל-solution, כל
    מחברות הכיתה מוגשות למודל כפתרון בית הספר וכולם מקבלים ציון מלא.
    """
    header = ["חותמת זמן", "כתובת אימייל", "קישור לתיקייה משותפת",
              "העלאת פתרונות התלמידים"]
    cm = fs.map_columns(header)
    assert cm["files"] == 3
    assert "solution" not in cm


# ─── מה המורה ענה ────────────────────────────────────────────────────────────

def test_uploaded_solution_becomes_file_ids():
    p = _parse(solution=UPLOAD_URL)
    assert p["solution_file_ids"] == [UPLOAD_ID]
    assert p["solution_text"] == ""
    assert p["solution_in_folder"] is False


def test_two_uploaded_pages_keep_their_order():
    second = "https://drive.google.com/open?id=1SeCoNdPaGeIdAbCdEfGhIjKlMnOp"
    p = _parse(solution=f"{UPLOAD_URL}, {second}")
    assert p["solution_file_ids"] == [UPLOAD_ID, "1SeCoNdPaGeIdAbCdEfGhIjKlMnOp"]


def test_typed_solution_becomes_text():
    typed = "סעיף א: x=4. סעיף ב: המשולשים חופפים לפי צלע-זווית-צלע."
    p = _parse(solution=typed)
    assert p["solution_text"] == typed
    assert p["solution_file_ids"] == []


def test_solution_attached_in_folder_is_not_itself_the_solution():
    """המשפט "מצורף בתיקייה" הוא הפניה, לא פתרון — הוא לא מוגש למודל."""
    p = _parse(solution="מצורף בתיקייה")
    assert p["solution_text"] == ""
    assert p["solution_in_folder"] is True
    assert p["solution_label"] == "מצורף בתיקייה"


def test_no_answer_leaves_everything_empty():
    p = _parse()
    assert p["solution_file_ids"] == []
    assert p["solution_text"] == ""
    assert p["solution_in_folder"] is False


def test_a_form_without_the_question_still_parses():
    """טפסים ישנים (ושל אנגלית) לא נשברים — השאלה אופציונלית."""
    header = ["חותמת זמן", "כתובת אימייל", "קישור לתיקייה משותפת"]
    p = fs.parse_row(["2026-08-05", "a@b.net", FOLDER_URL],
                     fs.map_columns(header))
    assert p["solution_file_ids"] == []
    assert p["solution_text"] == ""


# ─── קבצים בתיקייה ───────────────────────────────────────────────────────────

def test_solution_filenames_are_recognised():
    for name in ["פתרון.pdf", "פתרון המבחן.pdf", "פיתרון מלא.jpg",
                 "תשובות סופיות.pdf", "solution.pdf", "Answer Key.PDF",
                 "teacher_solution.png"]:
        assert fs.looks_like_solution(name), name


def test_student_scans_are_not_mistaken_for_the_solution():
    for name in ["דני כהן.pdf", "scan_0007.jpg", "מחברת 12.pdf",
                 "כיתה ט1 עמוד 3.png", "מחוון.pdf"]:
        assert not fs.looks_like_solution(name), name


# ─── מה מגיע למודל ───────────────────────────────────────────────────────────

class _FakeImage:
    """עומד במקום PIL.Image — build_solution רק סופר ומעביר הלאה."""


def test_build_solution_prefers_the_attachment_over_typed_prose(monkeypatch):
    monkeypatch.setattr(math_core, "solution_pages",
                        lambda data, ext: [_FakeImage(), _FakeImage()])
    text, imgs, source = checker.build_solution(
        {"solution_text": "כתבתי גם פה משהו"},
        [{"name": "פתרון.pdf", "ext": ".pdf", "data": b""}])
    assert len(imgs) == 2
    assert text == ""                      # הקובץ מנצח, הטקסט לא נשלח פעמיים
    assert "פתרון.pdf" in source


def test_build_solution_falls_back_to_typed_prose():
    text, imgs, source = checker.build_solution({"solution_text": "x=4"}, [])
    assert (text, imgs) == ("x=4", [])
    assert source == "typed into the form"


def test_build_solution_survives_an_unreadable_attachment():
    """קובץ פגום לא מפיל את הבדיקה — רק מאבד את הרמז."""
    def boom(data, ext):
        raise ValueError("corrupt")

    import unittest.mock as mock
    with mock.patch.object(math_core, "solution_pages", boom):
        text, imgs, source = checker.build_solution(
            {"solution_text": ""}, [{"name": "פתרון.pdf", "ext": ".pdf", "data": b""}])
    assert (text, imgs, source) == ("", [], "none")


def test_build_solution_caps_pages_across_several_files():
    import unittest.mock as mock
    files = [{"name": f"עמוד {i}.pdf", "ext": ".pdf", "data": b""} for i in range(5)]
    with mock.patch.object(math_core, "solution_pages",
                           lambda data, ext: [_FakeImage(), _FakeImage()]):
        _text, imgs, _source = checker.build_solution({}, files)
    assert len(imgs) == math_core.MAX_SOLUTION_PAGES


def test_no_solution_is_the_documented_default():
    assert checker.build_solution({}, []) == ("", [], "none")


# ─── הפרומפט ─────────────────────────────────────────────────────────────────

def test_the_solution_is_labelled_as_the_teachers_not_the_students():
    block = math_core._solution_text_block("x=4")
    assert "לא** חלק מעבודת התלמיד" in block or "לא" in block
    assert "x=4" in block


def test_the_prompt_permits_a_different_valid_method():
    """הכלל שמונע הורדת נקודות על דרך פתרון חלופית ותקפה."""
    block = math_core._solution_text_block("x=4")
    assert "דרך אחרת" in block
    assert "ניקוד מלא" in block


def test_image_note_names_the_students_page_as_the_last_one():
    note = math_core._solution_note_for_images(3)
    assert "3 התמונות הראשונות" in note
    assert "התמונה האחרונה" in note


def test_image_note_is_empty_when_there_is_no_solution():
    assert math_core._solution_note_for_images(0) == ""
    assert math_core._solution_text_block("") == ""
    assert math_core._solution_text_block("   ") == ""


def test_describe_solution_reports_both_kinds():
    assert math_core._describe_solution("", []) == "none"
    assert math_core._describe_solution("", [1, 2]) == "2 page(s)"
    assert "chars" in math_core._describe_solution("abc", [])


if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        if "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            print(f"  – {name} (דורש pytest)")
            continue
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
