"""טסטים ל-form_schema ול-drive_folder — הלוגיקה הטהורה של מיפוי טופס לפרמטרי בדיקה.

בלי רשת, בלי Sheets, בלי Drive, בלי קריאות למודל.

מריצים עם pytest אם הוא מותקן, אחרת:  python tests/test_form_schema.py

מה שונה מהטסטים של form-checker: אין כאן מודולי בגרות, אין רובריקות מצורפות
ואין שפת תרגיל. המחוון הוא טקסט חופשי, ולכן רוב הטסטים כאן בודקים שהטקסט הזה
מגיע למודל שלם — ושכשהמורה כותב "מצורף בתיקייה", המשפט הזה עצמו *לא* מוגש
כמחוון.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402

import drive_folder as df  # noqa: E402
import form_schema as fs  # noqa: E402
from haskala_grading import math_core  # noqa: E402


# טופס מתמטיקה מינימלי — רק מה שחייבים כדי לבדוק.
MIN_HEADER = [
    "Timestamp",
    "Email Address",
    "First name",
    "Last name",
    "קישור לתיקייה",
]

# הטופס המלא שאנחנו מציעים ב-FORM_SETUP.md.
HEADER = [
    "חותמת זמן",
    "כתובת אימייל",
    "שם המורה",
    "באיזה בית ספר את/ה מלמד/ת?",
    "שכבת הגיל של התלמידים",
    "מחוון הבדיקה — איך לנקד?",
    "נושא המבחן",
    "המבחנים לבדיקה (העלאת קבצים)",
    "קישור לתיקייה משותפת",
    "הערות ובקשות",
]

FOLDER_URL = "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
FOLDER_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
SCHEME = "סעיף א: 5 נק', סעיף ב: 10 נק', סעיף ג: 15 נק'"


# ─── מיפוי עמודות ────────────────────────────────────────────────────────────

def test_minimal_form_maps_completely():
    """טופס עם ארבע השאלות ההכרחיות בלבד עדיין ממופה במלואו."""
    cm = fs.map_columns(MIN_HEADER)
    assert cm["timestamp"] == 0
    assert cm["email"] == 1
    assert cm["first_name"] == 2
    assert cm["last_name"] == 3
    assert cm["folder_link"] == 4


def test_map_columns_basic():
    cm = fs.map_columns(HEADER)
    assert cm["timestamp"] == 0
    assert cm["email"] == 1
    assert cm["teacher_name"] == 2
    assert cm["school"] == 3
    assert cm["grade_level"] == 4
    assert cm["rubric"] == 5
    assert cm["instructions"] == 6
    assert cm["files"] == 7
    assert cm["folder_link"] == 8
    assert cm["comments"] == 9


def test_map_columns_longest_alias_wins():
    """'מחוון הבדיקה' ו'המבחנים לבדיקה' שניהם מכילים 'בדיק'.

    זה בדיוק המקרה שכלל ההתאמה-הארוכה-ביותר נועד לו: התאמה ראשונה-מנצחת
    הייתה ממפה את שתי העמודות לפי הסדר בטופס."""
    cm = fs.map_columns(HEADER)
    assert cm["rubric"] == 5
    assert cm["files"] == 7


def test_map_columns_survives_reordering():
    """מי שיבנה את הטופס יזיז שאלות. מיפוי לפי מיקום היה נשבר בשקט; לפי כותרת — לא."""
    shuffled = [HEADER[8], HEADER[0], HEADER[5], HEADER[1], HEADER[9]]
    cm = fs.map_columns(shuffled)
    assert cm["folder_link"] == 0
    assert cm["timestamp"] == 1
    assert cm["rubric"] == 2
    assert cm["email"] == 3
    assert cm["comments"] == 4


def test_map_columns_survives_rewording():
    reworded = [
        "חותמת זמן",
        "כתובת אימייל",
        "לפי איזה מחוון לנקד את המבחן?",
        "קישור לתיקיה עם הסריקות",
    ]
    cm = fs.map_columns(reworded)
    assert cm["rubric"] == 2
    assert cm["folder_link"] == 3


def test_map_columns_missing_required_raises():
    """בלי קישור לתיקייה אין מה לבדוק — וזו חייבת להיות נפילה רועשת."""
    try:
        fs.map_columns(["חותמת זמן", "כתובת אימייל"])
    except ValueError as e:
        assert "folder_link" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_live_math_form_header_maps():
    """שורת הכותרות האמיתית של 'Bdika - Math (Responses)', כפי שנקראה מהגיליון
    ב-5/8/2026. הועתקה, לא נוסחה מחדש.

    עד לתאריך הזה המיפוי נפל עליה עם ValueError: השאלה כתובה 'קישור לתקייה'
    בלי יו״ד, ואף אליאס לא תפס אותה. כלומר כל תשובה בטופס הייתה נדחית לפני
    הבדיקה. הטסט הזה קיים כדי שכתיב הכותרת בטופס החי לא יישבר שוב בשקט."""
    live = [
        "Timestamp",
        "Email Address",
        "שם פרטי",
        "שם משפחה",
        "התרגיל והפתרון המוצע",
        "קישור לתקייה בגוגל דרייב בה נמצאים התרגילים שיש לבדוק",
        "Comments and requests",
    ]
    cm = fs.map_columns(live)
    assert cm["timestamp"] == 0
    assert cm["email"] == 1
    assert cm["first_name"] == 2
    assert cm["last_name"] == 3
    assert cm["solution"] == 4
    assert cm["folder_link"] == 5
    assert cm["comments"] == 6


def test_folder_link_matches_both_spellings():
    """'תיקייה' ו'תקייה' — שני הכתיבים, כי מורים כותבים את שניהם."""
    for title in ("קישור לתיקייה משותפת", "קישור לתקייה משותפת",
                  "קישור לתיקיה משותפת", "קישור לתקיה משותפת"):
        cm = fs.map_columns(["חותמת זמן", "כתובת אימייל", title])
        assert cm["folder_link"] == 2, title


def test_map_columns_ignores_unknown_columns():
    """גיליון תשובות צובר עמודות שאדם הוסיף ביד. הן לא אמורות להתפרש כשדות."""
    cm = fs.map_columns(HEADER + ["ציון ידני", "טופל ע\"י"])
    assert "ציון ידני" not in cm
    assert len(cm) == 10


# ─── מחוון: טקסט חופשי, לא dropdown ──────────────────────────────────────────

def test_rubric_in_folder_detects_the_common_phrasings():
    for phrase in ("המחוון מצורף בתיקייה", "צירפתי מחוון", "see folder",
                   "the marking scheme is attached", "יש מחוון בקבצים"):
        assert fs.rubric_is_in_folder(phrase), phrase


def test_rubric_in_folder_does_not_fire_on_a_real_scheme():
    assert not fs.rubric_is_in_folder(SCHEME)
    assert not fs.rubric_is_in_folder("")


def test_a_pointer_to_the_folder_is_not_used_as_the_scheme():
    """הבאג שהטסט הזה מונע: להגיש למודל את המחרוזת 'מצורף בתיקייה' ככללי ניקוד."""
    cm = fs.map_columns(HEADER)
    row = _row(cm, rubric="המחוון מצורף בתיקייה")
    p = fs.parse_row(row, cm)
    assert p["rubric_in_folder"] is True
    assert p["rubric_text"] == ""
    assert "מצורף" not in fs.build_rubric(p)


def test_a_pasted_scheme_reaches_the_model_intact():
    cm = fs.map_columns(HEADER)
    p = fs.parse_row(_row(cm, rubric=SCHEME), cm)
    assert p["rubric_in_folder"] is False
    assert p["rubric_text"] == SCHEME
    assert SCHEME in fs.build_rubric(p)


# ─── בחירת מודל ──────────────────────────────────────────────────────────────

def test_resolve_model_defaults_to_sonnet5():
    assert fs.resolve_model_choice("") == "sonnet5"
    assert fs.resolve_model_choice("") == math_core.DEFAULT_MODEL


def test_resolve_model_accepts_key_label_and_alias():
    assert fs.resolve_model_choice("opus") == "opus"
    assert fs.resolve_model_choice(math_core.MODELS["opus"]) == "opus"
    assert fs.resolve_model_choice("Claude Opus") == "opus"
    assert fs.resolve_model_choice("gemini") == "gemini-flash"


def test_bare_sonnet_is_not_guessed():
    """ל-math_core יש גם sonnet5 וגם sonnet (4.6). ניחוש בין השניים היה מגיש
    למורה מודל אחר בלי לומר מילה, אז 'sonnet' לבד נופל לברירת המחדל."""
    assert fs.resolve_model_choice("sonnet") == math_core.DEFAULT_MODEL


def test_unknown_model_falls_back_rather_than_raising():
    assert fs.resolve_model_choice("gpt-9 ultra") == math_core.DEFAULT_MODEL


# ─── חילוץ מזהים ─────────────────────────────────────────────────────────────

def test_extract_folder_id_share_link():
    assert fs.extract_folder_id(FOLDER_URL + "?usp=drive_link") == FOLDER_ID


def test_extract_folder_id_address_bar_link():
    url = "https://drive.google.com/drive/u/0/folders/" + FOLDER_ID
    assert fs.extract_folder_id(url) == FOLDER_ID


def test_extract_folder_id_bare_and_whitespace():
    assert fs.extract_folder_id("  " + FOLDER_ID + "  ") == FOLDER_ID


def test_extract_folder_id_rejects_junk():
    assert fs.extract_folder_id("אין לי תיקייה") == ""
    assert fs.extract_folder_id("") == ""


def test_extract_file_ids_forms_format():
    cell = ("https://drive.google.com/open?id=1aaaaaaaaaaaaaaaaaaaaaaaa, "
            "https://drive.google.com/open?id=1bbbbbbbbbbbbbbbbbbbbbbbb")
    assert fs.extract_file_ids(cell) == ["1aaaaaaaaaaaaaaaaaaaaaaaa",
                                         "1bbbbbbbbbbbbbbbbbbbbbbbb"]


def test_extract_file_ids_preserves_order_and_dedups():
    cell = ("https://drive.google.com/open?id=1aaaaaaaaaaaaaaaaaaaaaaaa, "
            "https://drive.google.com/open?id=1aaaaaaaaaaaaaaaaaaaaaaaa, "
            "https://drive.google.com/open?id=1bbbbbbbbbbbbbbbbbbbbbbbb")
    assert fs.extract_file_ids(cell) == ["1aaaaaaaaaaaaaaaaaaaaaaaa",
                                         "1bbbbbbbbbbbbbbbbbbbbbbbb"]


def test_extract_file_ids_empty():
    assert fs.extract_file_ids("") == []


# ─── parse_row ───────────────────────────────────────────────────────────────

def _row(cm, **over):
    """שורת גיליון מלאה, עם דריסות לפי שם שדה."""
    vals = {
        "timestamp": "8/5/2026 09:00:00",
        "email": "avishai@taded.org.il",
        "teacher_name": "אבישי שלוש",
        "school": "שמיר",
        "grade_level": "כיתה ט",
        "rubric": SCHEME,
        "instructions": "גאומטריה — חפיפת משולשים",
        "files": "",
        "folder_link": FOLDER_URL,
        "comments": "תודה!",
    }
    vals.update(over)
    row = [""] * (max(cm.values()) + 1)
    for key, idx in cm.items():
        row[idx] = vals.get(key, "")
    return row


def test_parse_row_full():
    cm = fs.map_columns(HEADER)
    p = fs.parse_row(_row(cm), cm)
    assert p["email"] == "avishai@taded.org.il"
    assert p["teacher_name"] == "אבישי שלוש"
    assert p["folder_id"] == FOLDER_ID
    assert p["grade_level"] == "כיתה ט"
    assert p["school"] == "שמיר"
    assert p["rubric_text"] == SCHEME
    assert p["model_key"] == "sonnet5"


def test_parse_row_short_row():
    """Sheets חותך תאים ריקים בסוף. שורה קצרה מהכותרת היא מצב תקין."""
    cm = fs.map_columns(HEADER)
    p = fs.parse_row(["8/5/2026", "a@b.com", "", "", "", "", "", "", FOLDER_URL], cm)
    assert p["folder_id"] == FOLDER_ID
    assert p["comments"] == ""
    assert p["rubric_text"] == ""


def test_parse_row_joins_first_and_last_name():
    cm = fs.map_columns(MIN_HEADER)
    p = fs.parse_row(["8/5/2026", "a@b.com", "אבישי", "שלוש", FOLDER_URL], cm)
    assert p["teacher_name"] == "אבישי שלוש"


# ─── build_rubric ────────────────────────────────────────────────────────────

def test_build_rubric_puts_the_scheme_first():
    """המילים של המורה נקראות ראשונות; ההקשר נוסף אחריהן."""
    cm = fs.map_columns(HEADER)
    out = fs.build_rubric(fs.parse_row(_row(cm), cm))
    assert out.startswith(SCHEME)


def test_build_rubric_includes_class_context():
    cm = fs.map_columns(HEADER)
    out = fs.build_rubric(fs.parse_row(_row(cm), cm))
    assert "כיתה ט" in out
    assert "שמיר" in out
    assert "חפיפת משולשים" in out


def test_build_rubric_fences_comments_and_context():
    """'תודה!' בשדה חובה חייב להיות מסומן, אחרת הוא נקרא ככלל ניקוד."""
    cm = fs.map_columns(HEADER)
    out = fs.build_rubric(fs.parse_row(_row(cm), cm))
    assert "לא חלק מהמחוון" in out
    assert out.index("לא חלק מהמחוון") > out.index(SCHEME)


def test_build_rubric_empty_when_nothing_supplied():
    """בלי מחוון ובלי הקשר — מחרוזת ריקה, ואז math_core מנקד לפי הדף."""
    cm = fs.map_columns(MIN_HEADER)
    p = fs.parse_row(["8/5/2026", "a@b.com", "", "", FOLDER_URL], cm)
    assert fs.build_rubric(p) == ""


# ─── הפרדת עבודות תלמידים ממסמכי רקע ────────────────────────────────────────
# הפילטר החינמי שמחליף קריאת מודל לכל קובץ: מסמך מודפס נושא שכבת טקסט
# ובלי תמונות; סריקה היא בדיוק ההפך.

def _typed_pdf(pages=1, text="Write a composition about your summer. " * 40):
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(40, 40, 550, 780), text, fontsize=11)
    buf = doc.tobytes()
    doc.close()
    return buf


def _scanned_pdf(pages=1):
    """PDF שכולו תמונה — מה שסורק מייצר."""
    doc = fitz.open()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 600, 800))
    pix.clear_with(255)
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(fitz.Rect(0, 0, 600, 800), pixmap=pix)
    buf = doc.tobytes()
    doc.close()
    return buf


def test_looks_typed_detects_a_typed_document():
    typed, why = df.looks_typed(_typed_pdf(), ".pdf")
    assert typed is True, why


def test_looks_typed_passes_a_scan_through_as_student_work():
    typed, why = df.looks_typed(_scanned_pdf(), ".pdf")
    assert typed is False, why


def test_looks_typed_never_skips_a_photo():
    """צילום בטלפון הוא עבודת תלמיד. אין מצב שבו מדלגים עליו."""
    assert df.looks_typed(b"\xff\xd8\xff", ".jpg")[0] is False
    assert df.looks_typed(b"\x89PNG", ".png")[0] is False


def test_looks_typed_biases_towards_student_work_on_junk():
    """קובץ פגום לא מסווג כמסמך רקע — שיפול בצינור הבדיקה עם שגיאה אמיתית."""
    assert df.looks_typed(b"not a pdf at all", ".pdf")[0] is False


def test_page_count_matches_the_pdf():
    assert df.page_count(_scanned_pdf(pages=3), ".pdf") == 3
    assert df.page_count(b"\xff\xd8\xff", ".jpg") == 1


# ─── שמות הדוחות ─────────────────────────────────────────────────────────────

def test_report_name_keeps_the_student_file_name():
    """תיקייה עם שלושים דוחות שימושית רק אם כל אחד אומר של מי הוא."""
    assert df.report_name("רותם כהן.pdf") == "רותם כהן — checked"


def test_report_name_strips_path_hostile_characters():
    assert "/" not in df.report_name("a/b:c.pdf")


def test_report_name_survives_an_empty_name():
    assert df.report_name("").startswith("work")


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
