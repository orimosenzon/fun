"""טסטים ל-form_schema ול-drive_folder — הלוגיקה הטהורה של מיפוי טופס לפרמטרי בדיקה.

בלי רשת, בלי Sheets, בלי Drive, בלי קריאות למודל.

מריצים עם pytest אם הוא מותקן, אחרת:  python tests/test_form_schema.py
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402

import drive_folder as df  # noqa: E402
import form_schema as fs  # noqa: E402
from haskala_grading import core  # noqa: E402


# הטופס שאבישי בנה בפועל — שיתוף תיקייה, בלי העלאת קבצים.
# זו הכותרת שקובעת: אם הטסטים כאן עוברים והשירות עדיין נופל, הבעיה היא
# בהרשאות או ברשת, לא במיפוי.
AVISHAI_HEADER = [
    "Timestamp",
    "Email Address",
    "First name",
    "Last name",
    "Use rubric ",
    "Link to Folder",
    "Comments and requests",
]

# טופס עשיר יותר, בעברית — התרחיש שאליו form_schema נבנה במקור, עם עמודת
# תיקייה שנוספה. נשמר כדי שההתאמה הגמישה תמשיך להיבדק.
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
    "קישור לתיקייה משותפת",
]

FOLDER_URL = "https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
FOLDER_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"


# ─── הטופס האמיתי ────────────────────────────────────────────────────────────

def test_avishai_form_maps_completely():
    """כל שאלה בטופס של אבישי מגיעה לשדה שהצינור באמת קורא."""
    cm = fs.map_columns(AVISHAI_HEADER)
    assert cm["timestamp"] == 0
    assert cm["email"] == 1
    assert cm["first_name"] == 2
    assert cm["last_name"] == 3
    assert cm["rubric"] == 4
    assert cm["folder_link"] == 5
    assert cm["comments"] == 6


def test_avishai_form_parse_row():
    cm = fs.map_columns(AVISHAI_HEADER)
    row = ["03/08/2026 09:14:00", "avishai@shamir.org.il", "Avishai", "Shalosh",
           "Module G 16582", FOLDER_URL, "Weak class, please be gentle"]
    p = fs.parse_row(row, cm)
    assert p["email"] == "avishai@shamir.org.il"
    assert p["teacher_name"] == "Avishai Shalosh"      # שתי העמודות מאוחדות
    assert p["module"] == "g"
    assert p["rubric_id"] == "moe-writing-module-g"
    assert p["folder_id"] == FOLDER_ID
    assert p["comments"] == "Weak class, please be gentle"


def test_avishai_form_all_four_modules_resolve():
    """הבאג שהיה כאן: כל ארבע האפשרויות נפלו לרובריקת ברירת המחדל,
    שהיא סולם אחר לגמרי (30 נק' של כיתה ט' במקום 40 של מודול G)."""
    expected = {
        "Module G 16582": "moe-writing-module-g",
        "Module F 16584": "moe-writing-module-f",
        "Module D 016484": "moe-writing-module-d",
        "Module C 016382": "moe-writing-module-c",
    }
    for label, rubric_id in expected.items():
        assert fs.resolve_rubric_choice(label) == (rubric_id, False), label
        assert core.load_rubric(rubric_id), f"{rubric_id} is not bundled"


def test_module_rubrics_carry_the_right_word_counts():
    """המחוון הנכון עם טבלת אורך שגויה עדיין מוריד נקודות לא נכון."""
    expected = {"c": (70, 90), "d": (80, 100), "f": (100, 120), "g": (120, 140)}
    for letter, (lo, hi) in expected.items():
        rubric = core.load_rubric(f"moe-writing-module-{letter}")
        rule = core.resolve_word_count_rule(rubric)
        assert (rule["min"], rule["max"]) == (lo, hi), letter


# ─── זיהוי מודול ─────────────────────────────────────────────────────────────

def test_resolve_module_by_code():
    assert fs.resolve_module("Module G 16582") == "g"
    assert fs.resolve_module("16584") == "f"
    assert fs.resolve_module("מודול D — 016484") == "d"


def test_resolve_module_leading_zero_optional():
    """משרד החינוך כותב 016382 בגיליון אחד ו-16382 באחר."""
    assert fs.resolve_module("016382") == fs.resolve_module("16382") == "c"


def test_resolve_module_by_letter_when_no_code():
    assert fs.resolve_module("Module F") == "f"
    assert fs.resolve_module("מודול C") == "c"


def test_resolve_module_none_when_absent():
    assert fs.resolve_module("") is None
    assert fs.resolve_module("Modular thinking") is None      # לא גבול מילה
    assert fs.resolve_module(core.list_rubrics()[0]["name"]) is None


def test_resolve_module_code_beats_stray_letter():
    """'Module G 16582' מכיל גם את האות G וגם קוד. הקוד חד-משמעי."""
    assert fs.resolve_module("Grade module f 16582") == "g"


# ─── חילוץ מזהה תיקייה ───────────────────────────────────────────────────────

def test_extract_folder_id_share_link():
    assert fs.extract_folder_id(FOLDER_URL + "?usp=sharing") == FOLDER_ID


def test_extract_folder_id_address_bar_link():
    """מורה שמעתיק משורת הכתובת מקבל /u/0/ בדרך."""
    url = f"https://drive.google.com/drive/u/0/folders/{FOLDER_ID}"
    assert fs.extract_folder_id(url) == FOLDER_ID


def test_extract_folder_id_open_style():
    assert fs.extract_folder_id(f"https://drive.google.com/open?id={FOLDER_ID}") == FOLDER_ID


def test_extract_folder_id_bare_and_whitespace():
    assert fs.extract_folder_id(f"  {FOLDER_ID}  ") == FOLDER_ID


def test_extract_folder_id_rejects_junk():
    assert fs.extract_folder_id("") == ""
    assert fs.extract_folder_id(None) == ""
    assert fs.extract_folder_id("אני אשלח אחר כך") == ""


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
    shuffled = [HEADER[8], HEADER[0], HEADER[5], HEADER[1], HEADER[7], HEADER[9]]
    cm = fs.map_columns(shuffled)
    assert cm["files"] == 0
    assert cm["timestamp"] == 1
    assert cm["rubric"] == 2
    assert cm["email"] == 3
    assert cm["exercise_lang"] == 4
    assert cm["folder_link"] == 5


def test_map_columns_survives_rewording():
    reworded = ["Timestamp", "Email Address", "מה שם בית הספר שלך?",
                "אנא צרפו את הקבצים", "כיתה", "הדביקו קישור לתיקייה"]
    cm = fs.map_columns(reworded)
    assert cm["timestamp"] == 0
    assert cm["email"] == 1
    assert cm["school"] == 2
    assert cm["files"] == 3
    assert cm["grade_level"] == 4
    assert cm["folder_link"] == 5


def test_map_columns_missing_required_raises():
    """חסרה עמודת תיקייה — כישלון רועש, לא בדיקה של חצי שורה."""
    try:
        fs.map_columns(["חותמת זמן", "כתובת אימייל", "שם המורה"])
    except ValueError as e:
        assert "folder_link" in str(e)
    else:
        raise AssertionError("expected ValueError for a missing required column")


def test_map_columns_ignores_unknown_columns():
    cm = fs.map_columns(HEADER + ["הערות פנימיות", "ציון ידני"])
    assert len(cm) == 10


# ─── חילוץ מזהי קבצים (נתיב ההעלאה, שנשאר נתמך) ─────────────────────────────

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
           "אנגלית", "https://drive.google.com/open?id=" + "1" * 33, FOLDER_URL]
    p = fs.parse_row(row, cm)
    assert p["email"] == "dana@shamir.org.il"
    assert p["teacher_name"] == "דנה"
    assert p["school"] == "שמיר תל אביב"
    assert p["grade_level"] == "כיתה ט"
    assert p["exercise_lang"] == "en"
    assert p["rubric_id"] == core.list_rubrics()[0]["id"]
    assert p["file_ids"] == ["1" * 33]
    assert p["folder_id"] == FOLDER_ID


def test_parse_row_short_row():
    """Sheets חותך תאים ריקים בסוף — שורה קצרה מהכותרת היא המצב הרגיל."""
    cm = fs.map_columns(HEADER)
    p = fs.parse_row(["02/08/2026 09:14:00", "dana@shamir.org.il"], cm)
    assert p["email"] == "dana@shamir.org.il"
    assert p["school"] == ""
    assert p["file_ids"] == []
    assert p["folder_id"] == ""
    assert p["model_key"] == core.DEFAULT_MODEL


# ─── בניית ה"שאלה" ───────────────────────────────────────────────────────────

def test_build_question_includes_context():
    q = fs.build_question({"instructions": "Write a letter.",
                           "grade_level": "כיתה ז", "school": "שמיר"})
    assert q.startswith("Write a letter.")   # ניסוח המורה נשאר ראשון
    assert "כיתה ז" in q
    assert "שמיר" in q


def test_build_question_includes_module_and_comments():
    q = fs.build_question({"instructions": "Write a letter.", "module": "g",
                           "comments": "Weak class, please be gentle"})
    assert q.startswith("Write a letter.")
    assert "מודול G" in q
    assert "Weak class" in q


def test_build_question_comments_are_fenced():
    """הערות המורה הן שדה חובה בטופס, ולכן לרוב ייכתב שם 'תודה'.
    התיוג הוא מה שמונע מהמודל לקרוא אותן כהוראות המשימה."""
    q = fs.build_question({"instructions": "Write a letter.", "comments": "תודה רבה!"})
    assert "הערות מהמורה" in q
    assert q.index("Write a letter.") < q.index("תודה רבה!")


def test_build_question_instructions_only():
    assert fs.build_question({"instructions": "Write a letter."}) == "Write a letter."


def test_build_question_empty():
    assert fs.build_question({}) == ""


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
