"""טסטים להמרת LaTeX ליוניקוד בדוח הוורד.

הרקע: דוח וורד אמיתי שנוצר ב-5/8/2026 הציג למורה את מקור ה-LaTeX כטקסט גולמי,
למשל השורה

    \\triangle A_1=\\triangle A_2,\\ \\triangle BLA \\cong \\triangle BLA

דוח ה-HTML מעולם לא סבל מזה כי הוא מרנדר עם KaTeX — רק מסלול הוורד.

כל המחרוזות שנבדקות כאן נלקחו מאותו דוח אמיתי, לא הומצאו.

מריצים עם pytest, אחרת:  python tests/test_latex_unicode.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from haskala_grading.math_report import latex_to_unicode as u  # noqa: E402


# ─── השורות מהדוח האמיתי ─────────────────────────────────────────────────────

def test_the_line_that_started_this():
    assert u(r"\triangle A_1=\triangle A_2,\ \triangle BLA \cong \triangle BLA") \
        == "△A₁=△A₂, △BLA ≅ △BLA"


def test_geometry_symbols_from_the_real_report():
    assert u(r"BC \parallel AN") == "BC ∥ AN"
    assert u(r"AN=AN_2") == "AN=AN₂"
    assert u(r"\angle MLB=180°") == "∠MLB=180°"
    assert u(r"\triangle MLB,\ ML=BL") == "△MLB, ML=BL"


def test_hebrew_inside_text_survives():
    """הפרומפט מבקש LaTeX, והמודל עוטף מילים עבריות ב-\\text{}.

    התוכן חייב לעבור כמו שהוא — אם הוא היה מתפרש כמתמטיקה, כל אות עברית
    הייתה הופכת לפקודה לא מוכרת.
    """
    got = u(r"\triangle ALM,\ \angle ALM=90°,\ AL \text{ חוצה זווית } A,"
            r"\ MAN \text{ מחולק לזוויות שוות, } BC \perp BC")
    assert got == "△ALM, ∠ALM=90°, AL חוצה זווית A, MAN מחולק לזוויות שוות, BC ⊥ BC"


def test_final_answer_line():
    assert u(r"\angle MLB = 180°\ (\text{סכום זוויות במשולש})") \
        == "∠MLB = 180° (סכום זוויות במשולש)"


# ─── מה שכבר עבד קודם חייב להמשיך לעבוד ──────────────────────────────────────

def test_plain_algebra_is_untouched():
    """הנימוק המקורי בקוד היה "the strings are simple, e.g. x^2-6x+10".

    זה היה נכון לאלגברה, וחייב להישאר נכון.
    """
    assert u("x^2-6x+10") == "x²-6x+10"
    assert u("AB=BM") == "AB=BM"
    assert u("") == ""
    assert u(None) == ""


def test_fractions_and_roots():
    assert u(r"\frac{1}{2}") == "1/2"
    assert u(r"\frac{a+b}{2}") == "(a+b)/2"
    assert u(r"\sqrt{x^2+y^2}") == "√(x²+y²)"
    assert u(r"\sqrt{2}") == "√2"


def test_all_fraction_variants():
    """`\\tfrac` הופיע בדוח ייצור אמיתי (6/8/2026) ויצא "BC = tfrac12 AN" —
    רק `\\frac` ו-`\\dfrac` היו מטופלים, והשאר נפל לענף שמסיר את הלוכסן.

    ארבע הווריאציות נראות זהות בטקסט רגיל; ההבדל בין display ל-inline הוא
    עניין של רינדור ואין לו משמעות כאן."""
    assert u(r"BC = \tfrac{1}{2} AN") == "BC = 1/2 AN"
    assert u(r"BC = \tfrac12 AN") == "BC = 1/2 AN"      # בלי סוגריים, כמו ש-LaTeX קורא
    assert u(r"\dfrac{a+b}{c}") == "(a+b)/c"
    assert u(r"\cfrac{1}{2}") == "1/2"


def test_scripts_are_all_or_nothing():
    """תת-כתיב מתורגם רק אם כל התווים ניתנים להמרה.

    תרגום חלקי גרוע מכלום: "x²ʸ" ליד "x^(2y)" באותו דוח נקראים כשני ביטויים
    שונים.
    """
    assert u("x^{2n}") == "x²ⁿ"
    assert u("x^{2k}") == "x^(2k)"      # k חסר ביוניקוד — נופל לצורת ASCII
    assert u("A_1") == "A₁"


def test_degrees_written_as_a_superscript_command():
    """הדרך הרשמית לכתוב מעלות ב-LaTeX. הופיע בדוח אמיתי כ-"90^circ"."""
    assert u(r"\angle ALM = 90^\circ") == "∠ALM = 90°"
    assert u(r"x^\alpha") == "x^α"


def test_unknown_commands_degrade_readably():
    """פקודה לא מוכרת מאבדת את הלוכסן ולא מפילה כלום."""
    assert "widehat" in u(r"\widehat{ABC}")
    assert "\\" not in u(r"\widehat{ABC}")


def test_relations_get_spaces_prefixes_do_not():
    """△ נצמד לאות שאחריו, ≅ מקבל רווח משני צדדיו."""
    assert u(r"\triangle ABC \cong \triangle DEF") == "△ABC ≅ △DEF"
    assert u(r"a\leq b") == "a ≤ b"


def test_unbalanced_braces_do_not_raise():
    assert u(r"\frac{a}{b") != ""
    assert u("x^{2") != ""


if __name__ == "__main__":
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
