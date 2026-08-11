"""טסטים לשני השינויים ב-lib/haskala_grading מ-11/8/2026.

שניהם באו מאבישי:
  • ספירת מילים — רק כתב יד, בלי טקסט מודפס מדף הבחינה
  • טקסט שצריך להימחק — בסוגריים, קו חוצה והדגשה צהובה

בלי רשת ובלי קריאות למודל.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from haskala_grading import core, report  # noqa: E402


# ─── word count: handwriting only ───────────────────────────────────────────

def _page(*lines):
    """lines are (text, printed) pairs."""
    return [{"page": 1, "lines": [{"text": t, "printed": p} for t, p in lines]}]


def test_printed_lines_are_not_counted():
    """The exam's own question used to count as the student's writing."""
    pages = _page(("Write about your summer vacation in 70-90 words", True),
                  ("Last summer I went to Greece with my family", False))
    assert core.count_words(pages) == 9


def test_a_page_that_is_all_printed_counts_zero():
    pages = _page(("Ministry of Education", True), ("Module C 016382", True))
    assert core.count_words(pages) == 0


def test_lines_without_the_flag_are_still_counted():
    """Back-compat: results stored before the flag existed, and callers that
    build pages by hand, must not lose their text to a field they never set."""
    pages = [{"page": 1, "lines": [{"text": "one two three"}]}]
    assert core.count_words(pages) == 3


def test_the_deduction_this_was_really_about():
    """A sixty-word answer under a 70–90 rule earns a deduction. Forty words of
    printed prompt used to push it to a hundred and cancel that."""
    student = " ".join(["word"] * 60)
    printed = " ".join(["instructions"] * 40)
    assert core.count_words(_page((printed, True), (student, False))) == 60
    # and the old behaviour, for contrast
    assert core.count_words(_page((printed, False), (student, False))) == 100


def test_the_ocr_prompt_asks_for_the_flag():
    """The count is only honest while the OCR is actually classifying lines."""
    prompt = core.build_ocr_prompt("en") if hasattr(core, "build_ocr_prompt") else None
    if prompt is None:                      # prompt builder is named differently
        import re
        src = open(core.__file__, encoding="utf-8").read()
        assert '"printed"' in src and "מודפסת" in src
    else:
        assert "printed" in prompt


# ─── deletions: bracketed, struck through, yellow ───────────────────────────

def _evaluation(delete: bool):
    return {"criteria": [{
        "name": "Language Use", "color": "3366CC", "score": 5, "max_score": 10,
        "level": "fair", "feedback": "", "feedback_secondary": "",
        "issues": [{"line_ref": "p1-l0", "quote": "very very",
                    "comment": "repeated word", "delete": delete}],
    }]}


_PAGES = [{"page": 1, "lines": [{"text": "It was very very good", "printed": False}]}]


def test_delete_flag_reaches_the_span():
    spans = report.resolve_issue_spans(_PAGES, _evaluation(True))
    assert spans[(1, 0)][0]["delete"] is True
    spans = report.resolve_issue_spans(_PAGES, _evaluation(False))
    assert spans[(1, 0)][0]["delete"] is False


def _render(delete: bool):
    from docx import Document
    doc = Document()
    p = doc.add_paragraph()
    spans = report.resolve_issue_spans(_PAGES, _evaluation(delete))[(1, 0)]
    report._fill_paragraph_with_spans(p, _PAGES[0]["lines"][0]["text"], spans,
                                      include_notes=False)
    return p


def _shading(run):
    from docx.oxml.ns import qn
    rPr = run._r.find(qn("w:rPr"))
    shd = rPr.find(qn("w:shd")) if rPr is not None else None
    return shd.get(qn("w:fill")) if shd is not None else None


def test_deleted_text_is_bracketed_struck_and_yellow():
    marked = [r for r in _render(True).runs if r.text.startswith("[")]
    assert len(marked) == 1
    run = marked[0]
    assert run.text == "[very very]"
    assert run.font.strike is True
    assert _shading(run) == report.DELETE_HIGHLIGHT


def test_a_normal_issue_is_untouched_by_any_of_that():
    """The three marks mean "remove this". A correction that only needs
    rewriting must not be shown to a student as text to delete."""
    runs = _render(False).runs
    marked = [r for r in runs if r.text == "very very"]
    assert len(marked) == 1
    assert not marked[0].font.strike
    assert _shading(marked[0]) != report.DELETE_HIGHLIGHT
    assert not any(r.text.startswith("[") for r in runs)


def test_yellow_replaces_the_criterion_colour_rather_than_mixing():
    """Avishai's call: two shades over one span is unreadable, and the
    criterion is still named in the note that follows the span."""
    crit_tint = report._tint_hex("3366CC", mix=0.35)
    run = [r for r in _render(True).runs if r.text.startswith("[")][0]
    assert _shading(run) == report.DELETE_HIGHLIGHT != crit_tint


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except Exception as e:
                failures += 1
                print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
