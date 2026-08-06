"""math_core.py — ליבת בדיקת המתמטיקה, משותפת ל-math-checker ול-math-form-checker.

בדיקת מתמטיקה היא **לא** בעיית שורה-שורה כמו `core.py`. תרגיל מתמטי הוא יחידה
הוליסטית: סרטוטים, הוכחות גאומטריות ומעברים אלגבריים פרושים על פני הדף. לכן
ה-pipeline כאן שונה מזה של core.py:

    רינדור עמוד → תיקון אוריינטציה → ניתוח הוליסטי של העמוד → JSON מובנה

ומה שמשותף ל-core.py — רינדור PDF, downscale, JPEG, סיבוב, שגיאות — מיובא ממנו
ולא משוכפל. ההעתקה היחידה שנשארה בכוונה היא `detect_rotation`, כי הפרומפט כאן
מתאר עמוד מבחן במתמטיקה ולא חיבור באנגלית.

מקור: products/math-checker/core.py (הרפקטור מ-2026-07-03), שהוא בתורו פיצול של
ה-app.py המונוליטי שפרוס ב-HF Space `orimosenzon/haskala-math`. ANALYSIS_PROMPT,
ANALYSIS_SCHEMA ו-detect_rotation זהים לשתי הגרסאות — הרפקטור הזיז קוד, לא שינה
בדיקה.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time

from PIL import Image

from .core import (  # noqa: F401 — ACCEPTED_EXTS/RENDER_DPI re-exported for callers
    ACCEPTED_EXTS,
    ORIENT_MAX_EDGE,
    RENDER_DPI,
    _anthropic,
    _azure,
    _azure_deployment,
    _downscale,
    _gemini,
    _groq,
    _image_block_anthropic,
    _img_to_b64_jpeg,
    _jpeg_bytes,
    apply_rotation,
    count_pages,
    file_to_pages,
    humanize_error,
    iter_file_pages,
    pdf_to_pages,
)

log = logging.getLogger("haskala.math")

MAX_EDGE = 2200           # longest edge sent to the analysis model
PREVIEW_MAX_EDGE = 1400   # longest edge stored in the result for display

# Analysis models the teacher can pick. Holistic math/geometry grading is a hard
# reasoning task (proofs, 2D diagrams, multi-step algebra), so only strong
# reasoners do it well. The cheap / mini / free tiers stay exposed for testing
# but are flagged "(לא מומלץ)" — empirically they miss real calculation and
# logic errors on math pages.
MODELS = {
    "sonnet5":          "Claude Sonnet 5 · ברירת המחדל",
    "opus":             "Claude Opus 4.8 · איכות מרבית",
    # Renamed from the bare "sonnet" it carried in math-checker. With Sonnet 5
    # in the table, "sonnet" is ambiguous — and it was an exact key, so a
    # teacher typing the word they use colloquially for "the current Sonnet"
    # would have been handed 4.6 without a word. Unmapped, it warns and grades
    # with the default instead.
    "sonnet46":         "Claude Sonnet 4.6 · הדור הקודם",
    "gemini-flash":     "Gemini 2.5 Flash · מהיר וזול",
    "haiku":            "Claude Haiku 4.5 · הכי זול (לא מומלץ)",
    "gemini-lite":      "Gemini 2.5 Flash-Lite · זול מאוד (לא מומלץ)",
    "groq-scout":       "Groq Llama 4 Scout · חינמי (לא מומלץ)",
    "azure-gpt41-mini": "GPT-4.1-mini · Azure (לא מומלץ)",
}

_ANTHROPIC_IDS = {
    "sonnet5": "claude-sonnet-5",
    "opus":    "claude-opus-4-8",
    "sonnet46": "claude-sonnet-4-6",
    "haiku":   "claude-haiku-4-5-20251001",
}
_GEMINI_IDS = {
    "gemini-flash": "gemini-2.5-flash",
    "gemini-lite":  "gemini-2.5-flash-lite",
}
_GROQ_IDS = {
    "groq-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
}
_AZURE_KEYS = {"azure-gpt41-mini"}

# Sonnet 5 rather than the Opus 4.8 that the HF Space defaults to. Decided
# 2026-08-05 for math-form-checker: the form-driven flow grades unattended, so
# every scan that arrives spends budget with no human in the loop, and Opus is
# ~2.5× the input price on what are image-heavy pages. Same prompt and same
# schema — only the model differs — so grades are NOT guaranteed identical to
# the Space's. Opus stays one dropdown value away for anything that needs it.
DEFAULT_MODEL = "sonnet5"

# Orientation was Haiku until 2026-08-05, on the reasoning that a four-way
# choice is easy enough for the cheapest model. Measured against the five sample
# exam pages in /home/ori/avish/, Haiku got **0 of 5 right** — and the pages are
# all already upright, so every answer it gave rotated a perfectly readable page
# into an unreadable one. Worse, it was not even consistently wrong: the same
# page came back 90 on one call and 270 on the next, at both 900px and 1600px.
#
# That single step was responsible for most of what looked like grading quality
# problems. The analysis model was being handed sideways pages, which is why
# reports filled up with "לא קריא" and "קשה לעקוב", why scores sat at 3-4/10,
# and why re-running the same file produced 3.5 then 7 — the variance was the
# rotation lottery, not the grader.
#
# Sonnet 5 answered 0 on all five. It costs roughly $0.002 per page here against
# ~$0.06 for the analysis call, so this is a rounding error on the bill and the
# difference between a usable report and a worthless one.
#
# If this is ever revisited: measure against real pages before trusting a
# cheaper model, and remember that "no rotation needed" is the common case, so a
# detector that guesses is strictly worse than one that does nothing.
ORIENT_MODEL = "claude-sonnet-5"

# Adaptive thinking is ON by default on Sonnet 5 and Opus 5 (it was OFF on
# Opus 4.8, which is what this pipeline was tuned against), and max_tokens caps
# thinking *plus* response text together. Left alone, a 6000-token budget would
# be eaten by reasoning and the structured JSON would truncate mid-object. So
# every Anthropic call below disables thinking explicitly, which also keeps the
# per-page bill predictable. Raising this to adaptive is a real quality lever —
# but it needs a max_tokens rise and an A/B against the sample exams first.
_ANALYSIS_MAX_TOKENS = 6000


def resolve_model(key: str | None) -> tuple[str, str]:
    """(key, human label). An unknown key falls back to the default rather than
    raising — a stale form value must not lose a submission."""
    if key not in MODELS:
        if key:
            log.warning("unknown math model %r — falling back to %s", key, DEFAULT_MODEL)
        key = DEFAULT_MODEL
    return key, MODELS[key]


# ─── orientation ──────────────────────────────────────────────────────────────

def detect_rotation(img: Image.Image) -> int:
    """Clockwise rotation (0/90/180/270) needed to upright the page.

    core.py's equivalent asks about a handwritten composition; this one says
    "math exam" so the model weighs formulas and diagrams rather than prose
    baselines. 180° flips are common in scanned exam stacks and a
    portrait/landscape heuristic misses them entirely, so this stays a model
    call — but see ORIENT_MODEL for why it is emphatically not the cheapest
    model. Getting this wrong corrupts every downstream stage silently.

    The answer is verified rather than trusted. Measured on 2026-08-05, the raw
    ask gets 0° and 180° right but confuses the two 90s — a page rotated 90°
    clockwise comes back "90" when the correction needed is 270°. That is the
    classic "which way is the question asking" ambiguity, and no rewording of
    the prompt reliably settles it, so instead the candidate is applied and the
    result checked. An upright page (the overwhelmingly common case) still costs
    exactly one call, because verification is skipped when the answer is 0.
    """
    deg = _ask_rotation(img)
    if deg == 0:
        return 0
    if _looks_upright(apply_rotation(img, deg)):
        return deg
    # Only the 90s get a second chance: 180 has no counterpart to confuse it
    # with, so a failed check there means the page defeated the detector and
    # trying 180 again would just cost another call.
    if deg in (90, 270):
        alt = 360 - deg
        if _looks_upright(apply_rotation(img, alt)):
            log.info("orientation: %d° failed verification, using %d° instead", deg, alt)
            return alt
    log.warning("orientation: could not verify %d° — applying it anyway", deg)
    return deg


def _looks_upright(img: Image.Image) -> bool:
    """Yes/no check on an already-rotated page. Deliberately a different
    question from _ask_rotation: 'is this readable' has no direction to get
    backwards, which is the exact failure being corrected for."""
    msg = _anthropic().messages.create(
        model=ORIENT_MODEL,
        max_tokens=16,
        thinking={"type": "disabled"},
        messages=[{
            "role": "user",
            "content": [
                _image_block_anthropic(img, ORIENT_MAX_EDGE),
                {"type": "text", "text": (
                    "האם הטקסט בעמוד הסרוק הזה זקוף וקריא בכיוון הרגיל "
                    "(כלומר אפשר לקרוא אותו בלי להטות את הראש או להפוך את הדף)? "
                    "ענה מילה אחת בלבד: כן או לא."
                )},
            ],
        }],
    )
    txt = "".join(b.text for b in msg.content if b.type == "text")
    return "כן" in txt


def _ask_rotation(img: Image.Image) -> int:
    """The raw model answer, unverified. Split out so detect_rotation reads as
    the policy and this stays the single place the question is worded."""
    msg = _anthropic().messages.create(
        model=ORIENT_MODEL,
        max_tokens=16,
        # Same reason as the analysis call: adaptive thinking is on by default on
        # Sonnet 5, and max_tokens caps thinking and text together. A budget this
        # small would be swallowed whole, the text would come back empty, and the
        # fallback below would silently return 0 for every page — turning the fix
        # this model was chosen for back into a no-op that looks like it works.
        thinking={"type": "disabled"},
        messages=[{
            "role": "user",
            "content": [
                _image_block_anthropic(img, ORIENT_MAX_EDGE),
                {"type": "text", "text": (
                    "זהו עמוד סרוק של מבחן מתמטיקה בכתב יד (עברית + נוסחאות). "
                    "בכמה מעלות עם כיוון השעון צריך לסובב אותו כדי שהטקסט יהיה זקוף וקריא? "
                    "ענה במספר אחד בלבד: 0, 90, 180 או 270."
                )},
            ],
        }],
    )
    txt = "".join(b.text for b in msg.content if b.type == "text")
    for cand in ("180", "270", "90", "0"):
        if cand in txt:
            return int(cand)
    return 0


# ─── holistic math analysis ───────────────────────────────────────────────────

# Two rules below are load-bearing and were added 6/8/2026 after the first valid
# variance measurement (products/math-form-checker/variance_check.py). Both are
# there to make the *same page* produce the *same report* twice, which matters
# more to a teacher than any single grade being right.
#
# 1. "יחידת הבדיקה". The prompt never said what one entry in `problems` is, and
#    the id example listed "1" and "2א" side by side — both granularities were
#    legal. A six-page algebra exam came back with 17 problems on one run and 11
#    on the next two: sub-parts counted separately, then folded into their
#    parent exercise. That is not a score wobbling, it is a structurally
#    different document.
#
# 2. points_max when nothing is printed. The old wording said "הערך סביר
#    (ברירת מחדל 10)" — an estimate and a fixed default in the same breath. One
#    geometry page scored 6/10, 9/15, 6/10 across three runs: the same judgement
#    (60%) with an invented denominator. Pinning it to exactly 10 removes a
#    whole axis of variation for free.
#
# Measured noise floor before these rules, Sonnet 5, five real exam files, three
# runs: ~7 percentage points average, 10 worst. Re-measure before assuming any
# prompt change here helped.

ANALYSIS_PROMPT = r"""אתה בודק מבחנים במתמטיקה וגאומטריה (שכבת חטיבה/תיכון) בישראל.
לפניך עמוד סרוק של פתרון תלמיד בכתב יד. נתח את העמוד כיחידה שלמה — אל תחתוך לשורות.
שים לב לסרטוטים, הוכחות גאומטריות ומעברים אלגבריים שפרושים על פני הדף.

כללים:
• כל ביטוי מתמטי ב-LaTeX תקני (תוכן בלבד, ללא $, למשל: x^2-6x+10 , \triangle ABC \cong \triangle DEF).
• אם צעד שגוי — ok=false והסבר ב-comment מה הטעות (חישובית / שיטתית / הגיונית / כתיב מתמטי).
• אם משהו לא קריא — כתוב "[לא קריא]" באותו שדה.
• verdict לפי הפתרון כולו: correct / partial / incorrect / unclear.
• feedback: משוב קצר ובונה לתלמיד בעברית (2-3 משפטים). כלול מה טוב ומה צריך שיפור.
• יחידת הבדיקה — מה נחשב פריט אחד ברשימת problems:
  פריט אחד לכל **יחידה שנענית בנפרד**. אם לתרגיל יש סעיפים (א, ב, ג) — כל סעיף
  הוא פריט נפרד ו-id שלו הוא "3א", "3ב" וכולי. אם אין סעיפים — התרגיל כולו
  פריט אחד. אל תאחד סעיפים לפריט אחד, ואל תפצל סעיף יחיד לכמה פריטים.
  החלוקה חייבת להיות עקבית: אותו דף בדיוק חייב להניב את אותה רשימת פריטים בכל
  בדיקה, גם אם היא נעשית שוב מחר.
• ניקוד מספרי:
  – points_max: הניקוד המקסימלי לפריט. אם סופקה רובריקה/מחוון — לפיה בדיוק.
    אחרת לפי הניקוד המודפס ליד התרגיל או הסעיף בדף. אם אין ניקוד מודפס —
    **10 בדיוק**, ולא הערכה משלך.
  – points_earned: הניקוד שמגיע לתלמיד לפי איכות הפתרון. מספר בין 0 ל-points_max,
    חצאי נקודות מותרים. תן ניקוד חלקי הוגן לפתרון חלקי (אל תאפס בגלל טעות אחת).
• score_suggestion: נימוק קצר בעברית לניקוד שנתת, למשל "טעות חישוב בצעד האחרון (-3)"
  / "פתרון מלא ומנומק" / "לא ענה". זה נימוק, לא חובה לכלול בו מספר.
• id: מספר הפריט בדיוק כפי שמופיע בדף, כולל אות הסעיף אם יש (למשל "1", "3א",
  "3ב"). אם לא ברור — "?".
• topic: אחת מ: אלגברה / גאומטריה / חשבון / הסתברות / טריגונומטריה / מש"ב / אחר.

הניקוד שאתה מציע הוא **הצעה לבדיקת המורה** — המורה יאשר או יתקן. היה הוגן ועקבי."""

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "page_summary": {"type": "string"},
        "has_diagram": {"type": "boolean"},
        "diagram_description": {"type": "string"},
        "problems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":               {"type": "string"},
                    "topic":            {"type": "string"},
                    "statement_latex":  {"type": "string"},
                    "student_steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "latex":   {"type": "string"},
                                "ok":      {"type": "boolean"},
                                "comment": {"type": "string"},
                            },
                            "required": ["latex", "ok", "comment"],
                            "additionalProperties": False,
                        },
                    },
                    "final_answer_latex": {"type": "string"},
                    "verdict":            {"type": "string"},
                    "points_max":         {"type": "number"},
                    "points_earned":      {"type": "number"},
                    "score_suggestion":   {"type": "string"},
                    "feedback":           {"type": "string"},
                },
                "required": [
                    "id", "topic", "statement_latex", "student_steps",
                    "final_answer_latex", "verdict", "points_max", "points_earned",
                    "score_suggestion", "feedback",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["page_summary", "has_diagram", "diagram_description", "problems"],
    "additionalProperties": False,
}


def _strip_additional_props(node):
    """Gemini's response_schema is an OpenAPI subset that rejects the
    additionalProperties key our Anthropic schema carries. Drop it recursively."""
    if isinstance(node, dict):
        return {k: _strip_additional_props(v) for k, v in node.items()
                if k != "additionalProperties"}
    if isinstance(node, list):
        return [_strip_additional_props(v) for v in node]
    return node


_GEMINI_SCHEMA = _strip_additional_props(ANALYSIS_SCHEMA)

# Groq and Azure have no schema enforcement, only "respond with JSON" — so the
# shape has to be spelled out in the prompt itself.
_JSON_CONTRACT = (
    "\n\nהחזר JSON יחיד בלבד, בדיוק במבנה:\n"
    '{"page_summary":"...","has_diagram":true,"diagram_description":"...",'
    '"problems":[{"id":"1","topic":"אלגברה","statement_latex":"...",'
    '"student_steps":[{"latex":"...","ok":true,"comment":"..."}],'
    '"final_answer_latex":"...","verdict":"correct|partial|incorrect|unclear",'
    '"points_max":10,"points_earned":7,"score_suggestion":"...","feedback":"..."}]}'
)

_ANALYSIS_USER_TURN = "נתח את עמוד הפתרון וחזור JSON."


def _rubric_block(rubric: str) -> str:
    """Per-request rubric text, appended to the user turn (not the cached system
    prompt) so the teacher's grading key conditions scoring without busting the
    prompt cache."""
    rubric = (rubric or "").strip()
    if not rubric:
        return ""
    return (
        "\n\nרובריקה / מחוון מהמורה — נקד לפיה בדיוק (points_max ו-points_earned "
        "לכל תרגיל יתאימו לה):\n" + rubric
    )


# ─── the school's reference solution ─────────────────────────────────────────
# Added 2026-08-05 for Avishai's maths form, which offers "צירוף פתרון בית ספר":
# the teacher attaches the school's own worked solution so the model knows what
# the expected answer and method look like.
#
# The instruction below is longer than it looks like it needs to be, and every
# clause is load-bearing. Handed a worked solution with no framing, a vision
# model treats it as the definition of correctness and marks down any student
# who took a different route — which is precisely the failure mode a maths
# teacher will not forgive, because alternative valid methods are the norm in
# algebra and near-universal in geometry proofs. The reference is authoritative
# about the *answer*, advisory about the *path*.
#
# It also has to survive the mundane case where the reference covers problems
# this particular page does not (a solution sheet for the whole exam, one page
# of student work), which would otherwise invite the model to invent problems
# the student never attempted.

# "התרגיל והפתרון המוצע" is the wording of Avishai's question, and the intro
# matches it: what arrives may be the worked solution alone or the exam paper
# together with it. Saying "the exam and/or its proposed solution" covers both
# without ever describing a blank exam paper as a solution — which would invite
# the model to mark the student against an empty page.
_SOLUTION_INTRO = (
    "חומר עזר מהמורה — התרגיל ו/או הפתרון המוצע של בית הספר. לעיונך בלבד, "
    "הוא **לא** חלק מעבודת התלמיד:"
)

_SOLUTION_RULES = (
    "\n\nכיצד להשתמש בחומר הזה:\n"
    "• אם יש בו פתרון — הוא קובע מהי **התשובה הנכונה** ומהי הדרך המצופה.\n"
    "• אם יש בו רק את דף התרגיל בלי פתרון — השתמש בו כדי לקרוא את ניסוח "
    "התרגילים ואת הניקוד המודפס, ואל תתייחס אליו כאל תשובות.\n"
    "• הוא **דרך אחת** ולא היחידה. תלמיד שהגיע לתשובה נכונה בדרך אחרת ותקפה "
    "מקבל ניקוד מלא — אל תוריד נקודות על עצם הסטייה מהדרך שבפתרון.\n"
    "• נתח רק את התרגילים שמופיעים בדף התלמיד. אם הפתרון מכסה תרגילים שאינם "
    "בדף — התעלם מהם, ואל תמציא עבורם ערכים ב-problems.\n"
    "• אם ניסוח התרגיל בפתרון סותר את המודפס בדף התלמיד — הדף קובע.\n"
    "• אל תצטט את הפתרון ב-feedback כאילו התלמיד כתב אותו."
)


def _solution_text_block(solution_text: str) -> str:
    """The typed reference solution, as a labelled fence in the user turn.

    Fenced and labelled for the same reason build_rubric fences its blocks: an
    unlabelled wall of worked maths dropped next to the rubric reads as more
    marking criteria, and the model starts scoring the student against the
    rubric's *wording* rather than against the solution's answers."""
    solution_text = (solution_text or "").strip()
    if not solution_text:
        return ""
    return f"\n\n{_SOLUTION_INTRO}\n{solution_text}{_SOLUTION_RULES}"


def _solution_note_for_images(n: int) -> str:
    """The text that explains the solution *images* that precede the student's
    page in the same turn. Without it the model sees several pages and no way to
    tell whose handwriting is being graded — and the reference, being neat and
    correct, is the one it will happily award full marks to."""
    if n <= 0:
        return ""
    which = "התמונה הראשונה" if n == 1 else f"{n} התמונות הראשונות"
    return (f"\n\n{which} {'היא' if n == 1 else 'הן'} {_SOLUTION_INTRO} "
            f"עמוד התלמיד לבדיקה הוא **התמונה האחרונה**.{_SOLUTION_RULES}")


def analyze_page(img: Image.Image, model_key: str = DEFAULT_MODEL,
                 rubric: str = "", solution_text: str = "",
                 solution_imgs: list | None = None) -> dict:
    """Holistic full-page math analysis. Dispatches to the chosen provider;
    every backend returns the same structured JSON (ANALYSIS_SCHEMA shape).

    An optional rubric conditions the numeric scoring. An optional reference
    solution — typed text, scanned pages, or both — tells the model what the
    expected answer and method are; see _SOLUTION_RULES for how it is framed.
    """
    solution_imgs = list(solution_imgs or [])
    if model_key in _ANTHROPIC_IDS:
        return _analyze_anthropic(img, _ANTHROPIC_IDS[model_key], rubric,
                                  solution_text, solution_imgs)
    if model_key in _GEMINI_IDS:
        return _analyze_gemini(img, _GEMINI_IDS[model_key], rubric,
                               solution_text, solution_imgs)
    if model_key in _GROQ_IDS:
        return _analyze_groq(img, _GROQ_IDS[model_key], rubric,
                             solution_text, solution_imgs)
    if model_key in _AZURE_KEYS:
        return _analyze_azure(img, rubric, solution_text, solution_imgs)
    return _analyze_anthropic(img, _ANTHROPIC_IDS[DEFAULT_MODEL], rubric,
                              solution_text, solution_imgs)


def _analyze_anthropic(img: Image.Image, model_id: str, rubric: str = "",
                       solution_text: str = "",
                       solution_imgs: list | None = None) -> dict:
    solution_imgs = list(solution_imgs or [])

    # The reference pages go FIRST, and the last of them carries a cache
    # breakpoint. Every page of every student in one submission shares the same
    # reference, so this turns a per-page cost into a per-submission one — on a
    # class of thirty that is the difference between paying for the solution
    # once and paying for it thirty times. It only works because the blocks
    # before the breakpoint are byte-identical across those calls, which is why
    # the student's page must come after it and not before.
    content: list = [_image_block_anthropic(s, MAX_EDGE) for s in solution_imgs]
    if content:
        content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
    content.append(_image_block_anthropic(img, MAX_EDGE))
    content.append({
        "type": "text",
        "text": (_ANALYSIS_USER_TURN
                 + _solution_note_for_images(len(solution_imgs))
                 + _solution_text_block(solution_text)
                 + _rubric_block(rubric)),
    })

    response = _anthropic().messages.create(
        model=model_id,
        max_tokens=_ANALYSIS_MAX_TOKENS,
        thinking={"type": "disabled"},  # load-bearing — see _ANALYSIS_MAX_TOKENS
        system=[{
            "type": "text",
            "text": ANALYSIS_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        output_config={
            "format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA}
        },
        messages=[{"role": "user", "content": content}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _analyze_gemini(img: Image.Image, model_id: str, rubric: str = "",
                    solution_text: str = "",
                    solution_imgs: list | None = None) -> dict:
    from google.genai import types
    solution_imgs = list(solution_imgs or [])
    parts = [types.Part.from_bytes(data=_jpeg_bytes(s, MAX_EDGE),
                                   mime_type="image/jpeg")
             for s in solution_imgs]
    parts.append(types.Part.from_bytes(data=_jpeg_bytes(img, MAX_EDGE),
                                       mime_type="image/jpeg"))
    parts.append(_ANALYSIS_USER_TURN
                 + _solution_note_for_images(len(solution_imgs))
                 + _solution_text_block(solution_text)
                 + _rubric_block(rubric))
    response = _gemini().models.generate_content(
        model=model_id,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=ANALYSIS_PROMPT,
            response_mime_type="application/json",
            response_schema=_GEMINI_SCHEMA,
            # Gemini 2.5 thinking tokens count against this budget; keep it
            # generous or the JSON gets truncated mid-string before it closes.
            max_output_tokens=24000,
        ),
    )
    return json.loads(response.text)


def _analyze_openai_compatible(client_obj, model_id: str, img: Image.Image,
                               rubric: str = "", solution_text: str = "",
                               solution_imgs: list | None = None) -> dict:
    """Shared call shape for Groq and Azure (both OpenAI-compatible vision)."""
    solution_imgs = list(solution_imgs or [])

    def image_url(im):
        b64 = base64.standard_b64encode(_jpeg_bytes(im, MAX_EDGE)).decode()
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    parts = [image_url(s) for s in solution_imgs]
    parts.append(image_url(img))
    parts.append({
        "type": "text",
        "text": ("נתח את עמוד הפתרון."
                 + _solution_note_for_images(len(solution_imgs))
                 + _solution_text_block(solution_text)
                 + _rubric_block(rubric)
                 + _JSON_CONTRACT),
    })

    response = client_obj.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": ANALYSIS_PROMPT},
            {"role": "user", "content": parts},
        ],
        response_format={"type": "json_object"},
        max_tokens=8000,
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def _analyze_groq(img: Image.Image, model_id: str, rubric: str = "",
                  solution_text: str = "",
                  solution_imgs: list | None = None) -> dict:
    return _analyze_openai_compatible(_groq(), model_id, img, rubric,
                                      solution_text, solution_imgs)


def _analyze_azure(img: Image.Image, rubric: str = "", solution_text: str = "",
                   solution_imgs: list | None = None) -> dict:
    return _analyze_openai_compatible(
        _azure(), _azure_deployment(), img, rubric, solution_text, solution_imgs)


# ─── page pipeline ────────────────────────────────────────────────────────────

STAGE_LABELS = {
    "orient":  "מזהה סיבוב",
    "analyze": "מנתח מתמטיקה",
}
_STEPS_PER_PAGE = 2  # orient, analyze


MAX_SOLUTION_PAGES = 6


def solution_pages(file_bytes: bytes, ext: str, auto_orient: bool = True
                   ) -> list[Image.Image]:
    """Render one reference-solution file into analysis-resolution page images.

    Rendered once per submission rather than once per student page: the images
    are then reused verbatim across every analyze_page call, which is both what
    makes Anthropic prompt caching hit and what keeps a thirty-student folder
    from re-rendering the same solution thirty times.

    Orientation is corrected here for the same reason — one detection call
    per solution page for the whole submission. A sideways reference is worse than
    no reference: the model reads it badly and marks the student against a
    misread answer.

    Capped at MAX_SOLUTION_PAGES. A teacher who attaches the whole answer
    booklet would otherwise add its full page count to *every* student page's
    input, and the cost is multiplied by the class size.
    """
    imgs: list[Image.Image] = []
    for idx, img in enumerate(iter_file_pages(file_bytes, ext)):
        if idx >= MAX_SOLUTION_PAGES:
            log.warning("reference solution has more than %d pages — the rest "
                        "are ignored", MAX_SOLUTION_PAGES)
            break
        if auto_orient:
            img = apply_rotation(img, detect_rotation(img))
        imgs.append(_downscale(img, MAX_EDGE))
    return imgs


def process_stream(file_bytes: bytes, ext: str, auto_orient: bool = True,
                   model_key: str = DEFAULT_MODEL, model_label: str = "",
                   rubric: str = "", keep_imgs: bool = True,
                   solution_text: str = "", solution_imgs: list | None = None):
    """Generator: yields SSE-ready progress dicts, then a final result dict.

    keep_imgs=False (CLI / form-checker): לא צוברים את תמונות העמודים לטובת
    re-analysis — חוסך ~10MB לעמוד בזיכרון. ה-UI צריך אותן (סיבוב ידני), ה-batch לא.

    solution_text / solution_imgs: the school's reference solution, already
    resolved by the caller (see solution_pages). Passed through unchanged to
    every page's analyze_page call."""
    # רינדור עצל: עמוד אחד ב-200 DPI מוחזק בזיכרון בכל רגע (חשוב ל-PDF ארוך
    # על instance קטן). total נקרא ממבנה ה-PDF בלי לרנדר.
    total = count_pages(file_bytes, ext)
    pages_imgs = iter_file_pages(file_bytes, ext)
    solution_imgs = list(solution_imgs or [])
    short_label = (model_label.split("·")[0].strip() or "מודל")
    log.info("[job] %d page(s) to render at %d DPI; model=%s auto_orient=%s "
             "solution=%s",
             total, RENDER_DPI, model_key, auto_orient,
             _describe_solution(solution_text, solution_imgs))

    def progress(stage: str, page: int, completed: int) -> dict:
        # Emitted *before* each stage runs, so the label reflects the stage
        # currently executing (the analysis pass is the slow one).
        label = STAGE_LABELS.get(stage, stage)
        if stage == "analyze":
            label = f"{label} ({short_label})"
        return {
            "type": "progress",
            "page": page,
            "total_pages": total,
            "stage": stage,
            "label": label,
            "pct": round(100 * completed / max(1, total * _STEPS_PER_PAGE)),
        }

    results = []
    imgs = []  # oriented analysis-resolution images, kept for re-analysis
    for idx, img in enumerate(pages_imgs):
        p = idx + 1
        base = idx * _STEPS_PER_PAGE

        yield progress("orient", p, base)
        if auto_orient:
            t0 = time.time()
            rotation = detect_rotation(img)
            img = apply_rotation(img, rotation)
            log.info("[page %d] orient → %d° (%.1fs)", p, rotation, time.time() - t0)
        else:
            rotation = 0

        yield progress("analyze", p, base + 1)
        img = _downscale(img, MAX_EDGE)
        t0 = time.time()
        analysis = analyze_page(img, model_key, rubric,
                                solution_text, solution_imgs)
        log.info(
            "[page %d] model=%s rotation=%d° problems=%d verdicts=%s (%.1fs)",
            p, model_key, rotation,
            len(analysis.get("problems", [])),
            [pr.get("verdict") for pr in analysis.get("problems", [])],
            time.time() - t0,
        )

        results.append({
            "page": p,
            "rotation_applied": rotation,
            "image_b64": _img_to_b64_jpeg(img, PREVIEW_MAX_EDGE, quality=85),
            "analysis": analysis,
        })
        if keep_imgs:
            imgs.append(img)

    yield {"type": "result", "pages": results, "imgs": imgs}


def _describe_solution(solution_text: str, solution_imgs: list | None) -> str:
    """One-line log summary of what reference solution a run actually had.

    Worth its own function because "the grade looks wrong" is nearly always
    answered by whether the solution reached the model at all, and a folder that
    silently produced no reference looks identical in the logs otherwise."""
    bits = []
    if solution_imgs:
        bits.append(f"{len(solution_imgs)} page(s)")
    text = (solution_text or "").strip()
    if text:
        bits.append(f"{len(text)} chars of text")
    return " + ".join(bits) if bits else "none"


def check_file(file_bytes: bytes, ext: str, rubric: str = "",
               model_key: str = DEFAULT_MODEL, auto_orient: bool = True,
               solution_text: str = "", solution_imgs: list | None = None
               ) -> list[dict]:
    """One-shot wrapper for non-streaming callers (form-checker, CLI): runs the
    whole pipeline and returns just the page list that math_report expects."""
    _key, label = resolve_model(model_key)
    for ev in process_stream(file_bytes, ext, auto_orient, _key, label,
                             rubric, keep_imgs=False,
                             solution_text=solution_text,
                             solution_imgs=solution_imgs):
        if ev["type"] == "result":
            return ev["pages"]
    raise RuntimeError("process_stream ended without a result event")
