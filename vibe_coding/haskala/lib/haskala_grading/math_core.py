"""math_core.py — ליבת בדיקת המתמטיקה, משותפת ל-math-checker ול-math-form-checker.

בדיקת מתמטיקה היא **לא** בעיית שורה-שורה כמו `core.py`. תרגיל מתמטי הוא יחידה
הוליסטית: סרטוטים, הוכחות גאומטריות ומעברים אלגבריים פרושים על פני הדף. לכן
ה-pipeline כאן שונה מזה של core.py:

    רינדור עמוד → תיקון אוריינטציה (Haiku) → ניתוח הוליסטי של העמוד → JSON מובנה

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
ORIENT_MODEL = "claude-haiku-4-5-20251001"  # orientation is always Haiku (cheap)

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
    baselines. Always Haiku — cheap enough to run per page, and reliable for
    a four-way choice. 180° flips are common in scanned exam stacks and a
    portrait/landscape heuristic misses them entirely.
    """
    msg = _anthropic().messages.create(
        model=ORIENT_MODEL,
        max_tokens=10,
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

ANALYSIS_PROMPT = r"""אתה בודק מבחנים במתמטיקה וגאומטריה (שכבת חטיבה/תיכון) בישראל.
לפניך עמוד סרוק של פתרון תלמיד בכתב יד. נתח את העמוד כיחידה שלמה — אל תחתוך לשורות.
שים לב לסרטוטים, הוכחות גאומטריות ומעברים אלגבריים שפרושים על פני הדף.

כללים:
• כל ביטוי מתמטי ב-LaTeX תקני (תוכן בלבד, ללא $, למשל: x^2-6x+10 , \triangle ABC \cong \triangle DEF).
• אם צעד שגוי — ok=false והסבר ב-comment מה הטעות (חישובית / שיטתית / הגיונית / כתיב מתמטי).
• אם משהו לא קריא — כתוב "[לא קריא]" באותו שדה.
• verdict לפי הפתרון כולו: correct / partial / incorrect / unclear.
• feedback: משוב קצר ובונה לתלמיד בעברית (2-3 משפטים). כלול מה טוב ומה צריך שיפור.
• ניקוד מספרי:
  – points_max: הניקוד המקסימלי לתרגיל. אם סופקה רובריקה/מחוון — לפיה בדיוק.
    אחרת לפי הניקוד המודפס ליד התרגיל בדף, ואם אין — הערך סביר (ברירת מחדל 10).
  – points_earned: הניקוד שמגיע לתלמיד לפי איכות הפתרון. מספר בין 0 ל-points_max,
    חצאי נקודות מותרים. תן ניקוד חלקי הוגן לפתרון חלקי (אל תאפס בגלל טעות אחת).
• score_suggestion: נימוק קצר בעברית לניקוד שנתת, למשל "טעות חישוב בצעד האחרון (-3)"
  / "פתרון מלא ומנומק" / "לא ענה". זה נימוק, לא חובה לכלול בו מספר.
• id: מספר/אות התרגיל בדיוק כפי שמופיע בדף (למשל "1", "2א", "ב"). אם לא ברור — "?".
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


def analyze_page(img: Image.Image, model_key: str = DEFAULT_MODEL,
                 rubric: str = "") -> dict:
    """Holistic full-page math analysis. Dispatches to the chosen provider;
    every backend returns the same structured JSON (ANALYSIS_SCHEMA shape).
    An optional rubric conditions the numeric scoring."""
    if model_key in _ANTHROPIC_IDS:
        return _analyze_anthropic(img, _ANTHROPIC_IDS[model_key], rubric)
    if model_key in _GEMINI_IDS:
        return _analyze_gemini(img, _GEMINI_IDS[model_key], rubric)
    if model_key in _GROQ_IDS:
        return _analyze_groq(img, _GROQ_IDS[model_key], rubric)
    if model_key in _AZURE_KEYS:
        return _analyze_azure(img, rubric)
    return _analyze_anthropic(img, _ANTHROPIC_IDS[DEFAULT_MODEL], rubric)


def _analyze_anthropic(img: Image.Image, model_id: str, rubric: str = "") -> dict:
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
        messages=[{
            "role": "user",
            "content": [
                _image_block_anthropic(img, MAX_EDGE),
                {"type": "text", "text": _ANALYSIS_USER_TURN + _rubric_block(rubric)},
            ],
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _analyze_gemini(img: Image.Image, model_id: str, rubric: str = "") -> dict:
    from google.genai import types
    response = _gemini().models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=_jpeg_bytes(img, MAX_EDGE), mime_type="image/jpeg"),
            _ANALYSIS_USER_TURN + _rubric_block(rubric),
        ],
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
                               rubric: str = "") -> dict:
    """Shared call shape for Groq and Azure (both OpenAI-compatible vision)."""
    b64 = base64.standard_b64encode(_jpeg_bytes(img, MAX_EDGE)).decode()
    response = client_obj.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": ANALYSIS_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",
                 "text": "נתח את עמוד הפתרון." + _rubric_block(rubric) + _JSON_CONTRACT},
            ]},
        ],
        response_format={"type": "json_object"},
        max_tokens=8000,
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def _analyze_groq(img: Image.Image, model_id: str, rubric: str = "") -> dict:
    return _analyze_openai_compatible(_groq(), model_id, img, rubric)


def _analyze_azure(img: Image.Image, rubric: str = "") -> dict:
    return _analyze_openai_compatible(
        _azure(), _azure_deployment(), img, rubric)


# ─── page pipeline ────────────────────────────────────────────────────────────

STAGE_LABELS = {
    "orient":  "מזהה סיבוב (Haiku)",
    "analyze": "מנתח מתמטיקה",
}
_STEPS_PER_PAGE = 2  # orient, analyze


def process_stream(file_bytes: bytes, ext: str, auto_orient: bool = True,
                   model_key: str = DEFAULT_MODEL, model_label: str = "",
                   rubric: str = "", keep_imgs: bool = True):
    """Generator: yields SSE-ready progress dicts, then a final result dict.

    keep_imgs=False (CLI / form-checker): לא צוברים את תמונות העמודים לטובת
    re-analysis — חוסך ~10MB לעמוד בזיכרון. ה-UI צריך אותן (סיבוב ידני), ה-batch לא."""
    # רינדור עצל: עמוד אחד ב-200 DPI מוחזק בזיכרון בכל רגע (חשוב ל-PDF ארוך
    # על instance קטן). total נקרא ממבנה ה-PDF בלי לרנדר.
    total = count_pages(file_bytes, ext)
    pages_imgs = iter_file_pages(file_bytes, ext)
    short_label = (model_label.split("·")[0].strip() or "מודל")
    log.info("[job] %d page(s) to render at %d DPI; model=%s auto_orient=%s",
             total, RENDER_DPI, model_key, auto_orient)

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
        analysis = analyze_page(img, model_key, rubric)
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


def check_file(file_bytes: bytes, ext: str, rubric: str = "",
               model_key: str = DEFAULT_MODEL, auto_orient: bool = True) -> list[dict]:
    """One-shot wrapper for non-streaming callers (form-checker, CLI): runs the
    whole pipeline and returns just the page list that math_report expects."""
    _key, label = resolve_model(model_key)
    for ev in process_stream(file_bytes, ext, auto_orient, _key, label,
                             rubric, keep_imgs=False):
        if ev["type"] == "result":
            return ev["pages"]
    raise RuntimeError("process_stream ended without a result event")
