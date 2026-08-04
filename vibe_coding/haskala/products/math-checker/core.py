"""core.py — shim. הליבה עצמה עברה ל-`haskala_grading/math_core.py`.

עד 2026-08-05 הקובץ הזה החזיק את הליבה. כשנבנה `math-form-checker` הוא היה
עומד להיות ההעתק השני של אותו ANALYSIS_PROMPT ו-ANALYSIS_SCHEMA, אז הליבה
הועברה ל-`lib/haskala_grading/math_core.py` — מקור אמת יחיד לשני המוצרים.

הקובץ נשאר כדי ש-`app.py`, `check.py` ו-`poller.py` ימשיכו לעבוד עם
`import core` בלי שינוי. **אין לערוך כאן לוגיקה** — לערוך ב-lib/.

הערה על DEFAULT_MODEL: ב-lib ברירת המחדל היא Sonnet 5, כי היא נקבעה עבור
`math-form-checker` שבודק ללא אדם בלולאה ולכן רגיש לעלות. כאן — UI אינטראקטיבי
עם בורר מודל שהמורה רואה — נשמרת ברירת המחדל ההיסטורית Opus 4.8, כדי לא לשנות
בשקט את התנהגות ה-Space הפרוס.
"""
from __future__ import annotations

from haskala_grading.math_core import (  # noqa: F401
    ACCEPTED_EXTS,
    ANALYSIS_PROMPT,
    ANALYSIS_SCHEMA,
    MAX_EDGE,
    MODELS,
    ORIENT_MAX_EDGE,
    ORIENT_MODEL,
    PREVIEW_MAX_EDGE,
    RENDER_DPI,
    STAGE_LABELS,
    _ANALYSIS_USER_TURN,
    _ANTHROPIC_IDS,
    _AZURE_KEYS,
    _GEMINI_IDS,
    _GEMINI_SCHEMA,
    _GROQ_IDS,
    _JSON_CONTRACT,
    _downscale,
    _img_to_b64_jpeg,
    _jpeg_bytes,
    _rubric_block,
    _strip_additional_props,
    analyze_page,
    apply_rotation,
    check_file,
    count_pages,
    detect_rotation,
    file_to_pages,
    humanize_error,
    iter_file_pages,
    pdf_to_pages,
    process_stream,
)

# The Space's UI default stays Opus — see the module docstring.
DEFAULT_MODEL = "opus"


def resolve_model(key: str | None) -> tuple[str, str]:
    """Same as math_core.resolve_model, but an unknown or absent key falls back
    to this product's Opus default rather than the library's Sonnet 5.

    Deliberately not delegating to the library here: passing an unknown key
    through would land on Sonnet 5, silently changing what the Space grades
    with whenever a stale model name reaches it."""
    if key not in MODELS:
        return DEFAULT_MODEL, MODELS[DEFAULT_MODEL]
    return key, MODELS[key]
