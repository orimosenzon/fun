"""core.py — the checking core of oris-scanner (language checking), no Flask.

Based on the same pipeline architecture as math-checker/scan2 (render pages →
fix orientation → holistic analysis → structured JSON), with two key changes:
1. The checking content is language (written text, spelling/grammar/syntax/
   phrasing errors) instead of math.
2. The default model is Gemini, not Claude — for both analysis and
   orientation detection — so it never depends on an Anthropic key at all
   (that was the failure point last time: when Anthropic credit ran out,
   every call failed and counted as a 500 against Pub/Sub).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import time

import fitz  # PyMuPDF
from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv(override=True)

log = logging.getLogger("oris-scanner")

RENDER_DPI = 200          # A4 @ 200 DPI → ~1654×2339 px
MAX_EDGE = 2200           # longest edge sent to the model for analysis
ORIENT_MAX_EDGE = 900     # longest edge sent for fast orientation check
PREVIEW_MAX_EDGE = 1400   # longest edge stored in result for display/report

# Default model: Gemini (only requires GEMINI_API_KEY, not ANTHROPIC_API_KEY).
# Other options remain available (as in math-checker) in case we want to
# compare quality.
MODELS = {
    "gemini-flash":     "Gemini 2.5 Flash · ברירת מחדל",
    "gemini-lite":      "Gemini 2.5 Flash-Lite · זול וזריז יותר",
    "opus":             "Claude Opus 4.8 · איכות מרבית (Anthropic)",
    "sonnet":           "Claude Sonnet 4.6 · מהיר וזול יותר (Anthropic)",
    "haiku":            "Claude Haiku 4.5 · הכי זול (Anthropic, לא מומלץ)",
}

_ANTHROPIC_IDS = {
    "opus":   "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku":  "claude-haiku-4-5-20251001",
}
_GEMINI_IDS = {
    "gemini-flash": "gemini-2.5-flash",
    "gemini-lite":  "gemini-2.5-flash-lite",
}

DEFAULT_MODEL = "gemini-flash"
ORIENT_MODEL_KEY = "gemini-flash"  # orientation is also on Gemini (not Haiku/Anthropic)


def resolve_model(key: str | None) -> tuple[str, str]:
    k = key if key in MODELS else DEFAULT_MODEL
    return k, MODELS[k]

ACCEPTED_EXTS = (".pdf", ".jpg", ".jpeg", ".png")


# ─── file handling (same as math-checker) ─────────────────────────────────────

def pdf_to_pages(pdf_bytes: bytes) -> list[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    imgs = []
    for page in doc:
        pix = page.get_pixmap(dpi=RENDER_DPI)
        imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return imgs


def file_to_pages(data: bytes, ext: str) -> list[Image.Image]:
    if ext == ".pdf":
        return pdf_to_pages(data)
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    return [img.convert("RGB")]


def count_pages(data: bytes, ext: str) -> int:
    if ext != ".pdf":
        return 1
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return len(doc)
    finally:
        doc.close()


def iter_file_pages(data: bytes, ext: str):
    if ext != ".pdf":
        yield from file_to_pages(data, ext)
        return
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=RENDER_DPI)
            yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def _downscale(img: Image.Image, max_edge: int) -> Image.Image:
    long_edge = max(img.size)
    if long_edge <= max_edge:
        return img
    scale = max_edge / long_edge
    return img.resize(
        (round(img.width * scale), round(img.height * scale)),
        Image.LANCZOS,
    )


def _jpeg_bytes(img: Image.Image, max_edge: int | None = None, quality: int = 90) -> bytes:
    if max_edge:
        img = _downscale(img, max_edge)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _img_to_b64_jpeg(img: Image.Image, max_edge: int | None = None, quality: int = 90) -> str:
    return base64.standard_b64encode(_jpeg_bytes(img, max_edge, quality)).decode()


# ─── lazy provider clients ─────────────────────────────────────────────────────
_gemini_client = None
_anthropic_client = None


def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _image_block_anthropic(img: Image.Image, max_edge: int) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": _img_to_b64_jpeg(img, max_edge),
        },
    }


# ─── orientation detection (Gemini by default) ────────────────────────────────

_ORIENT_PROMPT = (
    "זהו עמוד סרוק של תרגיל שפה בכתב יד. "
    "בכמה מעלות עם כיוון השעון צריך לסובב אותו כדי שהטקסט יהיה זקוף וקריא? "
    "ענה במספר אחד בלבד: 0, 90, 180 או 270."
)


def detect_rotation(img: Image.Image, model_key: str = ORIENT_MODEL_KEY) -> int:
    if model_key in _GEMINI_IDS:
        from google.genai import types
        response = _gemini().models.generate_content(
            model=_GEMINI_IDS[model_key],
            contents=[
                types.Part.from_bytes(data=_jpeg_bytes(img, ORIENT_MAX_EDGE), mime_type="image/jpeg"),
                _ORIENT_PROMPT,
            ],
            config=types.GenerateContentConfig(max_output_tokens=10),
        )
        txt = response.text or ""
    else:
        msg = _anthropic().messages.create(
            model=_ANTHROPIC_IDS.get(model_key, _ANTHROPIC_IDS["haiku"]),
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": [_image_block_anthropic(img, ORIENT_MAX_EDGE),
                            {"type": "text", "text": _ORIENT_PROMPT}],
            }],
        )
        txt = "".join(b.text for b in msg.content if b.type == "text")

    for cand in ("180", "270", "90", "0"):
        if cand in txt:
            return int(cand)
    return 0


def apply_rotation(img: Image.Image, deg: int) -> Image.Image:
    if deg == 0:
        return img
    return img.rotate(-deg, expand=True, resample=Image.BILINEAR, fillcolor=(255, 255, 255))


# ─── holistic language analysis ────────────────────────────────────────────────

ANALYSIS_PROMPT = r"""אתה בודק תרגילי שפה (הבעה, דקדוק, איות, הבנת הנקרא) בבית ספר בישראל.
לפניך עמוד סרוק של תשובת תלמיד בכתב יד. נתח את העמוד כיחידה שלמה.

כללים:
• transcribed_text: תעתיק מדויק (ככל האפשר) של מה שהתלמיד כתב בפועל, כולל שגיאות איות
  כפי שהן (אל תתקן בתעתיק עצמו). אם משהו לא קריא — כתוב "[לא קריא]" באותו מקום.
• errors: רשימת שגיאות שנמצאו. לכל שגיאה:
  – type: אחת מ: איות / דקדוק / תחביר / ניסוח / תוכן / אחר.
  – quote: הציטוט המדויק מתוך מה שהתלמיד כתב (כפי שהוא, עם השגיאה).
  – correction: הצורה המתוקנת.
  – comment: הסבר קצר לתלמיד למה זו שגיאה ואיך לתקן.
• אם אין שגיאות בקטע מסוים — אל תמציא; פשוט אל תכלול אותו ב-errors.
• verdict לפי התשובה כולה: correct / partial / incorrect / unclear.
• feedback: משוב קצר ובונה לתלמיד בעברית (2-3 משפטים) — מה טוב ומה לשפר.
• ניקוד מספרי:
  – points_max: הניקוד המקסימלי לתרגיל. אם סופקה רובריקה/מחוון — לפיה בדיוק.
    אחרת לפי הניקוד המודפס ליד התרגיל בדף, ואם אין — הערך סביר (ברירת מחדל 10).
  – points_earned: הניקוד שמגיע לתלמיד. מספר בין 0 ל-points_max, חצאי נקודות מותרים.
    תן ניקוד חלקי הוגן (אל תאפס בגלל שגיאה אחת קטנה).
• score_suggestion: נימוק קצר בעברית לניקוד, למשל "שתי שגיאות איות, אחרת תשובה מלאה (-2)".
• id: מספר/אות התרגיל בדיוק כפי שמופיע בדף (למשל "1", "2א", "ב"). אם לא ברור — "?".
• exercise_prompt: השאלה/ההנחיה של התרגיל כפי שמופיעה בדף, אם קיימת (אחרת מחרוזת ריקה).

הניקוד שאתה מציע הוא **הצעה לבדיקת המורה** — המורה יאשר או יתקן. היה הוגן ועקבי."""

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "page_summary": {"type": "string"},
        "exercises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":               {"type": "string"},
                    "exercise_prompt":  {"type": "string"},
                    "transcribed_text": {"type": "string"},
                    "errors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type":       {"type": "string"},
                                "quote":      {"type": "string"},
                                "correction": {"type": "string"},
                                "comment":    {"type": "string"},
                            },
                            "required": ["type", "quote", "correction", "comment"],
                            "additionalProperties": False,
                        },
                    },
                    "verdict":          {"type": "string"},
                    "points_max":       {"type": "number"},
                    "points_earned":    {"type": "number"},
                    "score_suggestion": {"type": "string"},
                    "feedback":         {"type": "string"},
                },
                "required": [
                    "id", "exercise_prompt", "transcribed_text", "errors",
                    "verdict", "points_max", "points_earned",
                    "score_suggestion", "feedback",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["page_summary", "exercises"],
    "additionalProperties": False,
}


def _strip_additional_props(node):
    """Gemini's response_schema is an OpenAPI subset that rejects additionalProperties."""
    if isinstance(node, dict):
        return {k: _strip_additional_props(v) for k, v in node.items()
                if k != "additionalProperties"}
    if isinstance(node, list):
        return [_strip_additional_props(v) for v in node]
    return node


_GEMINI_SCHEMA = _strip_additional_props(ANALYSIS_SCHEMA)

_ANALYSIS_USER_TURN = "נתח את עמוד התשובה וחזור JSON."


def _rubric_block(rubric: str) -> str:
    rubric = (rubric or "").strip()
    if not rubric:
        return ""
    return (
        "\n\nרובריקה / מחוון מהמורה — נקד לפיה בדיוק (points_max ו-points_earned "
        "לכל תרגיל יתאימו לה):\n" + rubric
    )


def analyze_page(img: Image.Image, model_key: str = DEFAULT_MODEL,
                 rubric: str = "") -> dict:
    if model_key in _GEMINI_IDS:
        return _analyze_gemini(img, _GEMINI_IDS[model_key], rubric)
    if model_key in _ANTHROPIC_IDS:
        return _analyze_anthropic(img, _ANTHROPIC_IDS[model_key], rubric)
    return _analyze_gemini(img, _GEMINI_IDS[DEFAULT_MODEL], rubric)


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
            max_output_tokens=24000,
        ),
    )
    return json.loads(response.text)


def _analyze_anthropic(img: Image.Image, model_id: str, rubric: str = "") -> dict:
    response = _anthropic().messages.create(
        model=model_id,
        max_tokens=6000,
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


# ─── error handling ───────────────────────────────────────────────────────────

def humanize_error(exc: Exception) -> dict:
    """Turn an SDK exception into a user-readable Hebrew error payload."""
    raw = (str(exc) or repr(exc)).strip()
    tech = raw[:2000]
    msg = "תקלה לא צפויה"
    details = "אירעה שגיאה לא מזוהה. נסה שוב; אם זה חוזר צרף את הפרטים הטכניים."

    name = type(exc).__name__
    if "RateLimit" in name:
        msg, details = "חרגת ממכסת המודל לרגע זה", "נסה שוב בעוד דקה."
    elif "Authentication" in name or (isinstance(exc, KeyError) and "API_KEY" in str(exc.args[:1])):
        msg, details = "מפתח API לא תקף/חסר", "ודא ש-GEMINI_API_KEY (או ANTHROPIC_API_KEY) מוגדר נכון."
    elif "Timeout" in name or "Connection" in name:
        msg, details = "החיבור למודל נכשל", "ייתכן שיש בעיית רשת או שהשירות לא זמין כרגע. נסה שוב."
    elif isinstance(exc, (json.JSONDecodeError, ValueError)):
        msg, details = "המודל החזיר תשובה לא תקפה", "זו בדרך כלל בעיה זמנית — נסה שוב."

    return {"message": msg, "details": details, "technical": tech}


# ─── shared pipeline (yields progress events, then the result) ───────────────

STAGE_LABELS = {
    "orient":  "מזהה סיבוב",
    "analyze": "מנתח שפה",
}
_STEPS_PER_PAGE = 2  # orient, analyze


def process_stream(file_bytes: bytes, ext: str, auto_orient: bool,
                    model_key: str = DEFAULT_MODEL, model_label: str = "",
                    rubric: str = "", keep_imgs: bool = True):
    """Generator: yields SSE-ready progress dicts, then a final result dict."""
    total = count_pages(file_bytes, ext)
    pages_imgs = iter_file_pages(file_bytes, ext)
    short_label = (model_label.split("·")[0].strip() or "מודל")
    log.info("[job] %d page(s) to render at %d DPI; model=%s auto_orient=%s",
             total, RENDER_DPI, model_key, auto_orient)

    def progress(stage: str, page: int, completed: int) -> dict:
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
    imgs = []
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
            "[page %d] model=%s rotation=%d° exercises=%d verdicts=%s (%.1fs)",
            p, model_key, rotation,
            len(analysis.get("exercises", [])),
            [ex.get("verdict") for ex in analysis.get("exercises", [])],
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
