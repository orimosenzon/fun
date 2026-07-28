"""core.py — the checking core of oris-scanner (language checking), no Flask.

This is a direct port of the checking pipeline from haskala/products/checker/app.py
(deployed at HF Space orimosenzon/haskala-ocr): OCR each page to line-tagged text,
then evaluate the merged transcript against a structured JSON rubric in one call.
Ported near-verbatim, minus everything that only exists to support checker's
interactive browser UI:
  - the numbered-box overlay OCR mode (segmentation.py) — checker's own code
    documents numbered=False as "the clean-page variant used when no crops are
    needed (English flow)", which is exactly this product's case.
  - SSE streaming / progress events — this runs as a batch Cloud Function, not
    behind a live progress bar.
Rotation detection stays oris-scanner's own (Gemini-vision based, already
verified end-to-end) rather than checker's pixel-projection auto_rotate —
that's a different, unrelated preprocessing step, not "the checking" itself.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time

import fitz  # PyMuPDF
from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv(override=True)

log = logging.getLogger("oris-scanner")

RENDER_DPI = 200          # A4 @ 200 DPI → ~1654×2339 px

ACCEPTED_EXTS = (".pdf", ".jpg", ".jpeg", ".png")


# ─── file handling ─────────────────────────────────────────────────────────

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


# A full-page reference image kept beside the transcript. It's a visual aid
# for the reviewer (checker/app.py:255-260), not OCR input — a small JPEG is
# plenty and keeps the report from ballooning.
ORIGINAL_MAX_EDGE = 1800


def original_preview_b64(img: Image.Image) -> str:
    """Downscaled JPEG of the raw page — a reviewing aid, not OCR input."""
    return _img_to_b64_jpeg(img, ORIGINAL_MAX_EDGE, quality=85)


# ─── lazy provider clients ─────────────────────────────────────────────────
_gemini_client = None
_anthropic_client = None
_groq_client = None
_azure_client = None


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


def _groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


def _azure():
    global _azure_client
    if _azure_client is None:
        from openai import OpenAI
        base = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/"
        _azure_client = OpenAI(base_url=base, api_key=os.environ["AZURE_OPENAI_API_KEY"])
    return _azure_client


def _image_block_anthropic(img: Image.Image, max_edge: int) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": _img_to_b64_jpeg(img, max_edge),
        },
    }


# ─── model / provider registry (mirrors checker/app.py MODELS) ─────────────
# Same 5 provider keys as checker. All 4 providers' credentials are now wired
# into Cloud Run (Secret Manager: gemini-api-key, anthropic-api-key,
# groq-api-key, azure-openai-api-key), so switching is just this one constant.
#
# DEFAULT_MODEL applies when the assignment carries no [model: …] tag — see
# resolve_model below, which is how a teacher overrides it per assignment.
#
# Back to gemini on 2026-07-26: its free-tier daily quota has refilled. The
# 2026-07-22 switch to azure-gpt41-mini was a workaround for that quota being
# exhausted, alongside two provider problems that may well still stand —
# the Anthropic account was out of balance, and Groq had dropped
# llama-4-scout (its only vision-capable model) from this key entirely, 404
# model_not_found on every attempt. So [model: claude] / [model: groq-scout]
# are the two tags most likely to fail; azure-gpt41-mini is the known-good
# fallback if Gemini's quota runs dry again.
MODELS = {
    "claude": "Claude (Opus 4.8)",
    "gemini": "Gemini (2.5 Flash)",
    "gemini-lite": "Gemini (2.5 Flash-Lite)",
    "groq-scout": "Groq (Llama 4 Scout — חינמי)",
    "azure-gpt41-mini": "GPT-4.1-mini (Azure)",
}
DEFAULT_MODEL = "gemini"

MODEL_FILENAME_LABEL = {
    "claude":      "claude-opus-4-8",
    "gemini":      "gemini-2.5-flash",
    "gemini-lite": "gemini-2.5-flash-lite",
    "groq-scout":  "llama-4-scout",
    "azure-gpt41-mini": "gpt-4.1-mini-azure",
}

# ─── teacher-selectable model (tag inside the assignment description) ───────
# Classroom's API exposes no free metadata field on a courseWork, so the only
# place a teacher can hand us a per-assignment setting through the normal
# Classroom UI is text they already type. The convention: a tag anywhere in
# the assignment description —
#
#     [model: claude]        (or [מודל: claude], [model=claude])
#
# parsed out here, then *removed* from the text before the description is
# passed on as the rubric's "question", so the model never sees the tag and
# the evaluation isn't polluted by it. The tag stays visible to students in
# Classroom itself; that's the price of using the description as the channel.
#
# An unrecognised model name does NOT fall through silently — resolve_model()
# reports it so the caller can log a warning, because quietly grading with the
# wrong model is worse than grading with the default and saying so.
# The value must start with a non-space character, so a bracketed phrase that
# merely happens to read like "[model: ]" isn't mistaken for a tag and deleted
# out of the assignment text.
_MODEL_TAG_RE = re.compile(
    r"\[\s*(?:model|מודל)\s*[:=]\s*([^\]\n\s][^\]\n]*?)\s*\]",
    re.IGNORECASE,
)

# Spellings a teacher might plausibly type, mapped onto MODELS keys. Keys of
# MODELS itself are always accepted; this is only for the aliases.
_MODEL_ALIASES = {
    "claude-opus": "claude",
    "opus": "claude",
    "anthropic": "claude",
    "gemini-flash": "gemini",
    "gemini-2.5-flash": "gemini",
    "flash-lite": "gemini-lite",
    "gemini-flash-lite": "gemini-lite",
    "groq": "groq-scout",
    "llama": "groq-scout",
    "scout": "groq-scout",
    "azure": "azure-gpt41-mini",
    "gpt": "azure-gpt41-mini",
    "gpt-4.1-mini": "azure-gpt41-mini",
    "gpt41-mini": "azure-gpt41-mini",
}


def resolve_model(description: str | None) -> tuple[str, str | None, str | None]:
    """Reads a [model: …] tag out of an assignment description.

    Returns (model_key, cleaned_description, warning):
      model_key   — a valid MODELS key; DEFAULT_MODEL when no usable tag.
      cleaned_description — the description with every model tag removed
                    (None stays None), ready to be used as the question.
      warning     — None on success, otherwise a human-readable reason the
                    tag was ignored, for the caller to log.
    """
    if not description:
        return DEFAULT_MODEL, description, None

    matches = _MODEL_TAG_RE.findall(description)
    cleaned = _MODEL_TAG_RE.sub("", description)
    # collapse the blank lines the removed tag leaves behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() or None

    if not matches:
        return DEFAULT_MODEL, description, None

    raw = matches[0].strip()
    key = raw.lower().replace("_", "-")
    key = _MODEL_ALIASES.get(key, key)
    if key not in MODELS:
        return DEFAULT_MODEL, cleaned, (
            f"unknown model {raw!r} in assignment description — "
            f"falling back to {DEFAULT_MODEL}. Valid: {', '.join(sorted(MODELS))}"
        )
    if len(matches) > 1:
        return key, cleaned, (
            f"{len(matches)} model tags in the description — using the first ({key})"
        )
    return key, cleaned, None


_ANTHROPIC_MODEL = "claude-opus-4-8"

_GEMINI_MODEL_IDS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "gemini-lite": "gemini-2.5-flash-lite",
}


def _is_gemini(provider: str) -> bool:
    return provider in _GEMINI_MODEL_IDS


def _gemini_model_id(provider: str) -> str:
    return _GEMINI_MODEL_IDS.get(provider, "gemini-2.5-flash")


_GROQ_MODEL_IDS: dict[str, str] = {
    "groq-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
}


def _is_groq(provider: str) -> bool:
    return provider in _GROQ_MODEL_IDS


def _groq_model_id(provider: str) -> str:
    return _GROQ_MODEL_IDS.get(provider, "meta-llama/llama-4-scout-17b-16e-instruct")


_AZURE_PROVIDERS = {"azure-gpt41-mini"}


def _is_azure(provider: str) -> bool:
    return provider in _AZURE_PROVIDERS


def _azure_deployment() -> str:
    return os.environ["AZURE_OPENAI_DEPLOYMENT"]


# ─── languages (mirrors checker/app.py LANGS) ───────────────────────────────
LANGS: dict[str, dict] = {
    "he": {"name_he": "עברית",  "name_native": "עברית",   "dir": "rtl", "illegible": "[לא קריא]"},
    "en": {"name_he": "אנגלית", "name_native": "English", "dir": "ltr", "illegible": "[illegible]"},
    "ar": {"name_he": "ערבית",  "name_native": "العربية", "dir": "rtl", "illegible": "[غير مقروء]"},
}
DEFAULT_EXERCISE_LANG = "en"
DEFAULT_FEEDBACK_LANG = "he"


# ─── orientation detection ──────────────────────────────────────────────────
# Follows the same model_key as OCR/eval (passed in from check_pages) rather
# than a separately hardcoded provider — previously this always called Gemini
# regardless of DEFAULT_MODEL, so switching providers to work around a Gemini
# outage still silently hit Gemini here. Fixed 2026-07-22.

_ORIENT_PROMPT = (
    "זהו עמוד סרוק של תרגיל שפה בכתב יד. "
    "בכמה מעלות עם כיוון השעון צריך לסובב אותו כדי שהטקסט יהיה זקוף וקריא? "
    "ענה במספר אחד בלבד: 0, 90, 180 או 270."
)

ORIENT_MAX_EDGE = 900


def detect_rotation(img: Image.Image, model_key: str = DEFAULT_MODEL) -> int:
    if _is_gemini(model_key):
        from google.genai import types
        response = _gemini().models.generate_content(
            model=_gemini_model_id(model_key),
            contents=[
                types.Part.from_bytes(data=_jpeg_bytes(img, ORIENT_MAX_EDGE), mime_type="image/jpeg"),
                _ORIENT_PROMPT,
            ],
            config=types.GenerateContentConfig(max_output_tokens=10),
        )
        txt = response.text or ""
    elif _is_groq(model_key):
        response = _groq().chat.completions.create(
            model=_groq_model_id(model_key),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{_img_to_b64_jpeg(img, ORIENT_MAX_EDGE)}"}},
                    {"type": "text", "text": _ORIENT_PROMPT},
                ],
            }],
            max_tokens=10,
            temperature=0,
        )
        txt = response.choices[0].message.content or ""
    elif _is_azure(model_key):
        response = _azure().chat.completions.create(
            model=_azure_deployment(),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{_img_to_b64_jpeg(img, ORIENT_MAX_EDGE)}"}},
                    {"type": "text", "text": _ORIENT_PROMPT},
                ],
            }],
            max_tokens=10,
            temperature=0,
        )
        txt = response.choices[0].message.content or ""
    else:
        msg = _anthropic().messages.create(
            model=_ANTHROPIC_MODEL,
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


# ─── OCR (ported from checker/app.py:96-252, 535-699) ──────────────────────

MAX_EDGE = 2400  # checker's OCR_MAX_EDGE — downscaled JPEG sent to whichever model

_USER_TURN = "תמלל שורה-שורה והחזר JSON."


def build_ocr_prompt(exercise_lang: str, numbered: bool = True) -> str:
    """OCR system prompt parameterized by the exercise's primary language.

    numbered=True binds transcribed lines to a numbered-box overlay drawn by
    checker's segmentation.py — unused here (oris-scanner always calls with
    numbered=False, checker's own "clean page / English flow" variant). The
    signature and branches are kept identical to checker/app.py for fidelity."""
    lang = LANGS.get(exercise_lang, LANGS[DEFAULT_EXERCISE_LANG])
    primary = lang["name_native"]
    illegible = lang["illegible"]
    is_rtl_exercise = lang["dir"] == "rtl"
    rtl_emphasis = (
        f"השפה העיקרית ({primary}) נכתבת מימין לשמאל. שים לב מיוחד לכלל 9.\n\n"
        if is_rtl_exercise else ""
    )
    overlay_intro = (
        """על התמונה צוירו מלבנים כחולים ממוספרים (המספרים בירוק, משמאל
לכל מלבן). המלבנים והמספרים הם סימון סינתטי שהוסיפה תוכנה — הם אינם
חלק מהדף, אל תתמלל אותם.

"""
        if numbered else ""
    )
    overlay_rules = (
        """
כללי מיפוי למלבנים הממוספרים:
 • לכל שורה שתמללת ציין את n — מספר המלבן שמכיל אותה.
 • מלבן בלי כתב יד (ריק, קו דפוס, כתם, חפץ ברקע) — דלג עליו. אל
   תמציא טקסט כדי "למלא" מלבן.
 • שתי שורות כתובות באותו מלבן — החזר שני פריטים עם אותו n.
 • שורה כתובה שאינה מכוסה על ידי אף מלבן — החזר אותה עם n=null.
"""
        if numbered else ""
    )
    n_field = '"n": 7, ' if numbered else ""
    return f"""אתה מבצע OCR מדויק על צילום של תרגיל כתוב ביד.

{overlay_intro}השפה העיקרית של התרגיל היא {primary}. ייתכן שהדף יכיל גם שורות
בשפות אחרות (למשל הוראות מודפסות בעברית, אנגלית או ערבית מעל תשובות
בכתב יד). זהה את הכתב של כל שורה בנפרד ותמלל אותו כפי שהוא — אל
תתרגם ואל תמיר בין סקריפטים.

{rtl_emphasis}כללי תמלול קריטיים:
1. תמלל בדיוק את מה שכתוב — אל תתקן שום דבר.
2. שמור על שגיאות כתיב בדיוק כפי שהן.
3. כל שורה ויזואלית בתמונה = פריט אחד בפלט, מלמעלה למטה לפי הסדר.
4. שמור על כל סימני הפיסוק בדיוק כפי שנכתבו (כולל פיסוק חסר).
5. שמור על אותיות גדולות וקטנות באנגלית בדיוק כפי שנכתבו.
6. אם תו לא ברור, תמלל את מה שהכי דומה לצורה הכתובה.
7. אם מילה לא קריאה לחלוטין, כתוב במקומה בדיוק את התווית הזו (בשפת
   התרגיל): {illegible} — אל תכתוב תווית בעברית בתוך טקסט שאינו עברי.
8. שורות ריקות בתמונה — דלג עליהן (אל תפיק פריט לשורה ריקה).
9. סדר מילים בשפות RTL (עברית, ערבית): פלט תמיד ב-Unicode "logical
   order" — הסדר שבו הקורא קורא את הטקסט, לא הסדר התצוגתי. במחרוזת
   שלך, המילה הראשונה היא המילה הראשונה בקריאה, כלומר זו שמופיעה ימני
   ביותר על הדף (ולא משמאל). אסור להפוך את סדר המילים. ה-renderer
   ידאג להציג אותן נכון מימין לשמאל בעצמו.
   דוגמה (ערבית): שורה שכתוב בה visually "تفاحة على الطاولة" (= תפוח
   על השולחן) → במחרוזת ה-text שלך כתוב בדיוק "تفاحة على الطاولة",
   ולא "الطاولة على تفاحة".
{overlay_rules}
החזר את התוצאה כ-JSON תקין במבנה:
{{"lines": [{{{n_field}"text": "..."}}, ...]}}."""


OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": ["integer", "null"]},
                    "text": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lines"],
    "additionalProperties": False,
}


def ocr_jpeg_bytes(img: Image.Image) -> bytes:
    """Downscaled JPEG bytes — what actually goes to whichever model."""
    return _jpeg_bytes(img, MAX_EDGE, quality=90)


def _ocr_anthropic(img: Image.Image, exercise_lang: str, numbered: bool) -> dict:
    img_b64 = base64.standard_b64encode(ocr_jpeg_bytes(img)).decode("utf-8")
    response = _anthropic().messages.create(
        model=_ANTHROPIC_MODEL,
        max_tokens=8000,
        system=[
            {
                "type": "text",
                "text": build_ocr_prompt(exercise_lang, numbered),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={
            "format": {"type": "json_schema", "schema": OCR_SCHEMA}
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": _USER_TURN},
                ],
            }
        ],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _ocr_gemini(img: Image.Image, provider: str, exercise_lang: str, numbered: bool) -> dict:
    from google.genai import types

    # Gemini's response_schema is an OpenAPI subset — it rejects the
    # additionalProperties key that OCR_SCHEMA carries for Anthropic.
    gemini_schema = {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "nullable": True},
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
        },
        "required": ["lines"],
    }
    response = _gemini().models.generate_content(
        model=_gemini_model_id(provider),
        contents=[
            types.Part.from_bytes(
                data=ocr_jpeg_bytes(img), mime_type="image/jpeg"
            ),
            _USER_TURN,
        ],
        config=types.GenerateContentConfig(
            system_instruction=build_ocr_prompt(exercise_lang, numbered),
            response_mime_type="application/json",
            response_schema=gemini_schema,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=8000,
        ),
    )
    return json.loads(response.text)


def _ocr_groq(img: Image.Image, provider: str, exercise_lang: str, numbered: bool) -> dict:
    img_b64 = base64.standard_b64encode(ocr_jpeg_bytes(img)).decode("utf-8")
    response = _groq().chat.completions.create(
        model=_groq_model_id(provider),
        messages=[
            {"role": "system", "content": build_ocr_prompt(exercise_lang, numbered)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            _USER_TURN
                            + ' החזר JSON בפורמט הבא בלבד: '
                            '{"lines":[{"n":1,"text":"..."},...]}'
                        ),
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=8000,
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def _ocr_azure(img: Image.Image, exercise_lang: str, numbered: bool) -> dict:
    img_b64 = base64.standard_b64encode(ocr_jpeg_bytes(img)).decode("utf-8")
    response = _azure().chat.completions.create(
        model=_azure_deployment(),
        messages=[
            {"role": "system", "content": build_ocr_prompt(exercise_lang, numbered)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            _USER_TURN
                            + ' החזר JSON בפורמט הבא בלבד: '
                            '{"lines":[{"n":1,"text":"..."},...]}'
                        ),
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=8000,
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def ocr_page(
    img: Image.Image,
    provider: str = DEFAULT_MODEL,
    exercise_lang: str = DEFAULT_EXERCISE_LANG,
    numbered: bool = False,
) -> dict:
    """Returns {"lines": [{"n": int|None, "text": str}, ...]}.

    oris-scanner always calls this with numbered=False (no crop overlay —
    checker's "English flow" variant)."""
    if _is_gemini(provider):
        return _ocr_gemini(img, provider, exercise_lang, numbered)
    if _is_groq(provider):
        return _ocr_groq(img, provider, exercise_lang, numbered)
    if _is_azure(provider):
        return _ocr_azure(img, exercise_lang, numbered)
    return _ocr_anthropic(img, exercise_lang, numbered)


# ─── rubrics (ported from checker/app.py:1342-1394) ─────────────────────────

RUBRICS_DIR = os.path.join(os.path.dirname(__file__), "rubrics")

# The real "English Homework" Classroom assignment isn't tied to one specific
# prompt, so the generic CEFR-based rubric is the sane default (as opposed to
# rubric_with_question, which is scoped to one specific composition topic).
DEFAULT_RUBRIC_ID = "esl-9th-grade-analytical"


def _slugify(name: str) -> str:
    """Filename-safe slug. Keeps Hebrew/Arabic letters; replaces whitespace
    and forbidden filesystem chars with '-'."""
    import re
    s = re.sub(r"[\s/\\:*?\"<>|]+", "-", name.strip())
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "rubric"


def list_rubrics() -> list[dict]:
    """[{id, name}] for every .json file in rubrics/. id == filename stem."""
    if not os.path.isdir(RUBRICS_DIR):
        return []
    out = []
    for fname in sorted(os.listdir(RUBRICS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(RUBRICS_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            out.append({"id": fname[:-5], "name": data.get("name", fname[:-5])})
        except (json.JSONDecodeError, OSError):
            continue
    return out


def load_rubric(rubric_id: str) -> dict | None:
    """Returns {name, content, ...} or None if not found."""
    safe = _slugify(rubric_id)
    path = os.path.join(RUBRICS_DIR, f"{safe}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ─── rubric attached as a plain document ────────────────────────────────────
# Classroom's native rubric (courses.courseWork.rubrics) is a structured
# object, but teachers naturally reach for "attach a file" instead — that
# lands in courseWork.materials[] as an ordinary document with no special
# status, which is why an attached rubric PDF was previously ignored outright.
#
# Identifying *which* attachment is the rubric is the whole difficulty: an
# assignment may equally carry a worksheet or a reading. Three ways, checked in
# this order:
#   1. an explicit [rubric: <text>] tag in the description, matched against
#      attachment names — unambiguous, and mirrors the [model: …] convention;
#   2. otherwise, an attachment whose name looks like a rubric;
#   3. otherwise, a lone attachment is picked up anyway, but only as a
#      *candidate* (certain=False) — teachers do not rename their files to
#      "מחוון", so requiring the naming convention meant the feature almost
#      never fired in practice.
#
# A candidate is never trusted on its name alone. Its text is classified by
# classify_document() before use: only a document that really states grading
# criteria replaces the bundled rubric; an assignment brief is passed to the
# grader as context instead, leaving the bundled rubric in place. That routing
# is what makes step 3 safe — adopting an assignment brief *as* the rubric
# would silently throw away the score scale and the criteria, which is worse
# than ignoring the attachment altogether.
#
# With more than one unnamed attachment nothing is guessed: picking among
# several is exactly the ambiguity the naming convention exists to resolve, so
# the caller is warned to name one or add a [rubric: …] tag.
_RUBRIC_NAME_HINTS = ("rubric", "רובריקה", "מחוון")

_RUBRIC_TAG_RE = re.compile(
    r"\[\s*(?:rubric|רובריקה|מחוון)\s*[:=]\s*([^\]\n\s][^\]\n]*?)\s*\]",
    re.IGNORECASE,
)

# Below this many characters, a PDF's text layer is treated as absent (a
# scanned rubric, or one drawn entirely inside images) and vision OCR is used.
# Deliberately low: a scanned page yields 0-ish characters, while even a
# terse three-criterion rubric clears 100 comfortably. Set too high, compact
# but perfectly readable rubrics get sent for a needless — and billable —
# vision pass.
_MIN_RUBRIC_TEXT_LAYER = 100

# A rubric far longer than this is almost certainly the wrong attachment; keep
# the prompt bounded either way.
_MAX_RUBRIC_CHARS = 20000


def _material_files(materials: list[dict] | None) -> list[dict]:
    """The Drive-file attachments of a courseWork, as [{id, title}]. Links,
    YouTube videos and Forms are skipped — none can carry a rubric."""
    out = []
    for material in materials or []:
        drive_file = (material.get("driveFile") or {}).get("driveFile")
        if drive_file and drive_file.get("id"):
            out.append({"id": drive_file["id"], "title": drive_file.get("title") or ""})
    return out


def resolve_rubric_material(description: str | None,
                            materials: list[dict] | None) -> tuple[dict | None, str | None, str | None]:
    """Picks the attachment to use as the rubric.

    Returns (material, cleaned_description, warning) where material is
    {id, title, certain} or None, and cleaned_description has any [rubric: …]
    tag removed so it never reaches the grader as part of the question.

    certain=True means the teacher said so — an explicit [rubric: …] tag, or a
    filename that names itself a rubric — and the document is used as the
    rubric outright. certain=False is a lone unnamed attachment picked up on
    spec; the caller must run its text through classify_document() to decide
    whether it is a rubric or an assignment brief."""
    files = _material_files(materials)

    cleaned = description
    wanted = None
    if description:
        matches = _RUBRIC_TAG_RE.findall(description)
        if matches:
            wanted = matches[0].strip()
            cleaned = _RUBRIC_TAG_RE.sub("", description)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() or None

    if wanted:
        # Whole-word matches first, plain substring only as a fallback. A bare
        # [rubric: B] must select "rubric B.pdf" and not "rubric A.pdf", where
        # the letter b merely happens to sit inside the word "rubric".
        needle = wanted.lower()
        word = re.compile(rf"\b{re.escape(needle)}\b")
        for hits in ([f for f in files if word.search(f["title"].lower())],
                     [f for f in files if needle in f["title"].lower()]):
            if len(hits) == 1:
                return {**hits[0], "certain": True}, cleaned, None
            if len(hits) > 1:
                return {**hits[0], "certain": True}, cleaned, (
                    f"[rubric: {wanted}] matches {len(hits)} attachments "
                    f"({', '.join(repr(f['title']) for f in hits)}) — using the first; "
                    f"name it more precisely to disambiguate"
                )
        return None, cleaned, (
            f"[rubric: {wanted}] matches none of the attachments "
            f"({', '.join(repr(f['title']) for f in files) or 'none'}) — ignoring it"
        )

    hits = [f for f in files
            if any(h in f["title"].lower() for h in _RUBRIC_NAME_HINTS)]
    if len(hits) == 1:
        return {**hits[0], "certain": True}, cleaned, None
    if len(hits) > 1:
        return {**hits[0], "certain": True}, cleaned, (
            f"{len(hits)} attachments look like a rubric "
            f"({', '.join(repr(f['title']) for f in hits)}) — using the first; "
            f"add [rubric: <name>] to the description to choose explicitly"
        )

    # Nothing named itself a rubric. A single attachment is still worth
    # reading — but only as a candidate, to be classified before it is
    # allowed anywhere near the rubric slot.
    if len(files) == 1:
        return {**files[0], "certain": False}, cleaned, None
    if len(files) > 1:
        return None, cleaned, (
            f"{len(files)} attachments and none named like a rubric "
            f"({', '.join(repr(f['title']) for f in files)}) — not guessing which "
            f"one it is; add [rubric: <name>] to the description, or put "
            f"'מחוון' in the filename"
        )
    return None, cleaned, None


# ─── what kind of document did the teacher attach? ──────────────────────────
# Only reached for an attachment that did not name itself a rubric (see
# resolve_rubric_material step 3). The two outcomes are routed very
# differently — RUBRIC replaces the bundled rubric, ASSIGNMENT is appended to
# the question — so the prompt is written to make the distinction the model's
# only job, and the fallback on any doubt is ASSIGNMENT: adding context to the
# question can only inform the grade, while a wrong rubric silently destroys
# the score scale.
_DOC_KIND_PROMPT = (
    "לפניך טקסט של מסמך שמורה צירף למשימה בכיתה. קבע מה הוא:\n"
    "RUBRIC — מסמך שמגדיר כיצד להעריך ולנקד עבודה: קריטריונים להערכה, "
    "רמות ביצוע, טווחי ניקוד, סולם ציונים, תיאור מה נחשב תשובה טובה מול חלשה.\n"
    "ASSIGNMENT — כל דבר אחר: הגדרת התרגיל עצמו, הוראות לתלמיד, השאלה, "
    "טקסט קריאה, דף עבודה, רשימת מילים, דוגמאות.\n"
    "שים לב: מסמך שמסביר למורה כיצד לבנות מחוון הוא ASSIGNMENT, לא RUBRIC — "
    "הוא לא מכיל קריטריונים לתרגיל הזה.\n"
    "אם אינך בטוח — ענה ASSIGNMENT.\n"
    "ענה במילה אחת בלבד: RUBRIC או ASSIGNMENT."
)

# Enough to judge the document's nature; the full text is what actually gets
# used downstream. Keeps this classification call cheap and fast.
_DOC_KIND_SAMPLE_CHARS = 4000


def classify_document(text: str, model_key: str = DEFAULT_MODEL) -> str:
    """'rubric' if the text states grading criteria, otherwise 'assignment'.

    One short text-only call — no images — so it is far cheaper than the OCR
    and evaluation passes that follow, and it runs only when there is a
    submission actually being graded."""
    sample = (text or "").strip()[:_DOC_KIND_SAMPLE_CHARS]
    if not sample:
        return "assignment"

    user_turn = f"{_DOC_KIND_PROMPT}\n\n--- המסמך ---\n\n{sample}"

    if _is_gemini(model_key):
        from google.genai import types
        response = _gemini().models.generate_content(
            model=_gemini_model_id(model_key),
            contents=[user_turn],
            config=types.GenerateContentConfig(
                max_output_tokens=10,
                # Same reason as in evaluate_with_rubric: without this the
                # thinking budget eats the whole output and text comes back
                # empty, which here would silently mean "assignment".
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        txt = response.text or ""
    elif _is_groq(model_key):
        response = _groq().chat.completions.create(
            model=_groq_model_id(model_key),
            messages=[{"role": "user", "content": user_turn}],
            max_tokens=10,
            temperature=0,
        )
        txt = response.choices[0].message.content or ""
    elif _is_azure(model_key):
        response = _azure().chat.completions.create(
            model=_azure_deployment(),
            messages=[{"role": "user", "content": user_turn}],
            max_tokens=10,
            temperature=0,
        )
        txt = response.choices[0].message.content or ""
    else:
        msg = _anthropic().messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": user_turn}],
        )
        txt = "".join(b.text for b in msg.content if b.type == "text")

    return "rubric" if "RUBRIC" in (txt or "").upper() else "assignment"


def pdf_text_layer(pdf_bytes: bytes) -> str:
    """The PDF's embedded text, empty for a purely scanned document."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc).strip()
    finally:
        doc.close()


def rubric_text_from_document(data: bytes, ext: str, model_key: str = DEFAULT_MODEL,
                              exercise_lang: str = DEFAULT_EXERCISE_LANG) -> tuple[str, str]:
    """Extracts the rubric text from an attached document.

    Returns (text, method). A digital PDF is read straight off its text layer
    — free, instant and exact. Only a scanned one falls back to vision OCR,
    which costs a model call per page."""
    if ext == ".pdf":
        text = pdf_text_layer(data)
        if len(text) >= _MIN_RUBRIC_TEXT_LAYER:
            return text[:_MAX_RUBRIC_CHARS], "pdf text layer"

    lines = []
    for img in iter_file_pages(data, ext):
        ocr = ocr_page(_downscale(img, MAX_EDGE), model_key, exercise_lang, numbered=False)
        lines += [ln.get("text", "") for ln in ocr.get("lines", [])]
    return "\n".join(lines).strip()[:_MAX_RUBRIC_CHARS], f"vision OCR ({model_key})"


def rubric_from_classroom(classroom_rubric: dict, assignment_name: str = "") -> dict:
    """Converts a Classroom API Rubric object (courses.courseWork.rubrics —
    criteria[].levels[], each with title/description/points) into the same
    {name, content} shape as the bundled rubrics/*.json files, so it can be
    handed to evaluate_with_rubric like any other rubric."""
    lines = []
    for criterion in classroom_rubric.get("criteria", []):
        title = (criterion.get("title") or "").strip()
        c_desc = (criterion.get("description") or "").strip()
        header = f"Criterion — {title}" if title else "Criterion"
        if c_desc:
            header += f": {c_desc}"
        lines.append(header)
        levels = sorted(criterion.get("levels", []), key=lambda lv: lv.get("points") or 0, reverse=True)
        for level in levels:
            l_title = (level.get("title") or "").strip()
            l_desc = (level.get("description") or "").strip()
            points = level.get("points")
            points_str = f" ({points} pts)" if points is not None else ""
            line = f"  {l_title}{points_str}"
            if l_desc:
                line += f": {l_desc}"
            lines.append(line)
        lines.append("")
    return {"name": assignment_name or "Classroom rubric", "content": "\n".join(lines).strip()}


def count_words(pages: list[dict]) -> int:
    """Total number of words the student wrote across all pages/lines."""
    n = 0
    for page in pages:
        for line in page.get("lines", []):
            n += len(str(line.get("text", "")).split())
    return n


def helpful_words_usage(pages: list[dict], helpful_words: list[str]) -> dict | None:
    """Which of the rubric's suggested 'helpful words' actually appear in the
    student's transcript. Returns {used, unused, count, total} or None when
    the rubric has none."""
    if not helpful_words:
        return None
    text = pages_to_plain_text(pages).lower()
    used, unused = [], []
    for w in helpful_words:
        (used if str(w).lower() in text else unused).append(w)
    return {"used": used, "unused": unused, "count": len(used), "total": len(helpful_words)}


def pages_to_plain_text(pages: list[dict]) -> str:
    """Flatten the OCR'd pages into a single transcript for the evaluator.

    Each line is prefixed with its [p<page>-l<idx>] tag so the model can
    return per-issue references that we map back to highlights later."""
    chunks = []
    for page in pages:
        page_num = page.get("page", "?")
        chunks.append(f"--- עמוד {page_num} ---")
        for line_idx, line in enumerate(page.get("lines", [])):
            chunks.append(f"[p{page_num}-l{line_idx}] {line.get('text', '')}")
        chunks.append("")
    return "\n".join(chunks).strip()


# ─── evaluation (ported from checker/app.py:1420-1781) ─────────────────────

def build_eval_prompt(feedback_lang: str, exercise_lang: str,
                      question: str | None = None) -> str:
    """Evaluation system prompt parameterized by the two language choices.

    `feedback_lang` is the primary language for `feedback` / `overall_feedback`
    (what the teacher reads). `exercise_lang` is the language the student
    worked in — we ask for a parallel `feedback_secondary` / `overall_feedback_secondary`
    in that language. `question` is an optional task prompt; when provided the
    model also reports `question_answered` ("yes"/"no")."""
    f = LANGS.get(feedback_lang, LANGS[DEFAULT_FEEDBACK_LANG])
    e = LANGS.get(exercise_lang, LANGS[DEFAULT_EXERCISE_LANG])
    primary = f["name_native"]
    secondary = e["name_native"]
    LANG_DIRECTIVES = {
        "he": "כל הפידבק (feedback), ההערות (comment) ופסקת הסיכום (overall_feedback) "
              "ייכתבו בעברית בלבד.",
        "en": "ALL feedback text — every \"feedback\", \"comment\" and the "
              "\"overall_feedback\" summary — MUST be written in English only. "
              "Do NOT write any of this feedback in Hebrew, even though these "
              "instructions are in Hebrew.",
        "ar": "يجب كتابة كل الملاحظات (feedback) والتعليقات (comment) وفقرة "
              "الملخّص (overall_feedback) بالعربية فقط ، وليس بالعبرية.",
    }
    lang_directive = LANG_DIRECTIVES.get(feedback_lang, LANG_DIRECTIVES["he"])
    if feedback_lang == exercise_lang:
        secondary_block = (
            f'- לכל קריטריון: השדה "feedback_secondary" — מחרוזת ריקה "" '
            f"(שפת הפידבק זהה לשפת התרגיל ולכן אין צורך בתרגום).\n"
            f'- גם "overall_feedback_secondary" — מחרוזת ריקה "".'
        )
    else:
        secondary_block = (
            f'- לכל קריטריון: גם פידבק ב{secondary} ("feedback_secondary") — '
            f"תרגום נאמן של הפידבק שב-feedback, באותו אורך ובאותו תוכן "
            f"(לא להוסיף או להחסיר מידע).\n"
            f'- גם פסקת סיכום ב{secondary} ("overall_feedback_secondary") — '
            f"תרגום נאמן של פסקת הסיכום שב-overall_feedback, באותו אורך "
            f"ובאותו תוכן."
        )
    if question and question.strip():
        question_block = (
            f'\n- הרובריקה כוללת שאלה ספציפית שהתלמיד היה אמור לענות עליה:\n'
            f'  "{question.strip()}"\n'
            f'  קבע האם התלמיד אכן ענה על השאלה הזו (התייחס לנושא שנדרש), והחזר '
            f'שדה עליון "question_answered" = "yes" אם ענה, או "no" אם לא ענה / '
            f'כתב על נושא אחר.'
        )
        question_answered_field = '  "question_answered": "yes | no"\n'
    else:
        question_block = (
            '\n- אין שאלה ספציפית ברובריקה — החזר "question_answered": "" '
            "(מחרוזת ריקה)."
        )
        question_answered_field = '  "question_answered": ""\n'
    return f"""אתה בודק שיעורי בית בהתאם לרובריקה שתינתן לך.

‼️ שפת הפלט (חובה, גובר על כל השאר): {lang_directive}

תקבל:
1. רובריקה — מתארת קריטריונים, רמות וציונים אפשריים.
2. טקסט תרגיל של תלמיד — תמלול של כתב יד, יתכן עם שגיאות OCR. כל
   שורה מתויגת עם מזהה בצורה [p<page>-l<index>] בתחילתה. שפת התרגיל
   העיקרית היא {secondary}.

המשימה:
- זהה את כל הקריטריונים שמופיעים ברובריקה (השמות שלהם, והציון המקסימלי בכל אחד).
- העריך את התרגיל לפי הרובריקה — צא מנקודת הנחה שמה שנכתב הוא מה שהתלמיד התכוון אליו (אל תקטף נקודות על שגיאות OCR שנראות כמו טעויות תמלול).
- לכל קריטריון: ציון מספרי (1 עד max_score לפי הרובריקה) ופידבק קצר ב{primary} (1-3 משפטים).
- לכל קריטריון: שדה "level" — הרמה שהתלמיד השיג בקריטריון, אחד מ:
  "excellent" / "good" / "fair" / "needs_improvement". בחר את הרמה לפי
  תיאורי הרמות ברובריקה עצמה (אם קיימים), לא לפי חישוב יחס מספרי מהציון.
  אם הרובריקה לא מגדירה רמות כאלה — החזר מחרוזת ריקה "".
{secondary_block}
- לכל קריטריון: רשימת "issues" — דוגמאות ספציפיות לבעיות שמצאת
  בטקסט. לכל issue:
    * "line_ref": המזהה של השורה שבה הבעיה (לדוגמה "p1-l3").
    * "quote": ציטוט מילולי מדויק של הקטע הבעייתי מתוך אותה שורה —
      בדיוק כפי שהוא מופיע בתמלול (אותו רישיות, פיסוק ורווחים). זה
      חייב להיות תת-מחרוזת של השורה. ציטוט ברמת משפט/ביטוי, לא
      מילה בודדת ולא שורה שלמה אם רק חלק ממנה בעייתי.
    * "comment": הערה קצרה ב{primary} (3-15 מילים) שמסבירה למה זה בעייתי.
  אם אין בעיות בקריטריון מסוים, החזר רשימה ריקה.
  אם אותו משפט בעייתי בכמה קריטריונים — שייך אותו לקריטריון
  החמור/המהותי ביותר בלבד (לא לכמה במקביל).
- ציון כללי (אם הרובריקה מציינת mapping ל-CEFR/אחוז/אות — השתמש בו, אחרת תן את ממוצע הציונים).
- פסקת סיכום ב{primary} (2-4 משפטים) — חוזקות, חולשות, ומה כדאי לתלמיד לעבוד עליו.{question_block}

תזכורת אחרונה לפני הפלט — {lang_directive}

החזר JSON תקין במבנה:
{{
  "criteria": [
    {{
      "name": "...",
      "score": <int>,
      "max_score": <int>,
      "level": "excellent | good | fair | needs_improvement | (ריק אם אין רמות)",
      "feedback": "...",
      "feedback_secondary": "...",
      "issues": [
        {{"line_ref": "p1-l3", "quote": "...", "comment": "..."}}
      ]
    }}
  ],
  "overall_score": "<string — לדוגמה: B1, או 3.0/4, או 75%>",
  "overall_feedback": "...",
  "overall_feedback_secondary": "...",
{question_answered_field}}}"""


EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                    "max_score": {"type": "number"},
                    "level": {"type": "string"},
                    "feedback": {"type": "string"},
                    "feedback_secondary": {"type": "string"},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "line_ref": {"type": "string"},
                                "quote": {"type": "string"},
                                "comment": {"type": "string"},
                            },
                            "required": ["line_ref", "quote", "comment"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["name", "score", "max_score", "level", "feedback", "feedback_secondary", "issues"],
                "additionalProperties": False,
            },
        },
        "overall_score": {"type": "string"},
        "overall_feedback": {"type": "string"},
        "overall_feedback_secondary": {"type": "string"},
        "question_answered": {"type": "string"},
    },
    "required": ["criteria", "overall_score", "overall_feedback", "overall_feedback_secondary", "question_answered"],
    "additionalProperties": False,
}

# Groq's chat-completions API supports response_format={"type":"json_object"}
# but NOT json_schema, so the required shape is spelled out in the prompt instead.
_GROQ_EVAL_SHAPE_HINT = (
    'החזר אך ורק JSON תקין בפורמט הבא:\n'
    '{\n'
    '  "criteria": [\n'
    '    {\n'
    '      "name": "...",\n'
    '      "score": 0,\n'
    '      "max_score": 0,\n'
    '      "level": "excellent | good | fair | needs_improvement | (empty string if the rubric has no levels)",\n'
    '      "feedback": "...",\n'
    '      "feedback_secondary": "...",\n'
    '      "issues": [\n'
    '        {"line_ref": "p1-l3", "quote": "...", "comment": "..."}\n'
    '      ]\n'
    '    }\n'
    '  ],\n'
    '  "overall_score": "...",\n'
    '  "overall_feedback": "...",\n'
    '  "overall_feedback_secondary": "...",\n'
    '  "question_answered": "yes | no | (empty string if the rubric has no question)"\n'
    '}'
)


def evaluate_with_rubric(
    pages: list[dict],
    rubric: dict,
    provider: str,
    feedback_lang: str = DEFAULT_FEEDBACK_LANG,
    exercise_lang: str = DEFAULT_EXERCISE_LANG,
) -> dict:
    """Send transcript + rubric → structured per-criterion scores + feedback."""
    import report  # local import: report imports nothing from core, no cycle

    transcript = pages_to_plain_text(pages)
    user_turn = (
        f"רובריקה (שם: {rubric.get('name', '')}):\n\n"
        f"{rubric.get('content', '')}\n\n"
        f"--- טקסט התרגיל לבדיקה ---\n\n{transcript}\n\n"
        "החזר JSON לפי הסכימה."
    )
    eval_prompt = build_eval_prompt(feedback_lang, exercise_lang, rubric.get("question"))

    if _is_gemini(provider):
        from google.genai import types

        gemini_schema = {
            "type": "object",
            "properties": {
                "criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "score": {"type": "number"},
                            "max_score": {"type": "number"},
                            "level": {"type": "string"},
                            "feedback": {"type": "string"},
                            "feedback_secondary": {"type": "string"},
                            "issues": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "line_ref": {"type": "string"},
                                        "quote": {"type": "string"},
                                        "comment": {"type": "string"},
                                    },
                                    "required": ["line_ref", "quote", "comment"],
                                },
                            },
                        },
                        "required": ["name", "score", "max_score", "level", "feedback", "feedback_secondary", "issues"],
                    },
                },
                "overall_score": {"type": "string"},
                "overall_feedback": {"type": "string"},
                "overall_feedback_secondary": {"type": "string"},
                "question_answered": {"type": "string"},
            },
            "required": ["criteria", "overall_score", "overall_feedback", "overall_feedback_secondary", "question_answered"],
        }
        response = _gemini().models.generate_content(
            model=_gemini_model_id(provider),
            contents=[user_turn],
            config=types.GenerateContentConfig(
                system_instruction=eval_prompt,
                response_mime_type="application/json",
                response_schema=gemini_schema,
                # Gemini 2.5 Flash is a thinking model — without this it
                # silently eats most of max_output_tokens on internal
                # reasoning and the JSON gets truncated mid-string.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                max_output_tokens=12000,
            ),
        )
        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as e:
            log.error(
                "evaluate (gemini) JSON parse failed: %s | text_len=%d",
                e, len(response.text),
            )
            log.error("evaluate (gemini) raw response text:\n%s", response.text)
            raise
        return report.attach_colors(parsed)

    if _is_groq(provider):
        response = _groq().chat.completions.create(
            model=_groq_model_id(provider),
            messages=[
                {"role": "system", "content": eval_prompt},
                {"role": "user", "content": user_turn + "\n\n" + _GROQ_EVAL_SHAPE_HINT},
            ],
            response_format={"type": "json_object"},
            max_tokens=12000,
            temperature=0,
        )
        text = response.choices[0].message.content
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            log.error(
                "evaluate (groq) JSON parse failed: %s | text_len=%d", e, len(text),
            )
            log.error("evaluate (groq) raw response text:\n%s", text)
            raise
        return report.attach_colors(parsed)

    if _is_azure(provider):
        response = _azure().chat.completions.create(
            model=_azure_deployment(),
            messages=[
                {"role": "system", "content": eval_prompt},
                {"role": "user", "content": user_turn + "\n\n" + _GROQ_EVAL_SHAPE_HINT},
            ],
            response_format={"type": "json_object"},
            max_tokens=12000,
            temperature=0,
        )
        text = response.choices[0].message.content
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            log.error(
                "evaluate (azure) JSON parse failed: %s | text_len=%d", e, len(text),
            )
            log.error("evaluate (azure) raw response text:\n%s", text)
            raise
        return report.attach_colors(parsed)

    response = _anthropic().messages.create(
        model=_ANTHROPIC_MODEL,
        max_tokens=12000,
        system=[
            {
                "type": "text",
                "text": eval_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={
            "format": {"type": "json_schema", "schema": EVAL_SCHEMA}
        },
        messages=[{"role": "user", "content": user_turn}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        stop = getattr(response, "stop_reason", "?")
        usage = getattr(response, "usage", None)
        log.error(
            "evaluate JSON parse failed: %s | stop_reason=%s | usage=%s | text_len=%d",
            e, stop, usage, len(text),
        )
        log.error("evaluate raw response text:\n%s", text)
        raise
    return report.attach_colors(parsed)


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


# ─── pipeline orchestration ─────────────────────────────────────────────────

def check_pages(
    file_bytes_list: list[tuple[bytes, str]],
    rubric_id: str = DEFAULT_RUBRIC_ID,
    rubric_override: dict | None = None,
    question: str | None = None,
    model_key: str = DEFAULT_MODEL,
    feedback_lang: str = DEFAULT_FEEDBACK_LANG,
    exercise_lang: str = DEFAULT_EXERCISE_LANG,
    auto_orient: bool = True,
) -> tuple[list[dict], dict]:
    """OCRs every page of every file (auto-rotate + ocr_page(numbered=False)),
    merges them into one sequentially-numbered page list, then runs one
    evaluate_with_rubric call over the merged transcript.

    file_bytes_list: [(file_bytes, ext), ...] — ext one of ACCEPTED_EXTS.
    rubric_override: a ready-made {name, content} rubric dict (e.g. from
    rubric_from_classroom) that takes priority over rubric_id — used when
    the assignment has its own Classroom-attached rubric.
    question: the task/assignment description as the teacher wrote it in
    Classroom, if any — merged into the rubric so the model is evaluated
    against the actual assignment, not just the rubric's generic criteria.
    Returns (pages, evaluation).
    """
    rubric = rubric_override or load_rubric(rubric_id) or load_rubric(DEFAULT_RUBRIC_ID)
    if question:
        rubric = {**rubric, "question": question}

    t0 = time.monotonic()
    log.info("[check] start: files=%d model=%s rubric=%r auto_orient=%s langs=%s/%s",
             len(file_bytes_list), model_key, rubric.get("name"),
             auto_orient, exercise_lang, feedback_lang)

    pages: list[dict] = []
    for data, ext in file_bytes_list:
        for img in iter_file_pages(data, ext):
            if auto_orient:
                rotation = detect_rotation(img, model_key)
                img = apply_rotation(img, rotation)
            img = _downscale(img, MAX_EDGE)
            ocr = ocr_page(img, model_key, exercise_lang, numbered=False)
            page_num = len(pages) + 1
            pages.append({
                "page": page_num,
                "lines": [{"text": ln.get("text", "")} for ln in ocr.get("lines", [])],
                "original_b64": original_preview_b64(img),
            })
            log.info("[page %d] ocr'd, model=%s, lines=%d", page_num, model_key, len(pages[-1]["lines"]))

    evaluation = evaluate_with_rubric(pages, rubric, model_key, feedback_lang, exercise_lang)
    wc = count_words(pages)
    evaluation["word_count"] = wc
    hw = helpful_words_usage(pages, rubric.get("helpful_words") or [])
    if hw is not None:
        evaluation["helpful_words_usage"] = hw
    if rubric.get("criteria_grid"):
        evaluation["criteria_grid"] = rubric["criteria_grid"]
    log.info("[eval] model=%s rubric=%s criteria=%d overall=%s",
              model_key, rubric.get("name"), len(evaluation.get("criteria", [])),
              evaluation.get("overall_score"))
    log.info("[check] done in %.1fs: pages=%d words=%d", time.monotonic() - t0, len(pages), wc)
    return pages, evaluation
