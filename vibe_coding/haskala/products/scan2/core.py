"""core.py — the checking core of math-checker, no Flask.

Every consumer — the web UI (app.py), the CLI (check.py), and the Classroom
poller — calls the same pipeline: render pages → fix orientation (Haiku) →
holistic analysis (Opus) → structured JSON. There is intentionally no server
state here (JOBS/queues).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import time

import anthropic
import fitz  # PyMuPDF
from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv(override=True)

log = logging.getLogger("math-checker")

client = anthropic.Anthropic()

RENDER_DPI = 200          # A4 @ 200 DPI → ~1654×2339 px (well under Opus vision limit)
MAX_EDGE = 2200           # longest edge sent to Opus for analysis
ORIENT_MAX_EDGE = 900     # longest edge sent to Haiku for fast orientation check
PREVIEW_MAX_EDGE = 1400   # longest edge stored in result for display in browser

# Analysis models the teacher can pick in the UI. Holistic math/geometry
# grading is a hard reasoning task (proofs, 2D diagrams, multi-step algebra),
# so only strong reasoners do it well. The full provider line-up from
# products/checker is exposed for the current testing phase, but the cheap /
# mini / free tiers are flagged "(לא מומלץ)" — empirically they miss real
# calculation/logic errors on math pages. Opus is the quality default.
#
# Each UI key maps to (provider, model_id) so analyze_page can dispatch across
# Anthropic / Google / Groq / Azure (mirrors products/checker's abstraction).
MODELS = {
    "opus":             "Claude Opus 4.8 · איכות מרבית",
    "sonnet":           "Claude Sonnet 4.6 · מהיר וזול ~5×",
    "gemini-flash":     "Gemini 2.5 Flash · מהיר וזול",
    "haiku":            "Claude Haiku 4.5 · הכי זול (לא מומלץ)",
    "gemini-lite":      "Gemini 2.5 Flash-Lite · זול מאוד (לא מומלץ)",
    "groq-scout":       "Groq Llama 4 Scout · חינמי (לא מומלץ)",
    "azure-gpt41-mini": "GPT-4.1-mini · Azure (לא מומלץ)",
}

# provider kind + concrete model id per UI key
_ANTHROPIC_IDS = {
    "opus":   "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku":  "claude-haiku-4-5-20251001",
}
_GEMINI_IDS = {
    "gemini-flash": "gemini-2.5-flash",
    "gemini-lite":  "gemini-2.5-flash-lite",
}
_GROQ_IDS = {
    "groq-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
}
_AZURE_KEYS = {"azure-gpt41-mini"}

DEFAULT_MODEL = "opus"
ORIENT_MODEL  = "claude-haiku-4-5-20251001"  # orientation check is always Haiku (cheap)


def resolve_model(key: str | None) -> tuple[str, str]:
    """Map a UI model key to (canonical_key, hebrew_label). Falls back to default."""
    k = key if key in MODELS else DEFAULT_MODEL
    return k, MODELS[k]

ACCEPTED_EXTS = (".pdf", ".jpg", ".jpeg", ".png")


# ─── file handling ────────────────────────────────────────────────────────────

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
    """Number of pages without rendering any of them (cheap — only reads the PDF structure)."""
    if ext != ".pdf":
        return 1
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return len(doc)
    finally:
        doc.close()


def iter_file_pages(data: bytes, ext: str):
    """Like file_to_pages but lazy: renders one page at a time, so at peak
    only a single page's full render (200 DPI) is held in memory, not the
    whole document's."""
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


# ─── optional non-Claude providers (lazy: a missing key only bites if picked) ──
_gemini_client = None
_groq_client = None
_azure_client = None


def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


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


def _image_block(img: Image.Image, max_edge: int) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": _img_to_b64_jpeg(img, max_edge),
        },
    }


# ─── orientation detection ────────────────────────────────────────────────────

def detect_rotation(img: Image.Image) -> int:
    """Ask Haiku for the CW rotation (0/90/180/270) needed to upright the page.

    The checker's heuristic auto_rotate only handles portrait/landscape;
    it misses 180° flips that are common in scanned exam pages. Haiku is
    cheap enough to run per-page and reliable for simple orientation.
    """
    msg = client.messages.create(
        model=ORIENT_MODEL,
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": [
                _image_block(img, ORIENT_MAX_EDGE),
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


def apply_rotation(img: Image.Image, deg: int) -> Image.Image:
    if deg == 0:
        return img
    return img.rotate(-deg, expand=True, resample=Image.BILINEAR, fillcolor=(255, 255, 255))


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

# Groq / Azure are OpenAI-compatible and only support response_format=
# json_object (no schema), so the JSON shape is spelled out in the user turn.
_JSON_CONTRACT = (
    " החזר JSON בלבד, ללא טקסט נוסף, במבנה: "
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
    response = client.messages.create(
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
                _image_block(img, MAX_EDGE),
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
        _azure(), os.environ["AZURE_OPENAI_DEPLOYMENT"], img, rubric)


# ─── error handling ───────────────────────────────────────────────────────────

def humanize_error(exc: Exception) -> dict:
    """Turn an SDK exception into a user-readable Hebrew error payload."""
    raw = (str(exc) or repr(exc)).strip()
    tech = raw[:2000]

    if isinstance(exc, anthropic.RateLimitError):
        return {
            "message": "חרגת ממכסת Claude לרגע זה",
            "details": "Anthropic מגביל בקשות לדקה (RPM) וטוקנים לדקה (TPM). נסה שוב בעוד דקה.",
            "technical": tech,
        }
    if isinstance(exc, anthropic.AuthenticationError):
        return {
            "message": "מפתח Anthropic API לא תקף",
            "details": "ודא שהמשתנה ANTHROPIC_API_KEY מוגדר נכון ב-.env ושהמפתח עדיין פעיל.",
            "technical": tech,
        }
    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return {
            "message": "החיבור ל-Anthropic נכשל",
            "details": "ייתכן שיש בעיית רשת או ש-Anthropic לא זמין כרגע. נסה שוב בעוד דקה.",
            "technical": tech,
        }
    if isinstance(exc, KeyError) and exc.args and str(exc.args[0]) == "ANTHROPIC_API_KEY":
        return {
            "message": "מפתח Anthropic API חסר",
            "details": "המשתנה ANTHROPIC_API_KEY לא מוגדר. הוסף אותו ל-.env ואז restart.",
            "technical": tech,
        }
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return {
            "message": "המודל החזיר תשובה לא תקפה",
            "details": "לא הצלחנו לפענח את תשובת המודל. זו בדרך כלל בעיה זמנית — נסה שוב.",
            "technical": tech,
        }
    if isinstance(exc, anthropic.APIStatusError):
        return {
            "message": f"שגיאת API ({getattr(exc, 'status_code', '?')})",
            "details": "השירות החזיר תשובה לא תקינה. נסה שוב.",
            "technical": tech,
        }
    return {
        "message": "תקלה לא צפויה",
        "details": "אירעה שגיאה לא מזוהה. נסה שוב; אם זה חוזר צרף את הפרטים הטכניים.",
        "technical": tech,
    }


# ─── shared pipeline (yields progress events, then the result) ───────────────

STAGE_LABELS = {
    "orient":  "מזהה סיבוב (Haiku)",
    "analyze": "מנתח מתמטיקה",
}
_STEPS_PER_PAGE = 2  # orient, analyze


def process_stream(file_bytes: bytes, ext: str, auto_orient: bool,
                    model_key: str = DEFAULT_MODEL, model_label: str = "",
                    rubric: str = "", keep_imgs: bool = True):
    """Generator: yields SSE-ready progress dicts, then a final result dict.

    keep_imgs=False (CLI/poller): don't accumulate page images for
    re-analysis — saves ~10MB per page in memory. The UI needs them (manual
    rotation), the batch path doesn't."""
    # Lazy rendering: one page at 200 DPI is held in memory at a time
    # (important for long PDFs on a small instance). total is read from the
    # PDF structure without rendering.
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
    imgs = []  # oriented analysis-resolution images, kept server-side for re-analysis
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
