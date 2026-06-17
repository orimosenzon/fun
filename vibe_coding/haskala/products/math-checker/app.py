"""math-checker — בדיקת תרגילי מתמטיקה/גאומטריה בכתב יד (מוצר #2, חברת השכלה).

בניגוד לבודק-השפה (products/checker), כאן אין חיתוך שורה-שורה. תרגיל מתמטי
הוא יחידה הוליסטית — סרטוטים, הוכחות גאומטריות ומעברים אלגבריים פרושים ב-2D.
לכן העמוד המלא נשלח ל-Opus שמחזיר הבנה מובנית (JSON): רשימת תרגילים, ולכל
אחד את צעדי הפתרון ב-LaTeX, הערכת תקינות לכל צעד, ומשוב.
"""
from __future__ import annotations

import base64
import datetime
import hmac
import io
import json
import logging
import os
import queue
import threading
import time
import uuid

import anthropic
import fitz  # PyMuPDF
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image, ImageOps

load_dotenv(override=True)

LOG_PATH = os.path.join(os.path.dirname(__file__), "math-checker.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler()],
)
log = logging.getLogger("math-checker")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

DESKTOP_MODE = os.environ.get("MATH_CHECKER_DESKTOP") == "1"
BASIC_USER = os.environ.get("HASKALA_USER")
BASIC_PASS = os.environ.get("HASKALA_PASS")
if not DESKTOP_MODE and not (BASIC_USER and BASIC_PASS):
    log.warning(
        "running without Basic Auth — set HASKALA_USER/HASKALA_PASS "
        "before exposing this publicly"
    )


@app.before_request
def _require_basic_auth():
    if DESKTOP_MODE or not (BASIC_USER and BASIC_PASS):
        return
    if request.path.startswith("/static/"):
        return
    auth = request.authorization
    if (
        auth
        and auth.type == "basic"
        and hmac.compare_digest(auth.username or "", BASIC_USER)
        and hmac.compare_digest(auth.password or "", BASIC_PASS)
    ):
        return
    return Response(
        "נדרשת הזדהות",
        401,
        {"WWW-Authenticate": 'Basic realm="math-checker"'},
    )


client = anthropic.Anthropic()

RENDER_DPI = 200          # A4 @ 200 DPI → ~1654×2339 px (well under Opus vision limit)
MAX_EDGE = 2200           # longest edge sent to Opus for analysis
ORIENT_MAX_EDGE = 900     # longest edge sent to Haiku for fast orientation check
PREVIEW_MAX_EDGE = 1400   # longest edge stored in result for display in browser

ANALYSIS_MODEL = "claude-opus-4-8"
ORIENT_MODEL   = "claude-haiku-4-5-20251001"

ACCEPTED_EXTS = (".pdf", ".jpg", ".jpeg", ".png")

JOB_TTL = 3600  # seconds — results stay in memory for result refreshes
JOBS: dict[str, dict] = {}


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


def _downscale(img: Image.Image, max_edge: int) -> Image.Image:
    long_edge = max(img.size)
    if long_edge <= max_edge:
        return img
    scale = max_edge / long_edge
    return img.resize(
        (round(img.width * scale), round(img.height * scale)),
        Image.LANCZOS,
    )


def _img_to_b64_jpeg(img: Image.Image, max_edge: int | None = None, quality: int = 90) -> str:
    if max_edge:
        img = _downscale(img, max_edge)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode()


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
• score_suggestion: הצעת ניקוד טקסטואלית, למשל "מלא (10/10)" / "חלקי — טעות חישוב (7/10)" / "לא ענה (0)".
• id: מספר/אות התרגיל בדיוק כפי שמופיע בדף (למשל "1", "2א", "ב"). אם לא ברור — "?".
• topic: אחת מ: אלגברה / גאומטריה / חשבון / הסתברות / טריגונומטריה / מש"ב / אחר."""

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
                    "score_suggestion":   {"type": "string"},
                    "feedback":           {"type": "string"},
                },
                "required": [
                    "id", "topic", "statement_latex", "student_steps",
                    "final_answer_latex", "verdict", "score_suggestion", "feedback",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["page_summary", "has_diagram", "diagram_description", "problems"],
    "additionalProperties": False,
}


def analyze_page(img: Image.Image) -> dict:
    """Send a full page to Opus for holistic math analysis. Returns structured JSON."""
    response = client.messages.create(
        model=ANALYSIS_MODEL,
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
                {"type": "text", "text": "נתח את עמוד הפתרון וחזור JSON."},
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


# ─── streaming job runner ─────────────────────────────────────────────────────

STAGE_LABELS = {
    "render":  "מרנדר עמוד",
    "orient":  "מזהה סיבוב (Haiku)",
    "analyze": "מנתח מתמטיקה (Opus)",
}
_STEPS_PER_PAGE = 3  # render, orient, analyze


def _evict_stale_jobs():
    now = time.time()
    for jid in [k for k, v in JOBS.items() if now - v.get("ts", 0) > JOB_TTL]:
        JOBS.pop(jid, None)


def _process_stream(file_bytes: bytes, ext: str, auto_orient: bool):
    """Generator: yields SSE-ready progress dicts, then a final result dict."""
    pages_imgs = file_to_pages(file_bytes, ext)
    total = len(pages_imgs)
    done = 0

    def progress(stage: str, page: int) -> dict:
        nonlocal done
        done += 1
        return {
            "type": "progress",
            "page": page,
            "total_pages": total,
            "stage": stage,
            "label": STAGE_LABELS.get(stage, stage),
            "pct": round(100 * done / max(1, total * _STEPS_PER_PAGE)),
        }

    results = []
    for idx, img in enumerate(pages_imgs):
        p = idx + 1
        yield progress("render", p)

        if auto_orient:
            rotation = detect_rotation(img)
            img = apply_rotation(img, rotation)
        else:
            rotation = 0
        yield progress("orient", p)

        analysis = analyze_page(img)
        log.info(
            "[page %d] rotation=%d° problems=%d verdicts=%s",
            p, rotation,
            len(analysis.get("problems", [])),
            [pr.get("verdict") for pr in analysis.get("problems", [])],
        )

        results.append({
            "page": p,
            "rotation_applied": rotation,
            "image_b64": _img_to_b64_jpeg(img, PREVIEW_MAX_EDGE, quality=85),
            "analysis": analysis,
        })
        yield progress("analyze", p)

    yield {"type": "result", "pages": results}


def _run_job(job_id: str, file_bytes: bytes, ext: str, auto_orient: bool, filename: str):
    job = JOBS[job_id]
    q: queue.Queue = job["q"]
    try:
        for ev in _process_stream(file_bytes, ext, auto_orient):
            if ev["type"] == "result":
                job["pages"] = ev["pages"]
                job["filename"] = filename
                q.put({"type": "done"})
            else:
                q.put(ev)
    except Exception as e:
        log.exception("_run_job failed for job %s", job_id)
        q.put({"type": "error", **humanize_error(e)})
    finally:
        q.put({"type": "end"})


# ─── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze/start", methods=["POST"])
def analyze_start():
    _evict_stale_jobs()
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "לא הועלה קובץ"}), 400
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in ACCEPTED_EXTS:
        return jsonify({"error": f"סוג קובץ לא נתמך: {ext}"}), 400

    file_bytes = f.read()
    auto_orient = request.form.get("auto_orient", "1") == "1"
    filename = f.filename or "תרגיל"

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"q": queue.Queue(), "ts": time.time(), "pages": None, "filename": None}
    threading.Thread(
        target=_run_job,
        args=(job_id, file_bytes, ext, auto_orient, filename),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/analyze/stream/<job_id>")
def analyze_stream(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    def gen():
        q: queue.Queue = job["q"]
        while True:
            ev = q.get()
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if ev["type"] in ("end", "error"):
                break

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/result-data/<job_id>")
def result_data(job_id: str):
    """Return the analysis result as JSON (for client-side rendering after SSE)."""
    job = JOBS.get(job_id)
    if not job or not job.get("pages"):
        return jsonify({"error": "תוצאה לא נמצאה"}), 404
    return jsonify({"pages": job["pages"], "filename": job.get("filename", "")})


@app.route("/save/<job_id>")
def save_json(job_id: str):
    """Download the analysis result as a JSON file."""
    job = JOBS.get(job_id)
    if not job or not job.get("pages"):
        return "תוצאה לא נמצאה", 404
    filename = job.get("filename", "תרגיל")
    base = os.path.splitext(filename)[0]
    today = datetime.date.today().strftime("%Y%m%d")
    save_name = f"{base}_math_{today}.json"
    payload = json.dumps(
        {"pages": job["pages"], "filename": filename},
        ensure_ascii=False,
        indent=2,
    )
    return Response(
        payload,
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{save_name}"'},
    )


@app.route("/load", methods=["POST"])
def load_json():
    """Parse and return a previously saved JSON result file."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "לא הועלה קובץ"}), 400
    try:
        data = json.loads(f.read().decode("utf-8"))
        return jsonify({
            "pages": data.get("pages", []),
            "filename": data.get("filename", "שמור.json"),
        })
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"קובץ JSON לא תקין: {e}"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5051, threaded=True)
