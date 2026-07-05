"""math-checker — בדיקת תרגילי מתמטיקה/גאומטריה בכתב יד (מוצר #2, חברת השכלה).

בניגוד לבודק-השפה (products/checker), כאן אין חיתוך שורה-שורה. תרגיל מתמטי
הוא יחידה הוליסטית — סרטוטים, הוכחות גאומטריות ומעברים אלגבריים פרושים ב-2D.
לכן העמוד המלא נשלח ל-Opus שמחזיר הבנה מובנית (JSON): רשימת תרגילים, ולכל
אחד את צעדי הפתרון ב-LaTeX, הערכת תקינות לכל צעד, ומשוב.
"""
from __future__ import annotations

import datetime
import hmac
import json
import logging
import logging.handlers
import os
import queue
import threading
import time
import uuid

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from core import (ACCEPTED_EXTS, DEFAULT_MODEL, PREVIEW_MAX_EDGE, _img_to_b64_jpeg,
                  analyze_page, apply_rotation, humanize_error, process_stream,
                  resolve_model)
from report import build_result_docx, build_result_html

load_dotenv(override=True)

LOG_PATH = os.path.join(os.path.dirname(__file__), "math-checker.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        # rotation כדי שהלוג לא יגדל בלי הגבלה (5MB × 3 גיבויים)
        logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3),
        logging.StreamHandler(),
    ],
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


JOB_TTL = 3600  # seconds — results stay in memory for result refreshes
JOBS: dict[str, dict] = {}


# ─── streaming job runner ─────────────────────────────────────────────────────

def _evict_stale_jobs():
    now = time.time()
    for jid in [k for k, v in JOBS.items() if now - v.get("ts", 0) > JOB_TTL]:
        JOBS.pop(jid, None)


def _run_job(job_id: str, file_bytes: bytes, ext: str, auto_orient: bool, filename: str,
             model_key: str = DEFAULT_MODEL, model_label: str = "", rubric: str = ""):
    job = JOBS[job_id]
    q: queue.Queue = job["q"]
    try:
        for ev in process_stream(file_bytes, ext, auto_orient, model_key,
                                  model_label, rubric):
            if ev["type"] == "result":
                job["pages"] = ev["pages"]
                job["imgs"] = ev["imgs"]
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
    # no-store: the UI markup changes often during development and a stale
    # cached index.html silently sends old form values (e.g. auto_orient
    # defaulting on). Force the browser to always fetch fresh HTML.
    resp = Response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


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
    auto_orient = request.form.get("auto_orient", "0") == "1"
    model_key, model_label = resolve_model(request.form.get("model"))
    rubric = (request.form.get("rubric") or "").strip()
    filename = f.filename or "תרגיל"

    job_id = str(uuid.uuid4())
    log.info(
        "[upload] job=%s file=%r ext=%s size=%.1fKB model=%s auto_orient=%s rubric=%dch",
        job_id[:8], filename, ext, len(file_bytes) / 1024, model_key, auto_orient,
        len(rubric),
    )
    JOBS[job_id] = {"q": queue.Queue(), "ts": time.time(), "pages": None,
                    "imgs": None, "filename": None, "model": model_key,
                    "rubric": rubric}
    threading.Thread(
        target=_run_job,
        args=(job_id, file_bytes, ext, auto_orient, filename, model_key,
              model_label, rubric),
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


@app.route("/reanalyze", methods=["POST"])
def reanalyze():
    """Re-run analysis on a single page after the user manually rotated it.

    Body: {job_id, page, rotate}. `rotate` is extra CW degrees (0/90/180/270)
    to apply to the page's current orientation before re-analyzing.
    """
    data = request.get_json(silent=True) or {}
    job = JOBS.get(data.get("job_id"))
    if not job or not job.get("imgs"):
        return jsonify({"message": "התרגיל לא נמצא", "details": "ייתכן שהזמן הקצוב פג. טען מחדש."}), 404

    try:
        idx = int(data.get("page", 0)) - 1
    except (TypeError, ValueError):
        idx = -1
    if not (0 <= idx < len(job["imgs"])):
        return jsonify({"message": "מספר עמוד לא תקין"}), 400

    rotate = int(data.get("rotate", 0)) % 360
    img = apply_rotation(job["imgs"][idx], rotate) if rotate else job["imgs"][idx]

    model_key = job.get("model", DEFAULT_MODEL)
    try:
        analysis = analyze_page(img, model_key, job.get("rubric", ""))
    except Exception as e:
        log.exception("reanalyze failed")
        return jsonify(humanize_error(e)), 502

    new_rot = (job["pages"][idx].get("rotation_applied", 0) + rotate) % 360
    page_data = {
        "page": idx + 1,
        "rotation_applied": new_rot,
        "image_b64": _img_to_b64_jpeg(img, PREVIEW_MAX_EDGE, quality=85),
        "analysis": analysis,
    }
    job["imgs"][idx] = img
    job["pages"][idx] = page_data
    job["ts"] = time.time()
    log.info("[reanalyze page %d] rotate=%d° → rotation=%d° problems=%d",
             idx + 1, rotate, new_rot, len(analysis.get("problems", [])))
    return jsonify(page_data)


# fields the teacher may edit; everything else (image, latex, steps) stays as
# the model produced it and is never overwritten by a client update.
_EDITABLE_FIELDS = ("points_earned", "points_max", "verdict",
                    "score_suggestion", "feedback")


@app.route("/update/<job_id>", methods=["POST"])
def update_result(job_id: str):
    """Persist the teacher's edits (per-problem scores/feedback + approval) back
    onto the job so the exports reflect what the teacher confirmed, not the raw
    AI suggestion. Merges only whitelisted fields by position; the scan image
    and the model's transcription/steps are left intact."""
    job = JOBS.get(job_id)
    if not job or not job.get("pages"):
        return jsonify({"error": "תוצאה לא נמצאה"}), 404

    data = request.get_json(silent=True) or {}
    incoming = data.get("pages") or []
    for pi, page in enumerate(job["pages"]):
        if pi >= len(incoming):
            break
        in_problems = ((incoming[pi] or {}).get("analysis") or {}).get("problems") or []
        problems = (page.get("analysis") or {}).get("problems") or []
        for qi, pr in enumerate(problems):
            if qi >= len(in_problems):
                break
            src = in_problems[qi] or {}
            for fld in _EDITABLE_FIELDS:
                if fld in src:
                    pr[fld] = src[fld]

    if "approved" in data:
        job["approved"] = bool(data["approved"])
    job["ts"] = time.time()
    log.info("[update] job=%s approved=%s", job_id[:8], job.get("approved"))
    return jsonify({"ok": True, "approved": job.get("approved", False)})


@app.route("/result-data/<job_id>")
def result_data(job_id: str):
    """Return the analysis result as JSON (for client-side rendering after SSE)."""
    job = JOBS.get(job_id)
    if not job or not job.get("pages"):
        return jsonify({"error": "תוצאה לא נמצאה"}), 404
    return jsonify({"pages": job["pages"], "filename": job.get("filename", ""),
                    "approved": job.get("approved", False)})


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


def _export_basename(job: dict) -> str:
    base = os.path.splitext(job.get("filename", "תרגיל") or "תרגיל")[0]
    today = datetime.date.today().strftime("%Y%m%d")
    return f"{base}_math_{today}"


@app.route("/save/html/<job_id>")
def save_html(job_id: str):
    """Download the analysis result as a standalone HTML report (KaTeX math)."""
    job = JOBS.get(job_id)
    if not job or not job.get("pages"):
        return "תוצאה לא נמצאה", 404
    html_doc = build_result_html(job["pages"], job.get("filename", "תרגיל"),
                                 job.get("approved", False))
    return Response(
        html_doc,
        mimetype="text/html",
        headers={"Content-Disposition":
                 f'attachment; filename="{_export_basename(job)}.html"'},
    )


@app.route("/save/docx/<job_id>")
def save_docx(job_id: str):
    """Download the analysis result as a Word (.docx) document."""
    job = JOBS.get(job_id)
    if not job or not job.get("pages"):
        return "תוצאה לא נמצאה", 404
    docx_bytes = build_result_docx(job["pages"], job.get("filename", "תרגיל"),
                                   job.get("approved", False))
    return Response(
        docx_bytes,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition":
                 f'attachment; filename="{_export_basename(job)}.docx"'},
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
