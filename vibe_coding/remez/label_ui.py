"""
label_ui.py — ממשק גרפי לתיוג מילים.

הרצה: python label_ui.py
ואז: http://localhost:5057

זרימה: בוחרים תמונה (העלאה או רשימה), גוררים סביב מילים, מקלידים את התמלול הנכון.
כל מילה נשמרת אוטומטית ל-data/annotations/<name>.words.json בפורמט:
  {"image": "...", "image_size": [w,h], "image_path": "...",
   "words": [{"x": ..., "y": ..., "w": ..., "h": ..., "text": "..."}]}
"""
import hmac
import io
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from remez_lib import (
    DEFAULT_MODEL,
    MODELS,
    crop_word,
    extract_json,
    load_image_normalized,
    normalize_for_claude,
    transcribe,
)

# override=True so a stale ANTHROPIC/GEMINI key left in the shell doesn't
# shadow the real one from .env (already burned us once locally).
load_dotenv(override=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# Local dev keeps SAMPLES_DIR pointing at the private student-work folder.
# In the cloud (HF Space) it's unset and we work only with browser-uploaded
# images sitting in UPLOAD_DIR.
_samples_env = os.environ.get("SAMPLES_DIR", "").strip()
SAMPLES = Path(_samples_env) if _samples_env else None
UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR = Path(__file__).parent / "data" / "annotations"
ANN_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Public deployments must set REMEZ_USER / REMEZ_PASS so the OCR endpoints
# (which spend Anthropic credit) sit behind Basic Auth. Locally we leave
# them unset and the gate is skipped.
BASIC_USER = os.environ.get("REMEZ_USER")
BASIC_PASS = os.environ.get("REMEZ_PASS")
if not (BASIC_USER and BASIC_PASS):
    logging.getLogger("remez").warning(
        "running without Basic Auth — set REMEZ_USER/REMEZ_PASS before exposing publicly"
    )


@app.before_request
def _require_basic_auth():
    if not (BASIC_USER and BASIC_PASS):
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
        {"WWW-Authenticate": 'Basic realm="remez"'},
    )


def list_images():
    """Return (uploads, samples) — uploaded files first, bundled samples second."""
    seen = set()
    uploads = []
    for p in sorted(UPLOAD_DIR.iterdir()):
        if p.suffix.lower() in ALLOWED_EXT and p.name not in seen:
            uploads.append(p.name)
            seen.add(p.name)
    samples = []
    if SAMPLES and SAMPLES.exists():
        for p in sorted(SAMPLES.iterdir()):
            if p.suffix.lower() in ALLOWED_EXT and p.name not in seen:
                samples.append(p.name)
                seen.add(p.name)
    return uploads, samples


def resolve_image(name: str) -> Path:
    """Find an image path by filename. Uploads override samples on collision."""
    p = UPLOAD_DIR / name
    if p.exists():
        return p
    if SAMPLES:
        p = SAMPLES / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


@app.context_processor
def _inject_models():
    return {"models": MODELS, "default_model": DEFAULT_MODEL}


@app.route("/")
def index():
    name = request.args.get("image")
    if not name:
        uploads, samples = list_images()
        return render_template("index.html", uploads=uploads, samples=samples)
    try:
        image_version = int(resolve_image(name).stat().st_mtime)
    except FileNotFoundError:
        image_version = 0
    return render_template("label.html", image=name, image_version=image_version)


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "missing file"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False, "error": f"סוג קובץ לא נתמך: {ext}"}), 400
    base = secure_filename(Path(f.filename).stem) or "upload"
    name = f"{base}{ext}"
    target = UPLOAD_DIR / name
    if target.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{base}_{ts}{ext}"
        target = UPLOAD_DIR / name
    f.save(target)
    return jsonify({"ok": True, "image": name})


@app.route("/api/rotate/<name>", methods=["POST"])
def rotate(name):
    direction = request.args.get("dir", "cw")
    if direction not in {"cw", "ccw"}:
        return jsonify({"ok": False, "error": "dir חייב להיות cw או ccw"}), 400
    src = resolve_image(name)
    # If the file is a sample (outside UPLOAD_DIR), copy into UPLOAD_DIR
    # first so the original on disk stays untouched. resolve_image() and
    # list_images() prefer the UPLOAD_DIR copy, so the rotated version
    # is what the UI sees from now on.
    try:
        src.relative_to(UPLOAD_DIR)
        target = src
    except ValueError:
        target = UPLOAD_DIR / src.name
        if not target.exists():
            shutil.copy(src, target)

    img = Image.open(target)
    img = ImageOps.exif_transpose(img)
    # PIL rotate(): positive = CCW. So CW asks for -90.
    angle = -90 if direction == "cw" else 90
    rotated = img.rotate(angle, expand=True)
    if target.suffix.lower() in {".jpg", ".jpeg"} and rotated.mode != "RGB":
        rotated = rotated.convert("RGB")
    rotated.save(target)

    # Old word boxes and line-trans caches no longer match the pixels.
    stem = Path(name).stem
    for p in ANN_DIR.glob(f"{stem}.*"):
        p.unlink()
    return jsonify({"ok": True})


@app.route("/api/image/<path:name>")
def get_image(name):
    img = normalize_for_claude(load_image_normalized(str(resolve_image(name))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/load/<name>")
def load(name):
    stem = Path(name).stem
    path = ANN_DIR / f"{stem}.words.json"
    if path.exists():
        return jsonify(json.loads(path.read_text(encoding="utf-8")))
    src = resolve_image(name)
    img = normalize_for_claude(load_image_normalized(str(src)))
    return jsonify({
        "image": name,
        "image_size": list(img.size),
        "image_path": str(src.resolve()),
        "words": [],
    })


PROMPT_TASK = (
    "תמלל את כתב היד העברי בתמונה. עבור כל שורת כתב יד החזר את התמלול ואת המיקום האנכי שלה.\n\n"
    "החזר JSON בלבד, ללא הסבר:\n"
    '{"lines": [{"text": "<תמלול השורה>", "y_top": <int>, "y_bottom": <int>}, ...]}\n\n'
    "הנחיות:\n"
    "- דלג על תאריך, כותרת, וכל טקסט מודפס — רק כתב יד.\n"
    "- שורות מסודרות מלמעלה למטה, y_top < y_bottom.\n"
    "- y בפיקסלים של התמונה שהוצגה לך.\n"
    "- שמור על שגיאות כתיב ופיסוק כפי שכתוב.\n"
    "- שורה לא קריאה: text = \"[לא קריא]\"."
)


@app.route("/api/transcribe/<name>")
def transcribe_route(name):
    use_hints = request.args.get("hints") == "1"
    refresh = request.args.get("refresh") == "1"
    model_key = request.args.get("model", DEFAULT_MODEL)
    if model_key not in MODELS:
        model_key = DEFAULT_MODEL
    suffix = "linetrans_hinted" if use_hints else "linetrans"
    # Cache is per-(image, model, hinted/not) so flipping the model picker
    # never returns yesterday's Claude result for today's Gemini run.
    cache = ANN_DIR / f"{Path(name).stem}.{model_key}.{suffix}.json"

    if cache.exists() and not refresh:
        d = json.loads(cache.read_text(encoding="utf-8"))
        d["cached"] = True
        d["model"] = model_key
        return jsonify(d)

    src = resolve_image(name)
    img = normalize_for_claude(load_image_normalized(str(src)))
    parts: list = []
    hint_count = 0

    if use_hints:
        words_path = ANN_DIR / f"{Path(name).stem}.words.json"
        if words_path.exists():
            words = json.loads(words_path.read_text(encoding="utf-8")).get("words", [])
            for w in words:
                if not w.get("text", "").strip():
                    continue
                parts.append(("image", crop_word(img, w)))
                parts.append(("text", f"המילה הזו אומרת: {w['text']}"))
                hint_count += 1
            if hint_count:
                parts.append(("text", (
                    f"לעיל {hint_count} דוגמאות של מילים מאותו תלמיד עם התמלול הנכון. "
                    "השתמש בהן כדי לכייל את ההבנה שלך לכתב היד שלו. "
                    "אם אתה לא בטוח במילה — עדיף [לא קריא] מאשר ניחוש."
                )))

    parts.append(("image", img))
    parts.append(("text", PROMPT_TASK))

    raw = transcribe(parts, model_key=model_key, max_tokens=4000)
    data = extract_json(raw)
    data["hint_count"] = hint_count
    data["model"] = model_key
    data["cached"] = False
    cache.write_text(
        json.dumps({k: v for k, v in data.items() if k != "cached"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return jsonify(data)


@app.route("/api/save/<name>", methods=["POST"])
def save(name):
    stem = Path(name).stem
    path = ANN_DIR / f"{stem}.words.json"
    incoming = request.get_json()

    src = resolve_image(name)
    img = normalize_for_claude(load_image_normalized(str(src)))
    data = {
        "image": name,
        "image_size": list(img.size),
        "image_path": str(src.resolve()),
        "words": incoming.get("words", []),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "path": str(path), "count": len(data["words"])})


CLIENT_LOG = Path(__file__).parent / "logs" / "client.log"
CLIENT_LOG.parent.mkdir(exist_ok=True)


@app.route("/api/log", methods=["POST"])
def log_event():
    payload = request.get_json(silent=True) or {}
    line = f"{datetime.now().isoformat(timespec='seconds')}  {json.dumps(payload, ensure_ascii=False)}\n"
    with CLIENT_LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("→ http://localhost:5057")
    app.run(debug=False, host="127.0.0.1", port=5057)
