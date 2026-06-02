"""
label_ui.py — ממשק גרפי לתיוג מילים.

הרצה: python label_ui.py
ואז: http://localhost:5057

זרימה: בוחרים תמונה, גוררים סביב מילים, מקלידים את התמלול הנכון.
כל מילה נשמרת אוטומטית ל-data/annotations/<name>.words.json בפורמט:
  {"image": "...", "image_size": [w,h], "image_path": "...",
   "words": [{"x": ..., "y": ..., "w": ..., "h": ..., "text": "..."}]}
"""
import io
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file

from remez_lib import (
    MAX_TOKENS,
    MODEL,
    client,
    crop_word,
    extract_json,
    image_block,
    load_image_normalized,
    normalize_for_claude,
)

load_dotenv()

app = Flask(__name__)

SAMPLES = Path(os.environ["SAMPLES_DIR"])
ANN_DIR = Path(__file__).parent / "data" / "annotations"
ANN_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    name = request.args.get("image")
    if not name:
        images = sorted(SAMPLES.glob("*.jpg")) + sorted(SAMPLES.glob("*.jpeg"))
        return render_template("index.html", images=[p.name for p in images])
    return render_template("label.html", image=name)


@app.route("/api/image/<path:name>")
def get_image(name):
    img = normalize_for_claude(load_image_normalized(str(SAMPLES / name)))
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
    img = normalize_for_claude(load_image_normalized(str(SAMPLES / name)))
    return jsonify({
        "image": name,
        "image_size": list(img.size),
        "image_path": str((SAMPLES / name).resolve()),
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
def transcribe(name):
    use_hints = request.args.get("hints") == "1"
    refresh = request.args.get("refresh") == "1"
    suffix = "linetrans_hinted" if use_hints else "linetrans"
    cache = ANN_DIR / f"{Path(name).stem}.{suffix}.json"

    if cache.exists() and not refresh:
        d = json.loads(cache.read_text(encoding="utf-8"))
        d["cached"] = True
        return jsonify(d)

    img = normalize_for_claude(load_image_normalized(str(SAMPLES / name)))
    content = []
    hint_count = 0

    if use_hints:
        words_path = ANN_DIR / f"{Path(name).stem}.words.json"
        if words_path.exists():
            words = json.loads(words_path.read_text(encoding="utf-8")).get("words", [])
            for w in words:
                if not w.get("text", "").strip():
                    continue
                content.append(image_block(crop_word(img, w)))
                content.append({"type": "text", "text": f"המילה הזו אומרת: {w['text']}"})
                hint_count += 1
            if hint_count:
                content.append({"type": "text", "text": (
                    f"לעיל {hint_count} דוגמאות של מילים מאותו תלמיד עם התמלול הנכון. "
                    "השתמש בהן כדי לכייל את ההבנה שלך לכתב היד שלו. "
                    "אם אתה לא בטוח במילה — עדיף [לא קריא] מאשר ניחוש."
                )})

    content.append(image_block(img))
    content.append({"type": "text", "text": PROMPT_TASK})

    cli = client()
    resp = cli.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": content}],
    )
    raw = resp.content[0].text
    data = extract_json(raw)
    data["hint_count"] = hint_count
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

    img = normalize_for_claude(load_image_normalized(str(SAMPLES / name)))
    data = {
        "image": name,
        "image_size": list(img.size),
        "image_path": str((SAMPLES / name).resolve()),
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
