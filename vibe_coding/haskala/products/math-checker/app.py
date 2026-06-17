"""math-checker — בדיקת תרגילי מתמטיקה בכתב יד (מוצר #2 של חברת השכלה).

בניגוד לבודק-השפה (products/checker), כאן אין חיתוך שורה-שורה. תרגיל מתמטי הוא
יחידה הוליסטית — סרטוטים, הוכחות גאומטריות ומעברים אלגבריים פרושים ב-2D. לכן
העמוד המלא נשלח ל-Claude Vision שמחזיר הבנה מובנית (JSON): רשימת תרגילים, ולכל
אחד את צעדי הפתרון ב-LaTeX, הערכת תקינות לכל צעד, ומשוב.
"""
from __future__ import annotations

import base64
import io
import json
import os

import fitz  # pymupdf
from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image

load_dotenv()

app = Flask(__name__)
client = Anthropic()

ANALYSIS_MODEL = "claude-opus-4-8"      # ההבנה והבדיקה — איכות
ORIENT_MODEL = "claude-haiku-4-5"       # זיהוי סיבוב — זול ומהיר

RENDER_DPI = 200          # A4 ב-200dpi ≈ 1654×2339 px
MAX_EDGE = 2200           # צמצום הקצה הארוך לפני שליחה למודל


# ---------------------------------------------------------------- rendering

def file_to_pages(data: bytes, ext: str) -> list[Image.Image]:
    """PDF → תמונה לעמוד; צילום (jpg/png) → עמוד יחיד."""
    ext = ext.lower()
    if ext == ".pdf":
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=RENDER_DPI)
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        doc.close()
        return pages
    return [Image.open(io.BytesIO(data)).convert("RGB")]


def downscale(img: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    long_edge = max(img.size)
    if long_edge <= max_edge:
        return img
    scale = max_edge / long_edge
    return img.resize((round(img.width * scale), round(img.height * scale)))


def img_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _image_block(img: Image.Image) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": img_to_b64(img)},
    }


# ---------------------------------------------------------------- orientation

def detect_rotation(img: Image.Image) -> int:
    """מחזיר 0/90/180/270 — כמה מעלות עם כיוון השעון לסובב כדי שהדף יהיה זקוף.

    auto_rotate הרגיל מטפל רק ב-portrait/landscape ולא בהיפוך 180°, שזה בדיוק
    הכשל שנצפה בסריקות שנבדקו. כאן שואלים את Haiku ישירות.
    """
    small = downscale(img, 1100)
    msg = client.messages.create(
        model=ORIENT_MODEL,
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": [
                _image_block(small),
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
    # PIL.rotate מסובב נגד כיוון השעון; deg הוא עם כיוון השעון.
    return img.rotate(-deg, expand=True)


# ---------------------------------------------------------------- analysis

ANALYSIS_PROMPT = r"""אתה בודק מבחנים במתמטיקה (שכבת חטיבה/תיכון) בישראל. לפניך עמוד סרוק
של פתרון תלמיד בכתב יד. נתח את העמוד כיחידה שלמה — אל תחתוך לשורות. שים לב לסרטוטים,
הוכחות גאומטריות, ומעברים אלגבריים שפרושים על פני הדף.

החזר אך ורק JSON תקין במבנה הבא (בלי טקסט נוסף, בלי גדרות ```):
{
  "page_summary": "תיאור קצר של מה יש בעמוד",
  "has_diagram": true/false,
  "diagram_description": "תיאור הסרטוט אם קיים, אחרת ריק",
  "problems": [
    {
      "id": "מספר/אות התרגיל כפי שמופיע",
      "topic": "אלגברה/גאומטריה/...",
      "statement_latex": "ניסוח השאלה אם נראה בעמוד (LaTeX), אחרת ריק",
      "student_steps": [
        {"latex": "הצעד כפי שכתב התלמיד, ב-LaTeX", "ok": true/false, "comment": "הערה קצרה אם יש שגיאה"}
      ],
      "final_answer_latex": "התשובה הסופית של התלמיד (LaTeX)",
      "verdict": "correct" | "partial" | "incorrect" | "unclear",
      "score_suggestion": "הצעת ניקוד טקסטואלית, למשל 'מלא' / 'חלקי — טעות חישוב' ",
      "feedback": "משוב קצר ובונה לתלמיד בעברית"
    }
  ]
}

כללים:
- כל ביטוי מתמטי ב-LaTeX תקני (בלי $; רק תוכן, למשל: x^2 - 6x + 10).
- אם צעד שגוי — סמן ok=false והסבר ב-comment מה הטעות (חישובית/שיטתית).
- אם משהו לא קריא — כתוב "[לא קריא]" באותו שדה.
- verdict לפי הפתרון כולו: correct/partial/incorrect/unclear.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def analyze_page(img: Image.Image) -> dict:
    img = downscale(img)
    msg = client.messages.create(
        model=ANALYSIS_MODEL,
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": [_image_block(img), {"type": "text", "text": ANALYSIS_PROMPT}],
        }],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        return {"page_summary": "[נכשל פענוח JSON מהמודל]", "has_diagram": False,
                "diagram_description": "", "problems": [], "raw": raw}


# ---------------------------------------------------------------- routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "לא הועלה קובץ"}), 400
    data = f.read()
    ext = os.path.splitext(f.filename or "")[1] or ".pdf"
    auto_orient = request.form.get("auto_orient", "1") == "1"

    pages = file_to_pages(data, ext)
    results = []
    for i, img in enumerate(pages, 1):
        rotation = detect_rotation(img) if auto_orient else 0
        img = apply_rotation(img, rotation)
        analysis = analyze_page(img)
        results.append({
            "page": i,
            "rotation_applied": rotation,
            "image_b64": img_to_b64(downscale(img, 1400)),
            "analysis": analysis,
        })
    return jsonify({"pages": results})


if __name__ == "__main__":
    app.run(debug=True, port=5051, threaded=True)
