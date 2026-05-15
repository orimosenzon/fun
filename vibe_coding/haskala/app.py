import base64
import hashlib
import io
import json

import anthropic
import fitz
from dotenv import load_dotenv
from flask import Flask, render_template, request
from PIL import Image

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

client = anthropic.Anthropic()

# Render PDF at this DPI. Higher = better OCR but more tokens/cost.
# 200 DPI gives a good balance — A4 → ~1700×2340 px, well under Opus 4.7's 2576px limit.
RENDER_DPI = 200

OCR_SYSTEM_PROMPT = """אתה מבצע OCR מדויק על צילום של תרגיל כתוב ביד.

כללי תמלול קריטיים:
1. תמלל בדיוק את מה שכתוב — אל תתקן שום דבר.
2. שמור על שגיאות כתיב בדיוק כפי שהן.
3. כל שורה ויזואלית בתמונה = פריט אחד בפלט.
4. שמור על כל סימני הפיסוק בדיוק כפי שנכתבו (כולל פיסוק חסר).
5. שמור על אותיות גדולות וקטנות באנגלית בדיוק כפי שנכתבו.
6. אם תו לא ברור, תמלל את מה שהכי דומה לצורה הכתובה.
7. אם מילה לא קריאה לחלוטין, כתוב במקומה: [לא קריא]
8. שורות ריקות בתמונה — דלג עליהן (אל תפיק פריט לשורה ריקה).

עבור כל שורה, החזר:
- text: התמלול המדויק (כולל טעויות)
- y_top: פיקסל ה-y שבו מתחיל החלק העליון ביותר של הכתב היד (כולל אותיות גבוהות כמו ל, k, b, h, t, וקווים אלכסוניים שעולים מעל אזור האות הרגיל). זה לא הקו המודפס של המחברת מעל השורה — זה איפה שהדיו של התלמיד מתחיל.
- y_bottom: פיקסל ה-y שבו נגמר החלק הנמוך ביותר של הכתב יד (כולל זנבות יורדים כמו ך, ץ, ן, g, y, p, j). זה לא הקו המודפס של המחברת מתחת לשורה — זה איפה שהדיו של התלמיד נגמר.

חשוב: הטווח [y_top, y_bottom] חייב לכסות את כל הכתב היד של השורה בנדיבות. עדיף לכלול קצת רקע ריק מאשר לחתוך אותיות. בכתב יד עברי עם אנגלית מעורבת, גובה השורה הוא בדרך כלל 80-150 פיקסלים.

החזר את התוצאה כ-JSON תקין במבנה: {"lines": [{"text": "...", "y_top": N, "y_bottom": N}, ...]}.
הקואורדינטות חייבות להיות במערכת הפיקסלים של התמונה שניתנה לך."""

OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "y_top": {"type": "integer"},
                    "y_bottom": {"type": "integer"},
                },
                "required": ["text", "y_top", "y_bottom"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}


def pdf_to_page_images(pdf_bytes: bytes) -> list[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


def image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def ocr_page(img: Image.Image) -> list[dict]:
    img_b64 = image_to_b64(img, "PNG")

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=8000,
        system=[
            {
                "type": "text",
                "text": OCR_SYSTEM_PROMPT,
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
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"גודל התמונה: {img.width} × {img.height} פיקסלים. תמלל שורה-שורה והחזר JSON.",
                    },
                ],
            }
        ],
    )

    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return data["lines"]


def crop_line(img: Image.Image, y_top: int, y_bottom: int, padding: int = 30) -> str:
    h = img.height
    top = max(0, y_top - padding)
    bottom = min(h, y_bottom + padding)
    strip = img.crop((0, top, img.width, bottom))
    return image_to_b64(strip, "PNG")


def normalize_line_heights(lines: list[dict]) -> list[dict]:
    """If Claude returned a strip much thinner than the page median, expand it."""
    if not lines:
        return lines
    heights = [line["y_bottom"] - line["y_top"] for line in lines]
    heights = [h for h in heights if h > 0]
    if not heights:
        return lines
    sorted_h = sorted(heights)
    median = sorted_h[len(sorted_h) // 2]
    min_height = int(median * 0.6)
    for line in lines:
        h = line["y_bottom"] - line["y_top"]
        if h < min_height:
            center = (line["y_top"] + line["y_bottom"]) // 2
            line["y_top"] = center - min_height // 2
            line["y_bottom"] = center + min_height // 2
    return lines


def process_pdf(pdf_bytes: bytes) -> list[dict]:
    pages = pdf_to_page_images(pdf_bytes)
    results = []
    for page_idx, img in enumerate(pages):
        lines = ocr_page(img)
        lines = normalize_line_heights(lines)
        line_items = []
        for line in lines:
            y_top = max(0, min(img.height, int(line["y_top"])))
            y_bottom = max(0, min(img.height, int(line["y_bottom"])))
            if y_bottom <= y_top:
                continue
            line_items.append(
                {
                    "text": line["text"],
                    "image_b64": crop_line(img, y_top, y_bottom),
                }
            )
        results.append({"page": page_idx + 1, "lines": line_items})
    return results


def compute_doc_key(pages: list[dict]) -> str:
    """Stable hash of the line texts — used as localStorage key."""
    joined = "\n".join(
        line["text"]
        for page in pages
        for line in page.get("lines", [])
    )
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def render_result(pages: list[dict], filename: str, annotations: dict | None = None):
    server_data = {
        "doc_key": compute_doc_key(pages),
        "filename": filename,
        "annotations": annotations or {},
    }
    return render_template(
        "index.html",
        pages=pages,
        filename=filename,
        server_data_json=json.dumps(server_data, ensure_ascii=False),
    )


def parse_saved_json(raw: bytes) -> tuple[list[dict], str, dict]:
    """Returns (pages_for_template, filename, annotations_by_line_key).

    Strips annotations out of the per-line dicts (the template doesn't need
    them inline — they're delivered via server_data_json instead).
    """
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or "pages" not in data:
        raise ValueError("מבנה JSON לא תקין — חסר שדה 'pages'")

    pages_out = []
    annotations: dict[str, list] = {}
    for page in data["pages"]:
        page_num = page.get("page", len(pages_out) + 1)
        lines_out = []
        for line_idx, line in enumerate(page.get("lines", [])):
            text = line.get("text", "")
            image_b64 = line.get("image_b64", "")
            line_key = f"p{page_num}-l{line_idx}"
            line_annots = line.get("annotations") or []
            if line_annots:
                annotations[line_key] = line_annots
            lines_out.append({"text": text, "image_b64": image_b64})
        pages_out.append({"page": page_num, "lines": lines_out})

    filename = data.get("filename", "שמור.json")
    return pages_out, filename, annotations


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    saved = request.files.get("saved")
    if saved and saved.filename:
        raw = saved.read()
        if not raw:
            return render_template("index.html", error="קובץ ה-JSON ריק")
        try:
            pages, filename, annotations = parse_saved_json(raw)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
            return render_template("index.html", error=f"שגיאה בקריאת JSON: {e}")
        return render_result(pages, filename, annotations)

    file = request.files.get("pdf")
    if not file or not file.filename:
        return render_template("index.html", error="לא נבחר קובץ")

    if not file.filename.lower().endswith(".pdf"):
        return render_template("index.html", error="חייב להיות קובץ PDF")

    pdf_bytes = file.read()
    if not pdf_bytes:
        return render_template("index.html", error="הקובץ ריק")

    try:
        pages = process_pdf(pdf_bytes)
    except anthropic.APIError as e:
        return render_template("index.html", error=f"שגיאת API: {e.message}")
    except (json.JSONDecodeError, KeyError) as e:
        return render_template("index.html", error=f"שגיאת פענוח: {e}")

    return render_result(pages, file.filename)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
