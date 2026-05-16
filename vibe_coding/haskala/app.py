import base64
import hashlib
import io
import json
import logging
import os

import anthropic
import fitz
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request
from PIL import Image

load_dotenv()

# Pipeline diagnostics go to a file (not just stdout) so the segmentation
# behaviour can be inspected after a run without watching the console.
LOG_PATH = os.path.join(os.path.dirname(__file__), "haskala.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a"), logging.StreamHandler()],
)
log = logging.getLogger("haskala")

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
3. כל שורה ויזואלית בתמונה = פריט אחד בפלט, מלמעלה למטה לפי הסדר.
4. שמור על כל סימני הפיסוק בדיוק כפי שנכתבו (כולל פיסוק חסר).
5. שמור על אותיות גדולות וקטנות באנגלית בדיוק כפי שנכתבו.
6. אם תו לא ברור, תמלל את מה שהכי דומה לצורה הכתובה.
7. אם מילה לא קריאה לחלוטין, כתוב במקומה: [לא קריא]
8. שורות ריקות בתמונה — דלג עליהן (אל תפיק פריט לשורה ריקה).

החזר את התוצאה כ-JSON תקין במבנה: {"lines": [{"text": "..."}, ...]}."""

OCR_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
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
                        "text": "תמלל שורה-שורה והחזר JSON.",
                    },
                ],
            }
        ],
    )

    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return data["lines"]


def crop_line(img: Image.Image, y_top: int, y_bottom: int, padding: int = 0) -> str:
    h = img.height
    top = max(0, y_top - padding)
    bottom = min(h, y_bottom + padding)
    strip = img.crop((0, top, img.width, bottom))
    return image_to_b64(strip, "PNG")


def _smoothed_profile(
    img: Image.Image,
    ink_threshold: int = 180,
    smooth_window: int = 25,
    solid_row_ratio: float = 0.40,
    edge_margin_ratio: float = 0.04,
) -> np.ndarray:
    """Horizontal-projection ink profile, smoothed, with noise removed.

    solid_row_ratio: any row whose ink covers more than this fraction of the
        page width is zeroed out. Handwriting strokes are thin and gappy and
        never cover that much of a row — but a scan bar or heavy underline
        might.
    edge_margin_ratio: the top/bottom slice of the page is zeroed. A photo of
        a bound notebook almost always catches the spiral/clip and the desk
        shadow at the extreme top edge; its coverage is the *same* as a dense
        text row, so it can't be told apart by density — but real writing
        always has a page margin, so anything in the first/last few percent
        is noise. This was the root cause of the persistent off-by-one: the
        binding became a phantom first line and shifted every strip down one.
    """
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    profile = (gray < ink_threshold).astype(np.float32).sum(axis=1)
    profile[profile > w * solid_row_ratio] = 0.0
    margin = int(h * edge_margin_ratio)
    if margin > 0:
        profile[:margin] = 0.0
        profile[h - margin:] = 0.0
    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        profile = np.convolve(profile, kernel, mode="same")
    return profile


def deskew_page(img: Image.Image, max_angle: float = 6.0) -> tuple[Image.Image, float]:
    """Rotate the page so its text lines are horizontal.

    A phone photo of a notebook is almost always shot at a slight angle, so
    the ruled lines curve across the frame. A horizontal projection of a
    skewed page smears neighbouring lines into each other (shallow valleys —
    the root cause of merged/duplicated strips). We find the rotation that
    maximises profile contrast (std): when lines are level the projection has
    tall peaks and deep valleys, so its standard deviation peaks too.
    """
    def sharpness(angle: float) -> float:
        rot = img.rotate(
            angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255)
        )
        return float(_smoothed_profile(rot).std())

    best_angle, best = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + 0.01, 0.5):
        s = sharpness(float(a))
        if s > best:
            best_angle, best = float(a), s
    for a in np.arange(best_angle - 0.5, best_angle + 0.51, 0.1):
        s = sharpness(float(a))
        if s > best:
            best_angle, best = float(a), s
    angle = round(best_angle, 1)
    if abs(angle) < 0.1:
        return img, 0.0
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255)), angle


def _writing_region(profile: np.ndarray) -> tuple[int, int]:
    """First/last row of *sustained* ink — a thin stray mark won't start it."""
    h = len(profile)
    if profile.max() <= 0:
        return 0, h
    present = profile > profile.max() * 0.10
    min_run = 30
    top, bottom = 0, h
    run = 0
    for y in range(h):
        if present[y]:
            run += 1
            if run >= min_run:
                top = y - run + 1
                break
        else:
            run = 0
    run = 0
    for y in range(h - 1, -1, -1):
        if present[y]:
            run += 1
            if run >= min_run:
                bottom = y + run
                break
        else:
            run = 0
    return top, min(h, bottom)


def _prominence(region: np.ndarray, i: int) -> float:
    """Topographic prominence of peak `i` — how far it rises above the
    deeper of the two valleys separating it from any taller neighbour."""
    left = i
    left_min = region[i]
    while left > 0 and region[left] <= region[i]:
        left_min = min(left_min, region[left])
        left -= 1
    right = i
    right_min = region[i]
    while right < len(region) - 1 and region[right] <= region[i]:
        right_min = min(right_min, region[right])
        right += 1
    return float(region[i] - max(left_min, right_min))


def segment_into_lines(img: Image.Image, num_lines: int) -> list[tuple[int, int]]:
    """Cut the writing region into exactly `num_lines` strips.

    We know the line count from the OCR, so rather than guess it from a
    threshold (which merges close lines and splits sparse ones), we place one
    peak per written line in the ink profile and cut at the lowest point
    between consecutive peaks.

    The first and last lines are *anchored* to the strongest bump in the
    first/last pitch-window; the middle `num_lines - 2` are the most
    prominent peaks in between, spaced at least half a pitch apart. Anchoring
    is what keeps a faint short opening line (a name/signature) or a faint
    closing line from being out-competed by the dense body lines — without
    it, a dense line grabs two peaks and a faint edge line gets none, which
    shifts a whole block of strips by one.
    """
    if num_lines <= 0:
        return []
    profile = _smoothed_profile(img)
    top, bottom = _writing_region(profile)
    if num_lines == 1 or bottom - top < num_lines:
        return [(top, bottom)]

    region = profile[top:bottom]
    floor = profile.max() * 0.05

    # Dominant line pitch (notebook ruling) → minimum spacing between peaks.
    # 0.5 (not higher): some lines are written consecutively with a tight gap,
    # so a larger guard would merge two real lines and lose one peak.
    centred = region - region.mean()
    autocorr = np.correlate(centred, centred, "full")[len(centred) - 1:]
    pitch = 80 + int(np.argmax(autocorr[80:220])) if len(autocorr) > 220 else 100
    pitch = min(pitch, len(region))
    min_dist = max(15, int(pitch * 0.5))

    candidates = [
        i
        for i in range(1, len(region) - 1)
        if region[i] >= region[i - 1] and region[i] > region[i + 1] and region[i] > floor
    ]

    if num_lines >= 2 and len(candidates) >= num_lines:
        first = int(np.argmax(region[:pitch]))
        last = len(region) - pitch + int(np.argmax(region[-pitch:]))
        middle = sorted(
            (c for c in candidates if first + min_dist <= c <= last - min_dist),
            key=lambda i: -_prominence(region, i),
        )
        peaks = [first, last]
        for c in middle:
            if all(abs(c - p) >= min_dist for p in peaks):
                peaks.append(c)
            if len(peaks) == num_lines:
                break
        peaks.sort()

    if num_lines < 2 or len(candidates) < num_lines or len(peaks) < num_lines:
        # Not enough structure — fall back to even spacing in the region.
        step = (bottom - top) / num_lines
        return [
            (int(round(top + k * step)), int(round(top + (k + 1) * step)))
            for k in range(num_lines)
        ]

    bounds = [0]
    for a, b in zip(peaks, peaks[1:]):
        bounds.append(a + int(np.argmin(region[a:b])))
    bounds.append(len(region) - 1)
    segments = [(top + bounds[i], top + bounds[i + 1]) for i in range(num_lines)]
    heights = [b - a for a, b in segments]
    ink = [int(profile[a:b].sum()) for a, b in segments]
    med_ink = sorted(ink)[len(ink) // 2]
    near_empty = [i + 1 for i, v in enumerate(ink) if v < 0.35 * med_ink]
    log.info(
        "segment: region=(%d,%d) lines=%d pitch=%d peaks=%d "
        "heights min/med/max=%d/%d/%d near_empty_strips=%s",
        top, bottom, num_lines, pitch, len(peaks),
        min(heights), sorted(heights)[len(heights) // 2], max(heights),
        near_empty or "none",
    )
    return segments


def process_pdf(pdf_bytes: bytes) -> list[dict]:
    pages = pdf_to_page_images(pdf_bytes)
    results = []
    for page_idx, img in enumerate(pages):
        img, angle = deskew_page(img)
        texts = [line["text"] for line in ocr_page(img)]
        log.info("[page %d] deskew=%s° %d lines transcribed", page_idx + 1, angle, len(texts))
        bands = segment_into_lines(img, len(texts))
        line_items = [
            {"text": text, "image_b64": crop_line(img, y_top, y_bottom)}
            for text, (y_top, y_bottom) in zip(texts, bands)
        ]
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
