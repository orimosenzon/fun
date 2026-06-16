"""Numbered-overlay OCR protocol pilot.

Draws the detected line boxes + numbers on the page, sends it to a vision
model, and asks for {n, text} pairs — the model itself binds each
transcribed line to a detected region, killing the off-by-one class of bugs.

Usage:
    python3 segment_lab/pilot.py <file> [provider]   # provider: claude|gemini

Outputs in segment_lab/out/:
    <stem>_overlay.png   what the model saw
    <stem>_check.png     per-line strips + transcriptions for visual QA
    <stem>_pilot.json    the raw mapping
"""
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402
import lab  # noqa: E402

OUT = lab.OUT
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 26)
FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)

PROMPT = """\
אתה מתמלל דף תרגיל בכתב יד של תלמיד.
על התמונה צוירו מלבנים כחולים ממוספרים (המספרים בירוק, משמאל לכל מלבן).
המלבנים והמספרים הם סימון סינתטי שהוסף על ידי תוכנה — הם אינם חלק מהדף.

משימה: תמלל את כתב היד שורה אחר שורה, לפי הסדר מלמעלה למטה.
לכל שורה שתמללת ציין את מספר המלבן שמכיל אותה.

כללים:
- שמור על שגיאות כתיב ופיסוק של התלמיד בדיוק כפי שנכתבו.
- מלבן שאינו מכיל כתב יד (ריק, קו דפוס, כתם) — דלג עליו, אל תמציא טקסט.
- אם שתי שורות כתובות נופלות באותו מלבן, החזר אותן כשני פריטים עם אותו n.
- אם שורה כתובה לא מכוסה על ידי אף מלבן, החזר אותה עם n=null.
- מילה לא קריאה: כתוב [לא קריא] (בעברית), [illegible] (באנגלית), [غير مقروء] (בערבית) — לפי שפת הטקסט.

החזר JSON בלבד: {"lines":[{"n": <int|null>, "text": "<התמלול>"}]}"""


def make_overlay(img: Image.Image, boxes) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        d.rectangle([x0, y0, x1, y1], outline=(40, 110, 255), width=2)
        label = str(i + 1)
        tx = max(2, x0 - 38)
        ty = (y0 + y1) // 2 - 14
        d.rectangle([tx - 2, ty - 2, tx + 18 * len(label) + 2, ty + 28],
                    fill=(255, 255, 255))
        d.text((tx, ty), label, fill=(0, 150, 0), font=FONT)
    return out


def call_claude(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    resp = app.client.messages.create(
        model=app._ANTHROPIC_MODEL,
        max_tokens=8000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    txt = resp.content[0].text
    txt = txt[txt.index("{"):txt.rindex("}") + 1]
    return json.loads(txt)


def call_gemini(img: Image.Image) -> dict:
    from google import genai
    from google.genai import types
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    client = genai.Client(api_key=app.os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json"),
    )
    return json.loads(resp.text)


def render_check(img: Image.Image, boxes, mapping) -> Image.Image:
    """Strip + text per returned line, stacked — for visual verification."""
    rows = []
    pad = 6
    for item in mapping.get("lines", []):
        n = item.get("n")
        text = item.get("text", "")
        if n is not None and 1 <= n <= len(boxes):
            x0, y0, x1, y1 = boxes[n - 1]
            strip = img.crop((max(0, x0 - pad), max(0, y0 - pad),
                              min(img.width, x1 + pad), min(img.height, y1 + pad)))
        else:
            strip = Image.new("RGB", (600, 40), (255, 220, 220))
        rows.append((n, text, strip))

    W = 1100
    line_h = 30
    total_h = sum(min(r[2].height, 160) + line_h + 14 for r in rows) + 20
    canvas = Image.new("RGB", (W, total_h), (245, 242, 235))
    d = ImageDraw.Draw(canvas)
    y = 10
    for n, text, strip in rows:
        if strip.width > W - 20:
            r = (W - 20) / strip.width
            strip = strip.resize((W - 20, max(1, int(strip.height * r))))
        if strip.height > 160:
            strip = strip.resize((int(strip.width * 160 / strip.height), 160))
        canvas.paste(strip, (10, y))
        y += strip.height + 2
        d.text((10, y), f"[{n}] {text}"[:120], fill=(20, 20, 20), font=FONT_S)
        y += line_h + 12
    return canvas


def run(path: str, provider: str = "claude"):
    stem = Path(path).stem.replace(" ", "_") + f"_{provider}"
    img = lab.preprocess(lab.load_page(path))
    boxes = lab.detect_lines_v2(img)
    print(f"== {stem}: {len(boxes)} boxes")
    overlay = make_overlay(img, boxes)
    overlay.save(OUT / f"{stem}_overlay.png")
    mapping = call_gemini(overlay) if provider == "gemini" else call_claude(overlay)
    (OUT / f"{stem}_pilot.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=1))
    print(f"  model returned {len(mapping.get('lines', []))} lines")
    render_check(img, boxes, mapping).save(OUT / f"{stem}_check.png")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "claude")
