"""
annotate.py — שלב 1: יצירת ground truth לדף.

זרימה:
1. שולחים תמונה ל-Claude Vision.
2. מבקשים JSON של שורות עם {text, y_top, y_bottom}.
3. שומרים ל-data/annotations/<name>.gt.json.
4. אתה פותח את הקובץ ב-editor, מתקן טעויות בשדה text, שומר.

זה ה-ground truth שעליו ירוץ הניסוי.
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from remez_lib import (
    MAX_TOKENS,
    MODEL,
    client,
    extract_json,
    image_block,
    load_image_normalized,
    normalize_for_claude,
)

PROMPT = """זוהי תמונה של דף לימוד עברית בכתב יד של תלמיד.

תמלל כל שורה של כתב היד (לא של הטקסט המודפס/הקווים).
החזר JSON בלבד, ללא טקסט נוסף:

{
  "image_height_px": <גובה התמונה בפיקסלים, כפי שאתה רואה אותה>,
  "lines": [
    {"text": "<תמלול השורה>", "y_top": <פיקסלים מהקצה העליון>, "y_bottom": <פיקסלים מהקצה העליון>},
    ...
  ]
}

הנחיות:
- שמור על שגיאות כתיב, פיסוק ומבנה כפי שכתוב במקור.
- שורה שאתה לא מצליח לקרוא ברורה: text = "[לא קריא]".
- y_top/y_bottom חייבים להיות עקביים — y_top < y_bottom, ושורות סדורות מלמעלה למטה.
- אל תכלול את הטקסט המודפס בראש הדף (כותרות, תאריך מודפס, מסגרת).
"""


def annotate(image_path: Path) -> dict:
    img = load_image_normalized(str(image_path))
    norm = normalize_for_claude(img)
    print(f"image: {image_path.name}  original: {img.size}  sent: {norm.size}")

    cli = client()
    resp = cli.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [image_block(norm), {"type": "text", "text": PROMPT}],
        }],
    )
    raw = resp.content[0].text
    data = extract_json(raw)

    data["image_path"] = str(image_path.resolve())
    data["original_image_size"] = list(img.size)
    data["claude_image_size"] = list(norm.size)
    return data


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python annotate.py <image_path_or_filename>")
        sys.exit(1)

    load_dotenv()
    arg = sys.argv[1]

    p = Path(arg)
    if not p.exists():
        samples = Path(__import__("os").environ.get("SAMPLES_DIR", "."))
        candidate = samples / arg
        if candidate.exists():
            p = candidate
        else:
            print(f"file not found: {arg}")
            sys.exit(1)

    out_dir = Path(__file__).parent / "data" / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{p.stem}.gt.json"

    if out_path.exists():
        ans = input(f"{out_path.name} already exists. overwrite? [y/N] ")
        if ans.lower() != "y":
            print("aborted.")
            return

    data = annotate(p)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote: {out_path}")
    print(f"lines: {len(data['lines'])}")
    print("\n--- next ---")
    print(f"1. open the file: {out_path}")
    print("2. review each line's 'text' field, fix any mistakes")
    print("3. save the file")
    print(f"4. run: python experiment.py {out_path}")


if __name__ == "__main__":
    main()
