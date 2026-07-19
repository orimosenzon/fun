"""Parse re.zip (riddle photos named question_answer_hint) into app/riddles.js + app/images/.

Re-run this whenever new riddle photos are added to re.zip.
"""
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).parent
ZIP_PATH = ROOT / "re.zip"
APP_DIR = ROOT / "app"
IMAGES_DIR = APP_DIR / "images"
RIDDLES_JS = APP_DIR / "riddles.js"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

riddles = []
with zipfile.ZipFile(ZIP_PATH) as zf:
    for i, name in enumerate(zf.namelist(), start=1):
        stem = pathlib.Path(name).stem if pathlib.Path(name).suffix else name
        question, answer, hint = ((p.strip() for p in (stem.split("_", 2) + ["", ""])[:3]))
        image_name = f"riddle-{i:02d}.jpg"
        with zf.open(name) as src, open(IMAGES_DIR / image_name, "wb") as dst:
            dst.write(src.read())
        riddles.append({
            "image": f"images/{image_name}",
            "question": question,
            "answer": answer,
            "hint": hint,
        })

with open(RIDDLES_JS, "w", encoding="utf-8") as f:
    f.write("const RIDDLES = ")
    json.dump(riddles, f, ensure_ascii=False, indent=2)
    f.write(";\n")

print(f"Wrote {len(riddles)} riddles to {RIDDLES_JS}")
