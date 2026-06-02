"""
experiment.py — שלב 2: ניסוי על תיוגי מילים.

עבור כל k: בוחר k מילים אקראיות כרמזים, ולשאר (N-k) מבקש מ-Claude לתמלל
את חיתוך המילה (עם הרמזים כדוגמאות). מודד CER לכל מילה ומראה ממוצע וחציון.
"""
import argparse
import json
import random
import statistics
from pathlib import Path

from dotenv import load_dotenv

from remez_lib import (
    MAX_TOKENS,
    MODEL,
    cer,
    client,
    crop_word,
    image_block,
    load_image_normalized,
    normalize_for_claude,
)

SYSTEM = """תקבל חיתוך תמונה של מילה אחת בכתב יד עברי. תמלל אותה.

לפני המילה לתמלול, אם יש לך דוגמאות של מילים אחרות מאותו תלמיד עם התמלול הנכון — השתמש בהן כדי לכייל את ההבנה שלך לכתב היד שלו.

החזר רק את המילה עצמה, ללא הסברים, ללא מירכאות, ללא JSON, ללא prefix.
שמור על שגיאות כתיב כפי שכתוב במקור.
אם אינך מצליח לקרוא — החזר [לא קריא].
"""


def transcribe_word(cli, word_img, hint_pairs):
    content = []
    for img, text in hint_pairs:
        content.append(image_block(img))
        content.append({"type": "text", "text": f"המילה הזו: {text}"})
    content.append({"type": "text", "text": "תמלל את המילה הבאה:"})
    content.append(image_block(word_img))
    resp = cli.messages.create(
        model=MODEL,
        max_tokens=50,
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text.strip()


def run(annotation_path: Path, k_values: list[int], seed: int, out_dir: Path) -> None:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    img = normalize_for_claude(load_image_normalized(data["image_path"]))
    words = data["words"]
    n = len(words)
    print(f"image: {Path(data['image_path']).name}", flush=True)
    print(f"labeled words: {n}", flush=True)

    word_crops = [crop_word(img, w) for w in words]

    cli = client()
    rng = random.Random(seed)
    results = {}

    for k in k_values:
        if k >= n:
            print(f"k={k} skipped (need k<n, n={n})", flush=True)
            continue
        hint_idx = sorted(rng.sample(range(n), k)) if k > 0 else []
        hint_pairs = [(word_crops[i], words[i]["text"]) for i in hint_idx]
        non_hint = [i for i in range(n) if i not in hint_idx]
        print(f"\n--- k={k}  hints={hint_idx} ---", flush=True)

        per_word = []
        for i in non_hint:
            pred = transcribe_word(cli, word_crops[i], hint_pairs)
            score = cer(words[i]["text"], pred)
            per_word.append({"i": i, "gt": words[i]["text"], "pred": pred, "cer": score})
            print(f"  word {i:2d}  cer={score:.3f}  gt={words[i]['text']!r}  pred={pred!r}", flush=True)

        avg = statistics.mean(p["cer"] for p in per_word) if per_word else 0.0
        median = statistics.median(p["cer"] for p in per_word) if per_word else 0.0
        print(f"  avg CER={avg:.3f}  median CER={median:.3f}", flush=True)

        results[k] = {
            "hint_indices": hint_idx,
            "per_word": per_word,
            "avg_cer": avg,
            "median_cer": median,
        }

    print("\n=== summary ===", flush=True)
    print(f"{'k':>4} {'avg CER':>10} {'median CER':>12}  delta vs baseline", flush=True)
    base = results.get(0, {}).get("avg_cer")
    for k in k_values:
        if k not in results:
            continue
        r = results[k]
        delta = "—" if base is None or k == 0 else f"{(r['avg_cer'] - base):+.3f}"
        print(f"{k:>4} {r['avg_cer']:>10.3f} {r['median_cer']:>12.3f}  {delta}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{annotation_path.stem}.seed{seed}.json"
    out_file.write_text(
        json.dumps(
            {
                "annotation": str(annotation_path),
                "seed": seed,
                "k_values": k_values,
                "n_words": n,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved: {out_file}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=Path, help="path to .words.json")
    parser.add_argument("--k", default="0,3,5,10")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv()
    k_values = [int(x) for x in args.k.split(",")]
    out_dir = Path(__file__).parent / "results"
    run(args.annotation, k_values, args.seed, out_dir)


if __name__ == "__main__":
    main()
