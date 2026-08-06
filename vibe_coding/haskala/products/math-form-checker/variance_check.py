#!/usr/bin/env python3
"""כמה משתנה הציון כשמריצים את אותו קובץ שוב ושוב?

הרקע: עד 5/8/2026 שלב זיהוי הסיבוב היה שבור (ORIENT_MODEL היה Haiku, 0/5),
ולכן כל מדידת איכות שנעשתה לפני התאריך הזה מדדה בפועל את הגרלת הסיבוב ולא את
הבודק. אחרי התיקון אין *שום* מדידה תקפה, וזה הכלי שמייצר את הראשונה.

מה שנמדד כאן הוא רצפת הרעש: הטווח שבתוכו אותו קלט בדיוק מקבל ציונים שונים.
בלי המספר הזה אי אפשר לקרוא שום השוואה — הפרש של 2 נקודות בין שני מודלים לא
אומר כלום אם אותו מודל נע 3 נקודות מול עצמו. זו בדיוק הטעות שהובילה למסקנה
השגויה "סונט 4/10 מול אופוס 7/10".

    python variance_check.py                       # 3 הרצות, sonnet5, קבצי avish
    python variance_check.py --runs 5 --model opus
    python variance_check.py --files a.pdf b.pdf

הפלט: לכל קובץ — הציון בכל הרצה, הטווch, סטיית התקן ופסקי הדין. בסוף, סיכום.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from haskala_grading import math_core  # noqa: E402

DEFAULT_GLOBS = [
    "/home/ori/avish/Amir algebra /*.pdf",
    "/home/ori/avish/Amir geometric /*.pdf",
]


def page_score(pages: list[dict]) -> tuple[float, float, list[str]]:
    """(נקודות שהתקבלו, נקודות מקסימום, פסקי דין) על פני כל התרגילים בקובץ."""
    earned = total = 0.0
    verdicts: list[str] = []
    for page in pages:
        for pr in (page.get("analysis") or {}).get("problems", []):
            earned += float(pr.get("points_earned") or 0)
            total += float(pr.get("points_max") or 0)
            verdicts.append(str(pr.get("verdict") or "?"))
    return earned, total, verdicts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default=math_core.DEFAULT_MODEL)
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--out", default="variance_results.json")
    args = ap.parse_args()

    files = args.files or sorted(
        f for pat in DEFAULT_GLOBS for f in glob.glob(pat))
    if not files:
        print("no input files", file=sys.stderr)
        return 1

    _key, label = math_core.resolve_model(args.model)
    print(f"model: {label}   runs: {args.runs}   files: {len(files)}")
    print(f"≈ {len(files) * args.runs} analysis call(s)\n")

    results: dict[str, list[dict]] = {}
    t_start = time.time()

    for path in files:
        name = os.path.basename(path)
        # WITH the leading dot — core.count_pages compares against ".pdf"
        # literally, and a dotless "pdf" silently takes the image branch and
        # dies in PIL. drive_folder._ext_for uses the same convention.
        ext = os.path.splitext(path)[1].lower()
        blob = open(path, "rb").read()
        results[name] = []
        print(f"── {name}")
        for r in range(1, args.runs + 1):
            t0 = time.time()
            try:
                pages = math_core.check_file(blob, ext, model_key=args.model)
                earned, total, verdicts = page_score(pages)
                results[name].append({"run": r, "earned": earned, "max": total,
                                      "verdicts": verdicts})
                print(f"   run {r}: {earned:g}/{total:g}  {verdicts}  "
                      f"({time.time() - t0:.0f}s)")
            except Exception as e:                      # noqa: BLE001
                # הרצה שנפלה היא נתון בפני עצמו — לא סיבה להפיל את המדידה כולה.
                results[name].append({"run": r, "error": str(e)[:200]})
                print(f"   run {r}: FAILED — {str(e)[:120]}")
        print()

    # המדד הוא **אחוז**, לא נקודות. בלי מחוון המודל ממציא את points_max בכל
    # הרצה מחדש, אז 6/10 ו-9/15 הם אותו פסק דין בדיוק — והפרש בנקודות מוחלטות
    # היה סופר את זה כרעש של 3 נקודות. נמדד 6/8/2026: קובץ אחד קיבל 6/10, 9/15
    # ו-6/10 — 60% בשלוש ההרצות, כלומר יציבות מושלמת שהמדד הישן הסתיר.
    print("=" * 78)
    print(f"{'file':32} {'%':18} {'טווח':>7} {'סטיית תקן':>10} {'תרגילים':>9}")
    print("-" * 78)
    all_ranges = []
    unstable_count = []
    for name, runs in results.items():
        pcts = [100 * r["earned"] / r["max"]
                for r in runs if r.get("max")]
        if not pcts:
            print(f"{name[:32]:32} {'כל ההרצות נכשלו':18}")
            continue
        counts = [len(r["verdicts"]) for r in runs if "verdicts" in r]
        rng = max(pcts) - min(pcts)
        sd = statistics.stdev(pcts) if len(pcts) > 1 else 0.0
        all_ranges.append(rng)
        if len(set(counts)) > 1:
            unstable_count.append((name, counts))
        shown = "/".join(f"{p:.0f}" for p in pcts)
        print(f"{name[:32]:32} {shown:18} {rng:6.1f} {sd:10.2f} "
              f"{'/'.join(map(str, counts)):>9}")

    if all_ranges:
        print("-" * 78)
        print(f"רצפת הרעש — טווח ממוצע: {statistics.mean(all_ranges):.1f} נקודות אחוז")
        print(f"הטווח הגרוע ביותר:      {max(all_ranges):.1f} נקודות אחוז")
        print("\nכל הפרש בין מודלים או בין גרסאות פרומפט שקטן מהטווח הגרוע\n"
              "ביותר אינו אות אלא רעש.")

    if unstable_count:
        # חמור יותר מפיזור בציון: אם מספר התרגילים משתנה, מבנה הדוח עצמו
        # משתנה, והמורה מקבל דוח אחר לגמרי על אותו דף.
        print("\n⚠ מספר התרגילים שזוהו אינו יציב:")
        for name, counts in unstable_count:
            print(f"   {name}: {counts}")

    with open(args.out, "w") as fh:
        json.dump({"model": label, "runs": args.runs, "results": results},
                  fh, ensure_ascii=False, indent=2)
    print(f"\nנשמר: {args.out}   ({time.time() - t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
