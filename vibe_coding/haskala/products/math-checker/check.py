#!/usr/bin/env python3
"""
check.py — בדיקת תרגיל במצב headless (ללא ממשק).

מקבל קובץ תמונה (או PDF), בודק אותו מול רובריקה, ומשאיר קובץ תוצאה בתיקייה
הנוכחית. צורך את אותה ליבה שה-UI משתמש בה (core.process_stream) — מימוש אחד.

    ./venv/bin/python check.py exercise.jpg
    ./venv/bin/python check.py exercise.jpg --format all
    ./venv/bin/python check.py scan.pdf --rubric rubric.txt --model sonnet

שלב ראשון: הרובריקה היא ברירת המחדל (ריקה → המודל מנקד לפי הניקוד המודפס בדף,
או 10 כברירת מחדל). בעתיד תיכנס רובריקה כקלט (--rubric) וקלטים נוספים.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import core
import report


def _load_rubric(rubric_arg: str | None) -> str:
    """--rubric מקבל או נתיב לקובץ טקסט או טקסט ישיר. ריק = ברירת המחדל."""
    if not rubric_arg:
        return ""
    if os.path.isfile(rubric_arg):
        with open(rubric_arg, encoding="utf-8") as f:
            return f.read()
    return rubric_arg


def check_file(path: str, rubric: str = "", model_key: str = core.DEFAULT_MODEL,
               auto_orient: bool = True) -> list[dict]:
    """מריץ את ה-pipeline המשותף על קובץ אחד ומחזיר את רשימת ה-pages
    בדיוק במבנה ש-build_result_html/docx/compute_totals מצפים לו."""
    with open(path, "rb") as f:
        file_bytes = f.read()
    ext = os.path.splitext(path)[1].lower()

    _, model_label = core.resolve_model(model_key)
    for ev in core.process_stream(file_bytes, ext, auto_orient,
                                  model_key, model_label, rubric,
                                  keep_imgs=False):
        if ev["type"] == "progress":
            print(f"  [{ev['pct']:3d}%] עמוד {ev['page']}/{ev['total_pages']} — "
                  f"{ev['label']}", file=sys.stderr)
        elif ev["type"] == "result":
            return ev["pages"]
    raise RuntimeError("process_stream ended without a result event")


def main() -> int:
    ap = argparse.ArgumentParser(description="בדיקת תרגיל headless")
    ap.add_argument("image", help="קובץ תמונה או PDF לבדיקה")
    ap.add_argument("--rubric", help="רובריקה: נתיב לקובץ טקסט או טקסט ישיר "
                                     "(ברירת מחדל: ללא)")
    ap.add_argument("--model", default=core.DEFAULT_MODEL,
                    choices=sorted(core.MODELS.keys()),
                    help=f"מודל בדיקה (ברירת מחדל: {core.DEFAULT_MODEL})")
    ap.add_argument("--no-orient", action="store_true",
                    help="דלג על תיקון אוריינטציה אוטומטי")
    ap.add_argument("--format", default="html",
                    choices=["html", "docx", "json", "all"],
                    help="פורמט קובץ התוצאה (ברירת מחדל: html)")
    ap.add_argument("--out", default=".",
                    help="תיקיית יעד לקובץ התוצאה (ברירת מחדל: התיקייה הנוכחית)")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        print(f"קובץ לא נמצא: {args.image}", file=sys.stderr)
        return 1

    rubric = _load_rubric(args.rubric)
    try:
        pages = check_file(args.image, rubric, args.model, not args.no_orient)
    except Exception as e:
        err = core.humanize_error(e)
        print(f"שגיאה: {err['message']} — {err['details']}", file=sys.stderr)
        return 2

    filename = os.path.basename(args.image)
    base = os.path.splitext(filename)[0]
    today = datetime.date.today().strftime("%Y%m%d")
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"{base}_math_{today}")

    earned, total = report.compute_totals(pages)
    print(f"\nניקוד כולל: {report._fmt_pts(earned)} / {report._fmt_pts(total)}",
          file=sys.stderr)

    written: list[str] = []
    fmts = ["html", "docx", "json"] if args.format == "all" else [args.format]
    for fmt in fmts:
        if fmt == "html":
            out = f"{stem}.html"
            with open(out, "w", encoding="utf-8") as f:
                f.write(report.build_result_html(pages, filename))
        elif fmt == "docx":
            out = f"{stem}.docx"
            with open(out, "wb") as f:
                f.write(report.build_result_docx(pages, filename))
        else:  # json
            out = f"{stem}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"pages": pages, "filename": filename}, f,
                          ensure_ascii=False, indent=2)
        written.append(out)

    for out in written:
        print(f"נכתב: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
