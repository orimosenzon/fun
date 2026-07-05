#!/usr/bin/env python3
"""
poller.py — מעבר תקופתי על הגשות Classroom ובדיקה אוטומטית (מוצר math-checker).

הזרימה (עקרון-העל: המורה בלולאה — שום דבר לא נחשף לתלמיד אוטומטית):
  1. קריאת poller_config.json — אילו קורסים/מטלות בודקים (allowlist מפורש).
  2. לכל מטלה: משיכת הגשות במצב TURNED_IN שטרם נבדקו (לפי ה-ledger).
  3. הורדת הקבצים המצורפים מ-Drive לזיכרון → ליבת הבדיקה (core.process_stream).
  4. כתיבת דוח HTML ל-graded/<קורס>/<מטלה>/ ועדכון ה-ledger.
  5. אופציונלי (--push-grade): כתיבת ציון-טיוטה (draftGrade) — נראה למורה בלבד.

ה-ledger (poller_ledger.json) הופך את המעבר ל-idempotent: הגשה מסומנת לפי
submissionId+updateTime, כך שהגשה-מחדש של תלמיד נבדקת שוב אבל ריצה כפולה לא.

הרצה:
    ./venv/bin/python poller.py                 # מעבר יחיד (מתאים ל-cron)
    ./venv/bin/python poller.py --loop 300      # לולאה כל 5 דקות
    ./venv/bin/python poller.py --push-grade    # + כתיבת draftGrade (scopes רחבים)
    ./venv/bin/python poller.py --dry-run       # רק להראות מה היה נבדק
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
import time

from googleapiclient.discovery import build

import core
import gclass
import report

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(HERE, "poller_config.json")
LEDGER_FILE = os.path.join(HERE, "poller_ledger.json")

log = logging.getLogger("math-checker")

# סיומות שהליבה יודעת לעבד, לפי ה-mimeType של הקובץ ב-Drive
_MIME_EXT = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


# ─── pure helpers (מכוסים בטסטים) ────────────────────────────────────────────

def ext_for(mime_type: str | None, title: str | None) -> str | None:
    """סיומת לעיבוד לפי mimeType, עם fallback לסיומת שבשם הקובץ. None = לא נתמך."""
    if mime_type in _MIME_EXT:
        return _MIME_EXT[mime_type]
    ext = os.path.splitext(title or "")[1].lower()
    if ext == ".jpeg":
        ext = ".jpg"
    return ext if ext in (".pdf", ".jpg", ".png") else None


def merge_pages(page_lists: list[list[dict]]) -> list[dict]:
    """איחוד עמודים מכמה קבצים מצורפים לדוח אחד, עם מספור עמודים רציף."""
    merged: list[dict] = []
    for pages in page_lists:
        for p in pages:
            p = dict(p)
            p["page"] = len(merged) + 1
            merged.append(p)
    return merged


def scale_grade(earned: float, total: float, max_points: float | None) -> float | None:
    """המרת הניקוד של הרובריקה לסולם המטלה ב-Classroom (maxPoints).
    אם למטלה אין ניקוד או שהרובריקה ריקה — אין ציון לדחוף."""
    if not total or max_points is None:
        return None
    return round(earned / total * max_points, 1)


def safe_name(s: str) -> str:
    """שם קובץ בטוח: בלי תווים בעייתיים, בלי רווחים כפולים."""
    return re.sub(r"[^\w֐-׿.-]+", "_", s).strip("_") or "unnamed"


# ─── ledger ───────────────────────────────────────────────────────────────────

def load_ledger() -> dict:
    if os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ledger(ledger: dict) -> None:
    tmp = LEDGER_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)
    os.replace(tmp, LEDGER_FILE)  # כתיבה אטומית — ledger לא נהרס באמצע


# ─── config ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        sys.exit(
            f"חסר {CONFIG_FILE}. העתק את poller_config.example.json "
            "ל-poller_config.json וערוך את רשימת הקורסים."
        )
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


def _watched_works(works: list[dict], allow) -> list[dict]:
    """allow הוא "all" או רשימת כותרות/מזהים של מטלות."""
    if allow == "all":
        return works
    allow_set = {str(a).strip().lower() for a in allow}
    return [w for w in works
            if w["id"] in allow_set or w["title"].strip().lower() in allow_set]


# ─── main pass ────────────────────────────────────────────────────────────────

def check_submission(drive, sub: dict, model_key: str, rubric: str) -> list[dict] | None:
    """הורדת הקבצים המצורפים של הגשה ובדיקה. מחזיר pages או None אם אין מה לבדוק."""
    atts = sub.get("assignmentSubmission", {}).get("attachments", [])
    page_lists: list[list[dict]] = []
    for a in atts:
        df = a.get("driveFile")
        if not df:
            log.info("    מצורף לא-Drive (%s) — מדלג", list(a.keys()))
            continue
        meta = drive.files().get(fileId=df["id"], fields="mimeType,name").execute()
        ext = ext_for(meta.get("mimeType"), meta.get("name") or df.get("title"))
        if not ext:
            log.info("    %s: סוג לא נתמך (%s) — מדלג", df.get("title"), meta.get("mimeType"))
            continue
        data = gclass.download_drive_bytes(drive, df["id"])
        log.info("    %s (%.0fKB, %s)", df.get("title"), len(data) / 1024, ext)
        _, model_label = core.resolve_model(model_key)
        for ev in core.process_stream(data, ext, auto_orient=True,
                                      model_key=model_key, model_label=model_label,
                                      rubric=rubric, keep_imgs=False):
            if ev["type"] == "result":
                page_lists.append(ev["pages"])
    if not page_lists:
        return None
    return merge_pages(page_lists)


def push_draft_grade(classroom, course_id: str, work: dict, sub: dict,
                     earned: float, total: float) -> float | None:
    """כתיבת ציון-טיוטה. draftGrade נראה למורה בלבד עד שהוא מחזיר את ההגשה —
    בכוונה לא נוגעים ב-state של ההגשה ולא מחזירים לתלמיד."""
    grade = scale_grade(earned, total, work.get("maxPoints"))
    if grade is None:
        return None
    classroom.courses().courseWork().studentSubmissions().patch(
        courseId=course_id, courseWorkId=work["id"], id=sub["id"],
        updateMask="draftGrade", body={"draftGrade": grade},
    ).execute()
    return grade


def run_pass(cfg: dict, classroom, drive, push_grade: bool, dry_run: bool) -> int:
    """מעבר יחיד על כל הקורסים שבתצורה. מחזיר כמה הגשות נבדקו."""
    ledger = load_ledger()
    out_root = os.path.join(HERE, cfg.get("out_dir", "graded"))
    model_key = cfg.get("model", core.DEFAULT_MODEL)
    checked = 0

    for course_cfg in cfg.get("courses", []):
        course = gclass.find_course(classroom, course_cfg["name"])
        cid = course["id"]
        names = gclass.roster_names(classroom, cid)
        works = classroom.courses().courseWork().list(courseId=cid).execute() \
            .get("courseWork", [])
        rubric = ""
        if course_cfg.get("rubric_file"):
            with open(os.path.join(HERE, course_cfg["rubric_file"]), encoding="utf-8") as f:
                rubric = f.read()

        for w in _watched_works(works, course_cfg.get("coursework", "all")):
            subs = classroom.courses().courseWork().studentSubmissions().list(
                courseId=cid, courseWorkId=w["id"]
            ).execute().get("studentSubmissions", [])

            for sub in subs:
                if sub.get("state") != "TURNED_IN":
                    continue
                key = sub["id"]
                prev = ledger.get(key)
                if prev and prev.get("updateTime") == sub.get("updateTime"):
                    continue  # כבר נבדקה הגרסה הזו

                student = names.get(sub.get("userId", ""), sub.get("userId", "?"))
                tag = f"{course['name']} / {w['title']} / {student}"
                if dry_run:
                    print(f"[dry-run] היה נבדק: {tag}")
                    continue

                log.info("[poll] בודק: %s", tag)
                try:
                    pages = check_submission(drive, sub, model_key, rubric)
                except Exception:
                    log.exception("[poll] בדיקה נכשלה: %s — ממשיך להגשה הבאה", tag)
                    continue
                if pages is None:
                    ledger[key] = {"updateTime": sub.get("updateTime"),
                                   "checked_at": datetime.datetime.now().isoformat(),
                                   "status": "no_supported_attachments"}
                    save_ledger(ledger)
                    continue

                earned, total = report.compute_totals(pages)
                fname = f"{student}_{w['title']}"
                html = report.build_result_html(pages, fname)
                out_dir = os.path.join(out_root, safe_name(course["name"]),
                                       safe_name(w["title"]))
                os.makedirs(out_dir, exist_ok=True)
                today = datetime.date.today().strftime("%Y%m%d")
                out_path = os.path.join(out_dir, f"{safe_name(student)}_{today}.html")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(html)

                entry = {"updateTime": sub.get("updateTime"),
                         "checked_at": datetime.datetime.now().isoformat(),
                         "status": "checked", "report": out_path,
                         "earned": earned, "total": total}
                if push_grade:
                    try:
                        grade = push_draft_grade(classroom, cid, w, sub, earned, total)
                        entry["draft_grade"] = grade
                        log.info("[poll] draftGrade=%s → %s", grade, student)
                    except Exception:
                        log.exception("[poll] כתיבת draftGrade נכשלה: %s", tag)

                ledger[key] = entry
                save_ledger(ledger)
                checked += 1
                log.info("[poll] ✓ %s — ניקוד %s/%s → %s",
                         tag, report._fmt_pts(earned), report._fmt_pts(total), out_path)

    return checked


def main() -> int:
    ap = argparse.ArgumentParser(description="בדיקה אוטומטית של הגשות Classroom")
    ap.add_argument("--loop", type=int, metavar="SECONDS",
                    help="לרוץ בלולאה כל N שניות (ללא = מעבר יחיד)")
    ap.add_argument("--push-grade", action="store_true",
                    help="לכתוב ציון-טיוטה (draftGrade) להגשות שנבדקו")
    ap.add_argument("--dry-run", action="store_true",
                    help="רק להדפיס אילו הגשות היו נבדקות")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    scopes = gclass.SCOPES_RW if args.push_grade else gclass.SCOPES_RO
    creds = gclass.get_credentials(scopes)
    classroom = build("classroom", "v1", credentials=creds)
    drive = build("drive", "v3", credentials=creds)

    while True:
        n = run_pass(cfg, classroom, drive, args.push_grade, args.dry_run)
        print(f"מעבר הושלם — נבדקו {n} הגשות.")
        if args.loop is None:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
