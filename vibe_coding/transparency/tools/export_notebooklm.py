#!/usr/bin/env python3
"""ייצוא התמלולים לקבצי Markdown להעלאה ל-NotebookLM.

קובץ לכל חצי שנה (NotebookLM מגביל ל-50 מקורות במחברת). כל ישיבה נפתחת בכותרת עם
תאריך וקישור לסרטון, וכל פסקה מתחילה בקישור שפותח את הסרטון ברגע שבו נאמרה, ובשם
הדובר כפי שזוהה ב-tools/name_speakers.py. זיהוי ברמת ביטחון נמוכה מסומן "(משוער)".

    python3 tools/export_notebooklm.py   ->   private/notebooklm/*.md
"""
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.path.join(ROOT, "private")
OUT = os.path.join(PRIV, "notebooklm")
PARA_SECONDS = 90  # פסקה ארוכה מתפצלת, כדי שלכל פסקה יהיה קישור קרוב לרגע


def hms(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def label(spk, names):
    info = names.get(spk) or {}
    name, role = info.get("name"), info.get("role")
    if not name:
        return f"דובר לא מזוהה ({(spk or '?').replace('SPEAKER_', '#')})"
    role = f", {role}" if role and role not in ("לא ידוע", "אחר") else ""
    guess = " (משוער)" if info.get("confidence") == "low" else ""
    return f"{name}{role}{guess}"


def meeting_md(path):
    d = json.load(open(path, encoding="utf-8"))
    m = d["meeting"]
    sp = os.path.join(PRIV, "speakers", os.path.basename(path))
    names = json.load(open(sp, encoding="utf-8"))["speakers"] if os.path.exists(sp) else {}
    y, mo, da = m["date"].split("-")
    url = f"https://youtu.be/{m['id']}"
    lines = [f"# ישיבת מליאת המועצה פרדס חנה-כרכור, {int(da)}/{int(mo)}/{y}", "",
             f"סרטון מלא: {url}", ""]
    paras = []
    for s in d["segments"]:
        if (paras and paras[-1]["speaker"] == s["speaker"]
                and s["start"] - paras[-1]["start"] < PARA_SECONDS):
            paras[-1]["text"] += " " + s["text"]
        else:
            paras.append(dict(s))
    for p in paras:
        t = int(p["start"])
        lines.append(f"[{hms(t)}]({url}?t={t}) **{label(p['speaker'], names)}:** {p['text']}")
        lines.append("")
    return m["date"], "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    halves = defaultdict(list)
    src = sorted(glob.glob(os.path.join(PRIV, "transcripts_spk", "*.json")))
    for path in src:
        date, md = meeting_md(path)
        half = f"{date[:4]}-{'H1' if int(date[5:7]) <= 6 else 'H2'}"
        halves[half].append((date, md))
    for half, items in sorted(halves.items()):
        y, h = half.split("-")
        title = f"ישיבות מליאת המועצה פרדס חנה-כרכור, {'ינואר–יוני' if h == 'H1' else 'יולי–דצמבר'} {y}"
        head = (f"# {title}\n\nתמלול אוטומטי של השידורים ביוטיוב (מודל ivrit.ai), עם זיהוי דוברים "
                f"אוטומטי. ייתכנו שגיאות בתמלול ובשיוך הדוברים; לציטוט מדויק יש לצפות בסרטון "
                f"בקישור שליד כל פסקה.\n\n")
        body = "\n\n---\n\n".join(md for _, md in sorted(items))
        with open(os.path.join(OUT, f"meetings_{half}.md"), "w", encoding="utf-8") as f:
            f.write(head + body + "\n")
    print(f"{len(src)} meetings -> {len(halves)} files in {OUT}")


if __name__ == "__main__":
    main()
