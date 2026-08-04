"""math_report.py — בוני הדוחות של בדיקת המתמטיקה (HTML עצמאי + Word).

מקבל את רשימת ה-pages ש-`math_core.process_stream`/`check_file` מחזירים ומייצר
ממנה דוח. אין כאן קריאות AI ואין Flask, ואין import מ-math_core — כך ש-CLI או
poller יכולים לייצר דוח בלי לגרור את ה-pipeline, ואין מעגל import.

שני צרכנים:
  • products/math-checker  — ה-UI מציג HTML, ומייצא Word
  • products/math-form-checker — כותב Word לתיקיית ה-Drive של המורה

`approved` הוא ההבדל שחשוב ל-form-checker: הזרימה מהטופס היא חד-פעמית ואין בה
מסך שבו המורה מאשר ניקוד, ולכן היא תמיד מייצרת דוח עם approved=False — הכותרת
אומרת במפורש "טיוטה — טרם אושר ע\"י המורה", והניקוד מוצג כהצעה.
"""
from __future__ import annotations

import base64
import io
import logging

log = logging.getLogger("haskala.math")

# ─── export builders (HTML + Word) ──────────────────────────────────────────

VERDICT_HE = {"correct": "נכון ✓", "partial": "חלקי", "incorrect": "שגוי ✗", "unclear": "לא ברור"}
VERDICT_COLOR = {"correct": "1e7e34", "partial": "a0740a", "incorrect": "b54343", "unclear": "5a6b82"}


def _num(v):
    """Coerce a points value to a number, or None if missing/blank/non-numeric."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_pts(v) -> str:
    """Render a points number without a trailing .0 (7.0 → "7", 7.5 → "7.5")."""
    n = _num(v)
    if n is None:
        return "—"
    return str(int(n)) if n == int(n) else str(n)


def compute_totals(pages: list[dict]) -> tuple[float, float]:
    """Sum (earned, max) points across every problem on every page."""
    earned = total = 0.0
    for p in pages or []:
        for pr in ((p.get("analysis") or {}).get("problems") or []):
            mx = _num(pr.get("points_max"))
            if mx is None:
                continue
            total += mx
            earned += _num(pr.get("points_earned")) or 0
    return earned, total


def build_result_html(pages: list[dict], filename: str, approved: bool = False) -> str:
    """Standalone, self-contained HTML report. KaTeX is pulled from a CDN and
    auto-renders \\(...\\) so the LaTeX shows as real math; the scan images are
    embedded as base64 so the file opens anywhere with no server. Mirrors the
    web UI layout (verdict badges, ok/bad step coloring, comments, feedback)."""
    import html as _html

    def esc(s) -> str:
        return _html.escape(str(s if s is not None else ""))

    def math(latex) -> str:
        # KaTeX auto-render reads textContent, so escaping is safe (and needed
        # for inequalities like a<b). Wrapped in \( \) inline delimiters.
        return r"\(" + esc(latex) + r"\)"

    body: list[str] = []
    for p in pages or []:
        a = p.get("analysis") or {}
        rot = p.get("rotation_applied") or 0
        rot_lbl = f" · סובב {rot}°" if rot else ""
        body.append(f'<section class="page"><h2>עמוד {esc(p.get("page", "?"))}'
                    f'<span class="rot">{esc(rot_lbl)}</span></h2><div class="layout">')
        if p.get("image_b64"):
            body.append(f'<div class="imgwrap"><img src="data:image/jpeg;base64,'
                        f'{p["image_b64"]}" alt="עמוד {esc(p.get("page", ""))}"></div>')
        body.append('<div class="analysis">')
        if a.get("page_summary"):
            body.append(f'<div class="summary">{esc(a["page_summary"])}</div>')
        if a.get("has_diagram"):
            body.append(f'<div class="diagram">▣ סרטוט: {esc(a.get("diagram_description", ""))}</div>')
        problems = a.get("problems") or []
        if not problems:
            body.append('<div class="noprob">לא זוהו תרגילים בעמוד זה.</div>')
        for pr in problems:
            v = pr.get("verdict", "unclear")
            body.append('<div class="prob"><div class="prob-head">'
                        f'<span class="prob-id">תרגיל {esc(pr.get("id", "?"))}</span>'
                        f'<span class="topic">{esc(pr.get("topic", ""))}</span>'
                        f'<span class="badge b-{esc(v)}">{esc(VERDICT_HE.get(v, v))}</span></div>')
            if pr.get("statement_latex"):
                body.append(f'<div class="step statement" dir="ltr">{math(pr["statement_latex"])}</div>')
            for s in pr.get("student_steps") or []:
                cls = "bad" if s.get("ok") is False else ("ok" if s.get("ok") is True else "")
                body.append(f'<div class="step {cls}" dir="ltr">{math(s.get("latex", ""))}')
                if s.get("comment"):
                    body.append(f'<div class="step-comment">⚠ {esc(s["comment"])}</div>')
                body.append('</div>')
            if pr.get("final_answer_latex"):
                body.append(f'<div class="step final" dir="ltr"><b>תשובה:</b> {math(pr["final_answer_latex"])}</div>')
            if _num(pr.get("points_max")) is not None:
                body.append(
                    '<div class="score-row"><span class="pts">ניקוד: '
                    f'{_fmt_pts(pr.get("points_earned"))} / {_fmt_pts(pr.get("points_max"))}'
                    '</span>'
                    + (f'<span class="score-note">{esc(pr["score_suggestion"])}</span>'
                       if pr.get("score_suggestion") else "")
                    + '</div>')
            elif pr.get("score_suggestion"):
                body.append(f'<div class="score-row">{esc(pr["score_suggestion"])}</div>')
            if pr.get("feedback"):
                body.append(f'<div class="feedback">{esc(pr["feedback"])}</div>')
            body.append('</div>')
        body.append('</div></div></section>')

    head = """<!DOCTYPE html>
<html dir="rtl" lang="he"><head><meta charset="utf-8">
<title>בדיקת מתמטיקה — __TITLE__</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[{left:'\\\\(',right:'\\\\)',display:false},{left:'\\\\[',right:'\\\\]',display:true}],throwOnError:false});"></script>
<style>
  body{font-family:'Segoe UI',Arial,sans-serif;color:#2b3442;background:#f4f1ea;margin:0;padding:2rem;line-height:1.5}
  .doc-title{font-size:1.5rem;font-weight:800;color:#2e7286;margin:0 0 0.3rem}
  .doc-meta{color:#6e6a5c;margin:0 0 1.5rem;font-size:0.9rem}
  .page{background:#fff;border:1px solid #d8cfb6;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:1.5rem;box-shadow:0 1px 4px rgba(0,0,0,.06)}
  .page h2{margin:0 0 1rem;color:#2e7286;font-size:1.2rem}
  .rot{color:#6e6a5c;font-size:0.85rem;font-weight:400}
  .layout{display:flex;gap:1.4rem;align-items:flex-start;flex-wrap:wrap}
  .imgwrap{flex:0 0 320px;max-width:340px}
  .imgwrap img{width:100%;border:1px solid #d8cfb6;border-radius:8px}
  .analysis{flex:1;min-width:300px}
  .summary{font-size:0.95rem;color:#3a4250;margin-bottom:0.6rem}
  .diagram{font-size:0.85rem;color:#2e7286;background:#eef4f7;border-radius:6px;padding:0.3rem 0.6rem;margin-bottom:0.6rem;display:inline-block}
  .noprob{color:#6e6a5c;font-style:italic}
  .prob{border:1px solid #e7e0cd;border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.9rem}
  .prob-head{display:flex;gap:0.6rem;align-items:center;margin-bottom:0.5rem;flex-wrap:wrap}
  .prob-id{font-weight:700}
  .topic{font-size:0.78rem;color:#6e6a5c;background:#ece8db;border-radius:999px;padding:0.15rem 0.6rem}
  .badge{font-size:0.82rem;padding:0.18rem 0.7rem;border-radius:999px;font-weight:700}
  .b-correct{background:rgba(30,126,52,.15);color:#1e7e34}
  .b-partial{background:rgba(160,116,10,.15);color:#a0740a}
  .b-incorrect{background:rgba(181,67,67,.15);color:#b54343}
  .b-unclear{background:rgba(90,107,130,.15);color:#5a6b82}
  .step{padding:0.5rem 0.8rem;margin:0.3rem 0;border-radius:6px;border-inline-start:3px solid #d8cfb6;background:#f6f3eb;direction:ltr;text-align:left}
  .step.ok{border-inline-start-color:#1e7e34}
  .step.bad{border-inline-start-color:#b54343;background:#fff0f0}
  .step.statement{border-inline-start-color:#2e7286;background:#eef4f7}
  .step.final{border-inline-start-color:#b3924a;background:#faf5e9;font-weight:600}
  .step-comment{color:#b54343;font-size:0.85rem;margin-top:0.3rem;direction:rtl;text-align:right;font-weight:500}
  .score-row{color:#3a4250;font-size:0.9rem;margin:0.6rem 0 0.3rem;display:flex;gap:0.6rem;align-items:baseline;flex-wrap:wrap}
  .score-row .pts{font-weight:700;color:#2e7286}
  .score-row .score-note{color:#6e6a5c;font-size:0.85rem}
  .feedback{margin-top:0.5rem;padding-top:0.6rem;border-top:1px dashed #d8cfb6;font-size:0.93rem}
  .total-box{background:#fff;border:1px solid #d8cfb6;border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.5rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.8rem}
  .total-grade{font-size:1.3rem;font-weight:800;color:#2e7286}
  .approve-badge{font-size:0.9rem;font-weight:700;padding:0.3rem 0.9rem;border-radius:999px}
  .approve-yes{background:rgba(30,126,52,.15);color:#1e7e34}
  .approve-no{background:rgba(160,116,10,.15);color:#a0740a}
  @media print{body{background:#fff;padding:0}.page{box-shadow:none;break-inside:avoid}}
</style></head><body>
<div class="doc-title">בדיקת מתמטיקה — השכלה</div>
<div class="doc-meta">קובץ: __TITLE__ · __NPAGES__ עמודים</div>
__TOTAL_BOX__
"""
    npages = len(pages or [])
    earned, total = compute_totals(pages)
    if total > 0:
        badge = ('<span class="approve-badge approve-yes">✓ אושר ע"י המורה</span>'
                 if approved else
                 '<span class="approve-badge approve-no">טיוטה — טרם אושר</span>')
        total_box = (
            '<div class="total-box">'
            f'<span class="total-grade">ציון כולל: {_fmt_pts(earned)} / {_fmt_pts(total)}</span>'
            f'{badge}</div>')
    else:
        total_box = ""
    head = (head.replace("__TITLE__", _html.escape(filename))
                .replace("__NPAGES__", str(npages))
                .replace("__TOTAL_BOX__", total_box))
    return head + "\n".join(body) + "\n</body></html>"


def build_result_docx(pages: list[dict], filename: str, approved: bool = False) -> bytes:
    """Word (.docx) report. python-docx can't typeset LaTeX, so math is kept as
    monospace LTR text (the strings are simple, e.g. x^2-6x+10) — honest and
    editable. Each page carries its scan image, verdicts, comments, score and
    feedback. Hebrew paragraphs are right-aligned + bidi for correct RTL flow."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.shared import Inches, Pt, RGBColor

    def rtl(par):
        par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        pPr = par._p.get_or_add_pPr()
        bidi = OxmlElement("w:bidi")
        pPr.append(bidi)
        return par

    def mono(run):
        run.font.name = "Consolas"
        run.font.size = Pt(10.5)
        return run

    doc = Document()
    rtl(doc.add_heading("בדיקת מתמטיקה — השכלה", level=1))
    meta = rtl(doc.add_paragraph())
    meta.add_run(f"קובץ: {filename}").bold = True

    earned, total = compute_totals(pages)
    if total > 0:
        tp = rtl(doc.add_paragraph())
        tr = tp.add_run(f"ציון כולל: {_fmt_pts(earned)} / {_fmt_pts(total)}")
        tr.bold = True
        tr.font.size = Pt(14)
        tr.font.color.rgb = RGBColor(0x2E, 0x72, 0x86)
        sp = rtl(doc.add_paragraph())
        sr = sp.add_run('✓ אושר ע"י המורה' if approved else "טיוטה — טרם אושר ע\"י המורה")
        sr.bold = True
        sr.font.color.rgb = RGBColor(0x1E, 0x7E, 0x34) if approved else RGBColor(0xA0, 0x74, 0x0A)

    for idx, p in enumerate(pages or []):
        if idx > 0:
            doc.add_page_break()
        a = p.get("analysis") or {}
        rot = p.get("rotation_applied") or 0
        rot_lbl = f"  (סובב {rot}°)" if rot else ""
        rtl(doc.add_heading(f"עמוד {p.get('page', idx + 1)}{rot_lbl}", level=2))

        if p.get("image_b64"):
            try:
                pic = doc.add_paragraph()
                pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
                pic.add_run().add_picture(
                    io.BytesIO(base64.b64decode(p["image_b64"])), width=Inches(3.2))
            except Exception:
                log.warning("docx: failed to embed image for page %s", p.get("page"))

        if a.get("page_summary"):
            rtl(doc.add_paragraph(a["page_summary"]))
        if a.get("has_diagram"):
            d = rtl(doc.add_paragraph())
            r = d.add_run(f"▣ סרטוט: {a.get('diagram_description', '')}")
            r.italic = True
            r.font.color.rgb = RGBColor(0x2E, 0x72, 0x86)

        problems = a.get("problems") or []
        if not problems:
            ip = rtl(doc.add_paragraph())
            ip.add_run("לא זוהו תרגילים בעמוד זה.").italic = True

        for pr in problems:
            v = pr.get("verdict", "unclear")
            h = rtl(doc.add_heading(level=3))
            hr = h.add_run(f"תרגיל {pr.get('id', '?')}  ·  {pr.get('topic', '')}  ·  "
                           f"{VERDICT_HE.get(v, v)}")
            hr.font.color.rgb = RGBColor.from_string(VERDICT_COLOR.get(v, "5a6b82"))

            if pr.get("statement_latex"):
                sp = doc.add_paragraph()
                sp.alignment = WD_ALIGN_PARAGRAPH.LEFT
                sp.add_run("נתון: ").bold = True
                mono(sp.add_run(pr["statement_latex"]))

            for s in pr.get("student_steps") or []:
                mark = "✗ " if s.get("ok") is False else ("✓ " if s.get("ok") is True else "• ")
                stp = doc.add_paragraph()
                stp.alignment = WD_ALIGN_PARAGRAPH.LEFT
                mk = stp.add_run(mark)
                mk.bold = True
                mk.font.color.rgb = RGBColor(0xB5, 0x43, 0x43) if s.get("ok") is False \
                    else (RGBColor(0x1E, 0x7E, 0x34) if s.get("ok") is True else RGBColor(0x5A, 0x6B, 0x82))
                mono(stp.add_run(s.get("latex", "")))
                if s.get("comment"):
                    cp = rtl(doc.add_paragraph())
                    cr = cp.add_run(f"⚠ {s['comment']}")
                    cr.italic = True
                    cr.font.size = Pt(9.5)
                    cr.font.color.rgb = RGBColor(0xB5, 0x43, 0x43)

            if pr.get("final_answer_latex"):
                fp = doc.add_paragraph()
                fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
                fp.add_run("תשובה: ").bold = True
                mono(fp.add_run(pr["final_answer_latex"]))

            if _num(pr.get("points_max")) is not None:
                scp = rtl(doc.add_paragraph())
                scp.add_run(
                    f"ניקוד: {_fmt_pts(pr.get('points_earned'))} / "
                    f"{_fmt_pts(pr.get('points_max'))}").bold = True
                if pr.get("score_suggestion"):
                    note = scp.add_run(f"   ({pr['score_suggestion']})")
                    note.italic = True
                    note.font.size = Pt(9.5)
                    note.font.color.rgb = RGBColor(0x6E, 0x6A, 0x5C)
            elif pr.get("score_suggestion"):
                rtl(doc.add_paragraph()).add_run(pr["score_suggestion"]).bold = True
            if pr.get("feedback"):
                fb = rtl(doc.add_paragraph())
                fb.add_run(pr["feedback"]).italic = True

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
