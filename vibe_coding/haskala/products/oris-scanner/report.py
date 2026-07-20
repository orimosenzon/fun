"""report.py — בוני הדוחות של oris-scanner (HTML עצמאי + Word).

מבוסס על report.py של math-checker, מותאם לשדות בדיקת-שפה (אין LaTeX/KaTeX —
כל הטקסט הוא עברית/שפה רגילה, זורם RTL רגיל, לא LTR כפוי כמו נוסחאות).
"""
from __future__ import annotations

import base64
import io
import logging

log = logging.getLogger("oris-scanner")

VERDICT_HE = {"correct": "נכון ✓", "partial": "חלקי", "incorrect": "שגוי ✗", "unclear": "לא ברור"}
VERDICT_COLOR = {"correct": "1e7e34", "partial": "a0740a", "incorrect": "b54343", "unclear": "5a6b82"}


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_pts(v) -> str:
    n = _num(v)
    if n is None:
        return "—"
    return str(int(n)) if n == int(n) else str(n)


def compute_totals(pages: list[dict]) -> tuple[float, float]:
    earned = total = 0.0
    for p in pages or []:
        for ex in ((p.get("analysis") or {}).get("exercises") or []):
            mx = _num(ex.get("points_max"))
            if mx is None:
                continue
            total += mx
            earned += _num(ex.get("points_earned")) or 0
    return earned, total


def build_result_html(pages: list[dict], filename: str, approved: bool = False) -> str:
    import html as _html

    def esc(s) -> str:
        return _html.escape(str(s if s is not None else ""))

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
        exercises = a.get("exercises") or []
        if not exercises:
            body.append('<div class="noprob">לא זוהו תרגילים בעמוד זה.</div>')
        for ex in exercises:
            v = ex.get("verdict", "unclear")
            body.append('<div class="prob"><div class="prob-head">'
                        f'<span class="prob-id">תרגיל {esc(ex.get("id", "?"))}</span>'
                        f'<span class="badge b-{esc(v)}">{esc(VERDICT_HE.get(v, v))}</span></div>')
            if ex.get("exercise_prompt"):
                body.append(f'<div class="step statement">{esc(ex["exercise_prompt"])}</div>')
            if ex.get("transcribed_text"):
                body.append(f'<div class="step">{esc(ex["transcribed_text"])}</div>')
            for err in ex.get("errors") or []:
                body.append(
                    '<div class="step bad">'
                    f'<b>{esc(err.get("type", ""))}:</b> "{esc(err.get("quote", ""))}" → '
                    f'"{esc(err.get("correction", ""))}"'
                    + (f'<div class="step-comment">⚠ {esc(err["comment"])}</div>'
                       if err.get("comment") else "")
                    + '</div>'
                )
            if _num(ex.get("points_max")) is not None:
                body.append(
                    '<div class="score-row"><span class="pts">ניקוד: '
                    f'{_fmt_pts(ex.get("points_earned"))} / {_fmt_pts(ex.get("points_max"))}'
                    '</span>'
                    + (f'<span class="score-note">{esc(ex["score_suggestion"])}</span>'
                       if ex.get("score_suggestion") else "")
                    + '</div>')
            elif ex.get("score_suggestion"):
                body.append(f'<div class="score-row">{esc(ex["score_suggestion"])}</div>')
            if ex.get("feedback"):
                body.append(f'<div class="feedback">{esc(ex["feedback"])}</div>')
            body.append('</div>')
        body.append('</div></div></section>')

    head = """<!DOCTYPE html>
<html dir="rtl" lang="he"><head><meta charset="utf-8">
<title>בדיקת שפה — __TITLE__</title>
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
  .noprob{color:#6e6a5c;font-style:italic}
  .prob{border:1px solid #e7e0cd;border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.9rem}
  .prob-head{display:flex;gap:0.6rem;align-items:center;margin-bottom:0.5rem;flex-wrap:wrap}
  .prob-id{font-weight:700}
  .badge{font-size:0.82rem;padding:0.18rem 0.7rem;border-radius:999px;font-weight:700}
  .b-correct{background:rgba(30,126,52,.15);color:#1e7e34}
  .b-partial{background:rgba(160,116,10,.15);color:#a0740a}
  .b-incorrect{background:rgba(181,67,67,.15);color:#b54343}
  .b-unclear{background:rgba(90,107,130,.15);color:#5a6b82}
  .step{padding:0.5rem 0.8rem;margin:0.3rem 0;border-radius:6px;border-inline-start:3px solid #d8cfb6;background:#f6f3eb}
  .step.bad{border-inline-start-color:#b54343;background:#fff0f0}
  .step.statement{border-inline-start-color:#2e7286;background:#eef4f7;font-weight:600}
  .step-comment{color:#b54343;font-size:0.85rem;margin-top:0.3rem;font-weight:500}
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
<div class="doc-title">בדיקת שפה — השכלה</div>
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

    doc = Document()
    rtl(doc.add_heading("בדיקת שפה — השכלה", level=1))
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

        exercises = a.get("exercises") or []
        if not exercises:
            ip = rtl(doc.add_paragraph())
            ip.add_run("לא זוהו תרגילים בעמוד זה.").italic = True

        for ex in exercises:
            v = ex.get("verdict", "unclear")
            h = rtl(doc.add_heading(level=3))
            hr = h.add_run(f"תרגיל {ex.get('id', '?')}  ·  {VERDICT_HE.get(v, v)}")
            hr.font.color.rgb = RGBColor.from_string(VERDICT_COLOR.get(v, "5a6b82"))

            if ex.get("exercise_prompt"):
                sp = rtl(doc.add_paragraph())
                sp.add_run("שאלה: ").bold = True
                sp.add_run(ex["exercise_prompt"])

            if ex.get("transcribed_text"):
                tp = rtl(doc.add_paragraph())
                tp.add_run("תשובת התלמיד: ").bold = True
                tp.add_run(ex["transcribed_text"])

            for err in ex.get("errors") or []:
                ep = rtl(doc.add_paragraph())
                mk = ep.add_run(f"✗ {err.get('type', '')}: ")
                mk.bold = True
                mk.font.color.rgb = RGBColor(0xB5, 0x43, 0x43)
                ep.add_run(f"\"{err.get('quote', '')}\" ← \"{err.get('correction', '')}\"")
                if err.get("comment"):
                    cp = rtl(doc.add_paragraph())
                    cr = cp.add_run(f"⚠ {err['comment']}")
                    cr.italic = True
                    cr.font.size = Pt(9.5)
                    cr.font.color.rgb = RGBColor(0xB5, 0x43, 0x43)

            if _num(ex.get("points_max")) is not None:
                scp = rtl(doc.add_paragraph())
                scp.add_run(
                    f"ניקוד: {_fmt_pts(ex.get('points_earned'))} / "
                    f"{_fmt_pts(ex.get('points_max'))}").bold = True
                if ex.get("score_suggestion"):
                    note = scp.add_run(f"   ({ex['score_suggestion']})")
                    note.italic = True
                    note.font.size = Pt(9.5)
                    note.font.color.rgb = RGBColor(0x6E, 0x6A, 0x5C)
            elif ex.get("score_suggestion"):
                rtl(doc.add_paragraph()).add_run(ex["score_suggestion"]).bold = True
            if ex.get("feedback"):
                fb = rtl(doc.add_paragraph())
                fb.add_run(ex["feedback"]).italic = True

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
