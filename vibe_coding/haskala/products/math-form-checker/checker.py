"""checker.py — grading one submission's worth of maths files.

Sits between main.py (which owns Google, the ledger and the guardrails) and
haskala_grading.math_core (which owns the model calls). main.py hands it one
downloaded file at a time; it returns the .docx bytes and a report title.

WHAT THIS DOES DIFFERENTLY FROM form-checker's checker.py
─────────────────────────────────────────────────────────
The English product resolves a *bundled* rubric — four Ministry JSON files,
picked by Bagrut module code — and merges the teacher's task description into a
"question" that the evaluation is measured against. Neither idea transfers:

  • The maths rubric is free text. math_core appends it to the user turn, so
    there is nothing to load and nothing to match. build_context's only job is
    to find a מחוון *document* when the teacher said it was in the folder.
  • There is no "question" to grade against. The problems are printed on the
    scan; the model reads them from the page. Class context and the teacher's
    comments still condition the grade, and form_schema.build_rubric folds
    those into the rubric string.

  • There is one thing the English product has no equivalent of: the school's
    reference solution (added 2026-08-05). build_solution resolves it once per
    submission — attachment, folder file, or typed prose — and the pages travel
    with every student page of that submission.

So this file is much smaller than its English sibling, and that asymmetry is
the point rather than an omission.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import drive_folder
import form_schema
from haskala_grading import core as en_core
from haskala_grading import math_core, math_report

log = logging.getLogger("math-form-checker")


def _preview(text, limit: int) -> str:
    s = (text or "").strip()
    return s if len(s) <= limit else s[:limit] + f"… (+{len(s) - limit} chars)"


@dataclass
class FolderContext:
    """What the folder told us, resolved once per submission rather than per file."""
    rubric_text: str = ""
    rubric_source: str = ""
    rubric_name: str = "מחוון חופשי"
    context_docs: list = field(default_factory=list)   # [(name, kind)] for the summary
    # The school's reference solution, resolved once and reused for every
    # student in the submission. Pages rather than text: a maths solution is
    # diagrams and multi-line derivations, and the text layer of a printed one
    # comes back as reordered fragments with the geometry gone.
    solution_text: str = ""
    solution_imgs: list = field(default_factory=list)
    solution_source: str = "none"


def build_solution(params: dict, solution_files: list[dict] | None = None,
                   tag: str = "") -> tuple[str, list, str]:
    """(solution_text, solution page images, human-readable source).

    Precedence, highest first:
      1. files the teacher attached to the form's school-solution question;
      2. a file in the shared folder whose name says it is the solution;
      3. prose the teacher typed into that question instead of attaching a file;
      4. nothing.

    Attachments outrank typed prose because that is what the question asks for —
    a teacher who does both has attached the real thing and typed a note about
    it. Never raises: a solution that cannot be rendered is a lost hint, not a
    lost submission, so it is logged and grading continues without it, exactly
    as it did before this question existed.
    """
    files = list(solution_files or [])
    if files:
        imgs, names = [], []
        for f in files:
            try:
                pages = math_core.solution_pages(f["data"], f["ext"])
            except Exception as e:
                log.warning("%s could not render solution file %r (%s: %s) — "
                            "ignoring it", tag, f["name"], type(e).__name__,
                            str(e)[:200])
                continue
            if not pages:
                continue
            imgs.extend(pages)
            names.append(f["name"])
        if imgs:
            # The cap is enforced again across files, not only within one:
            # math_core.solution_pages limits a single attachment, and three
            # attachments of six pages would otherwise put eighteen reference
            # pages in front of every student page.
            if len(imgs) > math_core.MAX_SOLUTION_PAGES:
                log.warning("%s reference solution came to %d pages across %d "
                            "file(s) — keeping the first %d", tag, len(imgs),
                            len(names), math_core.MAX_SOLUTION_PAGES)
                imgs = imgs[:math_core.MAX_SOLUTION_PAGES]
            return "", imgs, f"attached to the form ({', '.join(names)})"
        log.warning("%s the teacher attached a school solution but none of it "
                    "could be rendered — grading without it", tag)

    typed = (params.get("solution_text") or "").strip()
    if typed:
        return typed, [], "typed into the form"

    return "", [], "none"


def build_context(params: dict, typed_docs: list[dict],
                  solution_files: list[dict] | None = None,
                  tag: str = "") -> FolderContext:
    """Resolve the מחוון and the school solution for one submission.

    מחוון precedence, highest first:
      1. the scheme the teacher pasted into the form;
      2. a typed document in the folder, when the form answer pointed there;
      3. nothing — math_core then scores by the points printed on the page.

    The form outranks the folder deliberately. A folder document is a guess
    about a file nobody described; the form answer is what the teacher actually
    wrote down. When they disagree, the answer wins.

    Never raises. A folder whose מחוון cannot be read still grades — it just
    grades against the printed points, which is the documented default.
    """
    sol_text, sol_imgs, sol_source = build_solution(params, solution_files, tag)
    log.info("%s reference solution: %s", tag, sol_source)

    def result(**kw) -> FolderContext:
        return FolderContext(solution_text=sol_text, solution_imgs=sol_imgs,
                             solution_source=sol_source, **kw)

    pasted = (params.get("rubric_text") or "").strip()
    if pasted:
        return result(
            rubric_text=pasted,
            rubric_source="the form's מחוון field",
            rubric_name="מחוון מהטופס",
            context_docs=[],
        )

    if not params.get("rubric_in_folder"):
        return result(
            rubric_text="",
            rubric_source="none — scoring by the points printed on the page",
            rubric_name="ללא מחוון",
            context_docs=[],
        )

    read: list[tuple[dict, str]] = []
    for doc in typed_docs:
        try:
            # Typed documents only — main.py has already split scans from text
            # documents, so this hits the PDF text layer and costs nothing. The
            # English default model is passed for the vision fallback that a
            # typed document should never reach; math model keys are not valid
            # for en_core.ocr_page and would raise if it ever did.
            text, method = en_core.rubric_text_from_document(
                doc["data"], doc["ext"], en_core.DEFAULT_MODEL)
        except Exception as e:
            log.warning("%s could not read מחוון document %r (%s: %s) — ignoring it",
                        tag, doc["name"], type(e).__name__, str(e)[:200])
            continue
        if not text.strip():
            continue
        log.info("%s מחוון document %r read via %s (%d chars)",
                 tag, doc["name"], method, len(text))
        read.append((doc, text))

    if not read:
        log.warning("%s teacher said the מחוון is in the folder, but no readable "
                    "typed document was found — scoring by the printed points", tag)
        return result(
            rubric_text="",
            rubric_source="the teacher said the מחוון is in the folder, but none "
                          "could be read — scoring by the points printed on the page",
            rubric_name="ללא מחוון",
            context_docs=[],
        )

    # Concatenated rather than picking one: a folder may hold a marking scheme
    # and a separate page of instructions, and there is no reliable way to rank
    # them without asking the model, which costs a call per file.
    doc_names = [d["name"] for d, _t in read]
    return result(
        rubric_text="\n\n".join(t.strip() for _d, t in read),
        rubric_source=f"document(s) in the folder ({', '.join(doc_names)})",
        rubric_name=doc_names[0],
        context_docs=[(n, "מחוון") for n in doc_names],
    )


def build_rubric(params: dict, ctx: FolderContext) -> str:
    """The full rubric string for math_core: the resolved scheme plus the class
    context. form_schema owns the shape; the folder's text is merged in as
    though the teacher had pasted it, because from the model's point of view
    that is exactly what it is."""
    merged = dict(params)
    merged["rubric_text"] = ctx.rubric_text
    return form_schema.build_rubric(merged)


def check_file(entry: dict, params: dict, ctx: FolderContext, rubric: str,
               tag: str = "") -> tuple[bytes, str]:
    """Grade one student's file. Returns (docx bytes, report title).

    Raises ValueError when the file yields nothing gradable — main.py records
    that against this file alone, so one unreadable scan does not take the rest
    of the class down with it.
    """
    t0 = time.monotonic()

    model_key, model_label = math_core.resolve_model(params.get("model_key"))

    log.info("%s ── checking parameters ──", tag)
    log.info("%s   file       : %r (%s, %.0f KB)", tag, entry["name"], entry["ext"],
             len(entry["data"]) / 1024)
    log.info("%s   teacher    : %s%s", tag, params.get("email") or "(no address)",
             f" ({params['teacher_name']})" if params.get("teacher_name") else "")
    log.info("%s   model      : %s (%s)", tag, model_key, model_label)
    log.info("%s   rubric src : %s", tag, ctx.rubric_source)
    log.info("%s   rubric     : %s", tag, _preview(rubric, 2000))
    log.info("%s   solution   : %s (%d page(s), %d chars)", tag,
             ctx.solution_source, len(ctx.solution_imgs), len(ctx.solution_text))
    log.info("%s ─────────────────────────", tag)

    pages = math_core.check_file(entry["data"], entry["ext"],
                                 rubric=rubric, model_key=model_key,
                                 solution_text=ctx.solution_text,
                                 solution_imgs=ctx.solution_imgs)
    if not pages:
        raise ValueError("no pages could be rendered from this file")

    problems = sum(len((p.get("analysis") or {}).get("problems") or []) for p in pages)
    if not problems:
        # Not an error: a blank or unreadable page is a legitimate result the
        # teacher needs to see. The report says so, rather than the submission
        # vanishing with only a log line to explain it.
        log.warning("%s no problems detected in %r — writing the report anyway",
                    tag, entry["name"])

    title = drive_folder.report_name(entry["name"])
    docx_bytes = math_report.build_result_docx(
        pages, entry["name"],
        # Always False. math-checker's UI has a flow where the teacher edits the
        # suggested points and confirms them; a form submission has no such
        # screen, so every report produced here is by definition unapproved and
        # says so on its first page.
        approved=False,
    )

    earned, total = math_report.compute_totals(pages)
    log.info("%s graded %d page(s), %d problem(s) of %r → %r "
             "(%s/%s pts, %.0f KB) in %.1fs",
             tag, len(pages), problems, entry["name"], title,
             math_report._fmt_pts(earned), math_report._fmt_pts(total),
             len(docx_bytes) / 1024, time.monotonic() - t0)
    return docx_bytes, title
