"""checker.py — one file from a shared folder → a graded Word report.

The glue layer. Decides which rubric applies to a folder, what the task was,
and then grades one student's work against both.

Deliberately narrower than oris-scanner's equivalent: there is no Classroom
here, so no structured Classroom rubric and no assignment materials to sift
through. The rubric comes from a dropdown the teacher answered, and the harder
question — what were the students actually asked to do? — is answered by the
typed documents sitting in the folder next to the scans.

WHY THE CONTEXT PASS EXISTS
───────────────────────────
The form has no "describe the task" question, and the grade depends on the task:
"fully on topic" is unscoreable without knowing the topic. But a teacher who
shares a folder of a class's compositions almost always shares the exam paper
with them. Reading the folder's typed documents once, before grading anything,
recovers the instructions the form never asked for — and costs two text-only
model calls per typed document, not per student.
"""
from __future__ import annotations

import logging
import time

import drive_folder
import form_schema
from haskala_grading import core, report

# No basicConfig here — main.py owns logging setup (structured JSON for Cloud
# Run, level from LOG_LEVEL). Configuring it from a module main.py imports would
# install a plain-text handler first and make main's own call a silent no-op.
log = logging.getLogger("form-checker-checker")


def _preview(text, limit):
    """Log-friendly rendering of a possibly long, possibly multi-line string:
    newlines shown as ⏎ so one log entry stays one line."""
    if not text:
        return "(none)"
    flat = " ⏎ ".join(str(text).splitlines())
    if len(flat) > limit:
        return f"{flat[:limit]}… [+{len(flat) - limit} chars]"
    return flat


# ─── folder context: the rubric and the task ────────────────────────────────

class FolderContext:
    """What a folder's typed documents tell us, computed once per submission.

    Held as an object rather than passed as four arguments because every field
    is derived together and every one of them is logged together — a surprising
    grade is diagnosed by reading this whole block at once, not one value at a
    time.
    """

    def __init__(self, rubric_override, rubric_source, rubric_name,
                 instructions, context_docs):
        self.rubric_override = rubric_override
        self.rubric_source = rubric_source
        self.rubric_name = rubric_name
        self.instructions = instructions
        self.context_docs = context_docs      # [(name, kind)] for the summary


def build_context(params: dict, typed_docs: list[dict], tag: str = "") -> FolderContext:
    """Resolve the rubric and the task description for one submission.

    Rubric precedence, highest first:
      1. the teacher said on the form that the rubric is among the files;
      2. the module they picked from the dropdown — an explicit, required answer;
      3. a document in the folder that reads as a rubric;
      4. the bundled default.

    The dropdown outranks the folder on purpose. A folder rubric is a guess about
    a file the teacher never said anything about; the dropdown is an answer they
    were required to give. When they disagree, the answer wins.

    Never raises. A folder whose context cannot be read still grades against the
    module rubric, which is the thing that actually determines the score.
    """
    read: list[tuple[dict, str, str]] = []       # (doc, text, kind)
    for doc in typed_docs:
        try:
            text, method = core.rubric_text_from_document(
                doc["data"], doc["ext"], params["model_key"])
        except Exception as e:
            log.warning("%s could not read context document %r (%s: %s) — ignoring it",
                        tag, doc["name"], type(e).__name__, str(e)[:200])
            continue
        if not text.strip():
            continue
        kind = core.classify_document(text, params["model_key"])
        log.info("%s context document %r read via %s (%d chars) — classified as %s",
                 tag, doc["name"], method, len(text), kind)
        read.append((doc, text, kind))

    folder_rubric = next(((d, t) for d, t, k in read if k == "rubric"), None)

    rubric_override, rubric_source = None, None
    if params.get("rubric_from_upload") and folder_rubric:
        doc, text = folder_rubric
        rubric_override = {"name": doc["name"], "content": text}
        rubric_source = f"a document in the folder ({doc['name']!r}), as the form asked"
    elif params.get("rubric_id"):
        loaded = core.load_rubric(params["rubric_id"])
        if loaded:
            rubric_override = loaded
            rubric_source = f"the form's dropdown ({params['rubric_id']})"
        else:
            log.error("%s rubric id %r resolved to no bundled rubric", tag,
                      params["rubric_id"])
    if rubric_override is None and folder_rubric:
        doc, text = folder_rubric
        rubric_override = {"name": doc["name"], "content": text}
        rubric_source = f"a document in the folder ({doc['name']!r})"
    if rubric_override is None:
        rubric_override = core.load_rubric(core.DEFAULT_RUBRIC_ID)
        rubric_source = (f"bundled default ({core.DEFAULT_RUBRIC_ID}) — no usable "
                         "rubric on the form or in the folder")

    # Everything that is not a rubric describes the task. Concatenated rather
    # than picking one: a folder may hold the exam paper and a separate page of
    # instructions, and there is no reliable way to rank them.
    instructions = "\n\n".join(t.strip() for _d, t, k in read
                               if k != "rubric" and t.strip())

    return FolderContext(
        rubric_override=rubric_override,
        rubric_source=rubric_source,
        rubric_name=(rubric_override or {}).get("name", core.DEFAULT_RUBRIC_ID),
        instructions=instructions,
        context_docs=[(d["name"], k) for d, _t, k in read],
    )


def build_question(params: dict, ctx: FolderContext) -> str:
    """The 'question' passed to core.check_pages.

    form_schema owns the shape of this; the folder's instructions are merged in
    as though the teacher had typed them into the form, because from the model's
    point of view that is exactly what they are."""
    merged = dict(params)
    existing = (params.get("instructions") or "").strip()
    if ctx.instructions:
        merged["instructions"] = "\n\n".join(p for p in (existing, ctx.instructions) if p)
    return form_schema.build_question(merged)


# ─── grading one file ───────────────────────────────────────────────────────

def check_file(entry: dict, params: dict, ctx: FolderContext, question: str,
               tag: str = "") -> tuple[bytes, str]:
    """Grade one student's file. Returns (docx bytes, report title).

    Raises ValueError when the file yields nothing gradable — main.py records
    that against this file alone, so one unreadable scan does not take the rest
    of the class down with it.
    """
    t0 = time.monotonic()

    log.info("%s ── checking parameters ──", tag)
    log.info("%s   file       : %r (%s, %.0f KB)", tag, entry["name"], entry["ext"],
             len(entry["data"]) / 1024)
    log.info("%s   teacher    : %s%s", tag, params.get("email") or "(no address)",
             f" ({params['teacher_name']})" if params.get("teacher_name") else "")
    log.info("%s   module     : %s", tag,
             (params.get("module") or "(not given)").upper())
    log.info("%s   model      : %s (%s)", tag, params["model_key"],
             core.MODELS.get(params["model_key"], "unknown key"))
    log.info("%s   language   : %s", tag, params["exercise_lang"])
    log.info("%s   rubric src : %s", tag, ctx.rubric_source)
    log.info("%s   rubric name: %r", tag, ctx.rubric_name)
    log.info("%s   question   : %s", tag, _preview(question, 1200))
    log.info("%s   rubric text: %s", tag,
             _preview((ctx.rubric_override or {}).get("content"), 2000))
    log.info("%s ─────────────────────────", tag)

    pages, evaluation = core.check_pages(
        [(entry["data"], entry["ext"])],
        rubric_override=ctx.rubric_override,
        question=question,
        model_key=params["model_key"],
    )

    title = drive_folder.report_name(entry["name"])
    docx_bytes = report.build_evaluation_docx(
        evaluation, title, ctx.rubric_name, pages=pages,
        feedback_lang=params["exercise_lang"],
        exercise_lang=params["exercise_lang"],
    )
    log.info("%s graded %d page(s) of %r → %r (%.0f KB) in %.1fs",
             tag, len(pages), entry["name"], title,
             len(docx_bytes) / 1024, time.monotonic() - t0)
    return docx_bytes, title
