"""checker.py — one Form response → a graded Word report.

The glue layer. Downloads the files the teacher uploaded, decides which rubric
applies, runs the shared grading pipeline, and returns the .docx bytes for
main.py to deliver.

Deliberately narrower than oris-scanner's equivalent: there is no Classroom
here, so no structured Classroom rubric and no assignment materials to sift
through. The rubric comes from a dropdown the teacher answered, and the only
ambiguous case is "the rubric is one of the files I uploaded", which is settled
by core.classify_document exactly as oris-scanner settles it.
"""
from __future__ import annotations

import io
import logging
import os
import re
import time

import fitz  # PyMuPDF — page counting only; core.py owns the rendering
from googleapiclient.http import MediaIoBaseDownload

import form_schema
from haskala_grading import core, report

# No basicConfig here — main.py owns logging setup (structured JSON for Cloud
# Run, level from LOG_LEVEL). Configuring it from a module main.py imports would
# install a plain-text handler first and make main's own call a silent no-op.
log = logging.getLogger("form-checker-checker")

_MIME_EXT = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


def _ext_for(mime_type, name):
    if mime_type in _MIME_EXT:
        return _MIME_EXT[mime_type]
    ext = os.path.splitext(name or "")[1].lower()
    if ext == ".jpeg":
        ext = ".jpg"
    return ext if ext in (".pdf", ".jpg", ".png") else None


def _download_bytes(service, file_id):
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def _preview(text, limit):
    """Log-friendly rendering of a possibly long, possibly multi-line string:
    newlines shown as ⏎ so one log entry stays one line."""
    if not text:
        return "(none)"
    flat = " ⏎ ".join(str(text).splitlines())
    if len(flat) > limit:
        return f"{flat[:limit]}… [+{len(flat) - limit} chars]"
    return flat


_FILENAME_UNSAFE_RE = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def _slug(s):
    """One filename component: no path-hostile characters, no runs of
    whitespace. Case and every other character are preserved — the address and
    the school name are meant to read back exactly as the teacher spells them,
    Hebrew and '@' included."""
    s = _FILENAME_UNSAFE_RE.sub("-", str(s))
    return re.sub(r"\s+", "_", s.strip()).strip("_")


# Drive's own limit is 255. Stopping short leaves room for the " (1)" Drive
# appends when a name repeats within a folder.
_MAX_FILENAME = 240


def report_filename(params: dict) -> str:
    """<teacher email>_<school>_<timestamp>.docx

    The timestamp is here on purpose, and this is where form-checker diverges
    from oris-scanner (which drops it by explicit request). A Classroom
    submission is one student's one attempt at one assignment, so a name
    collision is a resubmission and the two are told apart by date modified.
    A teacher trying the form out submits the same exercise repeatedly within
    minutes, and a mailbox with four identically-named attachments is unusable.
    """
    stamp = _slug(params.get("timestamp") or time.strftime("%Y-%m-%d_%H-%M"))
    parts = [params.get("email"), params.get("school"), stamp]
    name = "_".join(_slug(p) for p in parts if p)
    return name[:_MAX_FILENAME] + ".docx"


def _resolve_rubric(service, params, downloaded, tag):
    """(rubric_override | None, human description of where it came from).

    Precedence, highest first:
      1. the teacher said the rubric is one of the uploaded files — find it;
      2. the teacher picked a bundled rubric from the dropdown;
      3. neither — the bundled default.

    A rubric that cannot be read never takes the response down; falling back to
    the bundled default and grading is strictly better than returning nothing,
    and the log line says which happened.
    """
    if params.get("rubric_from_upload"):
        rubric = _rubric_from_uploads(service, params, downloaded, tag)
        if rubric:
            return rubric, f"a file the teacher uploaded ({rubric['name']!r})"
        log.warning("%s teacher chose %r but no uploaded file reads as a rubric — "
                    "falling back to the bundled default",
                    tag, form_schema.RUBRIC_FROM_UPLOAD)

    if params.get("rubric_id"):
        rubric = core.load_rubric(params["rubric_id"])
        if rubric:
            return rubric, f"the form's dropdown ({params['rubric_id']})"
        log.warning("%s rubric id %r resolved to no bundled rubric — using the default",
                    tag, params["rubric_id"])

    return None, (f"bundled default ({core.DEFAULT_RUBRIC_ID}) — "
                  "no rubric chosen on the form")


def _rubric_from_uploads(service, params, downloaded, tag):
    """Find the rubric among the files the teacher uploaded.

    Every uploaded file is a candidate: teachers do not name files "מחוון", so
    requiring the naming convention would mean the feature almost never fires —
    the same lesson core.resolve_rubric_material records for Classroom
    attachments. Each candidate's text is read and classified, and the first one
    that classifies as a rubric wins. That file is then dropped from the pages
    to grade, so the rubric is not also OCR'd as if it were student work.
    """
    for entry in list(downloaded):
        data, ext, name = entry["data"], entry["ext"], entry["name"]
        try:
            text, method = core.rubric_text_from_document(data, ext, params["model_key"])
        except Exception as e:
            log.warning("%s could not read %r as a document (%s: %s) — "
                        "treating it as student work",
                        tag, name, type(e).__name__, str(e)[:200])
            continue
        if not text.strip():
            continue
        kind = core.classify_document(text, params["model_key"])
        log.info("%s upload %r read via %s (%d chars) — classified as %s",
                 tag, name, method, len(text), kind)
        if kind == "rubric":
            downloaded.remove(entry)   # not student work — do not OCR it as pages
            return {"name": name, "content": text}
    return None


def _count_pages(entry) -> int:
    """Pages this file will render to, without rendering any of them.

    fitz reads a PDF's page count from its catalogue, so this costs a parse and
    no rasterisation — which is the point. The page count has to be known
    *before* core.check_pages starts, because by the time it is rendering
    page 300 the money is already being spent."""
    if entry["ext"] != ".pdf":
        return 1
    try:
        with fitz.open(stream=entry["data"], filetype="pdf") as doc:
            return doc.page_count
    except Exception:
        # Unreadable here means unreadable later too; let check_pages produce
        # the real error rather than guessing at one now.
        return 1


def check_form_response(service, params: dict, tag: str = "",
                        max_pages: int | None = None) -> tuple[bytes, str, dict]:
    """Grade one Form response.

    Returns (docx bytes, filename, evaluation) — main.py owns delivery, so
    nothing here writes to Drive or sends mail.

    max_pages bounds what one response can cost. Every page is a rotation call
    plus an OCR call plus its share of the evaluation, so a 300-page PDF is a
    three-figure bill from a single form submission.

    Raises ValueError when there is nothing gradable, or when the submission is
    over the page cap — main.py turns that into an explanation emailed to the
    teacher. Every other failure mode degrades: an unreadable rubric falls back
    to the default, an unreadable single file is skipped.
    """
    t0 = time.monotonic()

    downloaded = []
    for fid in params["file_ids"]:
        meta = service.files().get(fileId=fid, fields="mimeType, name, size").execute()
        ext = _ext_for(meta.get("mimeType"), meta.get("name"))
        if not ext:
            log.info("%s skipping upload %r — unsupported type %s",
                     tag, meta.get("name"), meta.get("mimeType"))
            continue
        data = _download_bytes(service, fid)
        log.info("%s downloaded %r (%s, %s, %.0f KB, id=%s)",
                 tag, meta.get("name"), meta.get("mimeType"), ext, len(data) / 1024, fid)
        downloaded.append({"data": data, "ext": ext, "name": meta.get("name") or fid})

    # These messages are emailed verbatim to the teacher (main.py turns a
    # ValueError into mailer.send_failure), so they are written in Hebrew and
    # say what to do about it — not phrased as log text.
    if not downloaded:
        raise ValueError(
            "אף אחד מ-%d הקבצים שצורפו אינו בפורמט שאפשר לבדוק. "
            "המערכת קוראת PDF, JPG ו-PNG. אם צילמת באייפון, הקובץ כנראה בפורמט "
            "HEIC — אפשר לשמור אותו כ-JPG ולשלוח שוב." % len(params["file_ids"]))

    rubric_override, rubric_source = _resolve_rubric(service, params, downloaded, tag)
    if not downloaded:
        # Everything uploaded turned out to be the rubric.
        raise ValueError("הקובץ היחיד שצורף הוא המחוון — לא צורפה עבודת תלמיד לבדיקה.")

    # After rubric resolution, so a rubric PDF's pages are not counted against
    # the student work's budget.
    if max_pages is not None:
        total = sum(_count_pages(d) for d in downloaded)
        if total > max_pages:
            raise ValueError(
                "התרגיל מכיל %d עמודים, והמערכת בודקת עד %d בכל שליחה. "
                "אפשר לפצל אותו לכמה שליחות." % (total, max_pages))
        log.info("%s %d page(s) to grade (cap %d)", tag, total, max_pages)

    _default = core.load_rubric(core.DEFAULT_RUBRIC_ID) or {}
    rubric_name = (rubric_override["name"] if rubric_override
                   else _default.get("name", core.DEFAULT_RUBRIC_ID))
    rubric_content = (rubric_override["content"] if rubric_override
                      else _default.get("content", ""))
    question = form_schema.build_question(params)

    # Everything that determines the grade, visible in the log *before* the slow
    # model calls start, so a surprising result can be traced to its inputs.
    # Ported from oris-scanner/checker.py, where it has repeatedly been the
    # difference between diagnosing a bad grade and guessing at it.
    log.info("%s ── checking parameters ──", tag)
    log.info("%s   teacher    : %s%s", tag, params.get("email") or "(no address)",
             f" ({params['teacher_name']})" if params.get("teacher_name") else "")
    log.info("%s   school     : %s", tag, params.get("school") or "(not given)")
    log.info("%s   grade level: %s", tag, params.get("grade_level") or "(not given)")
    log.info("%s   model      : %s (%s)", tag, params["model_key"],
             core.MODELS.get(params["model_key"], "unknown key"))
    log.info("%s   language   : %s", tag, params["exercise_lang"])
    log.info("%s   rubric src : %s", tag, rubric_source)
    log.info("%s   rubric name: %r", tag, rubric_name)
    log.info("%s   question   : %s", tag, _preview(question, 1200))
    log.info("%s   rubric text: %s", tag, _preview(rubric_content, 2000))
    log.info("%s   files      : %d page-source file(s)", tag, len(downloaded))
    log.info("%s ─────────────────────────", tag)

    pages, evaluation = core.check_pages(
        [(d["data"], d["ext"]) for d in downloaded],
        rubric_override=rubric_override,
        question=question,
        model_key=params["model_key"],
    )

    out_name = report_filename(params)
    docx_bytes = report.build_evaluation_docx(
        evaluation, out_name, rubric_name, pages=pages,
        feedback_lang=params["exercise_lang"],
        exercise_lang=params["exercise_lang"],
    )
    log.info("%s graded %d page(s) → %r (%.0f KB) in %.1fs",
             tag, len(pages), out_name, len(docx_bytes) / 1024, time.monotonic() - t0)
    return docx_bytes, out_name, evaluation
