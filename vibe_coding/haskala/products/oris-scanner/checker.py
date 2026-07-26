import io
import logging
import os
import time

from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

import core
import report

# No basicConfig here on purpose — main.py owns logging setup (structured
# JSON for Cloud Run, level from LOG_LEVEL). Configuring it at import time
# from a module that main.py imports would install a plain-text handler first
# and turn main's own configuration into a no-op.
log = logging.getLogger("oris-scanner-checker")

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
    """Log-friendly one-value rendering of a possibly long, possibly
    multi-line string: newlines shown as ⏎ so one log entry stays one line."""
    if not text:
        return "(none)"
    flat = " ⏎ ".join(str(text).splitlines())
    if len(flat) > limit:
        return f"{flat[:limit]}… [+{len(flat) - limit} chars]"
    return flat


def check_hw(service, file_ids, folder_id, classroom_rubric=None,
             assignment_description=None, assignment_name=None,
             model_key=None, tag=""):
    """Downloads each attached Drive file, runs them through the checking
    pipeline (OCR every page → evaluate the merged transcript against a
    rubric), and uploads the resulting Word report into folder_id.

    classroom_rubric: the assignment's own Rubric object from the Classroom
    API (courses.courseWork.rubrics), if the teacher attached one — takes
    priority over the bundled default rubric. assignment_description: the
    assignment's free-text definition as the teacher wrote it in Classroom,
    with any [model: …] tag already stripped by the caller, passed through as
    the rubric's "question" so the model evaluates against the actual task,
    not just generic criteria. model_key: the provider the teacher selected
    (see core.resolve_model); None means core.DEFAULT_MODEL.

    tag: a short correlation prefix (submission id etc.) stamped onto every
    log line, so one submission's trace can be followed through interleaved
    concurrent runs."""
    model_key = model_key or core.DEFAULT_MODEL
    t0 = time.monotonic()

    file_bytes_list = []
    first_name = None
    for fid in file_ids:
        meta = service.files().get(fileId=fid, fields="mimeType, name, size").execute()
        first_name = first_name or meta.get("name")
        ext = _ext_for(meta.get("mimeType"), meta.get("name"))
        if not ext:
            log.info("%s skipping attachment %r — unsupported type %s",
                     tag, meta.get("name"), meta.get("mimeType"))
            continue
        data = _download_bytes(service, fid)
        log.info("%s downloaded attachment %r (%s, %s, %.0f KB, id=%s)",
                 tag, meta.get("name"), meta.get("mimeType"), ext, len(data) / 1024, fid)
        file_bytes_list.append((data, ext))

    if not file_bytes_list:
        log.info("%s no supported attachments among %s — nothing to upload", tag, file_ids)
        return

    rubric_override = None
    if classroom_rubric and classroom_rubric.get("criteria"):
        rubric_override = core.rubric_from_classroom(classroom_rubric, assignment_name or "")

    rubric_name = (
        rubric_override["name"] if rubric_override
        else (core.load_rubric(core.DEFAULT_RUBRIC_ID) or {}).get("name", core.DEFAULT_RUBRIC_ID)
    )
    rubric_content = (
        rubric_override["content"] if rubric_override
        else (core.load_rubric(core.DEFAULT_RUBRIC_ID) or {}).get("content", "")
    )

    # The whole point of this block: everything that determines the grade is
    # visible in the log *before* the slow model calls start, so a surprising
    # result can be traced back to the inputs that produced it.
    log.info("%s ── checking parameters ──", tag)
    log.info("%s   assignment : %r", tag, assignment_name)
    log.info("%s   model      : %s (%s)", tag, model_key,
             core.MODELS.get(model_key, "unknown key"))
    log.info("%s   rubric src : %s", tag,
             "Classroom rubric attached to the assignment" if rubric_override
             else f"bundled default ({core.DEFAULT_RUBRIC_ID})")
    log.info("%s   rubric name: %r (%d criteria)", tag, rubric_name,
             len(classroom_rubric.get("criteria", [])) if rubric_override else 0)
    log.info("%s   question   : %s", tag, _preview(assignment_description, 1200))
    log.info("%s   rubric text: %s", tag, _preview(rubric_content, 2000))
    log.info("%s   files      : %d page-source file(s)", tag, len(file_bytes_list))
    log.info("%s ─────────────────────────", tag)

    pages, evaluation = core.check_pages(
        file_bytes_list,
        rubric_override=rubric_override,
        question=assignment_description,
        model_key=model_key,
    )

    stem = os.path.splitext(first_name or "report")[0]
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_name = f"{stem}_report_{ts}.docx"

    docx_bytes = report.build_evaluation_docx(
        evaluation, out_name, rubric_name, pages=pages,
        feedback_lang=core.DEFAULT_FEEDBACK_LANG, exercise_lang=core.DEFAULT_EXERCISE_LANG,
    )

    file_metadata = {"name": out_name, "parents": [folder_id]}
    media = MediaIoBaseUpload(
        io.BytesIO(docx_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    uploaded = service.files().create(
        body=file_metadata, media_body=media, fields="id").execute()
    log.info("%s uploaded report %r (id=%s) to folder %s — total %.1fs",
             tag, out_name, uploaded.get("id"), folder_id, time.monotonic() - t0)
