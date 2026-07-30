import io
import logging
import os
import re
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


# Google-native formats carry no bytes to download; they have to be exported.
# A Doc becomes plain text directly, which is exactly what a rubric needs.
_GOOGLE_EXPORTS = {
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
}


def load_material_rubric(service, material, model_key=None, tag=""):
    """Reads a document attached to the assignment into the same {name,
    content} shape as a bundled rubric, plus a "kind" saying how to use it.

    material: {id, title, certain} as picked by core.resolve_rubric_material.
    Returns None if the attachment can't be read at all, so the caller falls
    back to the bundled default rather than grading against nothing.

    kind is "rubric" when the document states grading criteria — it then
    replaces the bundled rubric — or "assignment" when it is a brief for the
    student, which is handed to the grader as extra context with the bundled
    rubric left in place. An attachment the teacher explicitly marked as a
    rubric (certain=True) is taken at their word and never classified."""
    model_key = model_key or core.DEFAULT_MODEL
    fid, title = material["id"], material.get("title") or "rubric"
    try:
        meta = service.files().get(fileId=fid, fields="mimeType, name").execute()
        mime, name = meta.get("mimeType"), meta.get("name") or title

        if mime in _GOOGLE_EXPORTS:
            export_mime, ext = _GOOGLE_EXPORTS[mime]
            data = service.files().export(fileId=fid, mimeType=export_mime).execute()
            text, method = data.decode("utf-8", "replace").strip(), "Google Doc export"
        else:
            ext = _ext_for(mime, name)
            if not ext:
                log.warning("%s rubric attachment %r is a %s — cannot read it as a rubric, "
                            "falling back to the bundled default", tag, name, mime)
                return None
            data = _download_bytes(service, fid)
            text, method = core.rubric_text_from_document(data, ext, model_key)

        if not text.strip():
            log.warning("%s attachment %r produced no text (%s) — "
                        "falling back to the bundled default", tag, name, method)
            return None

        if material.get("certain"):
            kind = "rubric"
            why = "named as a rubric by the teacher"
        else:
            kind = core.classify_document(text, model_key)
            why = f"classified as {kind} by {model_key}"

        log.info("%s attachment %r read via %s (%d chars) — using it as %s (%s)",
                 tag, name, method, len(text), kind, why)
        return {"name": name, "content": text, "kind": kind}
    except Exception as e:
        # A rubric we cannot read must never take the whole submission down;
        # grading with the bundled default is a far better outcome.
        log.warning("%s failed to read attachment %r (%s: %s) — "
                    "falling back to the bundled default",
                    tag, title, type(e).__name__, str(e)[:200])
        return None


_FILENAME_UNSAFE_RE = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def _report_filename(student_email, student_id, assignment_name, fallback_stem):
    """The report's name in Drive: the student's email and the assignment,
    so a teacher scanning the "תרגילים בדוקים" folder can tell whose work a
    report is without opening it (Avishai, 2026-07-30).

    Falls back to the Classroom userId when the email could not be resolved,
    and to the student's own uploaded filename when there is no id either —
    the old behaviour, which identified nobody but at least never collided.
    """
    clean = lambda s: _FILENAME_UNSAFE_RE.sub("-", str(s)).strip()
    who = clean(student_email or student_id or fallback_stem or "report")[:120]
    parts = [who]
    if assignment_name:
        # Only the assignment name is truncated. The timestamp has to survive
        # intact: it is what keeps a resubmission from overwriting the first
        # report, and Drive's 255-char cap is well within reach of a verbose
        # assignment title.
        parts.append(clean(assignment_name)[:80])
    parts.append(time.strftime("%Y%m%d-%H%M%S"))
    return "_".join(parts) + ".docx"


def check_hw(service, file_ids, folder_id, classroom_rubric=None,
             assignment_description=None, assignment_name=None,
             model_key=None, tag="", material_rubric=None,
             student_email=None, student_id=None):
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
    concurrent runs.

    student_email / student_id: who handed this in — used to name the report
    in Drive (see _report_filename). Both are optional; the report is still
    produced when neither can be resolved."""
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

    # An attached document that turned out to be an assignment brief rather
    # than a rubric never touches the rubric slot — it is appended to the
    # question, so the grader sees the actual task the student was set while
    # still scoring against a real rubric. This is the whole point of
    # classifying it: adopting a brief as the rubric would drop the score
    # scale and the criteria entirely.
    if material_rubric and material_rubric.get("kind") == "assignment":
        brief = material_rubric["content"]
        assignment_description = (
            f"{assignment_description}\n\n{brief}" if assignment_description else brief
        )
        log.info("%s attachment %r folded into the question as assignment context "
                 "(%d chars); rubric selection unaffected",
                 tag, material_rubric["name"], len(brief))
        material_rubric = None

    # Precedence: Classroom's own structured rubric wins, because it is the
    # most explicit statement of intent and is what the teacher sees in the
    # grading UI. An attached rubric document is the next best signal. Only
    # with neither does the bundled generic rubric apply.
    rubric_override = None
    if classroom_rubric and classroom_rubric.get("criteria"):
        rubric_override = core.rubric_from_classroom(classroom_rubric, assignment_name or "")
        rubric_source = "Classroom rubric attached to the assignment"
    elif material_rubric:
        rubric_override = material_rubric
        rubric_source = f"document attached to the assignment ({material_rubric['name']!r})"
    else:
        rubric_source = (f"bundled default ({core.DEFAULT_RUBRIC_ID}) — "
                         "no Classroom rubric and no rubric document attached")

    _default = core.load_rubric(core.DEFAULT_RUBRIC_ID) or {}
    rubric_name = (rubric_override["name"] if rubric_override
                   else _default.get("name", core.DEFAULT_RUBRIC_ID))
    rubric_content = (rubric_override["content"] if rubric_override
                      else _default.get("content", ""))

    # The whole point of this block: everything that determines the grade is
    # visible in the log *before* the slow model calls start, so a surprising
    # result can be traced back to the inputs that produced it.
    log.info("%s ── checking parameters ──", tag)
    log.info("%s   assignment : %r", tag, assignment_name)
    log.info("%s   student    : %s", tag,
             student_email or (f"id {student_id} (email unresolved)"
                               if student_id else "unidentified"))
    log.info("%s   model      : %s (%s)", tag, model_key,
             core.MODELS.get(model_key, "unknown key"))
    log.info("%s   rubric src : %s", tag, rubric_source)
    log.info("%s   rubric name: %r%s", tag, rubric_name,
             f" ({len(classroom_rubric['criteria'])} criteria)"
             if classroom_rubric and classroom_rubric.get("criteria") else "")
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
    out_name = _report_filename(student_email, student_id, assignment_name, stem)

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
