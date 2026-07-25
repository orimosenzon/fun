import io
import logging
import os
import time

from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

import core
import report

logging.basicConfig(level=logging.INFO)
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


def check_hw(service, file_ids, folder_id, classroom_rubric=None,
             assignment_description=None, assignment_name=None):
    """Downloads each attached Drive file, runs them through the checking
    pipeline (OCR every page → evaluate the merged transcript against a
    rubric), and uploads the resulting Word report into folder_id.

    classroom_rubric: the assignment's own Rubric object from the Classroom
    API (courses.courseWork.rubrics), if the teacher attached one — takes
    priority over the bundled default rubric. assignment_description: the
    assignment's free-text definition as the teacher wrote it in Classroom,
    passed through as the rubric's "question" so the model evaluates against
    the actual task, not just generic criteria.

    Model is currently fixed to core.DEFAULT_MODEL; check_pages() already
    takes it as a parameter, so a future entry point can expose it without
    further plumbing changes."""
    file_bytes_list = []
    first_name = None
    for fid in file_ids:
        meta = service.files().get(fileId=fid, fields="mimeType, name").execute()
        first_name = first_name or meta.get("name")
        ext = _ext_for(meta.get("mimeType"), meta.get("name"))
        if not ext:
            log.info("skipping %s - unsupported type %s", meta.get("name"), meta.get("mimeType"))
            continue
        data = _download_bytes(service, fid)
        file_bytes_list.append((data, ext))

    if not file_bytes_list:
        log.info("no supported attachments among %s - nothing to upload", file_ids)
        return

    rubric_override = None
    if classroom_rubric and classroom_rubric.get("criteria"):
        rubric_override = core.rubric_from_classroom(classroom_rubric, assignment_name or "")

    pages, evaluation = core.check_pages(
        file_bytes_list,
        rubric_override=rubric_override,
        question=assignment_description,
    )
    rubric_name = (
        rubric_override["name"] if rubric_override
        else (core.load_rubric(core.DEFAULT_RUBRIC_ID) or {}).get("name", core.DEFAULT_RUBRIC_ID)
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
    service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    log.info("uploaded %s to folder %s", out_name, folder_id)
