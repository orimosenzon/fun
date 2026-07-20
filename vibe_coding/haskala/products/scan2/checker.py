import io
import logging
import os
import time

from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

import core
import report

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scan2-checker")

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


def _merge_pages(page_lists):
    merged = []
    for pages in page_lists:
        for p in pages:
            p = dict(p)
            p["page"] = len(merged) + 1
            merged.append(p)
    return merged


def check_hw(service, file_ids, folder_id):
    """Downloads each attached Drive file, runs it through the math-checker
    core pipeline, merges the pages into one report, and uploads the result
    as a Word document into folder_id."""
    page_lists = []
    first_name = None
    _, model_label = core.resolve_model(core.DEFAULT_MODEL)
    for fid in file_ids:
        meta = service.files().get(fileId=fid, fields="mimeType, name").execute()
        first_name = first_name or meta.get("name")
        ext = _ext_for(meta.get("mimeType"), meta.get("name"))
        if not ext:
            log.info("skipping %s - unsupported type %s", meta.get("name"), meta.get("mimeType"))
            continue
        data = _download_bytes(service, fid)
        for ev in core.process_stream(data, ext, auto_orient=True,
                                       model_key=core.DEFAULT_MODEL,
                                       model_label=model_label, keep_imgs=False):
            if ev["type"] == "result":
                page_lists.append(ev["pages"])

    if not page_lists:
        log.info("no supported attachments among %s - nothing to upload", file_ids)
        return

    pages = _merge_pages(page_lists)
    stem = os.path.splitext(first_name or "report")[0]
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_name = f"{stem}_report_{ts}.docx"

    docx_bytes = report.build_result_docx(pages, out_name)

    file_metadata = {"name": out_name, "parents": [folder_id]}
    media = MediaIoBaseUpload(
        io.BytesIO(docx_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    log.info("uploaded %s to folder %s", out_name, folder_id)
