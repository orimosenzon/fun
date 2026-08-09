"""upload.py — read the one file a teacher attached to the demo form.

form-checker's drive_folder.py is 291 lines because a folder is a live, shared,
writable thing: it has to be resolved from a pasted link, checked for Editor
access, listed, filtered, and written back into. A form upload is none of that.
Forms has already put the file in the form owner's Drive, the responses sheet
already holds its id, and nothing is ever written back. What is left is: fetch
it, and refuse it politely if it is not something we can read.

WHERE THE FILE ACTUALLY LIVES
─────────────────────────────
Uploads land in the Drive of whoever owns the *form*, not the person filling it
in — here, exam@bdika.net. This service reads as WORKSPACE_SUBJECT
(ori@bdika.net) via domain-wide delegation, and reaches those files because the
"Bdika forms" folder is shared inside the Workspace. That is inherited access,
so it breaks silently if the form is ever moved out of that folder. UploadError
below names that case explicitly, because a 404 here otherwise reads as "the
teacher deleted their file".
"""
from __future__ import annotations

import datetime
import io
import logging
import os

import fitz
from googleapiclient.http import MediaIoBaseDownload

log = logging.getLogger("demo-checker")

_SHARED = {"supportsAllDrives": True}

# What core.check_pages can rasterise. Anything else is refused before it costs
# anything — notably HEIC, which is what an iPhone produces by default and the
# single most likely thing to arrive from a teacher photographing a page.
_MIME_EXT = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

_FORMAT_HELP = {
    ".heic": "HEIC (the iPhone default). In the Files app, share the photo as "
             "JPEG, or take the photo with Settings → Camera → Formats set to "
             "'Most Compatible'.",
    ".docx": "a Word document. This tool reads scans and photographs of "
             "handwritten work, not typed text.",
    ".doc": "a Word document. This tool reads scans and photographs of "
            "handwritten work, not typed text.",
    ".txt": "a text file. This tool reads scans and photographs of handwritten "
            "work, not typed text.",
}


class UploadError(Exception):
    """A submission we understand but cannot grade. The message is written for a
    teacher to read — it goes out verbatim in the failure mail, so it must say
    what to do next rather than what went wrong internally."""


def ext_for(mime_type: str, name: str) -> str | None:
    """Gradable file extension, or None if we cannot read this format."""
    if mime_type in _MIME_EXT:
        return _MIME_EXT[mime_type]
    ext = os.path.splitext(name or "")[1].lower()
    if ext == ".jpeg":
        ext = ".jpg"
    return ext if ext in (".pdf", ".jpg", ".png") else None


def fetch(drive, file_id: str) -> dict:
    """{id, name, ext, data} for one uploaded file.

    Raises UploadError with teacher-readable text for the two things that
    actually happen: a format we cannot read, and a file we cannot see."""
    try:
        meta = drive.files().get(
            fileId=file_id, fields="id,name,mimeType,size", **_SHARED).execute()
    except Exception as e:
        log.warning("cannot read uploaded file %s (%s: %s)",
                    file_id, type(e).__name__, str(e)[:200])
        raise UploadError(
            "We could not open the file you uploaded. If it was removed from "
            "Drive after you submitted the form, please upload it again."
        ) from e

    name = meta.get("name") or file_id
    ext = ext_for(meta.get("mimeType", ""), name)
    if ext is None:
        actual = os.path.splitext(name)[1].lower()
        detail = _FORMAT_HELP.get(actual, f"a {meta.get('mimeType')} file")
        raise UploadError(
            f"The file you uploaded ({name}) is {detail} "
            "Please upload a PDF, JPG or PNG and submit the form again.")

    data = _download(drive, file_id)
    log.info("fetched upload %s %r (%s, %.0f KB)",
             file_id, name, ext, len(data) / 1024)
    return {"id": file_id, "name": name, "ext": ext, "data": data}


def _download(drive, file_id: str) -> bytes:
    request = drive.files().get_media(fileId=file_id, **_SHARED)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def page_count(data: bytes, ext: str) -> int:
    """Pages this file will render to, without rendering any of them — the cap
    has to be enforced before the money is spent, not after."""
    if ext != ".pdf":
        return 1
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return doc.page_count or 1
    except Exception:
        return 1


_FORM_TS_FORMATS = ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S")


def report_name(source_name: str, timestamp: str = "") -> str:
    """Attachment filename. Carries a timestamp, unlike oris-scanner's.

    A teacher trying the demo sends the same page three times in ten minutes to
    see whether the grade is stable. Three attachments with identical names in
    one inbox is a mailbox they cannot navigate.

    The stamp is reformatted rather than stripped of punctuation: Forms writes
    "8/8/2026 10:00:00", and simply keeping the digits yields "882026100000",
    which is both unreadable and ambiguous about where the date ends. Sorting
    by name should also sort by time, hence year-first."""
    stem = os.path.splitext(source_name or "work")[0].strip() or "work"
    stamp = ""
    for fmt in _FORM_TS_FORMATS:
        try:
            stamp = datetime.datetime.strptime(timestamp.strip(), fmt).strftime("%Y-%m-%d_%H%M")
            break
        except (ValueError, AttributeError):
            continue
    return f"{stem}_bdika_{stamp}.docx" if stamp else f"{stem}_bdika.docx"
