"""drive_folder.py — the shared folder as both input and output.

The form does not collect uploads. A teacher shares a Drive folder, and this
module is everything that follows from that: reading the folder, deciding which
of its files are student work, and writing the results back into it.

WHY A FOLDER AND NOT UPLOADS
────────────────────────────
Two reasons, both the teacher's. Results land where they already keep the work,
instead of in an inbox they have to file by hand; and a whole class is one
submission instead of thirty. The cost is that we no longer control what arrives
— an upload question restricted to PDF/JPG/PNG guaranteed the shape of its
input, and a folder guarantees nothing. Everything below exists to survive that.

THE PERMISSION THAT MATTERS
───────────────────────────
Writing results back needs EDITOR on the folder, not Viewer. Teachers share
read-only by reflex, and the failure would otherwise surface deep in delivery,
after every page has already been paid for. check_access() is called before any
grading starts for exactly that reason.
"""
from __future__ import annotations

import io
import logging
import os
import re

import fitz  # PyMuPDF — text-layer inspection only; core.py owns the rendering
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

log = logging.getLogger("form-checker-drive")

# The subfolder results are written into, promised by name in the form's own
# description ("a new folder named 'Bdika' will be created…"). Changing it means
# changing the form text too, so it is a constant rather than an env var.
OUTPUT_FOLDER_NAME = os.environ.get("OUTPUT_FOLDER_NAME", "Bdika")

FOLDER_MIME = "application/vnd.google-apps.folder"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
GDOC_MIME = "application/vnd.google-apps.document"

_MIME_EXT = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

# Every Drive call carries these. Without them a folder shared from a Shared
# Drive — which is how a school with Workspace will typically share — returns
# nothing at all, with no error to say why.
_SHARED = {"supportsAllDrives": True, "includeItemsFromAllDrives": True}


class FolderError(Exception):
    """A folder we cannot use, with a message written for the teacher to read.

    main.py mails these back verbatim, so they say what to do rather than what
    went wrong internally."""


# ─── access ─────────────────────────────────────────────────────────────────

def check_access(drive, folder_id: str) -> dict:
    """Folder metadata, or FolderError explaining what the teacher must change.

    Checks three things in the order they fail in practice: the link resolves,
    it points at a folder rather than a file, and we may write into it.
    """
    try:
        meta = drive.files().get(
            fileId=folder_id,
            fields="id, name, mimeType, capabilities(canAddChildren,canListChildren)",
            **{"supportsAllDrives": True},
        ).execute()
    except HttpError as e:
        if e.resp.status in (403, 404):
            raise FolderError(
                "לא הצלחנו לפתוח את התיקייה שקישרת. ודא/י שהתיקייה שותפה עם "
                "%s עם הרשאת עריכה (Editor), ושהקישור הוא לתיקייה ולא לקובץ בודד."
                % os.environ.get("SHARE_WITH", "כתובת השירות")) from e
        raise

    if meta.get("mimeType") != FOLDER_MIME:
        raise FolderError(
            "הקישור ששלחת מצביע על קובץ בודד ולא על תיקייה (%r). "
            "יש לשתף תיקייה שבתוכה נמצאות העבודות." % meta.get("name"))

    caps = meta.get("capabilities") or {}
    if not caps.get("canListChildren", True):
        raise FolderError(
            "אין לנו הרשאה לראות את תוכן התיקייה %r. יש לשתף אותה עם %s."
            % (meta.get("name"), os.environ.get("SHARE_WITH", "כתובת השירות")))
    if not caps.get("canAddChildren", False):
        # Deliberately fatal rather than a fallback to email. The teacher was
        # told in the form that results appear in their folder; grading the whole
        # class and then quietly putting it somewhere else is worse than saying
        # "fix the sharing and resubmit", which costs nothing.
        raise FolderError(
            "התיקייה %r שותפה איתנו לצפייה בלבד, ולכן אי אפשר לכתוב אליה את "
            "תוצאות הבדיקה. יש לשנות את השיתוף עם %s להרשאת עריכה (Editor) "
            "ולשלוח את הטופס שוב."
            % (meta.get("name"), os.environ.get("SHARE_WITH", "כתובת השירות")))
    return meta


# ─── listing ────────────────────────────────────────────────────────────────

def _ext_for(mime_type, name):
    if mime_type in _MIME_EXT:
        return _MIME_EXT[mime_type]
    ext = os.path.splitext(name or "")[1].lower()
    if ext == ".jpeg":
        ext = ".jpg"
    return ext if ext in (".pdf", ".jpg", ".png") else None


def list_candidates(drive, folder_id: str) -> tuple[list[dict], list[dict]]:
    """(gradable-format files, everything else) directly inside the folder.

    Not recursive, and that is a decision rather than an omission. A teacher's
    folder tree is unknowable — one nested Drive shortcut to "my documents" and
    a submission becomes a bill. One level is what the form describes ("each
    file which was in the shared folder"), and it is bounded by what the teacher
    can see when they look at the folder.

    Our own output folder is excluded, so re-submitting the same folder does not
    feed last run's reports back in as student work.
    """
    files, skipped = [], []
    page_token = None
    while True:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            orderBy="name_natural",
            pageSize=200,
            pageToken=page_token,
            **_SHARED,
        ).execute()

        for f in resp.get("files", []):
            if f["mimeType"] == FOLDER_MIME:
                continue                      # including our own output folder
            ext = _ext_for(f["mimeType"], f["name"])
            if ext is None:
                skipped.append({**f, "reason": "format %s" % f["mimeType"]})
                continue
            files.append({**f, "ext": ext})

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    log.info("folder %s: %d gradable-format file(s), %d in other formats",
             folder_id, len(files), len(skipped))
    return files, skipped


def download(drive, file_id: str) -> bytes:
    request = drive.files().get_media(fileId=file_id, **{"supportsAllDrives": True})
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


# ─── which files are student work ───────────────────────────────────────────
# The form promises analysis of "each file which was in the shared folder and
# included handwritten work". So a folder is expected to hold two kinds of
# thing: scans of what students wrote, and typed documents about the task — the
# exam paper, the instructions, sometimes the rubric.
#
# Telling them apart with a model call would cost per file. It does not need
# one. A typed document carries a text layer and no page images; a scan is the
# exact opposite. fitz reads both straight out of the PDF structure, for free.

# Per page. A scan that has been through OCR carries text *and* images, and the
# image test is what keeps it on the student-work side where it belongs.
_TYPED_TEXT_CHARS_PER_PAGE = 400


def looks_typed(data: bytes, ext: str) -> tuple[bool, str]:
    """(is a typed document, why).

    Biased towards "student work": a misfiled scan costs a wasted report, while
    a skipped scan means a student silently gets nothing back. Only a PDF with
    a substantial text layer and no images is called typed.
    """
    if ext != ".pdf":
        return False, "image file — a photograph of written work"
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            pages = doc.page_count or 1
            text_chars = sum(len(p.get_text().strip()) for p in doc)
            images = sum(len(p.get_images(full=False)) for p in doc)
    except Exception as e:
        # Unreadable here is unreadable in check_pages too; let the grading path
        # produce the real error instead of guessing at a classification.
        return False, f"could not inspect ({type(e).__name__})"

    per_page = text_chars / pages
    if per_page >= _TYPED_TEXT_CHARS_PER_PAGE and images == 0:
        return True, (f"{per_page:.0f} chars/page of selectable text and no page "
                      f"images over {pages} page(s)")
    return False, (f"{per_page:.0f} chars/page of selectable text, {images} image(s) "
                   f"over {pages} page(s)")


def page_count(data: bytes, ext: str) -> int:
    """Pages this file will render to, without rendering any of them.

    Known before grading starts, because by the time check_pages is rendering
    page 300 the money is already spent."""
    if ext != ".pdf":
        return 1
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return doc.page_count or 1
    except Exception:
        return 1


# ─── writing results back ───────────────────────────────────────────────────

def ensure_output_folder(drive, parent_id: str, name: str = OUTPUT_FOLDER_NAME) -> str:
    """Id of the results subfolder, reusing it if a previous run made one.

    Find-or-create rather than create: a teacher who submits the same folder
    twice should get a second report beside the first, not a second folder with
    the same name — which Drive allows, and which is indistinguishable to a
    human looking at it.
    """
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    resp = drive.files().list(
        q=(f"'{parent_id}' in parents and trashed = false "
           f"and mimeType = '{FOLDER_MIME}' and name = '{escaped}'"),
        fields="files(id, name)", pageSize=1, **_SHARED,
    ).execute()
    found = resp.get("files", [])
    if found:
        log.info("reusing existing output folder %r (id=%s)", name, found[0]["id"])
        return found[0]["id"]

    created = drive.files().create(
        body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
        fields="id", **{"supportsAllDrives": True},
    ).execute()
    log.info("created output folder %r (id=%s) in %s", name, created["id"], parent_id)
    return created["id"]


_UNSAFE_RE = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def report_name(source_name: str) -> str:
    """Report title for one source file: the source's own name, checked.

    Named after the student's file rather than the teacher's email, because the
    audience changed with the delivery. A folder of thirty reports is only
    usable if each one says whose it is, and the file name is the only thing we
    have that does — students' names are in the handwriting, not the metadata.
    """
    stem = os.path.splitext(source_name or "")[0] or "work"
    return _UNSAFE_RE.sub("-", stem).strip()[:180] + " — checked"


def write_report(drive, folder_id: str, name: str, docx_bytes: bytes) -> str:
    """Upload the report as a real .docx file. Returns the id.

    THIS IS WHERE THE MATHS PRODUCT DIVERGES FROM form-checker, which uploads
    the same bytes with mimeType=GDOC_MIME so Drive converts them to a native
    Google Doc — nicer to open, and correct for an English composition report.

    A maths report cannot take that conversion. Its content is LaTeX kept as
    monospace LTR runs (x^2-6x+10, \\triangle ABC \\cong \\triangle DEF) laid
    out inside an otherwise right-to-left Hebrew document, plus an embedded
    scan of the page. Drive's importer re-flows exactly those things: the
    monospace runs lose their LTR direction and the formulas come out reversed
    or interleaved with the surrounding Hebrew, which turns a correct report
    into an unreadable one. Keeping the .docx costs the teacher a download and
    keeps the mathematics legible.
    """
    filename = name if name.lower().endswith(".docx") else f"{name}.docx"
    media = MediaIoBaseUpload(io.BytesIO(docx_bytes), mimetype=DOCX_MIME,
                              resumable=False)
    created = drive.files().create(
        body={"name": filename, "parents": [folder_id]},   # no mimeType → no conversion
        media_body=media, fields="id, webViewLink",
        **{"supportsAllDrives": True},
    ).execute()
    log.info("wrote report %r (id=%s, %.0f KB)",
             filename, created["id"], len(docx_bytes) / 1024)
    return created["id"]


def folder_link(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"
