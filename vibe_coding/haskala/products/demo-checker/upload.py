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
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

log = logging.getLogger("demo-checker")

_SHARED = {"supportsAllDrives": True}

FOLDER_MIME = "application/vnd.google-apps.folder"
GDOC_MIME = "application/vnd.google-apps.document"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# The subfolder graded work is written into, created beside the uploads. Named
# to match form-checker and math-form-checker rather than described ("checked
# exercises") — a teacher who later uses more than one of these products should
# find the same folder name in all of them.
OUTPUT_FOLDER_NAME = os.environ.get("OUTPUT_FOLDER_NAME", "Bdika")

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
    """{id, name, ext, parent, data} for one uploaded file.

    `parent` is the Forms "(File responses)" folder — where every upload from
    this question is collected — and is what the graded report's folder is
    created beside. Read here rather than configured, because Forms creates that
    folder itself and its id is not something anyone types in.

    Raises UploadError with teacher-readable text for the two things that
    actually happen: a format we cannot read, and a file we cannot see."""
    try:
        meta = drive.files().get(
            fileId=file_id, fields="id,name,mimeType,size,parents", **_SHARED).execute()
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
    parents = meta.get("parents") or []
    log.info("fetched upload %s %r (%s, %.0f KB, parent=%s)",
             file_id, name, ext, len(data) / 1024, parents[0] if parents else "?")
    return {"id": file_id, "name": name, "ext": ext, "data": data,
            "parent": parents[0] if parents else ""}


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


# ─── writing the graded report back ─────────────────────────────────────────

def ensure_output_folder(drive, parent_id: str, name: str = OUTPUT_FOLDER_NAME) -> str:
    """Id of the results subfolder beside the uploads, reusing an existing one.

    Find-or-create rather than create: this runs once per submission, and a new
    "Bdika (1)" every time a teacher tries the demo would bury the uploads
    folder in near-identical directories."""
    q = (f"'{parent_id}' in parents and name = '{name}' and "
         f"mimeType = '{FOLDER_MIME}' and trashed = false")
    found = drive.files().list(q=q, fields="files(id,name)", pageSize=10,
                               includeItemsFromAllDrives=True, **_SHARED).execute()
    for f in found.get("files", []):
        log.info("reusing output folder %r (%s)", name, f["id"])
        return f["id"]

    created = drive.files().create(
        body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
        fields="id", **_SHARED).execute()
    log.info("created output folder %r (%s) under %s", name, created["id"], parent_id)
    return created["id"]


def write_report(drive, folder_id: str, name: str, docx_bytes: bytes) -> str:
    """Upload the .docx and let Drive convert it to a native Google Doc.

    Converted rather than left as a .docx because the point is that the teacher
    can *edit* it — the grade is a suggestion, and a Word file in Drive opens
    read-only in a viewer until it is explicitly converted. This is safe here in
    a way it is not in math-form-checker, whose reports carry LaTeX that Drive's
    importer mangles inside an RTL document."""
    media = MediaIoBaseUpload(io.BytesIO(docx_bytes), mimetype=DOCX_MIME,
                              resumable=False)
    created = drive.files().create(
        body={"name": os.path.splitext(name)[0], "parents": [folder_id],
              "mimeType": GDOC_MIME},
        media_body=media, fields="id,webViewLink", **_SHARED).execute()
    log.info("wrote report %r as a Google Doc (%s)", name, created["id"])
    return created["id"]


def share(drive, file_id: str, email: str, role: str = "writer") -> bool:
    """Grant one address access to the report. True if it stuck.

    Never raises. A share that fails must not lose a report that is already
    graded and already in the folder — the mail still carries the .docx, so the
    teacher gets their result either way and the log says what to fix.

    sendNotificationEmail is off: our own mail is already on its way and says
    more than Drive's would, and two arrivals for one submission reads as a
    system that has lost track of itself."""
    try:
        drive.permissions().create(
            fileId=file_id,
            body={"type": "user", "role": role, "emailAddress": email},
            sendNotificationEmail=False,
            **_SHARED).execute()
        log.info("shared %s with %s as %s", file_id, email, role)
        return True
    except Exception as e:
        # Overwhelmingly: the address has no Google account, so there is nobody
        # to grant a permission to. Not worth failing a submission over.
        log.warning("could not share %s with %s as %s (%s: %s)",
                    file_id, email, role, type(e).__name__, str(e)[:200])
        return False


def share_with_domain(drive, file_id: str, domain: str, role: str = "writer") -> bool:
    """Give everyone in `domain` access to the report. True if it stuck.

    Avishai's spec asks for the Doc to be "shared with bdika.net as editor" —
    the domain, not one mailbox. That is the difference between the team being
    able to look at a report when a teacher writes in about it, and only
    whichever account happened to create it being able to.

    Same never-raises contract as share(): a graded report already in the folder
    must not be lost to a sharing policy."""
    try:
        drive.permissions().create(
            fileId=file_id,
            body={"type": "domain", "role": role, "domain": domain},
            sendNotificationEmail=False,
            **_SHARED).execute()
        log.info("shared %s with the %s domain as %s", file_id, domain, role)
        return True
    except Exception as e:
        # Typically a Workspace sharing policy that forbids domain-wide links.
        log.warning("could not share %s with the %s domain (%s: %s)",
                    file_id, domain, type(e).__name__, str(e)[:200])
        return False


def doc_link(file_id: str) -> str:
    return f"https://docs.google.com/document/d/{file_id}/edit"


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
