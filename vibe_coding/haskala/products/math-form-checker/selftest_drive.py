"""selftest_drive.py — prove the Drive layer works against real Google, safely.

Everything in form-checker up to now has been tested against a fake Drive. That
validates the logic and none of the assumptions: whether domain-wide delegation
actually grants the Drive scope, whether a folder's capabilities come back the
way check_access expects, whether Drive really converts our .docx into a Google
Doc. Those only fail in production, which is the worst place to find out.

This creates a scratch folder in the service account's Drive, exercises the real
code against it, prints what happened, and trashes it again. No model calls, so
it costs nothing but a few API round trips.

    python selftest_drive.py            # run and clean up
    python selftest_drive.py --keep     # leave the folder to look at by hand
    python selftest_drive.py --folder ID  # run against an EXISTING folder,
                                          # read-only — for testing a folder a
                                          # teacher actually shared with us

Prerequisite: application-default credentials.  ./connect --adc
"""
from __future__ import annotations

import io
import sys
import time

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

import drive_folder
import main
from haskala_grading import report


def _typed_pdf():
    """Stands in for the exam paper: a text layer, no images."""
    import fitz
    doc = fitz.open()
    doc.new_page().insert_textbox(
        fitz.Rect(40, 40, 550, 780),
        "Module G — Writing Task\n\nWrite a composition of 120-140 words about a "
        "person who has influenced your life. Explain who they are and describe "
        "the influence they have had on you. " * 6,
        fontsize=11)
    b = doc.tobytes(); doc.close(); return b


def _scanned_pdf():
    """Stands in for a student's scan: one full-page image, no text layer."""
    import fitz
    doc = fitz.open()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 600, 800))
    pix.clear_with(250)
    doc.new_page().insert_image(fitz.Rect(0, 0, 600, 800), pixmap=pix)
    b = doc.tobytes(); doc.close(); return b


def _upload(drive, parent, name, data, mime):
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)
    return drive.files().create(
        body={"name": name, "parents": [parent]}, media_body=media,
        fields="id", supportsAllDrives=True).execute()["id"]


def _sample_docx():
    """A real report, built by the real builder — so the Drive conversion is
    tested on the actual bytes we ship, not on a toy file."""
    evaluation = {
        "criteria": [
            {"name": "Content and Organization", "score": 7, "max": 8,
             "feedback": "Well on topic and clearly organized."},
            {"name": "Vocabulary", "score": 8, "max": 10, "feedback": "Good range."},
            {"name": "Language Use", "score": 12, "max": 16,
             "feedback": "Some tense inconsistencies."},
            {"name": "Mechanics", "score": 5, "max": 6, "feedback": "Minor spelling."},
        ],
        "overall_score": "32/40",
        "overall_feedback": "A solid composition. Watch past-tense consistency.",
        "issues": [],
    }
    pages = [{"page": 1, "text": "My grandmother taught me patience. " * 15,
              "image_b64": None, "issues": []}]
    return report.build_evaluation_docx(
        evaluation, "selftest", "MoE Writing Rubric — Module G (16582)",
        pages=pages, feedback_lang="en", exercise_lang="en")


def main_():
    keep = "--keep" in sys.argv
    existing = None
    if "--folder" in sys.argv:
        existing = sys.argv[sys.argv.index("--folder") + 1]

    print("── credentials ──")
    creds = main.get_workspace_credentials()
    print(f"   impersonating : {main.WORKSPACE_SUBJECT}")
    print(f"   mail scope    : {'granted' if main._extra_scopes_granted else 'NOT granted'}")
    drive = build("drive", "v3", credentials=creds)

    who = drive.about().get(fields="user(emailAddress)").execute()
    print(f"   acting as     : {who['user']['emailAddress']}")

    if existing:
        print(f"\n── inspecting existing folder {existing} (read-only) ──")
        meta = drive_folder.check_access(drive, existing)
        print(f"   name          : {meta.get('name')!r}")
        print(f"   canAddChildren: {(meta.get('capabilities') or {}).get('canAddChildren')}")
        work, other = drive_folder.list_candidates(drive, existing)
        for f in work:
            data = drive_folder.download(drive, f["id"])
            typed, why = drive_folder.looks_typed(data, f["ext"])
            kind = "context document" if typed else "STUDENT WORK"
            print(f"   • {f['name']!r} → {kind}  ({why})")
        for f in other:
            print(f"   • {f['name']!r} → skipped ({f['reason']})")
        print("\n(no folder was created and nothing was written — read-only run)")
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = drive.files().create(
        body={"name": f"bdika-selftest-{stamp}",
              "mimeType": drive_folder.FOLDER_MIME},
        fields="id").execute()["id"]
    print(f"\n── scratch folder {root} ──")

    try:
        _upload(drive, root, "Module G exam paper.pdf", _typed_pdf(), "application/pdf")
        _upload(drive, root, "Rotem Cohen.pdf", _scanned_pdf(), "application/pdf")
        _upload(drive, root, "Noa Levi.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 64, "image/jpeg")
        _upload(drive, root, "teacher notes.txt", b"ignore me", "text/plain")
        print("   uploaded 4 test files")

        print("\n── check_access ──")
        meta = drive_folder.check_access(drive, root)
        caps = meta.get("capabilities") or {}
        print(f"   name           : {meta.get('name')!r}")
        print(f"   canListChildren: {caps.get('canListChildren')}")
        print(f"   canAddChildren : {caps.get('canAddChildren')}")

        print("\n── list_candidates ──")
        work, other = drive_folder.list_candidates(drive, root)
        for f in work:
            data = drive_folder.download(drive, f["id"])
            typed, why = drive_folder.looks_typed(data, f["ext"])
            print(f"   • {f['name']!r} → {'context document' if typed else 'STUDENT WORK'}")
            print(f"       {why}")
        for f in other:
            print(f"   • {f['name']!r} → skipped ({f['reason']})")

        print("\n── ensure_output_folder ──")
        out = drive_folder.ensure_output_folder(drive, root)
        again = drive_folder.ensure_output_folder(drive, root)
        print(f"   created  : {out}")
        print(f"   reused   : {again}  ({'✅ same' if out == again else '❌ DUPLICATED'})")

        print("\n── write_report (docx → Google Doc) ──")
        doc_id = drive_folder.write_report(
            drive, out, drive_folder.report_name("Rotem Cohen.pdf"), _sample_docx())
        written = drive.files().get(
            fileId=doc_id, fields="name, mimeType, webViewLink").execute()
        converted = written["mimeType"] == drive_folder.GDOC_MIME
        print(f"   name     : {written['name']!r}")
        print(f"   mimeType : {written['mimeType']}")
        print(f"   converted: {'✅ yes, a real Google Doc' if converted else '❌ NO — still a .docx'}")
        print(f"   open     : {written.get('webViewLink')}")

        if keep:
            print(f"\n📂 kept: {drive_folder.folder_link(root)}")
    finally:
        if not keep:
            # Trash, not delete: only the folder this run created, and trashing
            # is reversible from the Drive UI if something here was wrong.
            drive.files().update(fileId=root, body={"trashed": True}).execute()
            print(f"\n🧹 scratch folder {root} moved to trash")


if __name__ == "__main__":
    main_()
