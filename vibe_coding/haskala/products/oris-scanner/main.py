import datetime

import functions_framework

import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.auth import iam
from google.cloud import firestore
from googleapiclient.discovery import build

import checker

# ─── dedup ledger (Firestore) ──────────────────────────────────────────────────
# This is the main architectural fix versus scan2: there, run_workspace_scan()
# reprocessed every submission on every trigger (no dedup), and every Pub/Sub
# redelivery (because the ack deadline was too short) ran another full scan —
# a chain that burned hundreds of thousands of requests.
#
# Here: before processing a submission, atomically "claim" it in Firestore
# (create fails if it already exists). If it's already "done" — skip (this is
# the dedup itself). If it's "in_progress" and fresh — another instance is
# probably handling it right now, skip. If "in_progress" but stale (a claim
# stuck from a crashed run) — reclaim it to allow a genuine retry.
_STALE_CLAIM_MINUTES = 20  # beyond this, a stuck claim is considered abandoned and reclaimable

_db = None


def _firestore():
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


def _try_claim_submission(subm_id: str) -> bool:
    """Returns True if the submission should be processed now (either it
    wasn't handled yet, or an old claim was abandoned), False if it should be
    skipped (already done, or another instance is working on it right now)."""
    doc_ref = _firestore().collection("oris_scanner_submissions").document(subm_id)
    now = datetime.datetime.now(datetime.timezone.utc)

    @firestore.transactional
    def _txn(transaction):
        snap = doc_ref.get(transaction=transaction)
        if not snap.exists:
            transaction.set(doc_ref, {"status": "in_progress", "claimed_at": now})
            return True
        data = snap.to_dict() or {}
        if data.get("status") == "done":
            return False
        claimed_at = data.get("claimed_at")
        if claimed_at is not None:
            age_minutes = (now - claimed_at).total_seconds() / 60
            if age_minutes < _STALE_CLAIM_MINUTES:
                return False  # another instance is working on it now, not abandoned
        # stale/abandoned claim — reclaim it to allow a genuine retry
        transaction.set(doc_ref, {"status": "in_progress", "claimed_at": now})
        return True

    return _txn(_firestore().transaction())


def _mark_done(subm_id: str):
    doc_ref = _firestore().collection("oris_scanner_submissions").document(subm_id)
    doc_ref.set({
        "status": "done",
        "completed_at": datetime.datetime.now(datetime.timezone.utc),
    }, merge=True)


def _get_or_create_coursework_folder(drive_service, cw_id: str, folder_name: str) -> str:
    """Returns the Drive folder id for this assignment's results — a
    subfolder of RESULTS_FOLDER_ID created once per courseWork and reused on
    every later run, so all reports for the same assignment land together.
    The mapping is kept in Firestore (mirrors the submission dedup ledger)
    so every instance/run agrees on the same folder.

    Not fully race-proof: two overlapping runs discovering the same brand
    new assignment for the first time could each create a folder. Accepted —
    worst case is a harmless duplicate folder, not a correctness bug, and it
    only matters once per assignment's lifetime."""
    doc_ref = _firestore().collection("oris_scanner_coursework_folders").document(cw_id)
    snap = doc_ref.get()
    if snap.exists:
        return snap.to_dict()["folder_id"]

    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [RESULTS_FOLDER_ID],
    }
    folder = drive_service.files().create(body=metadata, fields="id").execute()
    folder_id = folder["id"]
    doc_ref.set({"folder_id": folder_id, "name": folder_name})
    return folder_id


# ─── Google Workspace access (same as scan2: same service account + delegation) ──

def get_workspace_credentials():
    SCOPES = [
        'https://www.googleapis.com/auth/classroom.courses.readonly',
        'https://www.googleapis.com/auth/classroom.coursework.students',
        'https://www.googleapis.com/auth/drive'
    ]
    source_creds, project_id = google.auth.default()
    source_creds.refresh(Request())
    sa_email = "sainter@master-gecko-500709-t0.iam.gserviceaccount.com"
    teacher_email = "ori@bdika.net"
    signer = iam.Signer(Request(), source_creds, sa_email)
    delegated_creds = service_account.Credentials(
        signer,
        sa_email,
        token_uri="https://oauth2.googleapis.com/token",
        subject=teacher_email,
        scopes=SCOPES
    )
    return delegated_creds


# Dedicated results folder for oris-scanner (no longer the hardcoded folder
# shared with scan2/math) — updated per Ori's request, 2026-07-20:
# https://drive.google.com/drive/folders/1zzlOq6_UKZJUz33LvzRDqu5F9H-NCmE3
RESULTS_FOLDER_ID = '1zzlOq6_UKZJUz33LvzRDqu5F9H-NCmE3'

# Grading always uses core.DEFAULT_RUBRIC_ID (a fixed bundled rubric, not the
# courseWork's free-text description/maxPoints) — see core.check_pages.
#
# Scope: every courseWork item in the "integration" course only, where the
# delegated account (teacher_email above) is the teacher. Temporary
# narrowing (2026-07-22) — that course mixes language and non-language
# (geometry) assignments with no subject metadata to tell them apart yet, so
# for now we cap the blast radius to this one known course instead of every
# course the teacher has. TODO: real subject/language filter, then drop this.
TARGET_COURSE_NAME = "integration"


def run_workspace_scan():
    creds = get_workspace_credentials()
    drive_service = build('drive', 'v3', credentials=creds)
    classroom_service = build('classroom', 'v1', credentials=creds)

    res_courses = classroom_service.courses().list(pageSize=100, teacherId='me').execute()
    courses = [c for c in res_courses.get('courses', []) if c.get('name') == TARGET_COURSE_NAME]
    for course in courses:
        res_cw = classroom_service.courses().courseWork().list(courseId=course.get('id')).execute()
        course_work = res_cw.get('courseWork', [])
        for cw in course_work:
            res_sub = classroom_service.courses().courseWork().studentSubmissions().list(
                courseId=course.get('id'), courseWorkId=cw.get('id')).execute()
            subms = res_sub.get('studentSubmissions', [])
            if not subms:
                continue  # no submissions yet — don't create a results folder for nothing

            cw_folder_id = _get_or_create_coursework_folder(
                drive_service, cw.get('id'), f"{course.get('name')} — {cw.get('title')}")

            for subm in subms:
                subm_id = subm.get('id')
                if not _try_claim_submission(subm_id):
                    continue  # dedup: already handled, or another instance is on it right now

                file_ids = []
                attachments = (subm.get('assignmentSubmission') or {}).get('attachments', [])
                for attachment in attachments:
                    drive_file = attachment.get('driveFile')
                    if drive_file:
                        file_ids.append(drive_file.get('id'))

                checker.check_hw(drive_service, file_ids, cw_folder_id)
                _mark_done(subm_id)


@functions_framework.http
def process_my_drive_files(request):
    run_workspace_scan()
    return ("ok", 200)


if __name__ == '__main__':
    run_workspace_scan()
