import datetime
import json
import logging
import os
import sys
import time

import functions_framework

import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.auth import iam
from google.cloud import firestore
from googleapiclient.discovery import build

import checker
import core

# ─── logging ───────────────────────────────────────────────────────────────
# Everything lands on stdout, which Cloud Run forwards to Cloud Logging —
# read it with ./logs.sh (see that script for the useful filters).
#
# Two levels, deliberately: INFO is the story of a submission actually being
# picked up and graded, and stays quiet in steady state. DEBUG is the
# once-a-minute background chatter — polls that found nothing new, drafts not
# yet handed in, submissions already graded and skipped by the dedup ledger.
# At a 1-minute polling interval that chatter would otherwise bury the one
# trace worth reading. Set LOG_LEVEL=DEBUG on the service to see it all.


class _CloudLoggingFormatter(logging.Formatter):
    """Emits one JSON object per line. Cloud Run recognises that shape and
    lifts "severity" and "message" into the log entry itself.

    This is not cosmetic: printed as plain text, every line — DEBUG through
    ERROR alike — arrives in Cloud Logging as severity DEFAULT, so a
    `severity>=INFO` or `severity>=WARNING` query silently matches none of
    them. Structured output is what makes logs.sh's level filtering work."""

    def format(self, record):
        return json.dumps({
            "severity": record.levelname,
            "message": super().format(record),
            "logger": record.name,
        }, ensure_ascii=False)  # keep Hebrew readable rather than \uXXXX


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_CloudLoggingFormatter())
# force=True: checker/core are imported above and a library import may already
# have installed a default handler, which would make this call a silent no-op.
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                    handlers=[_handler], force=True)

# Third-party loggers that drown out our own trace. discovery_cache alone
# prints two useless lines per poll at INFO; the rest only bite at DEBUG,
# where every HTTP request and connection gets a line and the scan's own
# story becomes unreadable — which would defeat the point of LOG_LEVEL=DEBUG.
for _noisy in ("googleapiclient.discovery_cache", "googleapiclient.discovery",
               "urllib3.connectionpool", "google.auth", "google_auth_httplib2"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("oris-scanner-scan")

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
            log.debug("[subm %s] skip — already graded at %s", subm_id, data.get("completed_at"))
            return False
        claimed_at = data.get("claimed_at")
        if claimed_at is not None:
            age_minutes = (now - claimed_at).total_seconds() / 60
            if age_minutes < _STALE_CLAIM_MINUTES:
                log.debug("[subm %s] skip — another run claimed it %.1f min ago",
                          subm_id, age_minutes)
                return False  # another instance is working on it now, not abandoned
            log.info("[subm %s] reclaiming abandoned claim (%.1f min old, stale after %d)",
                     subm_id, age_minutes, _STALE_CLAIM_MINUTES)
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


def _get_or_create_checked_folder(drive_service, cw_id: str, anchor_file_id: str) -> str:
    """Returns the Drive folder id for this assignment's checked reports — a
    subfolder named CHECKED_FOLDER_NAME created once per courseWork, nested
    inside the submission's own folder (the one Classroom already created for
    this assignment, found via anchor_file_id's parent), and reused on every
    later run so all reports for the same assignment land together. The
    mapping is kept in Firestore (mirrors the submission dedup ledger) so
    every instance/run agrees on the same folder.

    Not fully race-proof: two overlapping runs discovering the same brand
    new assignment for the first time could each create a folder. Accepted —
    worst case is a harmless duplicate folder, not a correctness bug, and it
    only matters once per assignment's lifetime."""
    doc_ref = _firestore().collection("oris_scanner_coursework_folders").document(cw_id)
    snap = doc_ref.get()
    if snap.exists:
        folder_id = snap.to_dict()["folder_id"]
        log.info("reusing results folder %r (id=%s) for courseWork %s",
                 CHECKED_FOLDER_NAME, folder_id, cw_id)
        return folder_id

    anchor = drive_service.files().get(fileId=anchor_file_id, fields="parents").execute()
    parents = anchor.get("parents") or [FALLBACK_RESULTS_FOLDER_ID]
    if not anchor.get("parents"):
        log.warning("attachment %s has no resolvable parent — using fallback results folder %s",
                    anchor_file_id, FALLBACK_RESULTS_FOLDER_ID)

    metadata = {
        "name": CHECKED_FOLDER_NAME,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parents[0]],
    }
    folder = drive_service.files().create(body=metadata, fields="id").execute()
    folder_id = folder["id"]
    doc_ref.set({"folder_id": folder_id, "name": CHECKED_FOLDER_NAME})
    log.info("created results folder %r (id=%s) under %s for courseWork %s",
             CHECKED_FOLDER_NAME, folder_id, parents[0], cw_id)
    return folder_id


# ─── Google Workspace access (same as scan2: same service account + delegation) ──

def get_workspace_credentials():
    SCOPES = [
        'https://www.googleapis.com/auth/classroom.courses.readonly',
        'https://www.googleapis.com/auth/classroom.coursework.students',
        'https://www.googleapis.com/auth/classroom.topics.readonly',
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


# Checked reports now nest inside the assignment's own submission folder
# (resolved per-courseWork from a submitted file's parent — see
# _get_or_create_checked_folder), not a separate shared results tree.
# Updated per Ori's request, 2026-07-22.
CHECKED_FOLDER_NAME = "תרגילים בדוקים"

# Fallback parent, used only if a submitted file has no resolvable parent
# (unexpected for Classroom attachments, but cheap to guard against so a
# report is never silently dropped). Previously the sole results root:
# https://drive.google.com/drive/folders/1zzlOq6_UKZJUz33LvzRDqu5F9H-NCmE3
FALLBACK_RESULTS_FOLDER_ID = '1zzlOq6_UKZJUz33LvzRDqu5F9H-NCmE3'

# Per-assignment grading settings, all read off the courseWork itself:
#   rubric — the Classroom rubric attached to the assignment if there is one,
#            otherwise the bundled core.DEFAULT_RUBRIC_ID.
#   question — the assignment's free-text description.
#   model  — a [model: <key>] tag inside that description; see
#            core.resolve_model for the syntax and the accepted keys.
#
# Scope: every courseWork item, in any course the delegated account
# (teacher_email above) teaches, whose Classroom "topic" is named
# TARGET_TOPIC_NAME. Replaces the earlier TARGET_COURSE_NAME course-level
# restriction (2026-07-22 stopgap, since the "integration" course mixed
# language and geometry assignments with no other way to tell them apart) —
# this is the real subject/language filter that stopgap's TODO called for.
TARGET_TOPIC_NAME = "English Assignment"


def _get_target_topic_id(classroom_service, course_id: str, course_name: str = ""):
    """Returns the topicId of TARGET_TOPIC_NAME within this course, or None
    if the course has no topic by that name (in which case none of its
    courseWork can match, so the caller skips the course entirely)."""
    res = classroom_service.courses().topics().list(courseId=course_id).execute()
    names = [t.get('name') for t in res.get('topic', [])]
    for topic in res.get('topic', []):
        if topic.get('name') == TARGET_TOPIC_NAME:
            log.debug("course %r: topic %r found (id=%s)",
                      course_name, TARGET_TOPIC_NAME, topic.get('topicId'))
            return topic.get('topicId')
    # The single most common reason a submission is never picked up, so it is
    # logged with the topics that *do* exist rather than a bare "no match".
    log.info("course %r: no topic named %r — skipping. Topics in this course: %s",
             course_name, TARGET_TOPIC_NAME, names or "(none)")
    return None


def _get_classroom_rubric(classroom_service, course_id: str, cw_id: str) -> dict | None:
    """Returns the Rubric object (courses.courseWork.rubrics) attached to
    this assignment, or None if the teacher didn't attach one. Classroom
    allows at most one rubric per courseWork. No extra scope needed —
    rubrics.list is covered by the classroom.coursework.students scope
    already requested in get_workspace_credentials()."""
    res = classroom_service.courses().courseWork().rubrics().list(
        courseId=course_id, courseWorkId=cw_id).execute()
    rubrics = res.get('rubrics', [])
    return rubrics[0] if rubrics else None


def run_workspace_scan():
    t0 = time.monotonic()
    graded = 0

    creds = get_workspace_credentials()
    drive_service = build('drive', 'v3', credentials=creds)
    classroom_service = build('classroom', 'v1', credentials=creds)

    res_courses = classroom_service.courses().list(pageSize=100, teacherId='me').execute()
    courses = res_courses.get('courses', [])
    log.debug("scan start — %d course(s) taught by the delegated account, looking for topic %r",
              len(courses), TARGET_TOPIC_NAME)

    for course in courses:
        course_name = course.get('name')
        target_topic_id = _get_target_topic_id(classroom_service, course.get('id'), course_name)
        if target_topic_id is None:
            continue  # this course has no "English Assignment" topic at all

        res_cw = classroom_service.courses().courseWork().list(courseId=course.get('id')).execute()
        all_cw = res_cw.get('courseWork', [])
        course_work = [cw for cw in all_cw if cw.get('topicId') == target_topic_id]
        log.debug("course %r: %d of %d assignments are under topic %r",
                  course_name, len(course_work), len(all_cw), TARGET_TOPIC_NAME)

        for cw in course_work:
            # Same for every student on this assignment — fetched once per
            # courseWork, not per submission.
            classroom_rubric = _get_classroom_rubric(classroom_service, course.get('id'), cw.get('id'))

            # The teacher picks the grading model with a [model: …] tag in the
            # assignment description; resolve_model strips it back out so the
            # tag itself never reaches the grader as part of the question.
            model_key, assignment_description, model_warning = core.resolve_model(cw.get('description'))
            if model_warning:
                log.warning("assignment %r: %s", cw.get('title'), model_warning)

            # A rubric may instead be attached as an ordinary document. Only
            # identified here (a name match, no network); actually downloading
            # and reading it is deferred until there is a submission to grade,
            # because this scan runs every minute and the vast majority of
            # runs have nothing to do.
            rubric_material, assignment_description, rubric_warning = core.resolve_rubric_material(
                assignment_description, cw.get('materials'))
            if rubric_warning:
                log.warning("assignment %r: %s", cw.get('title'), rubric_warning)
            material_rubric = None
            material_rubric_loaded = False

            res_sub = classroom_service.courses().courseWork().studentSubmissions().list(
                courseId=course.get('id'), courseWorkId=cw.get('id')).execute()
            subms = res_sub.get('studentSubmissions', [])
            if not subms:
                log.debug("assignment %r: no submissions yet", cw.get('title'))
                continue  # no submissions yet — don't create a results folder for nothing

            eligible = [s for s in subms if s.get('state') in ('TURNED_IN', 'RETURNED')]
            log.debug("assignment %r: %d submission(s), %d handed in — states: %s",
                      cw.get('title'), len(subms), len(eligible),
                      ", ".join(sorted({s.get('state') or '?' for s in subms})))

            for subm in subms:
                # Attachments show up on the submission (and land in the
                # assignment's Drive folder) as soon as the student attaches
                # a file — well before they press "Hand in". Only TURNED_IN
                # (and RETURNED, e.g. a resubmission after the teacher
                # reopened it) reflects an actual formal submission; NEW/
                # CREATED/RECLAIMED_BY_STUDENT are drafts. Skip without
                # claiming, so once the student does submit, the next poll
                # picks it up normally.
                if subm.get('state') not in ('TURNED_IN', 'RETURNED'):
                    log.debug("[subm %s] skip — state %s, not handed in yet",
                              subm.get('id'), subm.get('state'))
                    continue

                subm_id = subm.get('id')
                if not _try_claim_submission(subm_id):
                    continue  # dedup: already handled, or another instance is on it right now

                tag = f"[subm {subm_id}]"
                log.info("%s detected: course=%r assignment=%r state=%s student=%s late=%s",
                         tag, course_name, cw.get('title'), subm.get('state'),
                         subm.get('userId'), subm.get('late', False))

                file_ids = []
                attachments = (subm.get('assignmentSubmission') or {}).get('attachments', [])
                for attachment in attachments:
                    drive_file = attachment.get('driveFile')
                    if drive_file:
                        file_ids.append(drive_file.get('id'))

                if not file_ids:
                    log.info("%s no Drive attachments (%d attachment(s) of other kinds) — "
                             "marking done, nothing to grade", tag, len(attachments))
                    _mark_done(subm_id)
                    continue  # nothing to check, and no attachment to anchor a folder on

                log.info("%s %d attachment(s) to check — starting", tag, len(file_ids))

                # Read once per assignment per scan, not once per student:
                # the rubric document is identical for all of them, and
                # reading it can cost a vision call per page.
                if rubric_material and not material_rubric_loaded:
                    material_rubric = checker.load_material_rubric(
                        drive_service, rubric_material, model_key, tag)
                    material_rubric_loaded = True

                cw_folder_id = _get_or_create_checked_folder(drive_service, cw.get('id'), file_ids[0])
                checker.check_hw(
                    drive_service, file_ids, cw_folder_id,
                    classroom_rubric=classroom_rubric,
                    assignment_description=assignment_description,
                    assignment_name=cw.get('title'),
                    model_key=model_key,
                    tag=tag,
                    material_rubric=material_rubric,
                )
                _mark_done(subm_id)
                graded += 1
                log.info("%s done", tag)

    level = log.info if graded else log.debug
    level("scan finished in %.1fs — %d submission(s) graded", time.monotonic() - t0, graded)


@functions_framework.http
def process_my_drive_files(request):
    try:
        run_workspace_scan()
    except Exception:
        # Cloud Scheduler only ever sees the status code, so an unhandled
        # error would otherwise leave nothing but a 500 to debug from.
        log.exception("scan failed")
        raise
    return ("ok", 200)


if __name__ == '__main__':
    run_workspace_scan()
