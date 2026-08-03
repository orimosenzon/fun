"""main.py — form-checker: grade a shared folder of exercises via a Google Form.

A teacher fills in a Google Form — naming the Bagrut module and pasting a link
to a Drive folder of scanned work — and the reports appear back inside that same
folder, in a subfolder called Bdika. This service is what happens in between.

Cloud Scheduler calls this every 5 minutes. Each run reads the Form's responses
spreadsheet, claims any row it has not already handled, grades every piece of
student work in the linked folder with the shared haskala_grading pipeline, and
writes one Google Doc per file back into the folder.

WHY A SHARED FOLDER AND NOT FILE UPLOADS
────────────────────────────────────────
Both reasons are the teacher's. Results land where they already keep the work
rather than in an inbox they have to file by hand, and a whole class is one
submission instead of thirty. It also sidesteps the prerequisite that used to
sink this design: Forms stores uploads in the *form owner's* Drive, so a form
built in the wrong Workspace put every file somewhere this service could not
see, with no fix short of rebuilding the form. A folder shared directly with our
service account has no such constraint — it works no matter who owns the form.

What it costs is control over the input. An upload question restricted to
PDF/JPG/PNG guaranteed the shape of what arrived; a folder guarantees nothing,
which is why drive_folder classifies before anything is graded and why the caps
below are enforced against the folder rather than trusted from it.

THE ONE HARD PREREQUISITE
─────────────────────────
The folder must be shared with WORKSPACE_SUBJECT as an EDITOR — Viewer is not
enough, because results are written back in. drive_folder.check_access checks
this before any page is paid for and mails the teacher if it fails.

The form also needs "allow responses from outside the organization" turned on,
or teachers at other schools cannot submit at all.
"""
import datetime
import hashlib
import io
import json
import logging
import os
import sys
import time

import functions_framework

import google.auth
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2 import service_account
from google.auth import iam
from google.cloud import firestore
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

import checker
import drive_folder
import form_schema
import mailer

# ─── logging ───────────────────────────────────────────────────────────────
# Ported from oris-scanner. Everything lands on stdout, which Cloud Run
# forwards to Cloud Logging — read it with ./logs.sh.
#
# Two levels, deliberately: INFO is the story of a response actually being
# graded and stays quiet in steady state. DEBUG is the every-5-minutes
# background chatter — polls that found nothing new, rows already graded and
# skipped by the ledger.


class _CloudLoggingFormatter(logging.Formatter):
    """Emits one JSON object per line. Cloud Run recognises that shape and
    lifts "severity" and "message" into the log entry itself.

    Not cosmetic: printed as plain text, every line — DEBUG through ERROR
    alike — arrives in Cloud Logging as severity DEFAULT, so a `severity>=INFO`
    query silently matches none of them. Structured output is what makes
    logs.sh's level filtering work."""

    def format(self, record):
        return json.dumps({
            "severity": record.levelname,
            "message": super().format(record),
            "logger": record.name,
        }, ensure_ascii=False)  # keep Hebrew readable rather than \uXXXX


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_CloudLoggingFormatter())
# force=True: the modules above are imported first and a library import may
# already have installed a default handler, which would make this a no-op.
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                    handlers=[_handler], force=True)

for _noisy in ("googleapiclient.discovery_cache", "googleapiclient.discovery",
               "urllib3.connectionpool", "google.auth", "google_auth_httplib2"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("form-checker")


# ─── configuration ─────────────────────────────────────────────────────────

# The Form's linked responses spreadsheet. Set at deploy time; the id is the
# long token in the sheet's URL between /d/ and /edit.
RESPONSES_SHEET_ID = os.environ.get("RESPONSES_SHEET_ID", "")

# Where reports go when the teacher's own folder cannot be written to. Almost
# never used now that results go back into the shared folder — check_access
# rejects a read-only folder up front rather than grading it and then looking
# for somewhere to put the results — but a graded result is never discarded.
FALLBACK_RESULTS_FOLDER_ID = os.environ.get("FALLBACK_RESULTS_FOLDER_ID", "")

# ─── cost guardrails ───────────────────────────────────────────────────────
# A form open to teachers is an open door to spend, and this project has twice
# paid for the lesson: ~$20 burned in July when run_workspace_scan had no
# dedup, and ~$26 on a Pub/Sub retry storm. Everything below exists to bound
# what one response, or one bad actor, can cost.
#
# The dominant cost is per page: rotation detection + OCR + a share of the
# evaluation call, roughly $0.04–0.06 per page on Claude Sonnet 5. So the page
# cap, not the file cap, is the one that actually bounds the bill.
#
# THE ONE THAT BOUNDS THE BILL, while we are still testing.
#
# At most this many pieces of student work are graded per folder — the rest are
# left for later and the teacher is told so. A *limit*, not a refusal: a teacher
# who shares a class of thirty gets three reports and an explanation, not a
# rejection. Refusing would be the wrong shape now that one folder is a whole
# class, since there is nothing for them to fix.
#
# Set to 3 for the pilot. Raise it once the grades have been eyeballed and the
# cost per work is known for real rather than estimated.
#
# It composes with the per-file ledger to give a useful property for free:
# resubmitting the same folder grades the NEXT three, because the first three
# are already recorded as done. So a class of thirty can be worked through
# deliberately, three at a time, without any extra machinery.
MAX_WORKS_PER_FOLDER = int(os.environ.get("MAX_WORKS_PER_FOLDER", "3"))

# These two are sanity bounds, not cost bounds — MAX_WORKS_PER_FOLDER above is
# what actually limits spend now. Listing and downloading cost bandwidth and a
# few seconds, not model calls, so the file limit is set where "this is not a
# class folder, something is wrong" becomes true rather than where the money is.
# Keeping it at a class-sized 15 would refuse a real class of thirty outright,
# which is the one thing this design is supposed to make unnecessary.
MAX_FILES_PER_RESPONSE = int(os.environ.get("MAX_FILES_PER_RESPONSE", "60"))
# Applied to the works actually selected for grading, so it never refuses a
# folder over pages we were not going to read anyway.
MAX_PAGES_PER_RESPONSE = int(os.environ.get("MAX_PAGES_PER_RESPONSE", "40"))

# Rows processed in one invocation. A backlog is drained over several polls
# rather than in one run that blows the Scheduler's 600s attempt deadline and
# gets retried from the top — the ledger makes the partial progress safe.
MAX_ROWS_PER_RUN = int(os.environ.get("MAX_ROWS_PER_RUN", "5"))

# Who may spend our model budget. Both are unset by default, which allows
# anyone with the form link — fine for a closed pilot, and logged loudly at
# startup so it is a decision rather than an oversight.
#   FORM_CHECKER_ALLOWED_DOMAINS — comma-separated, e.g. "bdika.net,shamir.org.il"
#   FORM_CHECKER_PASSPHRASE — a short code on the form, shared with the teachers
ALLOWED_DOMAINS = [d.strip().lower() for d in
                   os.environ.get("FORM_CHECKER_ALLOWED_DOMAINS", "").split(",") if d.strip()]
PASSPHRASE = os.environ.get("FORM_CHECKER_PASSPHRASE", "").strip()


# ─── dedup ledger (Firestore) ──────────────────────────────────────────────
# Ported wholesale from oris-scanner, where it is the fix for the incident that
# produced ~585,000 requests over six days. The reasoning carries over exactly:
# before processing a row, atomically claim it. Already done → skip. Claimed and
# fresh → another instance has it. Claimed and stale → a crashed run, reclaim.
#
# One difference. oris-scanner keys on the Classroom submission id and versions
# by updateTime, because a submission's id is stable while its content changes.
# A spreadsheet row has no id at all, so the key *is* the content hash: an
# edited response hashes differently, becomes a new ledger entry, and is graded
# again — which is the behaviour we want and gets it without a version field.
_STALE_CLAIM_MINUTES = 20   # beyond this, a stuck claim is abandoned and reclaimable

# Without a ceiling, a permanently-failing row is reclaimed every 20 minutes
# forever, each attempt paying for a full OCR pass. That is not hypothetical:
# on 2026-07-28 one submission OOM-killed oris-scanner's container ~40 times
# overnight, and at Sonnet 5 rates would have cost ~$1.70 on a single file,
# unattended. Past this ceiling the row is parked instead.
_MAX_GRADING_ATTEMPTS = 3

_db = None


def _firestore():
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


def row_key(params: dict) -> str:
    """Stable id for one response: a hash of the fields that define it.

    Timestamp and email identify the submission; the file list is included so
    that a teacher who edits their response to swap the attachment is treated
    as a new submission rather than silently skipped as already-graded.

    Not the row number — inserting or deleting a row above shifts every number
    below it, which would re-grade the entire sheet."""
    material = "|".join([
        params.get("timestamp", ""),
        params.get("email", "").lower(),
        params.get("folder_id", ""),
        ",".join(params.get("file_ids", [])),
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def file_key(params: dict, f: dict) -> str:
    """Stable id for "this file, graded this way".

    The row ledger alone is not enough once a response is a folder. A folder is
    live: the teacher adds five more scans on Sunday and submits the form again,
    which is a new row with a new timestamp and therefore a new row key. Without
    a second ledger at file level, that second submission re-grades and re-pays
    for all thirty of the originals to produce five new reports.

    Keyed on the file's own id and modifiedTime, so an edited or rescanned file
    is correctly treated as new work — and on the rubric, so a teacher who
    realises they picked Module C instead of Module G can fix the dropdown,
    resubmit, and actually get regraded. Deliberately NOT keyed on the row: the
    whole point is that it survives across submissions of the same folder.
    """
    material = "|".join([
        params.get("folder_id", ""),
        f.get("id", ""),
        f.get("modifiedTime", ""),
        params.get("rubric_id") or "",
        params.get("module") or "",
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _file_already_graded(key: str) -> bool:
    snap = _firestore().collection("form_checker_files").document(key).get()
    return bool(snap.exists and (snap.to_dict() or {}).get("status") == "done")


def _mark_file_done(key: str, folder_id: str, file_id: str, report_id: str = ""):
    _firestore().collection("form_checker_files").document(key).set({
        "status": "done",
        "folder_id": folder_id,
        "file_id": file_id,
        "report_id": report_id,
        "completed_at": datetime.datetime.now(datetime.timezone.utc),
    })


def _try_claim_row(key: str) -> bool:
    """True if this row should be processed now — never handled, or an
    abandoned claim worth retrying — and False if it should be skipped."""
    doc_ref = _firestore().collection("form_checker_responses").document(key)
    now = datetime.datetime.now(datetime.timezone.utc)

    # Decided inside the transaction, logged outside it: a transaction body may
    # be retried on contention, which would emit the line more than once.
    @firestore.transactional
    def _txn(transaction):
        snap = doc_ref.get(transaction=transaction)

        def claim_at(attempt: int) -> dict:
            return {"status": "in_progress", "claimed_at": now, "attempts": attempt}

        if not snap.exists:
            transaction.set(doc_ref, claim_at(1))
            return True, None
        data = snap.to_dict() or {}

        if data.get("status") == "failed":
            return False, ("debug", "parked after %d failed attempt(s) — edit the "
                           "response to retry" % (data.get("attempts") or _MAX_GRADING_ATTEMPTS))

        if data.get("status") == "done":
            return False, ("debug", "already graded at %s" % data.get("completed_at"))

        attempts = data.get("attempts") or 1
        claimed_at = data.get("claimed_at")
        if claimed_at is not None:
            age_minutes = (now - claimed_at).total_seconds() / 60
            if age_minutes < _STALE_CLAIM_MINUTES:
                return False, ("debug", "another run claimed it %.1f min ago" % age_minutes)
            if attempts >= _MAX_GRADING_ATTEMPTS:
                transaction.set(doc_ref, {"status": "failed", "failed_at": now,
                                          "attempts": attempts}, merge=True)
                return False, ("error",
                               "giving up after %d attempt(s) that never completed — "
                               "parking it. Something makes this response fail every time "
                               "(check for an out-of-memory kill)." % attempts)
            transaction.set(doc_ref, claim_at(attempts + 1))
            return True, ("info",
                          "reclaiming abandoned claim (%.1f min old, stale after %d) — "
                          "attempt %d of %d"
                          % (age_minutes, _STALE_CLAIM_MINUTES, attempts + 1,
                             _MAX_GRADING_ATTEMPTS))
        if attempts >= _MAX_GRADING_ATTEMPTS:
            transaction.set(doc_ref, {"status": "failed", "failed_at": now,
                                      "attempts": attempts}, merge=True)
            return False, ("error", "giving up after %d attempt(s) that never completed"
                           % attempts)
        transaction.set(doc_ref, claim_at(attempts + 1))
        return True, None

    should_process, note = _txn(_firestore().transaction())
    if note:
        level, message = note
        getattr(log, level)("[row %s] %s", key[:8], message)
    return should_process


def _mark_done(key: str, outcome: str = "graded"):
    _firestore().collection("form_checker_responses").document(key).set({
        "status": "done",
        "outcome": outcome,
        "completed_at": datetime.datetime.now(datetime.timezone.utc),
    }, merge=True)


# ─── Google Workspace access ───────────────────────────────────────────────
# Same service account and same domain-wide delegation entry as oris-scanner.

# Scopes this service cannot do its job without.
BASE_SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',  # read the responses sheet
    'https://www.googleapis.com/auth/drive',                  # download the uploads
]

# Delivery. Kept separate from BASE_SCOPES because domain-wide delegation is
# ALL-OR-NOTHING per token request — see get_workspace_credentials. Without this
# scope the service still grades and still writes reports to Drive; it just
# cannot mail them. That degradation is the entire reason for the split.
EXTRA_SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',
]

SA_EMAIL = "sainter@master-gecko-500709-t0.iam.gserviceaccount.com"

# The Workspace user this service impersonates: whose Drive it reads, and who
# outgoing mail is from. It is also the address teachers must share their folder
# with, which is why it is worth having a dedicated account rather than a
# person's — the address ends up printed on the form, in every error message,
# and in whatever teachers write down. A personal address there is impossible to
# change later without breaking every form already in circulation.
#
# Overridable so the switch to that dedicated account is a redeploy rather than
# a code change; until it is provisioned in Workspace, the default keeps the
# existing delegation working. Whatever it is set to must be listed as the
# subject the service account may impersonate.
WORKSPACE_SUBJECT = os.environ.get("WORKSPACE_SUBJECT", "ori@bdika.net")

# What teachers are told to share with. Normally the same address; kept separate
# because a Workspace alias or group can front the real mailbox. drive_folder
# reads it from the environment so its teacher-facing messages name it too.
SHARE_WITH = os.environ.get("SHARE_WITH", WORKSPACE_SUBJECT)
os.environ["SHARE_WITH"] = SHARE_WITH
# The service account's OAuth client id — the key the Workspace admin console's
# domain-wide-delegation entry is filed under. Logged with the scope warning so
# whoever reads it can go straight to the right row.
SA_CLIENT_ID = "114647217076557059736"

# Per-instance memo of whether the delegation entry actually grants
# EXTRA_SCOPES. None = not yet determined. A denial is remembered only for
# EXTRA_RECHECK_SECONDS, not forever: that expiry is what makes an
# admin-console fix land without a redeploy.
_extra_scopes_granted: bool | None = None
_extra_scopes_checked_at: float = 0.0
EXTRA_RECHECK_SECONDS = 600


def _delegated_credentials(scopes):
    source_creds, _project_id = google.auth.default()
    source_creds.refresh(Request())
    signer = iam.Signer(Request(), source_creds, SA_EMAIL)
    return service_account.Credentials(
        signer,
        SA_EMAIL,
        token_uri="https://oauth2.googleapis.com/token",
        subject=WORKSPACE_SUBJECT,
        scopes=scopes,
    )


def get_workspace_credentials():
    """Delegated credentials for the Workspace account that owns the form.

    Requests EXTRA_SCOPES on top of BASE_SCOPES, and drops them if the
    domain-wide-delegation entry does not grant them.

    That fallback is the whole point of this function. Domain-wide delegation is
    all-or-nothing per token request: if the service account's entry in the
    Workspace admin console does not list *every* scope asked for, the token
    endpoint rejects the entire request with `unauthorized_client` — it does not
    hand back the authorized subset. Listing two extra scopes took oris-scanner
    down completely on 2026-07-30 (revision 00030): no token, no API call,
    nothing graded. Email delivery must never be able to do that to grading,
    hence the two-tier request.

    Note this is per *token request*, so form-checker asking for a scope
    oris-scanner does not have cannot affect oris-scanner. Adding scopes to the
    shared delegation entry is likewise additive and safe.
    """
    global _extra_scopes_granted, _extra_scopes_checked_at

    stale = (time.monotonic() - _extra_scopes_checked_at) >= EXTRA_RECHECK_SECONDS
    if _extra_scopes_granted is not False or stale:
        creds = _delegated_credentials(BASE_SCOPES + EXTRA_SCOPES)
        was = _extra_scopes_granted
        try:
            # Refresh eagerly: the failure has to surface here, where it can be
            # retried without the extra scopes, rather than at whichever API
            # call happens to be first.
            creds.refresh(Request())
        except RefreshError as e:
            _extra_scopes_granted = False
            _extra_scopes_checked_at = time.monotonic()
            if was is not False:
                # Only on a change of state: at one poll every 5 minutes this
                # would otherwise repeat the same paragraph into the log all day.
                log.warning(
                    "delegation does not grant the mail scope (%s) — continuing "
                    "without it; reports will be written to Drive instead of "
                    "emailed, and teachers will not be told they are ready. To fix, "
                    "add %s to the service account's domain-wide delegation in the "
                    "Workspace admin console (client id %s). Rechecked every %ds, so "
                    "no redeploy is needed once it is there.",
                    str(e)[:200], ", ".join(EXTRA_SCOPES), SA_CLIENT_ID,
                    EXTRA_RECHECK_SECONDS)
        else:
            _extra_scopes_granted = True
            _extra_scopes_checked_at = time.monotonic()
            if was is not True:
                log.info("delegation grants the mail scope — reports will be emailed")
            return creds

    creds = _delegated_credentials(BASE_SCOPES)
    creds.refresh(Request())
    return creds


# ─── admission control ─────────────────────────────────────────────────────

def _admit(params: dict, tag: str) -> str | None:
    """None if this response may be graded, otherwise the reason it may not.

    A returned reason is written for a teacher to read: everything except the
    access checks is emailed back to them. The access checks deliberately say
    nothing to anyone — replying would turn the form into a way to send mail to
    an arbitrary address, and telling a would-be abuser which check they failed
    only helps them pass it.
    """
    if not params.get("email"):
        return "__silent__ no email address on the row"

    if ALLOWED_DOMAINS:
        domain = params["email"].rsplit("@", 1)[-1].lower()
        if domain not in ALLOWED_DOMAINS:
            log.warning("%s rejecting %s — domain %r is not in FORM_CHECKER_ALLOWED_DOMAINS",
                        tag, params["email"], domain)
            return "__silent__ domain not allowed"

    if PASSPHRASE and params.get("passphrase", "").strip() != PASSPHRASE:
        log.warning("%s rejecting %s — wrong or missing passphrase", tag, params["email"])
        return "__silent__ bad passphrase"

    if not params.get("folder_id"):
        if params.get("folder_link"):
            return ("לא הצלחנו לזהות קישור לתיקייה בתשובה %r. "
                    "יש להעתיק את הקישור מכפתור השיתוף של התיקייה ב-Drive, "
                    "בצורה https://drive.google.com/drive/folders/…"
                    % params["folder_link"][:120])
        return ("לא צוין קישור לתיקייה בטופס, ולכן אין מה לבדוק. "
                "יש לשתף תיקייה עם %s בהרשאת עריכה ולהדביק את הקישור אליה."
                % SHARE_WITH)

    return None


# ─── the scan ──────────────────────────────────────────────────────────────

def _read_responses(sheets_service) -> tuple[list[str], list[list[str]]]:
    """(header row, data rows) from the form's responses sheet.

    No sheet name in the range, so this reads the first sheet — which is where
    Forms puts responses. A:ZZ rather than A:Z because a form with more than 26
    questions is entirely plausible and the failure would be a silently missing
    column."""
    resp = sheets_service.spreadsheets().values().get(
        spreadsheetId=RESPONSES_SHEET_ID, range="A:ZZ").execute()
    values = resp.get("values", [])
    if not values:
        return [], []
    return values[0], values[1:]


def _fallback_write(drive_service, name, docx_bytes, tag) -> str:
    """Last resort when the teacher's own folder cannot be written to.

    Rare by construction — check_access rejects a read-only folder before
    anything is graded — so this is for the case where the sharing is revoked
    mid-run. The report is already paid for; parking it somewhere we control
    costs nothing and keeps it recoverable by hand.
    """
    if not FALLBACK_RESULTS_FOLDER_ID:
        log.error("%s report %r could not be written to the teacher's folder and no "
                  "FALLBACK_RESULTS_FOLDER_ID is configured — the report is lost. "
                  "Set that env var.", tag, name)
        return ""
    media = MediaIoBaseUpload(io.BytesIO(docx_bytes), mimetype=drive_folder.DOCX_MIME)
    uploaded = drive_service.files().create(
        body={"name": name, "parents": [FALLBACK_RESULTS_FOLDER_ID]},
        media_body=media, fields="id").execute()
    log.warning("%s report %r parked in the fallback folder (id=%s) — the teacher "
                "cannot see it there", tag, name, uploaded["id"])
    return uploaded["id"]


def _collect_folder(drive_service, params, tag):
    """Download and sort one folder's contents into (student work, context docs).

    Downloading before classifying is deliberate: the text-layer test needs the
    bytes, and a download is cents-free next to a page of OCR. What it must not
    do is download an unbounded folder, so the file cap is applied to the listing
    first — before any bytes move.

    Raises drive_folder.FolderError with a teacher-readable message.
    """
    meta = drive_folder.check_access(drive_service, params["folder_id"])
    candidates, wrong_format = drive_folder.list_candidates(
        drive_service, params["folder_id"])

    if not candidates:
        raise drive_folder.FolderError(
            "התיקייה %r לא מכילה קבצים שאפשר לבדוק. המערכת קוראת PDF, JPG ו-PNG. "
            "אם צילמת באייפון, הקבצים כנראה בפורמט HEIC — אפשר לשמור אותם כ-JPG "
            "ולשתף שוב." % meta.get("name"))

    if len(candidates) > MAX_FILES_PER_RESPONSE:
        raise drive_folder.FolderError(
            "בתיקייה %r יש %d קבצים, והמערכת בודקת עד %d בכל שליחה. "
            "אפשר לפצל אותם לכמה תיקיות ולשלוח את הטופס פעם אחת לכל תיקייה."
            % (meta.get("name"), len(candidates), MAX_FILES_PER_RESPONSE))

    work, context_docs, pages = [], [], 0
    for f in candidates:
        data = drive_folder.download(drive_service, f["id"])
        entry = {**f, "data": data}
        typed, why = drive_folder.looks_typed(data, f["ext"])
        if typed:
            log.info("%s %r is a context document, not student work — %s",
                     tag, f["name"], why)
            context_docs.append(entry)
            continue
        pages += drive_folder.page_count(data, f["ext"])
        work.append(entry)

    if not work:
        raise drive_folder.FolderError(
            "כל הקבצים בתיקייה %r נראים כמסמכים מודפסים (דף הבחינה, הוראות או "
            "מחוון) ולא כעבודות תלמידים בכתב יד. יש להוסיף לתיקייה את הסריקות "
            "של העבודות." % meta.get("name"))

    log.info("%s folder %r: %d work file(s) / %d page(s), %d context document(s), "
             "%d file(s) in unsupported formats",
             tag, meta.get("name"), len(work), pages, len(context_docs),
             len(wrong_format))
    return meta, work, context_docs, wrong_format


def _select_works(work, params, tag):
    """(works to grade now, already graded, deferred to a later submission).

    Order of operations matters and is not obvious. The ledger filter runs
    BEFORE the MAX_WORKS_PER_FOLDER cut, so the cap applies to what is still
    outstanding rather than to the folder as a whole. Cutting first would mean a
    resubmission re-selects the same three files, finds all three already done,
    and grades nothing — the folder would be permanently stuck at three.
    """
    pending, already = [], 0
    for entry in work:
        if _file_already_graded(file_key(params, entry)):
            log.debug("%s %r already graded — skipping", tag, entry["name"])
            already += 1
        else:
            pending.append(entry)

    deferred = []
    if MAX_WORKS_PER_FOLDER and len(pending) > MAX_WORKS_PER_FOLDER:
        deferred = pending[MAX_WORKS_PER_FOLDER:]
        pending = pending[:MAX_WORKS_PER_FOLDER]
        log.info("%s grading %d of %d outstanding work file(s) — "
                 "MAX_WORKS_PER_FOLDER is %d; the rest are left for a "
                 "resubmission", tag, len(pending), len(pending) + len(deferred),
                 MAX_WORKS_PER_FOLDER)
    return pending, already, deferred


def _process_folder(drive_service, gmail_service, params, tag) -> str:
    """Grade one submission's folder end to end. Returns an outcome for the ledger.

    Every per-file failure is caught and recorded rather than raised: one corrupt
    scan in a class of thirty must not cost the other twenty-nine their reports,
    and the teacher is told exactly which file failed in the summary mail.
    """
    meta, work, context_docs, wrong_format = _collect_folder(drive_service, params, tag)
    pending, already, deferred = _select_works(work, params, tag)

    # Counted over the selected works only, so a big folder is never refused
    # over pages we were never going to read.
    pages = sum(drive_folder.page_count(e["data"], e["ext"]) for e in pending)
    if pages > MAX_PAGES_PER_RESPONSE:
        raise drive_folder.FolderError(
            "העבודות שנבחרו לבדיקה בתיקייה %r מכילות %d עמודים, והמערכת בודקת "
            "עד %d בכל שליחה. אפשר לפצל אותן לכמה תיקיות ולשלוח את הטופס פעם "
            "אחת לכל תיקייה." % (meta.get("name"), pages, MAX_PAGES_PER_RESPONSE))

    ctx = checker.build_context(params, context_docs, tag=tag)
    question = checker.build_question(params, ctx)
    out_folder_id = drive_folder.ensure_output_folder(
        drive_service, params["folder_id"])

    results = {
        "graded": [], "failed": [], "already": already,
        "deferred": [e["name"] for e in deferred],
        "skipped": [(f["name"], f["reason"]) for f in wrong_format],
        "context": ctx.context_docs,
        "task_known": bool(ctx.instructions or params.get("instructions")),
    }

    for entry in pending:
        fkey = file_key(params, entry)
        try:
            docx_bytes, title = checker.check_file(entry, params, ctx, question, tag=tag)
        except Exception as e:
            log.exception("%s could not grade %r", tag, entry["name"])
            results["failed"].append((entry["name"], f"{type(e).__name__}: {str(e)[:200]}"))
            continue

        try:
            report_id = drive_folder.write_report(
                drive_service, out_folder_id, title, docx_bytes)
        except Exception as e:
            log.warning("%s could not write %r into the teacher's folder (%s: %s)",
                        tag, title, type(e).__name__, str(e)[:200])
            report_id = _fallback_write(drive_service, title, docx_bytes, tag)
            if not report_id:
                results["failed"].append((entry["name"], "הדוח נוצר אך לא ניתן היה לשמור אותו"))
                continue
        results["graded"].append(entry["name"])
        _mark_file_done(fkey, params["folder_id"], entry["id"], report_id)

    if gmail_service is not None:
        mailer.send_summary(gmail_service, WORKSPACE_SUBJECT, params, results,
                            ctx.rubric_name, drive_folder.folder_link(out_folder_id),
                            tag)
    else:
        log.warning("%s no mail scope — %d report(s) are in the teacher's folder but "
                    "they have not been told", tag, len(results["graded"]))

    log.info("%s done: %d graded, %d already done, %d deferred, %d failed, %d skipped",
             tag, len(results["graded"]), results["already"],
             len(results["deferred"]), len(results["failed"]),
             len(results["skipped"]))
    if results["graded"]:
        return "graded"
    return "nothing-new" if results["already"] else "no-results"


def run_form_scan(dry_run: bool = False):
    """One pass over the responses sheet."""
    if not RESPONSES_SHEET_ID:
        raise RuntimeError(
            "RESPONSES_SHEET_ID is not set — nothing to poll. Set it to the id of the "
            "Form's linked responses spreadsheet (the token between /d/ and /edit in "
            "its URL) via --set-env-vars in deploy.sh.")

    if not ALLOWED_DOMAINS and not PASSPHRASE:
        log.warning("no FORM_CHECKER_ALLOWED_DOMAINS and no FORM_CHECKER_PASSPHRASE — "
                    "anyone with the form link can spend model budget. Fine for a closed "
                    "pilot; set one of them before the link circulates.")

    creds = get_workspace_credentials()
    sheets_service = build('sheets', 'v4', credentials=creds)
    drive_service = build('drive', 'v3', credentials=creds)
    gmail_service = (build('gmail', 'v1', credentials=creds)
                     if _extra_scopes_granted else None)

    header, rows = _read_responses(sheets_service)
    if not header:
        log.info("responses sheet is empty — nothing to do")
        return
    colmap = form_schema.map_columns(header)
    log.debug("column map: %s", colmap)
    log.debug("%d response row(s) in the sheet", len(rows))

    processed = 0
    for row_num, row in enumerate(rows, start=2):   # +2: header is row 1
        if processed >= MAX_ROWS_PER_RUN:
            log.info("reached MAX_ROWS_PER_RUN (%d) — the rest will be picked up by the "
                     "next poll", MAX_ROWS_PER_RUN)
            break

        params = form_schema.parse_row(row, colmap)
        key = row_key(params)
        tag = f"[row {key[:8]}]"

        if dry_run:
            print(f"\n── sheet row {row_num}  key={key[:8]} ──")
            for k in ("timestamp", "email", "teacher_name", "module", "rubric_id",
                      "folder_id", "exercise_lang", "model_key", "comments"):
                print(f"   {k:18}: {params.get(k)!r}")
            print(f"   {'question':18}: {form_schema.build_question(params)[:200]!r}")
            refusal = _admit(params, tag)
            print(f"   {'admitted':18}: {'yes' if refusal is None else refusal}")
            continue

        if not _try_claim_row(key):
            continue
        processed += 1

        refusal = _admit(params, tag)
        if refusal is not None:
            if refusal.startswith("__silent__"):
                log.info("%s not graded: %s", tag, refusal.replace("__silent__ ", ""))
            else:
                log.info("%s not graded: %s", tag, refusal)
                if gmail_service is not None:
                    mailer.send_failure(gmail_service, WORKSPACE_SUBJECT, params, refusal, tag)
            _mark_done(key, outcome="rejected")
            continue

        log.info("%s grading folder %s from %s (sheet row %d)",
                 tag, params["folder_id"], params["email"], row_num)
        try:
            outcome = _process_folder(drive_service, gmail_service, params, tag)
        except drive_folder.FolderError as e:
            # A submission we understand but cannot use — a folder shared
            # read-only, the wrong link, nothing gradable inside. The teacher's
            # problem to fix, so tell them rather than retrying it three times.
            log.info("%s not graded: %s", tag, e)
            if gmail_service is not None:
                mailer.send_failure(gmail_service, WORKSPACE_SUBJECT, params, str(e), tag)
            _mark_done(key, outcome="rejected")
            continue

        _mark_done(key, outcome=outcome)

    log.debug("scan finished — %d row(s) processed this run", processed)


@functions_framework.http
def process_form_responses(request):
    """Cloud Run entry point, called by Cloud Scheduler.

    Re-raises after logging: Scheduler only ever sees the status code, so a
    failure that is not logged here is a failure nobody can diagnose."""
    try:
        run_form_scan()
        return "ok", 200
    except Exception:
        log.exception("form scan failed")
        raise


if __name__ == "__main__":
    run_form_scan(dry_run="--dry-run" in sys.argv)
