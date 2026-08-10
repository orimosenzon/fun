"""main.py — demo-checker: one uploaded exercise in, one graded report by email.

A teacher we have never met uploads a single scan to the "Bdika Demo- English"
Google Form. A few minutes later the graded report arrives in their inbox as a
Word attachment, with a link to a short feedback survey. That loop — try it,
see it, tell us — is the product; the grading is the same engine as everywhere
else in haskala.

Cloud Scheduler calls this every 5 minutes. Each run reads the Form's responses
spreadsheet, claims any row it has not already handled, grades the uploaded file
with the shared haskala_grading pipeline, and emails the report back.

WHY THIS IS NOT form-checker
────────────────────────────
form-checker asks for a link to a shared Drive folder, which is right for a
teacher grading a whole class: results land where they already keep the work,
and thirty submissions become one. It is wrong for a first contact. Creating a
folder, sharing it with an address on a domain they have never heard of, and
granting Editor is three acts of trust before seeing a single thing the tool
can do. So this form asks for a file, and the report leaves by mail.

THE DIFFERENCE THAT SHAPES THE CODE
───────────────────────────────────
In the folder flow, mail is a courtesy: the report is already sitting in the
teacher's folder, and a Gmail failure costs a notification. Here mail is the
only channel that exists. A delivery failure is a total failure of the
submission, so _process_upload does not mark such a row done — it leaves it for
the ledger to retry, up to the three-attempt cap.

THE ONE HARD PREREQUISITE
─────────────────────────
Forms stores uploaded files in the Drive of whoever owns the *form*. This form
is owned by exam@bdika.net and the service reads as ori@bdika.net, which works
because both are inside the bdika.net Workspace and the containing "Bdika forms"
folder is shared. A form rebuilt under an outside account puts every upload
somewhere this service cannot see, and there is no fix short of rebuilding it.

The form also needs "allow responses from outside the organization" turned on,
which for this product is the entire point rather than a detail.
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
import form_schema
import mailer
import sheet_link
import upload

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

log = logging.getLogger("demo-checker")


# ─── configuration ─────────────────────────────────────────────────────────

# The Form's linked responses spreadsheet. Set at deploy time; the id is the
# long token in the sheet's URL between /d/ and /edit.
RESPONSES_SHEET_ID = os.environ.get("RESPONSES_SHEET_ID", "")

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# Where a graded report is parked when it could not be emailed. This is NOT a
# delivery path — the teacher never sees this folder — it is so that a report we
# already paid for survives long enough to be sent by hand or diagnosed. The
# submission is still treated as failed and retried; see _process_upload.
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
# THE ONE THAT BOUNDS THE BILL.
#
# A demo is one piece of work, so the cap is far tighter here than in
# form-checker, where a folder is a whole class. This is also the product whose
# link goes to people we have never met, which is exactly the population that
# should not be able to submit a 300-page PDF.
#
# Unlike the folder flow, going over is a *refusal* rather than a deferral:
# there is no folder to come back to and grade the rest of, so "we did some of
# it" has nothing to attach to. The teacher is told which cap they hit.
MAX_PAGES_PER_RESPONSE = int(os.environ.get("MAX_PAGES_PER_RESPONSE", "4"))

# Files per response. The form asks for one, but a File upload question can be
# configured to accept several, and the setting lives on Avishai's side of the
# fence rather than ours. Grading is capped at the first file either way — the
# rest are named in the failure notice so nothing disappears silently.
MAX_FILES_PER_RESPONSE = int(os.environ.get("MAX_FILES_PER_RESPONSE", "1"))

# Rows processed in one invocation. A backlog is drained over several polls
# rather than in one run that blows the Scheduler's 600s attempt deadline and
# gets retried from the top — the ledger makes the partial progress safe.
MAX_ROWS_PER_RUN = int(os.environ.get("MAX_ROWS_PER_RUN", "5"))

# Who may spend our model budget. Both are unset by default, which allows
# anyone with the form link — fine for a closed pilot, and logged loudly at
# startup so it is a decision rather than an oversight.
#   DEMO_CHECKER_ALLOWED_DOMAINS — comma-separated, e.g. "bdika.net,taded.org.il"
#   DEMO_CHECKER_PASSPHRASE — a short code on the form, shared with the teachers
#
# Note this product is the one where an open door is closest to intentional: the
# whole point is reaching teachers we do not know yet, so a domain allowlist
# defeats it. MAX_PAGES_PER_RESPONSE is what is really holding the line here,
# and it is set to 4 for that reason.
ALLOWED_DOMAINS = [d.strip().lower() for d in
                   os.environ.get("DEMO_CHECKER_ALLOWED_DOMAINS", "").split(",") if d.strip()]
PASSPHRASE = os.environ.get("DEMO_CHECKER_PASSPHRASE", "").strip()


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
        ",".join(params.get("file_ids", [])),
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


# form-checker keeps a SECOND ledger at file level, because there a response is
# a shared folder: the folder is live, the teacher adds five scans and submits
# again, and without it that resubmission re-grades and re-pays for all thirty
# originals. An upload is frozen at the moment of submission — one response is
# one file, forever — so the row ledger above is sufficient and the file ledger
# is deliberately absent rather than merely unused.


def _try_claim_row(key: str) -> bool:
    """True if this row should be processed now — never handled, or an
    abandoned claim worth retrying — and False if it should be skipped."""
    doc_ref = _firestore().collection("demo_checker_responses").document(key)
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
    _firestore().collection("demo_checker_responses").document(key).set({
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

# Writing the report link back into the responses sheet. Its OWN tier, not
# folded into EXTRA_SCOPES, and that separation is the whole point: delegation
# is all-or-nothing per token request, so bundling an ungranted write scope with
# gmail.send would take mail down with it — and mail is the only channel that
# reaches the teacher. Asking separately means a missing write scope costs a
# link in a spreadsheet and nothing else.
SHEET_WRITE_SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
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
#
# exam@bdika.net rather than ori@ for this product, on two counts. Avishai's
# spec says the mail comes from exam@, which is the address printed on the form
# and the one a teacher would reply to — a personal address there is impossible
# to change later without breaking every form in circulation. And exam@ is the
# account that OWNS the form, so it owns the uploads outright rather than
# reaching them through a folder share that breaks silently if the form is ever
# moved. Verified impersonable on 2026-08-10.
WORKSPACE_SUBJECT = os.environ.get("WORKSPACE_SUBJECT", "exam@bdika.net")

# What teachers are told to share with. Normally the same address; kept separate
# because a Workspace alias or group can front the real mailbox. drive_folder
# reads it from the environment so its teacher-facing messages name it too.
SHARE_WITH = os.environ.get("SHARE_WITH", WORKSPACE_SUBJECT)
os.environ["SHARE_WITH"] = SHARE_WITH

# Every report is also shared with this whole domain as editor. Set empty to
# turn that off.
TEAM_DOMAIN = os.environ.get("TEAM_DOMAIN", "bdika.net").strip()
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


_sheet_write_granted: bool | None = None
_sheet_write_checked_at: float = 0.0


def get_sheet_write_credentials():
    """Credentials that may write the responses sheet, or None if not granted.

    Deliberately a separate token request from get_workspace_credentials — see
    SHEET_WRITE_SCOPES. Same recheck-on-a-timer shape as the mail scope, so an
    admin adding the scope takes effect without a redeploy."""
    global _sheet_write_granted, _sheet_write_checked_at

    stale = (time.monotonic() - _sheet_write_checked_at) >= EXTRA_RECHECK_SECONDS
    if _sheet_write_granted is False and not stale:
        return None

    creds = _delegated_credentials(BASE_SCOPES + SHEET_WRITE_SCOPES)
    was = _sheet_write_granted
    try:
        creds.refresh(Request())
    except RefreshError as e:
        _sheet_write_granted = False
        _sheet_write_checked_at = time.monotonic()
        if was is not False:
            log.warning(
                "delegation does not grant the sheet-write scope (%s) — grading, "
                "Drive and mail are unaffected; the only thing missing is the "
                "report link in the responses sheet. To enable it, add %s to the "
                "service account's domain-wide delegation in the Workspace admin "
                "console (client id %s). Rechecked every %ds, so no redeploy is "
                "needed once it is there.",
                str(e)[:200], ", ".join(SHEET_WRITE_SCOPES), SA_CLIENT_ID,
                EXTRA_RECHECK_SECONDS)
        return None

    _sheet_write_granted = True
    _sheet_write_checked_at = time.monotonic()
    if was is not True:
        log.info("delegation grants the sheet-write scope — report links will be "
                 "written to the responses sheet")
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
            log.warning("%s rejecting %s — domain %r is not in DEMO_CHECKER_ALLOWED_DOMAINS",
                        tag, params["email"], domain)
            return "__silent__ domain not allowed"

    if PASSPHRASE and params.get("passphrase", "").strip() != PASSPHRASE:
        log.warning("%s rejecting %s — wrong or missing passphrase", tag, params["email"])
        return "__silent__ bad passphrase"

    # The missing-file case is handled in _collect_upload rather than here, so
    # that the teacher gets the same explain-and-retry mail as every other
    # fixable problem instead of a silent drop.
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


def _park_report(drive_service, name, docx_bytes, tag) -> str:
    """Keep a graded report that could not be emailed.

    Explicitly NOT a delivery path: the teacher has no access to this folder and
    will never see what lands in it. It exists so that a report we have already
    paid model calls for can be sent by hand or inspected, instead of being
    thrown away because Gmail returned a 403. The submission is still counted as
    failed by the caller."""
    if not FALLBACK_RESULTS_FOLDER_ID:
        log.error("%s report %r could not be emailed and no FALLBACK_RESULTS_FOLDER_ID "
                  "is configured — the report is lost. Set that env var.", tag, name)
        return ""
    media = MediaIoBaseUpload(io.BytesIO(docx_bytes), mimetype=_DOCX_MIME)
    uploaded = drive_service.files().create(
        body={"name": name, "parents": [FALLBACK_RESULTS_FOLDER_ID]},
        media_body=media, fields="id").execute()
    log.warning("%s report %r parked in the fallback folder (id=%s) — the teacher "
                "cannot see it there", tag, name, uploaded["id"])
    return uploaded["id"]


def _collect_upload(drive_service, params, tag) -> dict:
    """The one file to grade, fetched and within the caps.

    Raises upload.UploadError with text written for the teacher — every branch
    here is something they can fix and resubmit, so every branch has to say how.
    """
    file_ids = params.get("file_ids") or []
    if not file_ids:
        raise upload.UploadError(
            "No file was attached to your submission. Please fill in the form "
            "again and upload a scan or photo of the work.")

    if len(file_ids) > MAX_FILES_PER_RESPONSE:
        log.info("%s response carries %d files, grading the first %d",
                 tag, len(file_ids), MAX_FILES_PER_RESPONSE)

    entry = upload.fetch(drive_service, file_ids[0])

    pages = upload.page_count(entry["data"], entry["ext"])
    if pages > MAX_PAGES_PER_RESPONSE:
        raise upload.UploadError(
            f"The file you uploaded ({entry['name']}) has {pages} pages, and "
            f"this demo grades up to {MAX_PAGES_PER_RESPONSE} per submission. "
            "Please upload a single piece of student work and submit again.")
    entry["pages"] = pages

    if len(file_ids) > MAX_FILES_PER_RESPONSE:
        entry["ignored"] = len(file_ids) - MAX_FILES_PER_RESPONSE
    return entry


def _publish_report(drive_service, entry, filename, docx_bytes, params, tag) -> str:
    """Put the graded report in Drive as an editable Doc and share it.

    Returns its link, or "" if Drive could not be written at all. Since the mail
    stopped carrying an attachment this link IS the delivery, so an empty return
    is a failed submission rather than a degraded one — the caller treats it as
    such. The sharing calls below are still best-effort: a report the teacher
    cannot open is worth retrying, a report they can open but we cannot is not."""
    parent = entry.get("parent")
    if not parent:
        log.warning("%s the uploaded file reports no parent folder — nowhere to "
                    "put the report", tag)
        return ""
    try:
        out_folder = upload.ensure_output_folder(drive_service, parent)
        doc_id = upload.write_report(drive_service, out_folder, filename, docx_bytes)
    except Exception as e:
        log.warning("%s could not write the report into Drive (%s: %s)",
                    tag, type(e).__name__, str(e)[:200])
        return ""

    # The submitter first: this is the copy they are meant to correct.
    if params.get("email"):
        upload.share(drive_service, doc_id, params["email"], role="writer")
    # Then the whole team, per Avishai's spec — the domain rather than one
    # mailbox, so anyone here can open a report a teacher writes in about, not
    # only the account that happened to create it.
    if TEAM_DOMAIN:
        upload.share_with_domain(drive_service, doc_id, TEAM_DOMAIN, role="writer")
    return upload.doc_link(doc_id)


def _process_upload(drive_service, gmail_service, params, tag) -> tuple[str, str]:
    """Grade one uploaded file, publish it, and mail it back.

    Returns (ledger outcome, report link). Raises upload.UploadError for
    anything the teacher can fix. The outcome is "undelivered" when grading
    worked but the mail did not — the caller leaves such a row unfinished so the
    next poll retries it, because an ungraded submission and an undelivered one
    look identical from the teacher's side.
    """
    entry = _collect_upload(drive_service, params, tag)

    # No folder means no context documents: whatever the teacher typed into
    # "Assignment description" is the only description of the task there will
    # ever be. build_context still runs so the NO_TASK_NOTICE path is identical
    # to form-checker's, which is what keeps the model from inventing a topic.
    ctx = checker.build_context(params, [], tag=tag)
    question = checker.build_question(params, ctx)

    docx_bytes, _title = checker.check_file(entry, params, ctx, question, tag=tag)
    filename = upload.report_name(entry["name"], params.get("timestamp", ""))

    link = _publish_report(drive_service, entry, filename, docx_bytes, params, tag)

    if not link:
        # Since the attachment was dropped, the link IS the delivery. Sending
        # Avishai's "your checked document is ready" over a document that was
        # never written would be worse than silence — the teacher would click
        # into nothing and conclude the tool is broken. Park it and retry.
        log.error("%s graded %r but the Drive copy could not be written, and the "
                  "mail has nothing to point at — not sending", tag, entry["name"])
        _park_report(drive_service, filename, docx_bytes, tag)
        return "undelivered", ""

    if gmail_service is None:
        log.error("%s graded %r but there is no mail scope, and mail is the only "
                  "way anything reaches this teacher. Add gmail.send to the "
                  "domain-wide delegation entry.", tag, entry["name"])
        return "undelivered", link

    sent = mailer.send_report(gmail_service, WORKSPACE_SUBJECT, params, link, tag)
    if not sent:
        return "undelivered", link

    log.info("%s graded %r (%d page(s)), emailed %s%s",
             tag, entry["name"], entry.get("pages", 1), params["email"],
             " and published to Drive" if link else "")
    return "graded", link



def run_form_scan(dry_run: bool = False):
    """One pass over the responses sheet."""
    if not RESPONSES_SHEET_ID:
        raise RuntimeError(
            "RESPONSES_SHEET_ID is not set — nothing to poll. Set it to the id of the "
            "Form's linked responses spreadsheet (the token between /d/ and /edit in "
            "its URL) via --set-env-vars in deploy.sh.")

    if not ALLOWED_DOMAINS and not PASSPHRASE:
        log.info("no DEMO_CHECKER_ALLOWED_DOMAINS and no DEMO_CHECKER_PASSPHRASE — "
                 "the form is open to anyone with the link, which is what this "
                 "product is for. Spend is bounded by MAX_PAGES_PER_RESPONSE=%d.",
                 MAX_PAGES_PER_RESPONSE)

    creds = get_workspace_credentials()
    sheets_service = build('sheets', 'v4', credentials=creds)
    drive_service = build('drive', 'v3', credentials=creds)
    gmail_service = (build('gmail', 'v1', credentials=creds)
                     if _extra_scopes_granted else None)

    write_creds = get_sheet_write_credentials()
    sheets_write = build('sheets', 'v4', credentials=write_creds) if write_creds else None

    header, rows = _read_responses(sheets_service)
    if not header:
        log.info("responses sheet is empty — nothing to do")
        return
    colmap = form_schema.map_columns(header)
    log.debug("column map: %s", colmap)
    log.debug("%d response row(s) in the sheet", len(rows))

    # Located once per scan rather than per row: it is the same column for every
    # row, and on the first ever run it also writes the header.
    link_col = (sheet_link.find_or_create_column(sheets_write, RESPONSES_SHEET_ID, header)
                if sheets_write else None)

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
                      "file_ids", "instructions", "exercise_lang", "model_key",
                      "comments"):
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

        log.info("%s grading upload from %s (sheet row %d)",
                 tag, params["email"], row_num)
        try:
            outcome, link = _process_upload(drive_service, gmail_service, params, tag)
        except upload.UploadError as e:
            # A submission we understand but cannot use — a format we cannot
            # read, too many pages, a file that has since been deleted. All of
            # them are the teacher's to fix, so tell them rather than burning
            # three retries on something that will fail identically each time.
            log.info("%s not graded: %s", tag, e)
            if gmail_service is not None:
                mailer.send_failure(gmail_service, WORKSPACE_SUBJECT, params, str(e), tag)
            _mark_done(key, outcome="rejected")
            continue

        if outcome == "undelivered":
            # Deliberately NOT marked done. Mail is the only delivery channel
            # here, so an unsent report is an unfinished submission; leaving the
            # claim to lapse lets the next poll retry it, and the ledger's
            # three-attempt cap stops that becoming a loop. The regrade costs
            # another set of model calls, which is the right trade against a
            # teacher who submitted and heard nothing back.
            log.error("%s graded but not delivered — leaving the row for retry", tag)
            continue

        if link and sheets_write and link_col is not None:
            sheet_link.write_link(sheets_write, RESPONSES_SHEET_ID, row_num,
                                  link_col, link, tag)

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
