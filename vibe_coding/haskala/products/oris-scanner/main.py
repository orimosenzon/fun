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
# זו התיקון הארכיטקטוני המרכזי מול scan2: שם, run_workspace_scan() עיבד מחדש
# את כל ההגשות בכל טריגר (אין דה-דופליקציה), וכל redelivery של Pub/Sub (בגלל
# ack deadline קצר מדי) הריץ סריקה מלאה נוספת — שרשרת שרפה מאות אלפי בקשות.
#
# כאן: לפני עיבוד הגשה, "תופסים" אותה אטומית ב-Firestore (create נכשל אם כבר
# קיימת). אם היא כבר "done" — מדלגים (זה ה-dedup עצמו). אם היא "in_progress"
# וטרייה — כנראה instance אחר כבר מטפל בה עכשיו, מדלגים. אם "in_progress" אבל
# ישנה (claim תקוע מריצה שקרסה) — לוקחים אותה מחדש כדי לאפשר retry אמיתי.
_STALE_CLAIM_MINUTES = 20  # מעבר לזה, claim תקוע נחשב נטוש וניתן לתפיסה מחדש

_db = None


def _firestore():
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


def _try_claim_submission(subm_id: str) -> bool:
    """מחזיר True אם יש לעבד את ההגשה עכשיו (או שהיא לא טופלה, או ש-claim ישן
    ננטש), False אם יש לדלג (כבר done, או instance אחר עובד עליה כרגע)."""
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
                return False  # instance אחר עובד על זה עכשיו, לא נטוש
        # claim ישן/נטוש — תופסים מחדש כדי לאפשר retry אמיתי
        transaction.set(doc_ref, {"status": "in_progress", "claimed_at": now})
        return True

    return _txn(_firestore().transaction())


def _mark_done(subm_id: str):
    doc_ref = _firestore().collection("oris_scanner_submissions").document(subm_id)
    doc_ref.set({
        "status": "done",
        "completed_at": datetime.datetime.now(datetime.timezone.utc),
    }, merge=True)


# ─── Google Workspace access (זהה ל-scan2: אותו service account + האצלה) ──────

def get_workspace_credentials():
    SCOPES = [
        'https://www.googleapis.com/auth/classroom.courses.readonly',
        'https://www.googleapis.com/auth/classroom.coursework.students',
        'https://www.googleapis.com/auth/drive'
    ]
    source_creds, project_id = google.auth.default()
    source_creds.refresh(Request())
    sa_email = "sainter@master-gecko-500709-t0.iam.gserviceaccount.com"
    teacher_email = "yaron@bdika.net"
    signer = iam.Signer(Request(), source_creds, sa_email)
    delegated_creds = service_account.Credentials(
        signer,
        sa_email,
        token_uri="https://oauth2.googleapis.com/token",
        subject=teacher_email,
        scopes=SCOPES
    )
    return delegated_creds


# TODO: כרגע אותה תיקיית תוצאות של scan2 (מתמטיקה) — שווה לשקול תיקייה נפרדת
# לדוחות שפה כדי לא לערבב את הדוחות של שני המקצועות אצל ירון.
RESULTS_FOLDER_ID = '1kkHROa7DlrehNOFvqkwAEQtvTsWKwNS8'


def run_workspace_scan():
    creds = get_workspace_credentials()
    drive_service = build('drive', 'v3', credentials=creds)
    classroom_service = build('classroom', 'v1', credentials=creds)

    res_courses = classroom_service.courses().list(pageSize=100).execute()
    courses = res_courses.get('courses', [])
    for course in courses:
        res_cw = classroom_service.courses().courseWork().list(courseId=course.get('id')).execute()
        course_work = res_cw.get('courseWork', [])
        for cw in course_work:
            res_sub = classroom_service.courses().courseWork().studentSubmissions().list(
                courseId=course.get('id'), courseWorkId=cw.get('id')).execute()
            subms = res_sub.get('studentSubmissions', [])
            for subm in subms:
                subm_id = subm.get('id')
                if not _try_claim_submission(subm_id):
                    continue  # dedup: כבר טופלה, או instance אחר מטפל בה כרגע

                file_ids = []
                attachments = (subm.get('assignmentSubmission') or {}).get('attachments', [])
                for attachment in attachments:
                    drive_file = attachment.get('driveFile')
                    if drive_file:
                        file_ids.append(drive_file.get('id'))

                checker.check_hw(drive_service, file_ids, RESULTS_FOLDER_ID)
                _mark_done(subm_id)


@functions_framework.cloud_event
def process_my_drive_files(cloud_event):
    run_workspace_scan()


if __name__ == '__main__':
    run_workspace_scan()
