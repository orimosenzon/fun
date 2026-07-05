"""gclass.py — עזרי Google Classroom/Drive משותפים ל-pull.py (POC) ול-poller.py.

אימות OAuth desktop: דורש credentials.json (OAuth client מסוג "Desktop app")
בתיקייה זו. בהרצה הראשונה נפתח דפדפן לאישור; token.json נשמר לפעמים הבאות.
שינוי scopes ⇐ מחיקת token.json ואישור מחדש.
"""
from __future__ import annotations

import io
import os
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload

HERE = os.path.dirname(os.path.abspath(__file__))
CREDS_FILE = os.path.join(HERE, "credentials.json")
TOKEN_FILE = os.path.join(HERE, "token.json")

# קריאה בלבד — מספיק ל-pull.py ול-poller בלי כתיבת ציונים.
SCOPES_RO = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.coursework.students.readonly",
    "https://www.googleapis.com/auth/classroom.rosters.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# כמו RO אבל עם כתיבת ציוני-טיוטה (draftGrade) — ל-poller --push-grade.
SCOPES_RW = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.coursework.students",
    "https://www.googleapis.com/auth/classroom.rosters.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def get_credentials(scopes: list[str],
                    token_file: str = TOKEN_FILE,
                    creds_file: str = CREDS_FILE) -> Credentials:
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_file):
                sys.exit(
                    f"חסר {creds_file}.\n"
                    "צור OAuth client מסוג 'Desktop app' ב-GCP console, "
                    "הורד את ה-JSON ושמור אותו בשם credentials.json בתיקייה זו."
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_file, scopes)
            creds = flow.run_local_server(port=0)
        with open(token_file, "w") as f:
            f.write(creds.to_json())
    return creds


def list_courses(classroom) -> list[dict]:
    return classroom.courses().list(courseStates=["ACTIVE"]).execute().get("courses", [])


def find_course(classroom, name: str | None) -> dict:
    courses = list_courses(classroom)
    if not courses:
        sys.exit("לא נמצאו קורסים פעילים עבור החשבון הזה.")
    if name is None:
        print("קורסים פעילים:")
        for c in courses:
            print(f"  - {c['name']}  (id={c['id']})")
        sys.exit("הרץ שוב עם --course <שם>.")
    for c in courses:
        if c["name"].strip().lower() == name.strip().lower():
            return c
    sys.exit(f"לא נמצא קורס בשם '{name}'. קורסים: {[c['name'] for c in courses]}")


def roster_names(classroom, course_id: str) -> dict[str, str]:
    """userId → שם תצוגה של התלמיד."""
    names: dict[str, str] = {}
    page = None
    while True:
        resp = classroom.courses().students().list(
            courseId=course_id, pageToken=page
        ).execute()
        for s in resp.get("students", []):
            names[s["userId"]] = s["profile"]["name"]["fullName"]
        page = resp.get("nextPageToken")
        if not page:
            break
    return names


def download_drive_bytes(drive, file_id: str) -> bytes:
    """הורדת קובץ Drive לזיכרון (bytes) — בלי לגעת בדיסק."""
    request = drive.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def download_drive_file(drive, file_id: str, dest_path: str) -> None:
    with open(dest_path, "wb") as fh:
        fh.write(download_drive_bytes(drive, file_id))
