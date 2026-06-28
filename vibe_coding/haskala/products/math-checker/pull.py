#!/usr/bin/env python3
"""
pull.py — POC לאינטגרציית Google Classroom (bdika.net sandbox).

מטרה: להוכיח את הצד של גוגל מקצה לקצה, לוקאלית, בעלות 0:
  1. אימות OAuth (משתמש = מורה, אתה).
  2. מציאת קורס לפי שם.
  3. מעבר על coursework → studentSubmissions.
  4. הורדת הקובץ המצורף (Drive) של כל הגשה שהוגשה (TURNED_IN/RETURNED).

הרצה ראשונה:
    ./venv/bin/python pull.py --course integration

דורש credentials.json (OAuth client "Desktop app") בתיקייה זו.
בהרצה הראשונה ייפתח דפדפן לאישור; ייווצר token.json לפעמים הבאות.
"""
from __future__ import annotations

import argparse
import io
import os
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# scopes ל-POC קריאה בלבד. שינוי scope ⇐ מחיקת token.json ואישור מחדש.
SCOPES = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.coursework.students.readonly",
    "https://www.googleapis.com/auth/classroom.rosters.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

HERE = os.path.dirname(os.path.abspath(__file__))
CREDS_FILE = os.path.join(HERE, "credentials.json")
TOKEN_FILE = os.path.join(HERE, "token.json")
DOWNLOAD_DIR = os.path.join(HERE, "downloads")


def get_credentials() -> Credentials:
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDS_FILE):
                sys.exit(
                    f"חסר {CREDS_FILE}.\n"
                    "צור OAuth client מסוג 'Desktop app' ב-GCP console, "
                    "הורד את ה-JSON ושמור אותו בשם credentials.json בתיקייה זו."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return creds


def find_course(classroom, name: str | None) -> dict:
    courses = classroom.courses().list(courseStates=["ACTIVE"]).execute().get("courses", [])
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


def download_drive_file(drive, file_id: str, dest_path: str) -> None:
    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", help="שם הקורס (ללא = להדפיס רשימה)")
    ap.add_argument("--download", action="store_true", help="להוריד קבצים מצורפים ל-downloads/")
    args = ap.parse_args()

    creds = get_credentials()
    classroom = build("classroom", "v1", credentials=creds)
    drive = build("drive", "v3", credentials=creds)

    course = find_course(classroom, args.course)
    cid = course["id"]
    print(f"\n=== קורס: {course['name']} (id={cid}) ===")

    names = roster_names(classroom, cid)

    works = classroom.courses().courseWork().list(courseId=cid).execute().get("courseWork", [])
    if not works:
        print("אין coursework בקורס.")
        return

    if args.download:
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for w in works:
        print(f"\n--- תרגיל: {w['title']} (courseWorkId={w['id']}) ---")
        subs = classroom.courses().courseWork().studentSubmissions().list(
            courseId=cid, courseWorkId=w["id"]
        ).execute().get("studentSubmissions", [])
        for s in subs:
            who = names.get(s.get("userId", ""), s.get("userId", "?"))
            state = s.get("state", "?")
            atts = s.get("assignmentSubmission", {}).get("attachments", [])
            print(f"  • {who:25} state={state:12} attachments={len(atts)}")
            for a in atts:
                df = a.get("driveFile")
                if not df:
                    print(f"      (קובץ לא-Drive: {list(a.keys())})")
                    continue
                print(f"      drive: {df['title']}  (id={df['id']})")
                if args.download:
                    safe = f"{who}_{df['title']}".replace("/", "_").replace(" ", "_")
                    dest = os.path.join(DOWNLOAD_DIR, safe)
                    download_drive_file(drive, df["id"], dest)
                    print(f"        ↓ נשמר: {dest}")


if __name__ == "__main__":
    main()
