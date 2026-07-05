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
import os

from googleapiclient.discovery import build

# האימות והעזרים המשותפים חיים ב-gclass.py (משותף עם poller.py)
from gclass import (SCOPES_RO, download_drive_file, find_course,
                    get_credentials, roster_names)

HERE = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(HERE, "downloads")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", help="שם הקורס (ללא = להדפיס רשימה)")
    ap.add_argument("--download", action="store_true", help="להוריד קבצים מצורפים ל-downloads/")
    args = ap.parse_args()

    creds = get_credentials(SCOPES_RO)
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
