import base64
import functions_framework

import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.auth import iam
from googleapiclient.discovery import build

import checker

def get_workspace_credentials():
    # Scopes required to interact with Google Workspace APIs
    SCOPES = [
        'https://www.googleapis.com/auth/classroom.courses.readonly',
        'https://www.googleapis.com/auth/classroom.coursework.students',
        'https://www.googleapis.com/auth/drive'
    ]
    source_creds, project_id = google.auth.default()
    print("here XXXXXXXXXXXXXXXXXXXXXX")
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
    print(" there OOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
    return delegated_creds

def append_text(service, document_id, text):
    # Fetch the document structure to find the absolute end index
    doc = service.documents().get(documentId=document_id).execute()
    end_index = doc.get('body').get('content')[-1].get('endIndex') - 1

    # Format and execute the required batchUpdate payload internally
    requests = [{
        'insertText': {
            'location': {'index': end_index},
            'text': text
        }
    }]
    service.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()

def run_workspace_scan():
    # Generate the credentials
    creds = get_workspace_credentials()

    # Connect directly to the Google Drive and Classroom APIs
    drive_service = build('drive', 'v3', credentials=creds)
    doc_service = build('docs', 'v1', credentials=creds)
    classroom_service = build('classroom', 'v1', credentials=creds)
   # res1 = drive_service.files().list(pageSize=10).execute()


    res_courses = classroom_service.courses().list(pageSize=100).execute()
    courses = res_courses.get('courses', [])
    for course in courses:
            res_cw = classroom_service.courses().courseWork().list(courseId=course.get('id')).execute()
            course_work = res_cw.get('courseWork', [])
            for cw in course_work:
                #create or get a folder for the results
                folder_id = '1kkHROa7DlrehNOFvqkwAEQtvTsWKwNS8'
                res_sub = classroom_service.courses().courseWork().studentSubmissions().list(courseId=course.get('id'), courseWorkId=cw.get('id')).execute()
                subms = res_sub.get('studentSubmissions', [])
                for subm in subms:
                    # check if we checked this submission
                    file_ids = []
                    attachmets = subm.get('assignmentSubmission').get('attachments', [])
                    for attachmet in attachmets:
                        file_ids.append(attachmet.get('driveFile').get('id'))
                    checker.check_hw(drive_service,file_ids,folder_id)
                    append_text(service=doc_service, document_id='1BauqmJhsMiUe-hgN8htDHVo9t2av7wI5KxCGD0HcuoI', text="checked submission" + subm.get('id') + "  .\n")


@functions_framework.cloud_event
def process_my_drive_files(cloud_event):
    run_workspace_scan()

if __name__ == '__main__':
    run_workspace_scan()
