from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.discovery import build

# Load the service account credentials
SERVICE_ACCOUNT_FILE = './cred5.json'
SCOPES = ['https://www.googleapis.com/auth/documents', 'https://www.googleapis.com/auth/drive.file']

creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Build the Google Docs and Google Drive service
docs_service = build('docs', 'v1', credentials=creds)
drive_service = build('drive', 'v3', credentials=creds)

# Create a new Google Doc
document = {
    'title': 'Created_by_python'
}
doc = docs_service.documents().create(body=document).execute()
doc_id = doc.get('documentId')

print(f'Created document with ID: {doc_id}')

# Define the text to be added
requests = [
    {
        'insertText': {
            'location': {
                'index': 1,
            },
            'text': 'I made this doc with python. chatgpt helped me.'
        }
    }
]

# Add text to the document
result = docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()

print('Text added successfully!')



# # After creating the document, add this code:

def share_document(drive_service, file_id, user_email):
    drive_service.permissions().create(
        fileId=file_id,
        body={'type': 'user', 'role': 'writer', 'emailAddress': user_email},
        fields='id'
    ).execute()

share_document(drive_service, doc_id, 'orimosenzon@gmail.com')

print(f'Document shared with your personal account.')


